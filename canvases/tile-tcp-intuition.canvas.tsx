import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Table,
  Select,
  Stack,
  Stat,
  Text,
  Toggle,
  useEffect,
  useHostTheme,
  useMemo,
  useRef,
  useState,
} from "cursor/canvas";

// ---------------------------------------------------------------------------------------------
// Model (per CU). Constants from kernels/gemm/intra_wave/a16w16/v9_tile/WORKLOG.md.
// ---------------------------------------------------------------------------------------------
const MFMA_CYCLES = 16; // one v_mfma_f32_16x16x32 (bf16/fp16)
const REQ_BYTES = 1024; // one buffer_load_dwordx4 per wave: 64 lanes x 16 B
const TCP_ISSUE_CYCLES = 16; // TCP processing per 1 KB request
const FLOP_PER_CLK_CU = 4096;
const L_COLD = 550;
const L_WARM = 385;

type Preset = { id: string; label: string; bm: number; bn: number; bk: number; clock: number };
const PRESETS: Preset[] = [
  { id: "256x256", label: "256x256x64 (v9)", bm: 256, bn: 256, bk: 64, clock: 1.71 },
  { id: "256x128", label: "256x128x64", bm: 256, bn: 128, bk: 64, clock: 1.92 },
  { id: "128x256", label: "128x256x64", bm: 128, bn: 256, bk: 64, clock: 1.88 },
  { id: "128x128", label: "128x128x64", bm: 128, bn: 128, bk: 64, clock: 2.11 },
  { id: "128x128x128", label: "128x128x128", bm: 128, bn: 128, bk: 128, clock: 2.1 },
];

// Measured main-loop MFMA efficiency (ATT), full wave; cold K = 8192, warm K = 4096.
const MEASURED = [
  { tile: "256x256", i: 128, cold: 0.95, warm: 0.984 },
  { tile: "256x128", i: 85.3, cold: 0.643, warm: 0.886 },
  { tile: "128x256", i: 85.3, cold: 0.683, warm: undefined },
  { tile: "128x128", i: 64, cold: 0.448, warm: 0.662 },
  { tile: "128x128x128", i: 64, cold: 0.486, warm: 0.621 },
];

// pipe: how far ahead a CU may load. 0 = serial (load, wait, compute); N = N LDS buffers, k-step j's
// loads may issue once k-step j-N has released its buffer; PIPE_INF = no buffer limit (only the
// TCP slots). relEnd: a buffer is released at the end of its k-step's compute (operands read from
// LDS during the compute) instead of at its start (operands copied to registers up front).
const PIPE_SERIAL = 0;
const PIPE_INF = 99;
type Params = { bm: number; bn: number; bk: number; clock: number; L: number; qKB: number; pipe: number; relEnd: boolean };

function R_of(clock: number) {
  return FLOP_PER_CLK_CU * clock; // FLOP per ns = GFLOP/s
}

function derived(p: Params) {
  // 4 waves, one per SIMD, share the tile's 16x16 output blocks; each block needs BK/32 16x16x32
  // MFMAs per k-step. With 4 or more blocks (BM, BN >= 32 here) each wave gets blocks / 4, which is
  // (BM/2 / 16)(BN/2 / 16); with fewer, some SIMDs have no work.
  const blocks = (p.bm / 16) * (p.bn / 16);
  const active = Math.min(4, blocks);
  const mfmaPerWave = Math.ceil(blocks / active) * (p.bk / 32);
  const simdFrac = active / 4;
  const cycles = mfmaPerWave * MFMA_CYCLES;
  const C = cycles / p.clock; // ns per k-step
  const B = (p.bm + p.bn) * p.bk * 2; // bytes per k-step
  const Q = p.qKB * 1024;
  const need = B / C; // B/ns the MFMAs consume
  const deliver = Q / p.L; // B/ns the window can deliver
  const stepsNeeded = p.L / C;
  const stepsAllowed = Q / B;
  const windowCeil = Math.min(1, deliver / need);
  const issueCeil = Math.min(1, (TCP_BYTES_PER_CLK * p.clock) / need);
  const tIssue = (B / REQ_BYTES) * (TCP_ISSUE_CYCLES / p.clock); // ns to push one k-step's requests through the TCP
  const pipeCeil =
    p.pipe === PIPE_SERIAL
      ? C / (C + p.L + tIssue)
      : p.pipe >= PIPE_INF
        ? 1
        : Math.min(1, (p.pipe * C) / ((p.relEnd ? C : 0) + p.L + tIssue));
  // MFMA busy over the CU's 4 SIMDs: the SIMDs with work, times the lowest per-wave limit
  const ceiling = simdFrac * Math.min(windowCeil, issueCeil, pipeCeil);
  const intensity = (p.bm * p.bn) / (p.bm + p.bn);
  return {
    blocks, active, mfmaPerWave, simdFrac, cycles, C, B, Q, need, deliver, stepsNeeded, stepsAllowed,
    windowCeil, issueCeil, pipeCeil, tIssue, ceiling, intensity,
  };
}

type Req = { r: number; j: number; issue: number; done: number; slot: number };
type Step = { j: number; start: number; end: number; idleCause: "slots" | "latency" | "buffers" | "none" };

/**
 * Requests issue in order, at most one per TCP_ISSUE_CYCLES. Normally a request also needs a free
 * slot (at most Q / 1 KB outstanding). With `unlimited`, there is no slot limit and k-step j's
 * requests are paced to start at j x C, just in time for full speed: what full speed would need.
 * Each request is drawn in the lowest row free at its issue time, so the occupied height is the
 * number of kilobytes outstanding.
 */
function simulate(p: Params, horizon: number, unlimited = false) {
  const { C, B } = derived(p);
  const n = Math.max(1, Math.round(B / REQ_BYTES));
  const S = Math.max(1, Math.round(p.qKB));
  const gap = TCP_ISSUE_CYCLES / p.clock;
  const reqs: Req[] = [];
  const steps: Step[] = [];
  const done: number[] = [];
  const issue: number[] = [];
  const binding: string[] = [];
  const rowEnd: number[] = [];
  for (let j = 0; j < 6000; j++) {
    const stepAllowed =
      p.pipe === PIPE_SERIAL
        ? j >= 1
          ? steps[j - 1].end
          : 0
        : p.pipe < PIPE_INF && j >= p.pipe
          ? p.relEnd
            ? steps[j - p.pipe].end
            : steps[j - p.pipe].start
          : 0;
    for (let i = 0; i < n; i++) {
      const r = j * n + i;
      // unlimited: request i of k-step j is paced to j x C + i x C / n, spread over a k-step as v9's regions do
      const allowed = unlimited ? Math.max(j * C + (i * C) / n, stepAllowed) : stepAllowed;
      const tPrev = r > 0 ? issue[r - 1] + gap : 0;
      const tSlot = !unlimited && r >= S ? done[r - S] : 0;
      const t = Math.max(tPrev, tSlot, allowed);
      issue.push(t);
      done.push(t + p.L);
      binding.push(
        stepAllowed > 0 && t === stepAllowed && stepAllowed >= tSlot
          ? "buffers"
          : !unlimited && t === tSlot && r >= S
            ? "slots"
            : "issue",
      );
      let row = rowEnd.findIndex((e) => e <= t + 1e-6);
      if (row < 0) {
        row = rowEnd.length;
        rowEnd.push(0);
      }
      rowEnd[row] = t + p.L;
      reqs.push({ r, j, issue: t, done: t + p.L, slot: row });
    }
    const lastReq = j * n + n - 1;
    const prevEnd = j > 0 ? steps[j - 1].end : 0;
    const start = Math.max(prevEnd, done[lastReq]);
    let idleCause: Step["idleCause"] = "none";
    if (start > prevEnd + 1e-6 && j > 0) {
      const b = binding[lastReq];
      idleCause = b === "slots" ? "slots" : b === "buffers" ? "buffers" : "latency";
    }
    steps.push({ j, start, end: start + C, idleCause });
    if (start > horizon) break;
  }
  return { reqs, steps, n, S, rows: rowEnd.length };
}

// ---------------------------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------------------------
function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
  digits?: number;
}) {
  const t = useHostTheme();
  return (
    <Stack gap={4}>
      <Row justify="space-between">
        <Text size="small" tone="secondary">
          {props.label}
        </Text>
        <Text size="small" weight="medium">
          {props.value.toFixed(props.digits ?? 0)} {props.unit}
        </Text>
      </Row>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e: { target: HTMLInputElement }) => props.onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: t.accent.primary }}
      />
    </Stack>
  );
}

function PillGroup(props: { label: string; options: number[]; value: number; onChange: (v: number) => void }) {
  return (
    <Stack gap={4}>
      <Text size="small" tone="secondary">
        {props.label}
      </Text>
      <Row gap={6}>
        {props.options.map((o) => (
          <Pill key={o} size="sm" active={o === props.value} onClick={() => props.onChange(o)}>
            {o}
          </Pill>
        ))}
      </Row>
    </Stack>
  );
}

function pct(x: number) {
  return `${Math.round(x * 100)}%`;
}

// ---------------------------------------------------------------------------------------------
// Tile geometry: area = compute, strips = bytes
// ---------------------------------------------------------------------------------------------
function TileGeometry({ p }: { p: Params }) {
  const t = useHostTheme();
  const s = 150 / 256;
  const strip = Math.max(14, p.bk * s * 0.6);
  const x0 = 34;
  const y0 = 30;
  const ref = { bm: 256, bn: 256, bk: 64 };
  const area = (p.bm * p.bn) / (ref.bm * ref.bn);
  const bytes = ((p.bm + p.bn) * p.bk) / ((ref.bm + ref.bn) * ref.bk);
  const kArea = (p.bm * p.bn * p.bk) / (ref.bm * ref.bn * ref.bk);
  const w = p.bn * s;
  const h = p.bm * s;
  return (
    <Stack gap={8}>
      <svg width="100%" viewBox="0 0 260 230" role="img" aria-label="Tile geometry: compute area against loaded strips">
        {/* ghost 256x256 */}
        <rect x={x0 + strip} y={y0 + strip} width={256 * s} height={256 * s} fill="none" stroke={t.stroke.secondary} strokeDasharray="4 3" />
        <text x={x0 + strip + 256 * s - 2} y={y0 + strip + 256 * s + 12} fontSize={9} textAnchor="end" fill={t.text.quaternary}>
          256x256 outline
        </text>
        {/* B strip on top */}
        <rect x={x0 + strip} y={y0} width={w} height={strip - 2} fill={t.category.blue} opacity={0.55} />
        <text x={x0 + strip + w / 2} y={y0 - 5} fontSize={9.5} textAnchor="middle" fill={t.text.secondary}>
          B strip: BK x BN = {p.bk} x {p.bn}
        </text>
        {/* A strip on the left */}
        <rect x={x0} y={y0 + strip} width={strip - 2} height={h} fill={t.category.blue} opacity={0.55} />
        <text
          x={x0 - 6}
          y={y0 + strip + h / 2}
          fontSize={9.5}
          textAnchor="middle"
          fill={t.text.secondary}
          transform={`rotate(-90 ${x0 - 6} ${y0 + strip + h / 2})`}
        >
          A strip: BM x BK = {p.bm} x {p.bk}
        </text>
        {/* output tile */}
        <rect x={x0 + strip} y={y0 + strip} width={w} height={h} fill={t.accent.primary} opacity={0.22} stroke={t.accent.primary} />
        <text x={x0 + strip + w / 2} y={y0 + strip + h / 2 - 4} fontSize={10.5} textAnchor="middle" fill={t.text.primary}>
          {p.bm} x {p.bn}
        </text>
        <text x={x0 + strip + w / 2} y={y0 + strip + h / 2 + 10} fontSize={9} textAnchor="middle" fill={t.text.secondary}>
          area = compute
        </text>
      </svg>
      <Text size="small" tone="secondary">
        Per k-step, against 256x256x64: compute (area x BK) x{kArea.toFixed(2)}, bytes (strips) x{bytes.toFixed(2)}, so the MFMAs
        eat data {(bytes / kArea).toFixed(2)}x as fast{p.bk !== 64 ? ` (BK scales both; it cancels)` : ""}.
        {area < 1 && p.bk === 64 ? ` Area x${area.toFixed(2)}, strips only x${bytes.toFixed(2)}.` : ""}
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Timeline: TCP slots and the MFMA lane
// ---------------------------------------------------------------------------------------------
const WINDOW_NS = 3000;
const MAX_POS_NS = 15000; // the slider scrubs the window's left edge over [0, MAX_POS_NS] from kernel start

function Timeline({
  p,
  pos,
  hover,
  setHover,
  unlimited,
}: {
  p: Params;
  pos: number;
  hover: number | null;
  setHover: (j: number | null) => void;
  unlimited: boolean;
}) {
  const t = useHostTheme();
  const d = derived(p);
  const sim = useMemo(
    () => simulate(p, MAX_POS_NS + WINDOW_NS + 2000, unlimited),
    [p.bm, p.bn, p.bk, p.clock, p.L, p.qKB, p.pipe, p.relEnd, unlimited],
  );
  const steady = useMemo(() => {
    const warm = 30 * Math.max(d.C, (d.B * p.L) / d.Q);
    return { ...simulate(p, warm + 12000), warm };
  }, [p.bm, p.bn, p.bk, p.clock, p.L, p.qKB, p.pipe, p.relEnd]);
  const t0 = pos;
  const t1 = t0 + WINDOW_NS;
  const W = 780;
  const left = 70;
  const plotW = W - left - 10;
  const nRows = Math.min(200, Math.max(sim.S, sim.rows));
  const tall = nRows <= 32;
  const rowH = tall ? 11 : Math.max(2, 352 / nRows);
  const slotsH = rowH * nRows;
  const realH = rowH * Math.min(sim.S, nRows);
  // full-speed reference: outstanding bytes at full speed = (B / C) x L
  const fullKB = (d.need * p.L) / 1024;
  const fullSim = unlimited
    ? sim.steps.filter((s) => s.start >= 2 * p.L + d.C && s.end <= MAX_POS_NS)
    : [];
  const fullEff = fullSim.length > 1 ? (fullSim.length * d.C) / (fullSim[fullSim.length - 1].end - fullSim[0].start) : 1;
  const tcpRateBns = TCP_BYTES_PER_CLK * p.clock;
  const laneY = 24; // TCP issue lane: one tick per request entering the TCP
  const laneH = 14;
  const top = laneY + laneH + 12;
  // LDS buffer lane: which k-step holds each buffer, from its first request to its release
  const bufRowOf: number[] = [];
  {
    const ends: number[] = [];
    for (const s of sim.steps) {
      const first = sim.reqs[s.j * sim.n].issue;
      const release = p.pipe === PIPE_SERIAL || p.relEnd ? s.end : s.start;
      let row: number;
      if (p.pipe === PIPE_SERIAL) row = 0;
      else if (p.pipe < PIPE_INF) row = s.j % p.pipe;
      else {
        row = ends.findIndex((e) => e <= first + 1e-6);
        if (row < 0) {
          row = ends.length;
          ends.push(0);
        }
        ends[row] = release;
      }
      bufRowOf.push(row);
    }
  }
  const bufRows = Math.min(12, Math.max(1, ...bufRowOf.map((r) => r + 1)));
  const bufRowH = 11;
  const bufY = top + slotsH + 14;
  const bufH = bufRows * bufRowH;
  const mfmaY = bufY + bufH + 16;
  const gapNs = TCP_ISSUE_CYCLES / p.clock;
  const mfmaH = 24;
  const H = mfmaY + mfmaH + 36;
  const x = (tt: number) => left + ((tt - t0) / WINDOW_NS) * plotW;
  const clip = (v: number) => Math.min(left + plotW, Math.max(left, v));
  const vis = sim.reqs.filter((r) => r.done > t0 && r.issue < t1);
  const steps = sim.steps.filter((s) => s.end > t0 && s.start < t1);
  const lastReq = (j: number) => sim.reqs[j * sim.n + sim.n - 1];
  const hoverStep = hover !== null ? sim.steps[hover] : undefined;
  // steady-state efficiency from a separate, longer run
  const ss = steady.steps.filter((s) => s.start >= steady.warm && s.end <= steady.warm + 10000);
  const simEff = ss.length > 1 ? (ss.length * d.C) / (ss[ss.length - 1].end - ss[0].start) : 1;
  const idle: Record<Step["idleCause"], number> = { slots: 0, latency: 0, buffers: 0, none: 0 };
  for (let i = 1; i < ss.length; i++) {
    const g = ss[i].start - ss[i - 1].end;
    if (g > 0 && ss[i].idleCause !== "none") idle[ss[i].idleCause] += g;
  }
  const span = ss.length > 1 ? ss[ss.length - 1].end - ss[0].start : 1;
  const ticks = [];
  const tickStep = 500;
  for (let tt = Math.ceil(t0 / tickStep) * tickStep; tt <= t1; tt += tickStep) ticks.push(tt);
  const red = t.category.red;
  return (
    <Stack gap={8}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="TCP slots and MFMA lane over time">
        <text x={4} y={top + realH / 2} fontSize={10} fill={t.text.secondary}>
          TCP slots
        </text>
        <text x={4} y={top + realH / 2 + 12} fontSize={9} fill={t.text.tertiary}>
          Q = {sim.S} KB
        </text>
        <text x={4} y={top + realH / 2 + 23} fontSize={9} fill={t.text.tertiary}>
          ({sim.S} x 1 KB)
        </text>
        <rect x={left} y={top} width={plotW} height={slotsH} fill={t.fill.quaternary} />
        {nRows > sim.S && (
          <g>
            <rect x={left} y={top + realH} width={plotW} height={slotsH - realH} fill={red} opacity={0.12} />
            <line x1={left - 6} x2={left + plotW} y1={top + realH} y2={top + realH} stroke={red} strokeWidth={1.5} />
            <text x={4} y={top + realH + (slotsH - realH) / 2} fontSize={9.5} fill={red}>
              slots the
            </text>
            <text x={4} y={top + realH + (slotsH - realH) / 2 + 11} fontSize={9.5} fill={red}>
              TCP does
            </text>
            <text x={4} y={top + realH + (slotsH - realH) / 2 + 22} fontSize={9.5} fill={red}>
              not have
            </text>
          </g>
        )}
        {vis.map((r) => {
          const xa = clip(x(r.issue));
          const xb = clip(x(r.done));
          const w = Math.max(0.5, xb - xa - 1.5);
          const lit = hover === null || hover === r.j;
          return (
            <g key={r.r} onMouseEnter={() => setHover(r.j)} onMouseLeave={() => setHover(null)} style={{ cursor: "pointer" }}>
              <rect
                x={xa}
                y={top + r.slot * rowH + 0.5}
                width={w}
                height={Math.max(1, rowH - 1.5)}
                rx={tall ? 2 : 0.5}
                fill={t.category.blue}
                fillOpacity={lit ? (r.j % 2 === 0 ? 0.8 : 0.45) : 0.12}
                stroke={hover === r.j ? t.accent.primary : t.text.secondary}
                strokeOpacity={hover === r.j ? 1 : lit ? 0.7 : 0.2}
                strokeWidth={hover === r.j ? 1.2 : 0.7}
              />
              {tall && w > 20 && (
                <text x={xa + 3} y={top + r.slot * rowH + rowH - 3} fontSize={7.5} fill={t.text.primary} opacity={lit ? 0.95 : 0.3}>
                  k{r.j}
                </text>
              )}
            </g>
          );
        })}
        <text x={4} y={bufY + Math.min(bufH, 22) / 2 + 3} fontSize={10} fill={t.text.secondary}>
          LDS
        </text>
        <text x={4} y={bufY + Math.min(bufH, 22) / 2 + 14} fontSize={9} fill={t.text.tertiary}>
          {p.pipe === PIPE_SERIAL ? "serial" : p.pipe >= PIPE_INF ? `${bufRows} used` : `${p.pipe} buffer${p.pipe > 1 ? "s" : ""}`}
        </text>
        <rect x={left} y={bufY} width={plotW} height={bufH} fill={t.fill.quaternary} />
        {sim.steps
          .filter((s) => s.end > t0 && sim.reqs[s.j * sim.n].issue < t1)
          .map((s) => {
            const row = bufRowOf[s.j];
            if (row >= bufRows) return null;
            const first = sim.reqs[s.j * sim.n].issue;
            const arrive = sim.reqs[s.j * sim.n + sim.n - 1].done;
            const release = p.pipe === PIPE_SERIAL || p.relEnd ? s.end : s.start;
            const y = bufY + row * bufRowH + 0.5;
            const lit = hover === null || hover === s.j;
            const xa = clip(x(first));
            const xm = clip(x(Math.min(arrive, release)));
            const xb = clip(x(release));
            return (
              <g key={`b${s.j}`} onMouseEnter={() => setHover(s.j)} onMouseLeave={() => setHover(null)} style={{ cursor: "pointer" }}>
                <rect x={xa} y={y} width={Math.max(0.5, xm - xa)} height={bufRowH - 1.5} fill={t.category.blue} fillOpacity={lit ? 0.3 : 0.1} />
                <rect x={xm} y={y} width={Math.max(0, xb - xm - 1)} height={bufRowH - 1.5} fill={t.category.blue} fillOpacity={lit ? 0.85 : 0.2} />
                <rect
                  x={xa}
                  y={y}
                  width={Math.max(0.5, xb - xa - 1)}
                  height={bufRowH - 1.5}
                  fill="none"
                  stroke={hover === s.j ? t.accent.primary : t.text.secondary}
                  strokeOpacity={lit ? 0.7 : 0.2}
                  strokeWidth={hover === s.j ? 1.2 : 0.6}
                  rx={2}
                />
                {xb - xa > 18 && (
                  <text x={xa + 3} y={y + bufRowH - 3.5} fontSize={7.5} fill={t.text.primary} opacity={lit ? 0.95 : 0.3}>
                    k{s.j}
                  </text>
                )}
              </g>
            );
          })}
        <text x={4} y={mfmaY + 16} fontSize={10} fill={t.text.secondary}>
          MFMA
        </text>
        <rect x={left} y={mfmaY} width={plotW} height={mfmaH} fill={red} opacity={0.28} />
        {steps.map((s) => {
          const xa = clip(x(s.start));
          const xb = clip(x(s.end));
          const w = Math.max(0.5, xb - xa - 1);
          const lit = hover === null || hover === s.j;
          return (
            <g key={s.j} onMouseEnter={() => setHover(s.j)} onMouseLeave={() => setHover(null)} style={{ cursor: "pointer" }}>
              <rect
                x={xa}
                y={mfmaY}
                width={w}
                height={mfmaH}
                fill={t.accent.primary}
                opacity={lit ? 1 : 0.35}
                stroke={t.bg.editor}
                strokeWidth={1}
              />
              {w > 16 && (
                <text x={xa + w / 2} y={mfmaY + 15} fontSize={8.5} textAnchor="middle" fill={t.text.onAccent}>
                  k{s.j}
                </text>
              )}
            </g>
          );
        })}
        {hoverStep && hover !== null && (
          (() => {
            const lr = lastReq(hover);
            const xr = x(lr.done);
            const xs = x(hoverStep.start);
            if (xr < left || xs > left + plotW) return null;
            const yr = top + lr.slot * rowH + rowH / 2;
            return (
              <g pointerEvents="none">
                <line x1={xr} x2={xr} y1={yr} y2={mfmaY - 6} stroke={t.accent.primary} strokeDasharray="3 2" />
                <line x1={xr} x2={xs} y1={mfmaY - 6} y2={mfmaY - 6} stroke={t.accent.primary} strokeDasharray="3 2" />
                <line x1={xs} x2={xs} y1={mfmaY - 6} y2={mfmaY} stroke={t.accent.primary} />
                <text x={xr + 3} y={mfmaY - 9} fontSize={8.5} fill={t.accent.primary}>
                  k{hover}: last 1 KB arrives at {Math.round(lr.done)} ns, compute starts at {Math.round(hoverStep.start)} ns
                </text>
              </g>
            );
          })()
        )}
        {ticks.map((tt) => (
          <g key={tt}>
            <line x1={x(tt)} x2={x(tt)} y1={mfmaY + mfmaH} y2={mfmaY + mfmaH + 4} stroke={t.stroke.primary} />
            <text x={x(tt)} y={mfmaY + mfmaH + 15} fontSize={9} textAnchor="middle" fill={t.text.tertiary}>
              {Math.round(tt)}
            </text>
          </g>
        ))}
        <text x={left + plotW / 2} y={H - 4} fontSize={10} textAnchor="middle" fill={t.text.secondary}>
          time since the kernel started (ns)
        </text>
        {/* one latency bar for scale */}
        <line x1={left + plotW - (p.L / WINDOW_NS) * plotW} x2={left + plotW} y1={10} y2={10} stroke={t.text.secondary} />
        <text x={left + plotW - (p.L / WINDOW_NS) * plotW - 4} y={13} fontSize={9} textAnchor="end" fill={t.text.secondary}>
          one request = L = {p.L} ns
        </text>
        {/* TCP issue lane */}
        <text x={4} y={laneY + 10} fontSize={10} fill={t.text.secondary}>
          TCP issue
        </text>
        <rect x={left} y={laneY} width={plotW} height={laneH} fill={t.fill.quaternary} />
        {sim.reqs
          .filter((r) => r.r > 0 && r.issue > t0 && sim.reqs[r.r - 1].issue < t1)
          .map((r) => {
            const prev = sim.reqs[r.r - 1];
            const dt = r.issue - prev.issue;
            if (dt <= gapNs * 1.05) return null;
            const xa = clip(x(prev.issue + gapNs));
            const xb = clip(x(r.issue));
            return (
              <rect
                key={`w${r.r}`}
                x={xa}
                y={laneY}
                width={Math.max(0.5, xb - xa)}
                height={laneH}
                fill={unlimited ? t.fill.secondary : red}
                opacity={unlimited ? 1 : 0.35}
              />
            );
          })}
        {sim.reqs
          .filter((r) => r.issue >= t0 && r.issue <= t1)
          .map((r) => (
            <line
              key={`i${r.r}`}
              x1={x(r.issue)}
              x2={x(r.issue)}
              y1={laneY + 1}
              y2={laneY + laneH - 1}
              stroke={hover === r.j ? t.accent.primary : t.text.secondary}
              strokeWidth={hover === r.j ? 1.2 : 0.7}
            />
          ))}
        {hover !== null &&
          hover > 0 &&
          (() => {
            const first = sim.reqs[hover * sim.n];
            const prev = sim.reqs[hover * sim.n - 1];
            if (!first || !prev) return null;
            const xp = x(prev.issue);
            const xf = x(first.issue);
            if (xf < left || xp > left + plotW) return null;
            const cyc = (first.issue - prev.issue) * p.clock;
            const wait = Math.max(0, cyc - TCP_ISSUE_CYCLES);
            const yEnd = top + slotsH;
            return (
              <g pointerEvents="none">
                <line x1={xp} x2={xp} y1={laneY - 2} y2={yEnd} stroke={t.text.primary} strokeDasharray="2 2" />
                <line x1={xf} x2={xf} y1={laneY - 2} y2={yEnd} stroke={t.accent.primary} strokeDasharray="2 2" />
                <text x={Math.min(xf + 4, left + plotW - 330)} y={laneY - 4} fontSize={8.5} fill={t.text.primary}>
                  k{hover - 1} last issue to k{hover} first issue: {cyc.toFixed(0)} cycles = 16 (TCP processing)
                  {wait > 0.5
                    ? unlimited
                      ? ` + ${wait.toFixed(0)} paced (issued just in time${p.pipe < PIPE_INF ? ", or waiting for a buffer" : ""})`
                      : ` + ${wait.toFixed(0)} waiting for a free slot${p.pipe < PIPE_INF ? " or buffer" : ""}`
                    : ""}
                </text>
              </g>
            );
          })()}
      </svg>
      <Row gap={14} wrap>
        <Row gap={6} align="center">
          <svg width={14} height={10}>
            <rect width={14} height={10} fill={t.category.blue} opacity={0.85} />
          </svg>
          <Text size="small" tone="secondary">
            one bar = one 1 KB request, length L = {p.L} ns; kN = the k-step it feeds; {sim.n} bars = B = {d.B / 1024} KB per k-step
          </Text>
        </Row>
        <Row gap={6} align="center">
          <svg width={14} height={10}>
            <rect width={14} height={10} fill={t.accent.primary} />
          </svg>
          <Text size="small" tone="secondary">
            MFMA busy: one block = one k-step, length C = {Math.round(d.C)} ns
          </Text>
        </Row>
        <Row gap={6} align="center">
          <svg width={14} height={10}>
            <rect width={14} height={10} fill={red} opacity={0.28} />
          </svg>
          <Text size="small" tone="secondary">
            MFMA idle, waiting for data
          </Text>
        </Row>
        <Row gap={6} align="center">
          <svg width={14} height={10}>
            <line x1={4} x2={4} y1={0} y2={10} stroke={t.text.secondary} />
            <line x1={10} x2={10} y1={0} y2={10} stroke={t.text.secondary} />
          </svg>
          <Text size="small" tone="secondary">
            TCP issue: one tick per request, at least 16 cycles ({gapNs.toFixed(1)} ns) apart; red = waiting for a free slot or LDS
            buffer. Hover a k-step to measure the gap from the previous k-step&apos;s last issue
          </Text>
        </Row>
        <Row gap={6} align="center">
          <svg width={28} height={10}>
            <rect width={14} height={10} fill={t.category.blue} fillOpacity={0.3} />
            <rect x={14} width={14} height={10} fill={t.category.blue} fillOpacity={0.85} />
          </svg>
          <Text size="small" tone="secondary">
            LDS buffer: light = filling (its requests in flight), dark = full, until it is freed: at its k-step&apos;s compute start
            (operands copied to registers) or end (read during compute), per the setting; serial frees at the end. A k-step&apos;s loads
            cannot start until a buffer is free
          </Text>
        </Row>
      </Row>
      <Text size="small" tone="tertiary">
        {unlimited
          ? `Unlimited slots: k-step j's requests are spread over [j x C, (j+1) x C), just in time. Full speed keeps (B/C) x L = ${fullKB.toFixed(0)} KB outstanding (L/C = ${d.stepsNeeded.toFixed(2)} k-steps of ${d.B / 1024} KB); the TCP holds ${sim.S} KB. MFMA busy here: ${pct(fullEff)}${
              d.need > tcpRateBns ? ` (still short: the tile needs ${d.need.toFixed(0)} B/ns, above the TCP's processing rate of 64 B/clk = ${tcpRateBns.toFixed(0)} B/ns)` : ""
            }. `
          : ""}
        Hover any bar or MFMA block to highlight its k-step. Simulated, one CU: {sim.n} requests per k-step, each holding a slot for L;
        each request is drawn in the lowest free row, so the filled height is the kilobytes outstanding. A k-step computes once its last
        request returns and the previous k-step is done. Steady-state MFMA busy on the SIMDs with work {pct(simEff)} (formula{" "}
        {pct(d.ceiling / d.simdFrac)})
        {d.simdFrac < 1 ? `; only ${d.active} of 4 SIMDs have work, so the CU's MFMAs are busy ${pct(simEff * d.simdFrac)}` : ""}. Idle
        time: TCP slots full{" "}
        {pct(idle.slots / span)}, LDS buffers {pct(idle.buffers / span)}, latency with free slots {pct(idle.latency / span)}.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Number line: L/C against Q/B
// ---------------------------------------------------------------------------------------------
function WindowLine({ p }: { p: Params }) {
  const t = useHostTheme();
  const d = derived(p);
  const maxD = Math.max(2.5, d.stepsNeeded * 1.15, d.stepsAllowed * 1.15);
  const W = 420;
  const left = 16;
  const plotW = W - left - 16;
  const x = (v: number) => left + (v / maxD) * plotW;
  const y = 40;
  const ticks = [];
  for (let v = 0; v <= maxD + 1e-9; v += 0.5) ticks.push(v);
  const short = d.stepsNeeded > d.stepsAllowed;
  return (
    <Stack gap={6}>
      <svg width="100%" viewBox={`0 0 ${W} 92`} role="img" aria-label="k-steps in flight needed against allowed">
        <rect x={x(0)} y={y - 9} width={x(d.stepsAllowed) - x(0)} height={18} fill={t.fill.secondary} />
        <line x1={x(d.stepsAllowed)} x2={x(d.stepsAllowed)} y1={y - 13} y2={y + 13} stroke={t.text.primary} strokeWidth={2} />
        <text x={x(d.stepsAllowed)} y={y - 16} fontSize={9.5} textAnchor="middle" fill={t.text.primary}>
          Q/B = {d.stepsAllowed.toFixed(2)} allowed
        </text>
        {short ? (
          <rect x={x(d.stepsAllowed)} y={y - 4} width={x(d.stepsNeeded) - x(d.stepsAllowed)} height={8} fill={t.category.red} opacity={0.55} />
        ) : (
          <rect x={x(d.stepsNeeded)} y={y - 4} width={x(d.stepsAllowed) - x(d.stepsNeeded)} height={8} fill={t.category.green} opacity={0.55} />
        )}
        <path d={`M ${x(d.stepsNeeded) - 6} ${y + 14} L ${x(d.stepsNeeded) + 6} ${y + 14} L ${x(d.stepsNeeded)} ${y + 4} Z`} fill={t.text.primary} />
        <text x={x(d.stepsNeeded)} y={y + 26} fontSize={9.5} textAnchor="middle" fill={t.text.primary}>
          L/C = {d.stepsNeeded.toFixed(2)} needed
        </text>
        {ticks.map((v) => (
          <text key={v} x={x(v)} y={86} fontSize={8.5} textAnchor="middle" fill={t.text.tertiary}>
            {v.toFixed(1)}
          </text>
        ))}
      </svg>
      <Text size="small" tone="secondary">
        k-steps in flight. Hiding the latency needs L/C; the TCP holds Q/B.{" "}
        {short
          ? `Short by ${(d.stepsNeeded / d.stepsAllowed).toFixed(2)}x, so the MFMAs are busy at most ${pct(d.windowCeil)}.`
          : "The window is big enough: the latency is hidden."}
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Ceiling against tile intensity
// ---------------------------------------------------------------------------------------------
function CeilingCurve({ p }: { p: Params }) {
  const t = useHostTheme();
  const d = derived(p);
  const R = FLOP_PER_CLK_CU * p.clock; // FLOP per ns
  const W = 420;
  const H = 210;
  const left = 38;
  const bottom = 30;
  const plotW = W - left - 10;
  const plotH = H - bottom - 10;
  const maxI = 200;
  const x = (i: number) => left + (i / maxI) * plotW;
  const y = (e: number) => 10 + (1 - e) * plotH;
  const pts: string[] = [];
  for (let i = 0; i <= maxI; i += 2) pts.push(`${x(i)},${y(Math.min(1, (i * d.deliver) / R))}`);
  const ridge = R / d.deliver;
  return (
    <Stack gap={6}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="MFMA efficiency ceiling against tile intensity">
        {[0, 0.25, 0.5, 0.75, 1].map((e) => (
          <g key={e}>
            <line x1={left} x2={left + plotW} y1={y(e)} y2={y(e)} stroke={t.stroke.tertiary} />
            <text x={left - 4} y={y(e) + 3} fontSize={8.5} textAnchor="end" fill={t.text.tertiary}>
              {pct(e)}
            </text>
          </g>
        ))}
        {[0, 50, 100, 150, 200].map((i) => (
          <text key={i} x={x(i)} y={H - bottom + 12} fontSize={8.5} textAnchor="middle" fill={t.text.tertiary}>
            {i}
          </text>
        ))}
        <polyline points={pts.join(" ")} fill="none" stroke={t.accent.primary} strokeWidth={2} />
        {ridge <= maxI && (
          <g>
            <line x1={x(ridge)} x2={x(ridge)} y1={y(1)} y2={y(0)} stroke={t.stroke.secondary} strokeDasharray="3 3" />
            <text x={x(ridge) + 3} y={y(0.06)} fontSize={8.5} fill={t.text.tertiary}>
              ridge {Math.round(ridge)}
            </text>
          </g>
        )}
        {MEASURED.map((m) => (
          <g key={m.tile}>
            <circle cx={x(m.i)} cy={y(m.cold)} r={3.5} fill={t.text.secondary} />
            {m.warm !== undefined && <circle cx={x(m.i)} cy={y(m.warm)} r={3.5} fill="none" stroke={t.text.secondary} strokeWidth={1.4} />}
          </g>
        ))}
        <circle cx={x(d.intensity)} cy={y(d.windowCeil)} r={6} fill="none" stroke={t.accent.primary} strokeWidth={2} />
        <text x={x(d.intensity) + 8} y={y(d.windowCeil) + 12} fontSize={9} fill={t.text.primary}>
          {p.bm}x{p.bn}: I = {d.intensity.toFixed(0)}
        </text>
        <text x={left + plotW / 2} y={H - 3} fontSize={9.5} textAnchor="middle" fill={t.text.secondary}>
          tile intensity I = BM*BN/(BM+BN) (FLOP per byte loaded)
        </text>
        <text x={10} y={10 + plotH / 2} fontSize={9.5} textAnchor="middle" fill={t.text.secondary} transform={`rotate(-90 10 ${10 + plotH / 2})`}>
          MFMA efficiency
        </text>
      </svg>
      <Row gap={14} wrap>
        <Row gap={6} align="center">
          <svg width={18} height={8}>
            <line x1={0} x2={18} y1={4} y2={4} stroke={t.accent.primary} strokeWidth={2} />
          </svg>
          <Text size="small" tone="secondary">
            ceiling min(1, I x Q/L / R) at the current clock, L, Q
          </Text>
        </Row>
        <Row gap={6} align="center">
          <svg width={10} height={10}>
            <circle cx={5} cy={5} r={3.5} fill={t.text.secondary} />
          </svg>
          <Text size="small" tone="secondary">
            measured, cold
          </Text>
        </Row>
        <Row gap={6} align="center">
          <svg width={10} height={10}>
            <circle cx={5} cy={5} r={3.5} fill="none" stroke={t.text.secondary} strokeWidth={1.4} />
          </svg>
          <Text size="small" tone="secondary">
            measured, warm
          </Text>
        </Row>
      </Row>
      <Text size="small" tone="tertiary">
        Measured: ATT main-loop efficiency, full wave, GPU 2 (cold K = 8192 at each tile's own clock, 1.7-2.1 GHz; warm K = 4096 with
        inputs in the MALL). The curve is drawn for the slider settings.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Bandwidth ladder: the per-CU window against the shared hierarchy, all in GB/s per CU
// Ceilings measured in experiments/hbm_bandwidth_xcd (MI355X, all 8 XCDs).
// ---------------------------------------------------------------------------------------------
const TCP_BYTES_PER_CLK = 64; // the TCP processes one 64 B line per clock (153 GB/s at 2.4 GHz)
const L2_BYTES_PER_CLK_XCD = 2048; // L2 peak per XCD; 8 x 2 KB/clk = 256 CUs x 64 B/clk, matched by design
const HBM_CHIP_GBS = 6300;
const MALL_CHIP_GBS = 7900; // fabric-limited as the XCDs see it

function BandwidthLadder({ p }: { p: Params }) {
  const t = useHostTheme();
  const d = derived(p);
  const [hit, setHit] = useState(80);
  const [cus, setCus] = useState(256);
  const [stream, setStream] = useState(false);
  const [mall, setMall] = useState(false);
  const h = stream ? 0 : hit / 100;
  const demand = stream ? Infinity : d.need;
  const windowCap = d.deliver;
  const tcpRate = TCP_BYTES_PER_CLK * p.clock;
  const l2Share = (L2_BYTES_PER_CLK_XCD * 8 * p.clock) / cus;
  const beyondChip = mall ? MALL_CHIP_GBS : HBM_CHIP_GBS;
  const beyondShare = h >= 1 ? Infinity : beyondChip / (cus * (1 - h));
  const rows = [
    { key: "demand", label: stream ? "Demand: a stream wants all it can get" : "Demand: what the MFMAs consume (B / C)", v: demand, kind: "demand" },
    { key: "window", label: "L1 window: Q / L (slots held for the round trip)", v: windowCap, kind: "limit" },
    { key: "tcp", label: `L1 processing: 64 B/clk x ${p.clock.toFixed(2)} GHz`, v: tcpRate, kind: "limit" },
    { key: "l2", label: `L2 share: 8 x 2 KB/clk x clock / ${cus} CUs`, v: l2Share, kind: "limit" },
    {
      key: "beyond",
      label: `${mall ? "MALL" : "HBM"} share: ${(beyondChip / 1000).toFixed(1)} TB/s / (${cus} CUs x ${Math.round((1 - h) * 100)}% misses)`,
      v: beyondShare,
      kind: "limit",
    },
  ];
  const got = Math.min(demand, windowCap, tcpRate, l2Share, beyondShare);
  const binding = rows.reduce((a, r) => (r.v < a.v ? r : a), rows[0]);
  const maxV = 200;
  const W = 560;
  const left = 250;
  const plotW = W - left - 60;
  const rowH = 30;
  const H = rows.length * rowH + 40;
  const xv = (v: number) => left + (Math.min(v, maxV) / maxV) * plotW;
  const bindLabel =
    binding.key === "demand"
      ? "nothing: the demand is met"
      : binding.key === "window"
        ? "the L1 window (latency too long for the slots)"
        : binding.key === "tcp"
          ? "the L1's processing rate (the round trip is short enough that the slots keep it busy)"
          : binding.key === "l2"
            ? "L2 bandwidth (shared; equals the L1 rate at 256 CUs)"
            : `${mall ? "MALL" : "HBM"} bandwidth (shared)`;
  return (
    <Stack gap={12}>
      <Grid columns="minmax(0, 1fr) minmax(0, 2.2fr)" gap={24} align="start">
        <Stack gap={12}>
          <Slider label="L2 hit rate" value={stream ? 0 : hit} min={0} max={100} step={1} unit="%" onChange={setHit} />
          <Slider label="Active CUs" value={cus} min={8} max={256} step={8} unit="" onChange={setCus} />
          <Row gap={8} align="center">
            <Toggle checked={stream} onChange={setStream} />
            <Text size="small" tone="secondary">
              Memory-bound stream (no reuse: 0% hits, unbounded demand)
            </Text>
          </Row>
          <Row gap={8} align="center">
            <Toggle checked={mall} onChange={setMall} />
            <Text size="small" tone="secondary">
              Misses served by the MALL instead of HBM
            </Text>
          </Row>
          <Row gap={6} wrap>
            <Pill
              size="sm"
              onClick={() => {
                setStream(true);
                setCus(256);
              }}
            >
              stream, 256 CUs
            </Pill>
            <Pill
              size="sm"
              onClick={() => {
                setStream(true);
                setCus(32);
              }}
            >
              stream, 32 CUs
            </Pill>
            <Pill
              size="sm"
              onClick={() => {
                setStream(false);
                setHit(80);
                setCus(256);
              }}
            >
              GEMM, 80% hits
            </Pill>
            <Pill
              size="sm"
              onClick={() => {
                setStream(false);
                setHit(10);
                setCus(256);
              }}
            >
              GEMM, 10% hits
            </Pill>
          </Row>
        </Stack>
        <Stack gap={8}>
          <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-CU bandwidth demand against each ceiling">
            {[0, 50, 100, 150, 200].map((v) => (
              <g key={v}>
                <line x1={xv(v)} x2={xv(v)} y1={8} y2={H - 26} stroke={t.stroke.tertiary} />
                <text x={xv(v)} y={H - 14} fontSize={9} textAnchor="middle" fill={t.text.tertiary}>
                  {v}
                </text>
              </g>
            ))}
            <text x={left + plotW / 2} y={H - 2} fontSize={9.5} textAnchor="middle" fill={t.text.secondary}>
              GB/s per CU (= bytes per ns)
            </text>
            {rows.map((r, i) => {
              const y = 10 + i * rowH;
              const isBind = r.key === binding.key && r.kind === "limit";
              const fill = r.kind === "demand" ? t.category.blue : isBind ? t.accent.primary : t.fill.secondary;
              return (
                <g key={r.key}>
                  <text x={left - 8} y={y + 14} fontSize={9.5} textAnchor="end" fill={isBind ? t.text.primary : t.text.secondary}>
                    {r.label}
                  </text>
                  <rect x={left} y={y + 3} width={xv(r.v) - left} height={rowH - 12} fill={fill} opacity={r.kind === "demand" ? 0.6 : 1} />
                  <text x={xv(r.v) + 4} y={y + 15} fontSize={9.5} fill={t.text.primary}>
                    {Number.isFinite(r.v) ? `${r.v.toFixed(0)}${r.v > maxV ? " (off scale)" : ""}` : "unbounded"}
                  </text>
                </g>
              );
            })}
            <line x1={xv(got)} x2={xv(got)} y1={6} y2={H - 26} stroke={t.text.primary} strokeDasharray="4 3" />
            <text x={xv(got) + 3} y={8} fontSize={9} fill={t.text.primary}>
              delivered
            </text>
          </svg>
          <Row gap={28} wrap>
            <Stat value={`${got.toFixed(0)} GB/s`} label="delivered per CU" />
            <Stat value={`${((got * cus) / 1000).toFixed(1)} TB/s`} label={`chip, L2 to CUs (${cus} CUs)`} />
            <Stat value={`${((got * cus * (1 - h)) / 1000).toFixed(1)} TB/s`} label={`chip, from ${mall ? "MALL" : "HBM"}`} />
            {!stream && <Stat value={pct(d.simdFrac * Math.min(1, got / d.need))} label="MFMA efficiency ceiling" />}
          </Row>
          <Text size="small" weight="medium">
            Binds first: {bindLabel}.
          </Text>
        </Stack>
      </Grid>
      <Text size="small" tone="tertiary">
        An L1 gets the lower of two limits: its processing rate (64 B/clk) and its window (bytes in flight / round trip). L2 is sized to
        match the first (8 XCDs x 2 KB/clk = 256 CUs x 64 B/clk), so it never out-delivers the L1; the window is what drops a CU below
        that rate when the round trip is long (below about 32 KB / 153 GB/s = 210 ns the processing rate binds instead). Ceilings
        measured with a streaming kernel in experiments/hbm_bandwidth_xcd (all 8 XCDs, 2.38 GHz): L2 38.8 TB/s and 153 GB/s per CU,
        MALL 7.0-7.9 TB/s, HBM 6.2-6.4 TB/s, and about 60 GB/s per CU from HBM, 110 from the MALL. The window uses the L slider as the
        average round trip; real misses hold their slots far longer than hits, which is why 80% hits still give about 60 GB/s cold.
        The per-XCD link (1.35 TB/s) is not drawn: with round-robin placement HBM binds before it.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------
export default function TileTcpIntuition() {
  const t = useHostTheme();
  const [preset, setPreset] = useState("256x128");
  const [bm, setBm] = useState(256);
  const [bn, setBn] = useState(128);
  const [bk, setBk] = useState(64);
  const [clock, setClock] = useState(1.92);
  const [L, setL] = useState(L_COLD);
  const [qKB, setQ] = useState(32);
  const [pipe, setPipe] = useState("2");
  const [release, setRelease] = useState("start");
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState(0); // left edge of the timeline window, ns since kernel start
  const [speed, setSpeed] = useState("0.5");
  const [hover, setHover] = useState<number | null>(null);
  const [unlimited, setUnlimited] = useState(false);

  const posRef = useRef(pos);
  posRef.current = pos;

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const rate = parseFloat(speed); // ns of model time per ms of wall time (1 = 1 us per second)
    const tick = (now: number) => {
      const next = Math.min(MAX_POS_NS, posRef.current + (now - last) * rate);
      last = now;
      posRef.current = next;
      setPos(next);
      if (next >= MAX_POS_NS) setPlaying(false);
      else raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]);

  const applyPreset = (id: string) => {
    const pr = PRESETS.find((x) => x.id === id);
    if (!pr) return;
    setPreset(id);
    setBm(pr.bm);
    setBn(pr.bn);
    setBk(pr.bk);
    setClock(pr.clock);
  };
  const custom = (f: (v: number) => void) => (v: number) => {
    setPreset("custom");
    f(v);
  };

  const p: Params = { bm, bn, bk, clock, L, qKB, pipe: parseInt(pipe, 10), relEnd: release === "end" };
  const d = derived(p);
  const short = d.need > d.deliver;

  return (
    <Stack gap={20} style={{ padding: 4 }}>
      <Stack gap={6}>
        <H1>Why smaller tiles starve: tile shape against the 32 KB TCP window</H1>
        <Text tone="secondary">
          A CU must request bytes as fast as its MFMAs consume them. Compute per k-step grows with the tile's area, bytes with its
          edges, and the TCP lets at most 32 KB be in flight for one round trip. Move the sliders and watch the slots fill and the
          MFMA lane open gaps.
        </Text>
        <Text size="small" tone="tertiary">
          Source: v9_tile on GPU 2 (MI355X), WORKLOG sections 4, 7, 9. L = 550 ns cold and 385 ns warm are inferred from the measured
          ~60 and ~85 B/ns per CU with 32 KB in flight; 16 cycles per MFMA; 16 cycles of TCP processing per 1 KB request.
        </Text>
      </Stack>

      <Grid columns="minmax(0, 1.1fr) minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)" gap={16} align="start">
        <Stack gap={10}>
          <Stack gap={4}>
            <Text size="small" tone="secondary">
              Tile preset (sets the measured clock)
            </Text>
            <Select
              value={preset}
              onChange={applyPreset}
              options={[...PRESETS.map((x) => ({ value: x.id, label: x.label })), { value: "custom", label: "custom", disabled: true }]}
            />
          </Stack>
          <Row gap={16}>
            <PillGroup label="BM" options={[16, 32, 64, 128, 256]} value={bm} onChange={custom(setBm)} />
            <PillGroup label="BN" options={[16, 32, 64, 128, 256]} value={bn} onChange={custom(setBn)} />
            <PillGroup label="BK" options={[64, 128]} value={bk} onChange={custom(setBk)} />
          </Row>
        </Stack>
        <Stack gap={10}>
          <Slider label="Clock" value={clock} min={1.5} max={2.4} step={0.01} unit="GHz" digits={2} onChange={custom(setClock)} />
          <Text size="small" tone="tertiary">
            Faster clock: shorter k-steps, same latency in ns.
          </Text>
        </Stack>
        <Stack gap={10}>
          <Slider label="Round trip L" value={L} min={150} max={1200} step={5} unit="ns" onChange={setL} />
          <Row gap={6}>
            <Pill size="sm" active={L === L_COLD} onClick={() => setL(L_COLD)}>
              cold 550
            </Pill>
            <Pill size="sm" active={L === L_WARM} onClick={() => setL(L_WARM)}>
              warm 385
            </Pill>
          </Row>
        </Stack>
        <Stack gap={10}>
          <Slider label="TCP in flight Q (32 KB in hardware)" value={qKB} min={8} max={64} step={4} unit="KB" onChange={setQ} />
          <Stack gap={4}>
            <Text size="small" tone="secondary">
              Pipeline: how far ahead a CU may load
            </Text>
            <Select
              value={pipe}
              onChange={setPipe}
              options={[
                { value: String(PIPE_SERIAL), label: "Serial: load, wait, compute" },
                { value: "1", label: "1 LDS buffer" },
                { value: "2", label: "2 LDS buffers (v9)" },
                { value: "3", label: "3 LDS buffers" },
                { value: "4", label: "4 LDS buffers" },
                { value: String(PIPE_INF), label: "Unlimited buffers" },
              ]}
            />
            <Text size="small" tone="secondary">
              A buffer frees at
            </Text>
            <Select
              value={release}
              onChange={setRelease}
              options={[
                { value: "start", label: "compute start: operands copied to registers first (optimistic)" },
                { value: "end", label: "compute end: operands read from LDS during compute (pessimistic)" },
              ]}
            />
          </Stack>
        </Stack>
      </Grid>

      <Row gap={32} align="end" wrap>
        <Stat value={pct(d.ceiling)} label="MFMA efficiency ceiling" tone={d.ceiling >= 0.95 ? "success" : d.ceiling >= 0.6 ? "warning" : "danger"} />
        <Stat value={`${d.need.toFixed(0)} B/ns`} label="data the MFMAs consume  (B / C)" tone={short ? "danger" : undefined} />
        <Stat value={`${d.deliver.toFixed(0)} B/ns`} label="data the window delivers  (Q / L)" />
        <Stat value={`${Math.round(d.C)} ns`} label={`compute per k-step  (${d.cycles} cycles)`} />
        <Stat value={`${d.B / 1024} KB`} label="bytes per k-step" />
      </Row>

      <Card collapsible defaultOpen>
        <CardHeader trailing="live values for the current settings">Symbols and ratios</CardHeader>
        <CardBody style={{ padding: 0 }}>
          <Grid columns={2} gap={0} align="start">
            <Table
              framed={false}
              headers={["Symbol", "Meaning", "How it is computed", "Now"]}
              columnAlign={["left", "left", "left", "right"]}
              rows={[
                ["B", "bytes one CU loads per k-step", `(BM + BN) x BK x 2 = (${bm} + ${bn}) x ${bk} x 2`, `${d.B / 1024} KB (${d.B / 1024} requests)`],
                [
                  "C",
                  "MFMA time per k-step (4 waves in 2x2, one per SIMD, in parallel)",
                  `per wave: (16x16 blocks / 4 waves) x (BK / 32) MFMAs of 16x16x32 = ${d.mfmaPerWave}; x 16 cycles / ${clock.toFixed(2)} GHz`,
                  `${d.C.toFixed(0)} ns`,
                ],
                ["Q", "bytes the TCP lets one CU have in flight", "hardware: 32 KB (slider)", `${qKB} KB (${qKB} slots)`],
                ["L", "round trip of one request (issue to data)", "slider: 550 ns cold, 385 ns warm (inferred)", `${L} ns`],
                ["R", "one CU's peak MFMA rate", `4096 FLOP/clk x ${clock.toFixed(2)} GHz`, `${(R_of(clock) / 1000).toFixed(2)} TFLOPS`],
                ["I", "tile intensity: FLOP per byte loaded", `BM x BN / (BM + BN)`, `${d.intensity.toFixed(0)} FLOP/B`],
                [
                  "SIMDs",
                  "SIMDs with work: the tile's 16x16 output blocks shared by 4 waves",
                  `(BM/16) x (BN/16) = ${d.blocks} blocks`,
                  `${d.active} of 4`,
                ],
              ]}
            />
            <Table
              framed={false}
              headers={["Ratio", "What it shows", "Now"]}
              columnAlign={["left", "left", "right"]}
              rowTone={[
                undefined,
                undefined,
                undefined,
                undefined,
                short ? "danger" : "success",
                short ? "danger" : "success",
                d.issueCeil < 1 ? "danger" : "success",
                d.pipeCeil < 1 ? "danger" : "success",
                d.ceiling < 0.95 ? "danger" : "success",
              ]}
              rows={[
                ["B / C", "bandwidth the MFMAs consume (demand)", `${d.need.toFixed(0)} B/ns`],
                ["Q / L", "bandwidth the window can deliver (Little's law)", `${d.deliver.toFixed(0)} B/ns`],
                [
                  "L / C",
                  "k-steps of data that must be outstanding at any instant to hide the latency",
                  `${d.stepsNeeded.toFixed(2)} (= ${((d.stepsNeeded * d.B) / 1024).toFixed(0)} KB)`,
                ],
                ["Q / B", "k-steps of data the TCP can hold outstanding", `${d.stepsAllowed.toFixed(2)} (= ${qKB} KB)`],
                [
                  "(B/Q) x L  vs  C",
                  `time to stream one k-step through the window: ${(d.B / d.Q).toFixed(2)} window-fulls x L; must not exceed C`,
                  `${((d.B / d.Q) * L).toFixed(0)} vs ${d.C.toFixed(0)} ns`,
                ],
                [
                  "(Q/B) / (L/C)",
                  "window ceiling; the same as (Q/L) / (B/C), C / ((B/Q) x L) and I x (Q/L) / R",
                  `${pct(d.windowCeil)}${short ? "" : " (capped at 100%)"}`,
                ],
                [
                  "64 B/clk / (B / C_cyc)",
                  "TCP processing ceiling: 1 KB per 16 cycles, against what the tile needs per cycle",
                  pct(d.issueCeil),
                ],
                [
                  pipe === String(PIPE_SERIAL)
                    ? "C / (C + L + T_issue)"
                    : release === "end"
                      ? "N x C / (C + L + T_issue)"
                      : "N x C / (L + T_issue)",
                  pipe === String(PIPE_SERIAL)
                    ? "pipeline ceiling, serial: every k-step pays the round trip and the issue time"
                    : pipe === String(PIPE_INF)
                      ? "pipeline ceiling: unlimited buffers, no limit"
                      : release === "end"
                        ? `pipeline ceiling, ${pipe} LDS buffer${pipe === "1" ? "" : "s"} freed at compute end: loads lead their use by (${pipe} - 1) x C, which must cover L + T_issue (T_issue = ${d.tIssue.toFixed(0)} ns)${pipe === "1" ? "; 1 buffer is the same as serial" : ""}`
                        : `pipeline ceiling, ${pipe} LDS buffer${pipe === "1" ? "" : "s"} freed at compute start: loads lead their use by ${pipe} x C, which must cover L + T_issue (T_issue = ${d.tIssue.toFixed(0)} ns)`,
                  pct(d.pipeCeil),
                ],
                [
                  d.simdFrac < 1 ? `${d.active}/4 SIMDs x min of the three` : "min of the three",
                  "MFMA efficiency ceiling (the headline number): an upper bound; when two limits are close they compound, and the simulated timeline lands lower",
                  pct(d.ceiling),
                ],
              ]}
            />
          </Grid>
        </CardBody>
      </Card>

      <Grid columns="minmax(0, 1fr) minmax(0, 2.6fr)" gap={20} align="start">
        <Card>
          <CardHeader>The cause: area against edges</CardHeader>
          <CardBody>
            <TileGeometry p={p} />
          </CardBody>
        </Card>
        <Stack gap={8}>
          <H2>The effect: slots fill, MFMAs wait</H2>
          <Row gap={10} align="center" wrap>
            <Button
              variant="primary"
              onClick={() => {
                setPos(0);
                setPlaying(true);
              }}
            >
              Start from kernel start
            </Button>
            <Button
              variant="secondary"
              onClick={() => {
                if (!playing && pos >= MAX_POS_NS) setPos(0);
                setPlaying(!playing);
              }}
            >
              {playing ? "Pause" : "Play"}
            </Button>
            <Text size="small" tone="secondary">
              Speed
            </Text>
            <Select
              value={speed}
              onChange={setSpeed}
              options={[
                { value: "0.05", label: "50 ns / s" },
                { value: "0.1", label: "100 ns / s" },
                { value: "0.25", label: "250 ns / s" },
                { value: "0.5", label: "500 ns / s" },
                { value: "1", label: "1 us / s" },
                { value: "2", label: "2 us / s" },
                { value: "4", label: "4 us / s" },
              ]}
              style={{ width: 110 }}
            />
          </Row>
          <Stack gap={2}>
            <Row justify="space-between">
              <Text size="small" tone="secondary">
                Drag to scrub (window shows {WINDOW_NS} ns)
              </Text>
              <Text size="small" weight="medium">
                {Math.round(pos)} - {Math.round(pos + WINDOW_NS)} ns
              </Text>
            </Row>
            <input
              type="range"
              min={0}
              max={MAX_POS_NS}
              step={5}
              value={pos}
              onChange={(e: { target: HTMLInputElement }) => setPos(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: t.accent.primary }}
            />
          </Stack>
          <Row gap={8} align="center">
            <Toggle checked={unlimited} onChange={setUnlimited} />
            <Text size="small" tone="secondary">
              Unlimited slots: show what full speed would need (requests paced just in time, no 32 KB limit)
            </Text>
          </Row>
          <Timeline p={p} pos={pos} hover={hover} setHover={setHover} unlimited={unlimited} />
        </Stack>
      </Grid>

      <Grid columns={2} gap={24} align="start">
        <Stack gap={8}>
          <H2>Where the prefetch distance must sit</H2>
          <WindowLine p={p} />
        </Stack>
        <Stack gap={8}>
          <H2>The ceiling for any tile</H2>
          <CeilingCurve p={p} />
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Which limit binds: the per-CU window or the shared hierarchy</H2>
        <Text size="small" tone="secondary">
          Every ceiling in GB/s per CU: the shortest bar binds. A memory-bound stream spread over all CUs needs only about 25 GB/s per
          CU from HBM, under the window, so HBM binds. A small-tile GEMM needs about 90 GB/s per CU, mostly from L2, so the window
          binds long before L2 or HBM. Click the presets. Here the hit rate and the CU count are free inputs; for a whole GEMM the
          shape sets both, and shape-bound-intuition derives them from M, N, K, the tile and GROUP_SIZE_M.
        </Text>
        <BandwidthLadder p={p} />
      </Stack>

      <Callout tone="info" title="The chain in one line">
        <Text>
          Halving one side of the tile halves the compute per k-step but removes only a quarter of the bytes, so the CU needs 1.5x the
          bandwidth. The window's delivery rate, Q / L, does not change with the tile, so the slots fill and the MFMAs wait. Only a bigger
          tile, a shorter L (operands in the MALL), or a lower clock closes the gap. Three L1-side limits in all: the window (slots x 1 KB
          / L), the TCP's processing rate (64 B/clk; 64x64x64 exceeds it even with unlimited slots), and the pipeline depth (N LDS buffers
          let a CU load only N k-steps ahead; serial loads nothing ahead). At 32 KB the slots bind before 2 buffers do; turn on unlimited
          slots to see the buffers alone: with 2, 128x128x64 reaches 63% and 128x128x128 95%, which is why BK = 128 removed the vmcnt
          stalls. Step through Serial, 1, 2, 3 buffers to see how much lookahead each tile needs.
        </Text>
      </Callout>
      <Text size="small" tone={t.kind === "light" ? "tertiary" : "quaternary"}>
        Model per CU: each of the 4 waves (2x2) issues (BM/2 / 16)(BN/2 / 16)(BK / 32) MFMAs of 16x16x32 per k-step, 16 cycles each, on
        its own SIMD, so C = that count x 16 cycles / clock; B = (BM + BN) x BK x 2 bytes; ceiling = min(1, (Q/B) / (L/C)).
      </Text>
    </Stack>
  );
}
