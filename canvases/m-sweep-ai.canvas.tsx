import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Select,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
  useState,
} from "cursor/canvas";

// ---------------------------------------------------------------------------------------------
// Constants. Same chip numbers as shape-bound-intuition: 256 CUs x 4096 bf16 FLOP/clk, clocks
// measured under load in v9_tile, HBM 6.3 TB/s measured streaming read (8 TB/s spec).
// ---------------------------------------------------------------------------------------------
const NUM_CUS = 256;
const FLOP_PER_CLK_CU = 4096;

type Ridge = { id: string; label: string; short: string; clock: number; bw: number };
const RIDGES: Ridge[] = [
  { id: "spec", label: "Spec: 2.4 GHz, 8 TB/s", short: "spec", clock: 2.4, bw: 8.0 },
  { id: "t128", label: "128x128 under load: 2.11 GHz, 6.3 TB/s", short: "128x128", clock: 2.11, bw: 6.3 },
  { id: "t256", label: "256x256 under load: 1.71 GHz, 6.3 TB/s", short: "256x256", clock: 1.71, bw: 6.3 },
  { id: "meas", label: "Spec clock, measured HBM: 2.4 GHz, 6.3 TB/s", short: "measured HBM", clock: 2.4, bw: 6.3 },
];
const peakOf = (r: Ridge) => (NUM_CUS * FLOP_PER_CLK_CU * r.clock) / 1000; // TFLOPS
const ridgeOf = (r: Ridge) => peakOf(r) / r.bw; // FLOP/B
const BAND_LO = Math.min(...RIDGES.map(ridgeOf));
const BAND_HI = Math.max(...RIDGES.map(ridgeOf));

// ---------------------------------------------------------------------------------------------
// Model: bf16 in and out, each element crosses HBM once (compulsory traffic).
// ---------------------------------------------------------------------------------------------
/** 2MNK / 2(MK + NK + MN) = 1 / (1/M + 1/N + 1/K). */
function ai(M: number, N: number, K: number) {
  return 1 / (1 / M + 1 / N + 1 / K);
}
/** Large-M limit of the intensity, and also the M of the knee (where AI is half of it). */
function plateau(N: number, K: number) {
  return (N * K) / (N + K);
}
/** M where the intensity reaches the ridge R, or null if the plateau is at or below R. */
function crossM(N: number, K: number, R: number): number | null {
  const d = 1 / R - 1 / N - 1 / K;
  return d > 0 ? 1 / d : null;
}
/** Smallest N = K that makes a given M compute bound at ridge R, or null if no N, K can. */
function minNK(M: number, R: number): number | null {
  const d = 1 / R - 1 / M;
  return d > 0 ? 2 / d : null;
}
function attained(a: number, r: Ridge) {
  return Math.min(peakOf(r), a * r.bw);
}
function bytes(M: number, N: number, K: number) {
  return { a: 2 * M * K, b: 2 * N * K, c: 2 * M * N };
}

type Regime = "memory" | "depends" | "compute";
function regime(a: number): Regime {
  return a < BAND_LO ? "memory" : a > BAND_HI ? "compute" : "depends";
}
const REGIME_TEXT: Record<Regime, string> = {
  memory: "memory bound under every ridge",
  depends: "inside the ridge band: depends on clock and HBM",
  compute: "compute bound under every ridge",
};

// ---------------------------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------------------------
function fmtSize(v: number) {
  if (!Number.isFinite(v)) return "inf";
  return v >= 1024 && v % 1024 === 0 ? `${v / 1024}K` : `${Math.round(v)}`;
}
function fmtNum(v: number) {
  return Math.round(v).toLocaleString("en-US");
}
function fmtTF(v: number) {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}
function fmtUs(v: number) {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}
function pct(x: number) {
  return `${Math.round(x * 100)}%`;
}
function fmtMB(b: number) {
  const v = b / 1e6;
  return v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2);
}

// ---------------------------------------------------------------------------------------------
// Plot scaffolding (raw SVG, log axes), as in shape-bound-intuition
// ---------------------------------------------------------------------------------------------
function Blank() {
  return <g />;
}
type El = ReturnType<typeof Blank>;
type Axis = { lo: number; hi: number; log?: boolean; ticks: number[]; fmt?: (v: number) => string; label: string };
type Scales = { sx: (v: number) => number; sy: (v: number) => number; x0: number; x1: number; y0: number; y1: number };

function frac(a: Axis, v: number) {
  if (a.log) return v <= 0 ? -1 : (Math.log(v) - Math.log(a.lo)) / (Math.log(a.hi) - Math.log(a.lo));
  return (v - a.lo) / (a.hi - a.lo);
}

function Plot(props: {
  x: Axis;
  y: Axis;
  width?: number;
  height?: number;
  aria: string;
  onPick?: (v: number) => void;
  children: (s: Scales) => El;
}) {
  const t = useHostTheme();
  const W = props.width ?? 640;
  const H = props.height ?? 300;
  const x0 = 56;
  const x1 = W - 14;
  const y1 = 12;
  const y0 = H - 38;
  const cl = (f: number) => Math.min(1.01, Math.max(-0.01, f));
  const sx = (v: number) => x0 + cl(frac(props.x, v)) * (x1 - x0);
  const sy = (v: number) => y0 - cl(frac(props.y, v)) * (y0 - y1);
  const fx = props.x.fmt ?? ((v: number) => `${v}`);
  const fy = props.y.fmt ?? ((v: number) => `${v}`);
  const inv = (f: number) => {
    const a = props.x;
    return a.log ? Math.exp(Math.log(a.lo) + f * (Math.log(a.hi) - Math.log(a.lo))) : a.lo + f * (a.hi - a.lo);
  };
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={props.aria}>
      {props.y.ticks.map((v) => (
        <g key={`y${v}`}>
          <line x1={x0} x2={x1} y1={sy(v)} y2={sy(v)} stroke={t.stroke.tertiary} />
          <text x={x0 - 5} y={sy(v) + 3} fontSize={9} textAnchor="end" fill={t.text.tertiary}>
            {fy(v)}
          </text>
        </g>
      ))}
      {props.x.ticks.map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={y1} y2={y0} stroke={t.stroke.tertiary} />
          <text x={sx(v)} y={y0 + 13} fontSize={9} textAnchor="middle" fill={t.text.tertiary}>
            {fx(v)}
          </text>
        </g>
      ))}
      <line x1={x0} x2={x1} y1={y0} y2={y0} stroke={t.stroke.primary} />
      <line x1={x0} x2={x0} y1={y1} y2={y0} stroke={t.stroke.primary} />
      <text x={(x0 + x1) / 2} y={H - 6} fontSize={10} textAnchor="middle" fill={t.text.secondary}>
        {props.x.label}
      </text>
      <text x={12} y={(y0 + y1) / 2} fontSize={10} textAnchor="middle" fill={t.text.secondary} transform={`rotate(-90 12 ${(y0 + y1) / 2})`}>
        {props.y.label}
      </text>
      {props.children({ sx, sy, x0, x1, y0, y1 })}
      {props.onPick && (
        <rect
          x={x0}
          y={y1}
          width={x1 - x0}
          height={y0 - y1}
          fill="transparent"
          style={{ cursor: "crosshair" }}
          onClick={(e: { clientX: number; currentTarget: { getBoundingClientRect(): DOMRect } }) => {
            const b = e.currentTarget.getBoundingClientRect();
            props.onPick?.(inv((e.clientX - b.left) / b.width));
          }}
        />
      )}
    </svg>
  );
}

function poly(pts: [number, number][], s: Scales) {
  return pts.map(([x, y]) => `${s.sx(x).toFixed(1)},${s.sy(y).toFixed(1)}`).join(" ");
}

function LineSwatch(props: { color: string; dashed?: boolean; width?: number }) {
  return (
    <svg width={20} height={8}>
      <line x1={0} x2={20} y1={4} y2={4} stroke={props.color} strokeWidth={props.width ?? 2} strokeDasharray={props.dashed ? "4 3" : undefined} />
    </svg>
  );
}

function DotSwatch(props: { color: string; hollow?: boolean }) {
  return (
    <svg width={10} height={10}>
      <circle cx={5} cy={5} r={3.5} fill={props.hollow ? "none" : props.color} stroke={props.color} strokeWidth={1.2} />
    </svg>
  );
}

function LegendItem(props: { swatch: El; label: string }) {
  return (
    <Row gap={6} align="center">
      {props.swatch}
      <Text size="small" tone="secondary">
        {props.label}
      </Text>
    </Row>
  );
}

function PillGroup<T extends number | string>(props: {
  label: string;
  options: T[];
  value: T;
  onChange: (v: T) => void;
  fmt?: (v: T) => string;
}) {
  return (
    <Stack gap={4}>
      <Text size="small" tone="secondary">
        {props.label}
      </Text>
      <Row gap={4} wrap>
        {props.options.map((o) => (
          <Pill key={String(o)} size="sm" active={o === props.value} onClick={() => props.onChange(o)}>
            {props.fmt ? props.fmt(o) : String(o)}
          </Pill>
        ))}
      </Row>
    </Stack>
  );
}

function useRegimeColor() {
  const t = useHostTheme();
  return (r: Regime) => (r === "memory" ? t.category.blue : r === "depends" ? t.category.orange : t.category.green);
}

// ---------------------------------------------------------------------------------------------
// Sweep grids
// ---------------------------------------------------------------------------------------------
const M_LO = 1;
const M_HI = 16384;
const M_POW2 = Array.from({ length: 15 }, (_, i) => 2 ** i);
const M_SMOOTH = Array.from({ length: 141 }, (_, i) => 2 ** (i / 10));
const M_TICKS = [1, 4, 16, 64, 256, 1024, 4096, 16384];
const N_OPTS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
const K_OPTS = [1024, 2048, 4096, 8192, 16384, 32768];
const MUL_OPTS = [1, 2, 10];
const GRID_M = M_POW2.filter((m) => m >= 16);

function snapPow2(v: number) {
  return Math.max(M_LO, Math.min(M_HI, 2 ** Math.round(Math.log2(Math.max(1, v)))));
}

// ---------------------------------------------------------------------------------------------
// Panel 1: AI against M
// ---------------------------------------------------------------------------------------------
function AISweep(props: { N: number; K: number; mul: number; ridge: Ridge; M: number; onPickM: (m: number) => void }) {
  const t = useHostTheme();
  const rc = useRegimeColor();
  const { N, K, mul, ridge, M } = props;
  const R = ridgeOf(ridge);
  const Nc = N * mul;
  const P = plateau(N, K);
  const Pc = plateau(Nc, K);
  const mc = crossM(N, K, R);
  const cmpColor = t.category.purple;
  return (
    <Stack gap={8}>
      <Plot
        x={{ lo: M_LO, hi: M_HI, log: true, ticks: M_TICKS, fmt: fmtSize, label: "M (log scale, powers of 2; click to select)" }}
        y={{ lo: 1, hi: 32768, log: true, ticks: [1, 10, 100, 1000, 10000], fmt: fmtNum, label: "Arithmetic intensity (FLOP/B, log)" }}
        width={760}
        height={380}
        aria="Arithmetic intensity of a bf16 GEMM against M for fixed N and K, with the MI355X ridge band"
        onPick={(v) => props.onPickM(snapPow2(v))}
      >
        {(s) => (
          <g>
            <rect x={s.x0} y={s.sy(BAND_HI)} width={s.x1 - s.x0} height={s.sy(BAND_LO) - s.sy(BAND_HI)} fill={t.fill.tertiary} />
            <text x={s.x0 + 6} y={s.sy(BAND_HI) - 4} fontSize={9.5} fill={t.text.tertiary}>
              ridge band {Math.round(BAND_LO)} to {Math.round(BAND_HI)} FLOP/B
            </text>
            <line x1={s.x0} x2={s.x1} y1={s.sy(R)} y2={s.sy(R)} stroke={t.text.secondary} strokeWidth={1.2} />
            <text x={s.x1 - 4} y={s.sy(R) + 12} fontSize={9.5} textAnchor="end" fill={t.text.secondary}>
              selected ridge {Math.round(R)} ({ridge.short})
            </text>

            <polyline points={poly([[M_LO, M_LO], [M_HI, M_HI]], s)} fill="none" stroke={t.text.tertiary} strokeWidth={1} strokeDasharray="5 4" />
            <text x={s.sx(24)} y={s.sy(24) - 8} fontSize={9.5} fill={t.text.tertiary} transform={`rotate(-24 ${s.sx(24)} ${s.sy(24) - 8})`}>
              AI = M (weights dominate bytes)
            </text>
            <line x1={s.x0} x2={s.x1} y1={s.sy(P)} y2={s.sy(P)} stroke={t.accent.primary} strokeWidth={1} strokeDasharray="5 4" opacity={0.7} />
            <text x={s.x0 + 6} y={s.sy(P) - 4} fontSize={9.5} fill={t.accent.primary}>
              plateau NK/(N+K) = {fmtNum(P)}
            </text>
            {mul > 1 && (
              <g>
                <line x1={s.x0} x2={s.x1} y1={s.sy(Pc)} y2={s.sy(Pc)} stroke={cmpColor} strokeWidth={1} strokeDasharray="5 4" opacity={0.7} />
                <text x={s.x1 - 4} y={s.sy(Pc) - 4} fontSize={9.5} textAnchor="end" fill={cmpColor}>
                  plateau with N x {mul} = {fmtNum(Pc)}
                </text>
              </g>
            )}

            <line x1={s.sx(P)} x2={s.sx(P)} y1={s.sy(P / 2) - 14} y2={s.y0} stroke={t.accent.primary} strokeDasharray="2 3" opacity={0.7} />
            <text x={s.sx(P) + 4} y={s.y0 - 6} fontSize={9.5} fill={t.accent.primary}>
              knee M = {fmtNum(P)}
            </text>
            {mc !== null && mc <= M_HI && (
              <g>
                <circle cx={s.sx(mc)} cy={s.sy(R)} r={4} fill="none" stroke={t.text.primary} strokeWidth={1.4} />
                <text x={s.sx(mc) - 6} y={s.sy(R) - 8} fontSize={9.5} textAnchor="end" fill={t.text.primary}>
                  crosses at M = {fmtNum(mc)}
                </text>
              </g>
            )}

            {mul > 1 && (
              <polyline
                points={poly(M_SMOOTH.map((m) => [m, ai(m, Nc, K)] as [number, number]), s)}
                fill="none"
                stroke={cmpColor}
                strokeWidth={1.8}
              />
            )}
            <polyline
              points={poly(M_SMOOTH.map((m) => [m, ai(m, N, K)] as [number, number]), s)}
              fill="none"
              stroke={t.accent.primary}
              strokeWidth={2.6}
            />
            {mul > 1 &&
              M_POW2.map((m) => (
                <circle key={`c${m}`} cx={s.sx(m)} cy={s.sy(ai(m, Nc, K))} r={3} fill={t.bg.editor} stroke={cmpColor} strokeWidth={1.2} />
              ))}
            {M_POW2.map((m) => {
              const a = ai(m, N, K);
              return <circle key={m} cx={s.sx(m)} cy={s.sy(a)} r={4} fill={rc(regime(a))} stroke={t.bg.editor} strokeWidth={1.2} />;
            })}

            <g pointerEvents="none">
              <line x1={s.sx(M)} x2={s.sx(M)} y1={s.y1} y2={s.y0} stroke={t.accent.primary} strokeWidth={1} opacity={0.6} />
              <circle cx={s.sx(M)} cy={s.sy(ai(M, N, K))} r={6.5} fill="none" stroke={t.text.primary} strokeWidth={1.5} />
            </g>
          </g>
        )}
      </Plot>
      <Row gap={16} wrap>
        <LegendItem swatch={<LineSwatch color={t.accent.primary} width={3} />} label={`N = ${fmtSize(N)}, K = ${fmtSize(K)}`} />
        {mul > 1 && <LegendItem swatch={<LineSwatch color={cmpColor} />} label={`N x ${mul} = ${fmtSize(Nc)}, same K`} />}
        <LegendItem swatch={<LineSwatch color={t.text.tertiary} dashed width={1} />} label="straight-line limits" />
        <LegendItem swatch={<DotSwatch color={rc("memory")} />} label="memory bound under every ridge" />
        <LegendItem swatch={<DotSwatch color={rc("depends")} />} label="inside the ridge band" />
        <LegendItem swatch={<DotSwatch color={rc("compute")} />} label="compute bound under every ridge" />
      </Row>
      <Text size="small" tone="tertiary">
        AI = 2MNK / 2(MK + NK + MN), bf16 inputs and output, each byte across HBM once (the best case, so memory bound here is
        guaranteed and compute bound is only allowed). On log axes the curve is a slope-1 ramp hugging AI = M, then a plateau at
        NK/(N+K). The knee sits at M = NK/(N+K), where AI is exactly half the plateau. Multiplying N only moves the plateau, so it
        changes nothing left of the knee.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Readout for the selected M
// ---------------------------------------------------------------------------------------------
function ByteBar(props: { M: number; N: number; K: number }) {
  const t = useHostTheme();
  const b = bytes(props.M, props.N, props.K);
  const tot = b.a + b.b + b.c;
  const parts = [
    { key: "a", label: "A, M x K", v: b.a, color: t.category.blue },
    { key: "b", label: "B (weights), N x K", v: b.b, color: t.category.orange },
    { key: "c", label: "C, M x N", v: b.c, color: t.category.green },
  ];
  const W = 320;
  let x = 0;
  return (
    <Stack gap={6}>
      <svg width="100%" viewBox={`0 0 ${W} 16`} role="img" aria-label="Share of HBM bytes from A, B and C">
        {parts.map((p) => {
          const w = (p.v / tot) * W;
          const el = <rect key={p.key} x={x} y={0} width={Math.max(0, w)} height={16} fill={p.color} opacity={0.75} />;
          x += w;
          return el;
        })}
      </svg>
      <Row gap={12} wrap>
        {parts.map((p) => (
          <LegendItem key={p.key} swatch={<DotSwatch color={p.color} />} label={`${p.label}: ${pct(p.v / tot)}`} />
        ))}
      </Row>
    </Stack>
  );
}

function Readout(props: { M: number; N: number; K: number; mul: number; ridge: Ridge }) {
  const { M, N, K, mul, ridge } = props;
  const R = ridgeOf(ridge);
  const a = ai(M, N, K);
  const flops = 2 * M * N * K;
  const b = bytes(M, N, K);
  const tot = b.a + b.b + b.c;
  const tc = flops / (peakOf(ridge) * 1e6);
  const tm = tot / (ridge.bw * 1e6);
  const Nc = N * mul;
  const ac = ai(M, Nc, K);
  const bc = bytes(M, Nc, K);
  const totC = bc.a + bc.b + bc.c;
  const r = regime(a);
  return (
    <Card>
      <CardHeader trailing={`${fmtSize(M)} x ${fmtSize(N)} x ${fmtSize(K)}`}>Selected shape</CardHeader>
      <CardBody>
        <Stack gap={14}>
          <Row gap={24} wrap>
            <Stat value={`${fmtNum(a)}`} label="FLOP/B" tone={r === "memory" ? "info" : r === "compute" ? "success" : "warning"} />
            <Stat value={a < R ? "memory" : "compute"} label={`bound at ridge ${Math.round(R)}`} />
            <Stat value={`${fmtTF(attained(a, ridge))}`} label={`TFLOPS ceiling (${pct(attained(a, ridge) / peakOf(ridge))} of peak)`} />
          </Row>
          <Text size="small">Across all four ridges: {REGIME_TEXT[r]}.</Text>
          <Stack gap={4}>
            <Text size="small" weight="medium">
              Where the {fmtMB(tot)} MB of HBM traffic comes from
            </Text>
            <ByteBar M={M} N={N} K={K} />
          </Stack>
          <Text size="small" tone="secondary">
            Time at the roofline: compute {fmtUs(tc)} us, memory {fmtUs(tm)} us, so {tm > tc ? `memory takes ${(tm / tc).toFixed(1)}x` : `compute takes ${(tc / tm).toFixed(1)}x`} as long
            as the other.
          </Text>
          {mul > 1 && (
            <Text size="small" tone="secondary">
              With N x {mul}: FLOPs x {mul}, bytes x {(totC / tot).toFixed(2)} (A is unchanged; B and C grow x {mul}), so AI goes from{" "}
              {fmtNum(a)} to {fmtNum(ac)} ({ac / a - 1 < 0.005 ? "unchanged" : `+${pct(ac / a - 1)}`}).
              {b.b > 0.5 * tot ? " The weights already dominate the bytes, so the extra FLOPs bring their own bytes." : ""}
            </Text>
          )}
        </Stack>
      </CardBody>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------
// Panel 2: regime grid, M rows against N columns at fixed K
// ---------------------------------------------------------------------------------------------
function RegimeGrid(props: { K: number; ridge: Ridge; M: number; N: number; onPick: (m: number, n: number) => void }) {
  const t = useHostTheme();
  const rc = useRegimeColor();
  const { K, M, N } = props;
  const R = ridgeOf(props.ridge);
  const left = 54;
  const top = 30;
  const cw = 54;
  const ch = 22;
  const W = left + N_OPTS.length * cw + 4;
  const H = top + GRID_M.length * ch + 4;
  return (
    <Stack gap={8}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Arithmetic intensity for each M and N at fixed K, coloured by regime">
        <text x={left + (N_OPTS.length * cw) / 2} y={10} fontSize={10} textAnchor="middle" fill={t.text.secondary}>
          N
        </text>
        <text x={12} y={top + (GRID_M.length * ch) / 2} fontSize={10} textAnchor="middle" fill={t.text.secondary} transform={`rotate(-90 12 ${top + (GRID_M.length * ch) / 2})`}>
          M
        </text>
        {N_OPTS.map((n, j) => (
          <text key={n} x={left + j * cw + cw / 2} y={top - 6} fontSize={9.5} textAnchor="middle" fill={n === N ? t.text.primary : t.text.tertiary} fontWeight={n === N ? 600 : 400}>
            {fmtSize(n)}
          </text>
        ))}
        {GRID_M.map((m, i) => (
          <g key={m}>
            <text x={left - 6} y={top + i * ch + ch / 2 + 3} fontSize={9.5} textAnchor="end" fill={m === M ? t.text.primary : t.text.tertiary} fontWeight={m === M ? 600 : 400}>
              {fmtSize(m)}
            </text>
            {N_OPTS.map((n, j) => {
              const a = ai(m, n, K);
              const sel = m === M && n === N;
              return (
                <g key={n} style={{ cursor: "pointer" }} onClick={() => props.onPick(m, n)}>
                  <rect
                    x={left + j * cw + 1}
                    y={top + i * ch + 1}
                    width={cw - 2}
                    height={ch - 2}
                    rx={2}
                    fill={rc(regime(a))}
                    opacity={a >= R ? 0.42 : 0.2}
                  />
                  {sel && <rect x={left + j * cw + 1} y={top + i * ch + 1} width={cw - 2} height={ch - 2} rx={2} fill="none" stroke={t.text.primary} strokeWidth={1.6} />}
                  <text x={left + j * cw + cw / 2} y={top + i * ch + ch / 2 + 3} fontSize={9.5} textAnchor="middle" fill={t.text.primary}>
                    {fmtNum(a)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <Row gap={14} wrap>
        <LegendItem swatch={<DotSwatch color={rc("memory")} />} label={`below ${Math.round(BAND_LO)}: memory bound`} />
        <LegendItem swatch={<DotSwatch color={rc("depends")} />} label={`${Math.round(BAND_LO)} to ${Math.round(BAND_HI)}: depends`} />
        <LegendItem swatch={<DotSwatch color={rc("compute")} />} label={`above ${Math.round(BAND_HI)}: compute bound`} />
      </Row>
      <Text size="small" tone="tertiary">
        Each cell is the AI in FLOP/B at K = {fmtSize(K)}; darker cells clear the selected ridge of {Math.round(R)}. Click a cell to
        select that M and N. Reading down a column is the M sweep; reading across a row shows that once M is small, N stops
        mattering.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Panel 3: minimum N = K for each M
// ---------------------------------------------------------------------------------------------
const TABLE_M = [128, 256, 384, 512, 768, 1024, 2048, 4096, 8192, Infinity];

function MinNKTable(props: { ridge: Ridge }) {
  const rows = TABLE_M.map((m) => RIDGES.map((r) => minNK(m, ridgeOf(r))));
  return (
    <Stack gap={8}>
      <Table
        headers={["M", ...RIDGES.map((r) => `${r.short}: ${Math.round(ridgeOf(r))}${r.id === props.ridge.id ? " (selected)" : ""}`)]}
        columnAlign={["left", ...RIDGES.map(() => "right" as const)]}
        rowTone={rows.map((vals) =>
          vals.every((v) => v === null) ? "danger" : vals.some((v) => v === null || v > 2048) ? "warning" : "success",
        )}
        rows={TABLE_M.map((m, i) => [fmtSize(m), ...rows[i].map((v) => (v === null ? "never" : fmtNum(v)))])}
      />
      <Text size="small" tone="tertiary">
        Smallest N = K that makes M compute bound: N = 2 / (1/R - 1/M). Red: no N or K can. Amber: at least one ridge needs N = K
        above 2048 or can never get there. Green: N = K = 2048 is compute bound under every ridge. Even at infinite M, N = K must
        exceed twice the ridge.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Panel 4: attained TFLOPS against M
// ---------------------------------------------------------------------------------------------
function TflopsSweep(props: { N: number; K: number; mul: number; ridge: Ridge; M: number; onPickM: (m: number) => void }) {
  const t = useHostTheme();
  const { N, K, mul, ridge, M } = props;
  const peak = peakOf(ridge);
  const cmpColor = t.category.purple;
  const others = RIDGES.filter((r) => r.id !== ridge.id);
  return (
    <Stack gap={8}>
      <Plot
        x={{ lo: M_LO, hi: M_HI, log: true, ticks: M_TICKS, fmt: fmtSize, label: "M (log scale; click to select)" }}
        y={{ lo: 1, hi: 4000, log: true, ticks: [1, 3, 10, 30, 100, 300, 1000, 3000], label: "Roofline ceiling (TFLOPS, log)" }}
        width={520}
        height={320}
        aria="Roofline TFLOPS ceiling against M for the selected N and K"
        onPick={(v) => props.onPickM(snapPow2(v))}
      >
        {(s) => (
          <g>
            {others.map((r) => (
              <polyline
                key={r.id}
                points={poly(M_SMOOTH.map((m) => [m, attained(ai(m, N, K), r)] as [number, number]), s)}
                fill="none"
                stroke={t.text.tertiary}
                strokeWidth={1}
                opacity={0.6}
              />
            ))}
            <line x1={s.x0} x2={s.x1} y1={s.sy(peak)} y2={s.sy(peak)} stroke={t.text.secondary} strokeDasharray="4 3" />
            <text x={s.x0 + 6} y={s.sy(peak) - 4} fontSize={9.5} fill={t.text.secondary}>
              peak {fmtTF(peak)} TFLOPS ({ridge.short})
            </text>
            {mul > 1 && (
              <polyline
                points={poly(M_SMOOTH.map((m) => [m, attained(ai(m, N * mul, K), ridge)] as [number, number]), s)}
                fill="none"
                stroke={cmpColor}
                strokeWidth={1.8}
              />
            )}
            <polyline
              points={poly(M_SMOOTH.map((m) => [m, attained(ai(m, N, K), ridge)] as [number, number]), s)}
              fill="none"
              stroke={t.accent.primary}
              strokeWidth={2.6}
            />
            <g pointerEvents="none">
              <line x1={s.sx(M)} x2={s.sx(M)} y1={s.y1} y2={s.y0} stroke={t.accent.primary} strokeWidth={1} opacity={0.6} />
              <circle cx={s.sx(M)} cy={s.sy(attained(ai(M, N, K), ridge))} r={5} fill={t.accent.primary} stroke={t.bg.editor} strokeWidth={1.5} />
              <text x={s.sx(M) + (M > 1024 ? -8 : 8)} y={s.sy(attained(ai(M, N, K), ridge)) + 16} fontSize={10} textAnchor={M > 1024 ? "end" : "start"} fill={t.text.primary}>
                {fmtTF(attained(ai(M, N, K), ridge))} TFLOPS
              </text>
            </g>
          </g>
        )}
      </Plot>
      <Row gap={14} wrap>
        <LegendItem swatch={<LineSwatch color={t.accent.primary} width={3} />} label={`N = ${fmtSize(N)}, K = ${fmtSize(K)}, selected ridge`} />
        {mul > 1 && <LegendItem swatch={<LineSwatch color={cmpColor} />} label={`N x ${mul}, selected ridge`} />}
        <LegendItem swatch={<LineSwatch color={t.text.tertiary} width={1} />} label="same shape, other ridge presets" />
      </Row>
      <Text size="small" tone="tertiary">
        Ceiling = min(peak, AI x HBM bandwidth). Below the knee it grows linearly with M (each extra row of A reuses the same weights);
        the flat part is the compute peak. This is the roofline only: CU fill, wave tails and L2 hit rate come on top (see
        shape-bound-intuition).
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// L2 hit rate against the TCP window. Per-CU model from shape-bound-intuition: a CU can have at most
// Q = 32 KB of loads in flight (hits and misses alike), so it pulls at most Q / L_mean, where L_mean is
// the hit-weighted round trip. The chip's misses then reach HBM at CUs x Q x (1 - h) / L_mean at most.
// ---------------------------------------------------------------------------------------------
const Q_BYTES = 32 * 1024;
const HBM_MEAS = 6300; // B/ns, measured streaming read
const XCDS = 8;
const TCP_BYTES_PER_CLK = 64;
const L2_BYTES_PER_CLK_XCD = 2048;

type TileColor = "blue" | "cyan" | "green" | "yellow" | "pink" | "red";
type TileSpec = { id: string; label: string; bm: number; bn: number; bk: number; clock: number; color: TileColor; assumed?: boolean };
// Clocks measured under full load in v9_tile; the three small tiles are assumed.
const TILES: TileSpec[] = [
  { id: "256x256", label: "256x256x64", bm: 256, bn: 256, bk: 64, clock: 1.71, color: "blue" },
  { id: "256x128", label: "256x128x64", bm: 256, bn: 128, bk: 64, clock: 1.92, color: "cyan" },
  { id: "128x128", label: "128x128x64", bm: 128, bn: 128, bk: 64, clock: 2.11, color: "green" },
  { id: "64x64", label: "64x64x64", bm: 64, bn: 64, bk: 64, clock: 2.2, color: "yellow", assumed: true },
  { id: "32x128", label: "32x128x128", bm: 32, bn: 128, bk: 128, clock: 2.2, color: "pink", assumed: true },
  { id: "16x64", label: "16x64x128", bm: 16, bn: 64, bk: 128, clock: 2.2, color: "red", assumed: true },
];

type SMode = 1 | "auto";
type Lat = { hit: number; miss: number };
type CapKey = "mfma" | "window" | "tcp" | "l2" | "hbm";
const CAP_TEXT: Record<CapKey, string> = {
  mfma: "MFMA (padded tile compute)",
  window: "TCP window",
  tcp: "TCP rate",
  l2: "L2 bandwidth",
  hbm: "HBM bandwidth",
};

/** v9's GROUP_SIZE_M rule: the GM minimising the A + B strips p concurrent tiles need. */
function autoGm(bm: number, bn: number, tm: number, p: number) {
  let best = 1;
  let cost = Infinity;
  for (const gm of [1, 2, 4, 8, 16, 32]) {
    if (gm > Math.max(1, tm)) break;
    const c = gm * bm + Math.ceil(p / gm) * bn;
    if (c < cost) {
      cost = c;
      best = gm;
    }
  }
  return best;
}

/** Mean round trip for a window holding hits and misses in proportion h : 1 - h. */
function meanL(h: number, lat: Lat) {
  return h * lat.hit + (1 - h) * lat.miss;
}

/** Hit rate above which the busy CUs' windows cannot pull 6.3 TB/s of misses (0 if they never can). */
function hCrit(cus: number, lat: Lat) {
  const den = cus * Q_BYTES - HBM_MEAS * (lat.miss - lat.hit);
  if (den <= 0) return 0;
  const u = (HBM_MEAS * lat.hit) / den;
  return u >= 1 ? 0 : 1 - u;
}

function l2Model(M: number, N: number, K: number, tile: TileSpec, sMode: SMode, lat: Lat) {
  const bme = Math.min(tile.bm, M); // rows past M are masked off and load nothing
  const bne = Math.min(tile.bn, N);
  const tm = Math.ceil(M / tile.bm);
  const tn = Math.ceil(N / tile.bn);
  const tiles = tm * tn;
  const ks = Math.ceil(K / tile.bk);
  const S = sMode === "auto" ? Math.max(1, Math.min(ks, Math.floor(NUM_CUS / tiles))) : 1;
  const T = tiles * S;
  const cus = Math.min(T, NUM_CUS);
  const kPer = Math.ceil(ks / S);
  const B = (bme + bne) * tile.bk * 2; // bytes one unit requests per k-step
  const C = (tile.bm * tile.bn * tile.bk) / 2048 / tile.clock; // ns per k-step, MFMAs run the padded tile
  const need = B / C;
  const X = T * kPer * B; // L2 -> CU bytes
  const Uin = 2 * (M * K + N * K);
  const hMax = Math.max(0, 1 - Uin / X);
  const units = Math.ceil(cus / XCDS);
  const g = Math.max(1, Math.min(units, tiles));
  const gm = autoGm(bme, bne, tm, g);
  let bgm: number;
  let bgn: number;
  if (g >= tiles) {
    bgm = tm;
    bgn = tn;
  } else {
    bgm = Math.min(gm, tm, g);
    bgn = Math.min(tn, Math.ceil(g / bgm));
  }
  const h = Math.min(hMax, Math.max(0, 1 - (bgm * bme + bgn * bne) / (g * (bme + bne))));
  const L = meanL(h, lat);
  const caps: Record<CapKey, number> = {
    mfma: need,
    window: Q_BYTES / L,
    tcp: TCP_BYTES_PER_CLK * tile.clock,
    l2: (L2_BYTES_PER_CLK_XCD * XCDS * tile.clock) / cus,
    hbm: h >= 1 ? Infinity : HBM_MEAS / (cus * (1 - h)),
  };
  let bind: CapKey = "mfma";
  for (const k of ["window", "tcp", "l2", "hbm"] as CapKey[]) if (caps[k] < caps[bind]) bind = k;
  const d = caps[bind];
  const misses = X * (1 - h);
  const partialWrite = S > 1 ? S * M * N * 4 : 0; // one fp32 copy of C per K-slice, read back by the reduction
  const partials = 2 * partialWrite;
  const outWrite = 2 * M * N;
  const tLoop = X / (cus * d); // ns
  const tPost = (partials + outWrite) / HBM_MEAS; // ns, after the main loop
  const t = tLoop + tPost;
  return {
    tile, S, T, cus, bme, h, hCrit: hCrit(cus, lat), L, need, caps, bind, d, X, Uin, misses, partials, partialWrite, outWrite,
    partialPerUnit: S > 1 ? bme * bne * 4 : 0,
    allBytes: X + partials + outWrite,
    hbmBytes: misses + partials + outWrite,
    tLoop, tPost, t,
    windowHbm: (cus * Q_BYTES * (1 - h)) / L / 1000, // TB/s of misses the windows could pull
    hbm: (cus * d * (1 - h)) / 1000, // TB/s achieved in the main loop
    tflops: (2 * M * N * K) / t / 1000,
  };
}
type L2Result = ReturnType<typeof l2Model>;

function useTileColor() {
  const t = useHostTheme();
  return (c: TileColor) => t.category[c];
}

const L2_M = M_POW2.filter((m) => m >= 16 && m <= 4096);
const CU_TICKS = [4, 8, 16, 32, 64, 128, 256];

function PhaseChart(props: { rows: L2Result[]; path: L2Result[]; sel: string; lat: Lat; M: number; onPick: (id: string) => void }) {
  const t = useHostTheme();
  const tc = useTileColor();
  const { rows, path, lat } = props;
  const cuGrid = Array.from({ length: 61 }, (_, i) => 2 ** (2 + i / 10));
  const edge = cuGrid.map((c) => [c, hCrit(c, lat)] as [number, number]);
  const cur = rows.find((r) => r.tile.id === props.sel);
  return (
    <Stack gap={8}>
      <Plot
        x={{ lo: 4, hi: 256, log: true, ticks: CU_TICKS, label: "CUs with work (log scale)" }}
        y={{ lo: 0, hi: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], fmt: pct, label: "L2 hit rate" }}
        width={560}
        height={340}
        aria="L2 hit rate against busy CUs for each tile, with the region where the TCP windows cannot saturate HBM"
      >
        {(s) => (
          <g>
            <polygon points={poly([[4, 1], ...edge, [256, 1]], s)} fill={t.category.red} opacity={0.1} />
            <polyline points={poly(edge, s)} fill="none" stroke={t.category.red} strokeWidth={1.4} />
            <text x={s.sx(6)} y={s.sy(0.94)} fontSize={9.5} fill={t.text.secondary}>
              windows cannot pull 6.3 TB/s of misses
            </text>
            <text x={s.sx(256) - 4} y={s.sy(hCrit(256, lat)) + 14} fontSize={9.5} textAnchor="end" fill={t.category.red}>
              h* = {pct(hCrit(256, lat))} at 256 CUs
            </text>
            {cur && (
              <g>
                <polyline points={poly(path.map((r) => [r.cus, r.h] as [number, number]), s)} fill="none" stroke={tc(cur.tile.color)} strokeWidth={1} opacity={0.7} />
                {path.map((r, i) => (
                  <g key={L2_M[i]}>
                    <circle cx={s.sx(r.cus)} cy={s.sy(r.h)} r={2.2} fill={tc(cur.tile.color)} />
                    <text x={s.sx(r.cus) - 5} y={s.sy(r.h) + 3} fontSize={8.5} textAnchor="end" fill={L2_M[i] === props.M ? t.text.primary : t.text.tertiary}>
                      {fmtSize(L2_M[i])}
                    </text>
                  </g>
                ))}
              </g>
            )}
            {rows.map((r) => {
              const sat = r.hbm >= 0.98 * (HBM_MEAS / 1000);
              const isSel = r.tile.id === props.sel;
              return (
                <g key={r.tile.id} style={{ cursor: "pointer" }} onClick={() => props.onPick(r.tile.id)}>
                  <circle
                    cx={s.sx(r.cus)}
                    cy={s.sy(r.h)}
                    r={isSel ? 7 : 5.5}
                    fill={sat ? tc(r.tile.color) : t.bg.editor}
                    stroke={tc(r.tile.color)}
                    strokeWidth={isSel ? 2.4 : 1.6}
                  />
                </g>
              );
            })}
          </g>
        )}
      </Plot>
      <Row gap={12} wrap>
        {rows.map((r) => (
          <LegendItem key={r.tile.id} swatch={<DotSwatch color={tc(r.tile.color)} />} label={`${r.tile.id}: ${r.cus} CUs, ${pct(r.h)}`} />
        ))}
      </Row>
      <Text size="small" tone="tertiary">
        One dot per tile at M = {fmtSize(props.M)}: filled if it saturates HBM, hollow if something else binds first. Red: CUs x 32 KB x
        (1 - h) / L_mean is under 6.3 TB/s, so even if every CU kept its window full, the misses could not saturate HBM. The thin line
        is the selected tile ({props.sel}) as M goes from 16 to 4K, each point labelled with M.
      </Text>
    </Stack>
  );
}

function Ladder(props: { rows: L2Result[]; sel: string }) {
  const t = useHostTheme();
  const tc = useTileColor();
  const left = 92;
  const right = 210;
  const W = 760;
  const rowH = 30;
  const H = props.rows.length * rowH + 34;
  const ax: Axis = { lo: 10, hi: 1000, log: true, ticks: [], label: "" };
  const sx = (v: number) => left + Math.min(1.02, Math.max(0, frac(ax, v))) * (W - left - right);
  const ticks = [10, 20, 50, 100, 200, 500, 1000];
  const capStyle: Record<Exclude<CapKey, "mfma">, { dash?: string; color: string; w: number }> = {
    window: { color: t.text.primary, w: 2 },
    tcp: { color: t.text.tertiary, w: 1.4, dash: "4 3" },
    l2: { color: t.text.tertiary, w: 1.4, dash: "1.5 2.5" },
    hbm: { color: t.category.red, w: 2 },
  };
  return (
    <Stack gap={8}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-CU bandwidth each ceiling allows, for each tile">
        {ticks.map((v) => (
          <g key={v}>
            <line x1={sx(v)} x2={sx(v)} y1={4} y2={H - 26} stroke={t.stroke.tertiary} />
            <text x={sx(v)} y={H - 14} fontSize={9} textAnchor="middle" fill={t.text.tertiary}>
              {v}
            </text>
          </g>
        ))}
        <text x={left + (W - left - right) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill={t.text.secondary}>
          Per-CU load bandwidth, hits + misses (GB/s, log)
        </text>
        {props.rows.map((r, i) => {
          const y = 6 + i * rowH;
          const mid = y + (rowH - 8) / 2;
          const isSel = r.tile.id === props.sel;
          return (
            <g key={r.tile.id}>
              <text x={left - 8} y={mid + 3} fontSize={10} textAnchor="end" fill={isSel ? t.text.primary : t.text.secondary} fontWeight={isSel ? 600 : 400}>
                {r.tile.label}
              </text>
              <rect x={left} y={y + 4} width={Math.max(0, sx(r.d) - left)} height={rowH - 16} fill={tc(r.tile.color)} opacity={isSel ? 0.55 : 0.3} />
              {(Object.keys(capStyle) as Exclude<CapKey, "mfma">[]).map((k) => {
                const v = r.caps[k];
                if (!Number.isFinite(v) || v > 1000) return null;
                const st = capStyle[k];
                return <line key={k} x1={sx(v)} x2={sx(v)} y1={y + 1} y2={y + rowH - 9} stroke={st.color} strokeWidth={st.w} strokeDasharray={st.dash} />;
              })}
              {(() => {
                const x = sx(r.need);
                return <polygon points={`${x},${mid - 5} ${x + 5},${mid} ${x},${mid + 5} ${x - 5},${mid}`} fill={t.bg.editor} stroke={t.text.primary} strokeWidth={1.3} />;
              })()}
              <text x={W - right + 10} y={mid + 3} fontSize={10} fill={t.text.primary}>
                {r.d.toFixed(0)} GB/s: {CAP_TEXT[r.bind]}
              </text>
            </g>
          );
        })}
      </svg>
      <Row gap={14} wrap>
        <LegendItem
          swatch={
            <svg width={12} height={12}>
              <polygon points="6,1 11,6 6,11 1,6" fill="none" stroke={t.text.primary} strokeWidth={1.3} />
            </svg>
          }
          label="what the MFMAs could consume"
        />
        <LegendItem swatch={<LineSwatch color={t.text.primary} />} label="TCP window: 32 KB / L_mean" />
        <LegendItem swatch={<LineSwatch color={t.category.red} />} label="HBM share: 6.3 TB/s / (CUs x (1 - h))" />
        <LegendItem swatch={<LineSwatch color={t.text.tertiary} dashed width={1.4} />} label="TCP rate 64 B/clk" />
        <LegendItem swatch={<LineSwatch color={t.text.tertiary} width={1.4} />} label="L2 share (dotted): 2 KB/clk per XCD / CUs" />
      </Row>
      <Text size="small" tone="tertiary">
        Bar: what each busy CU actually gets, the lowest of the five. The HBM share rises with the hit rate, because only misses use HBM,
        while the window does not care whether a load hits. Once the red line passes the black one, the window binds before HBM does.
      </Text>
    </Stack>
  );
}

function L2Sweeps(props: { sweeps: { tile: TileSpec; res: L2Result[] }[]; M: number; sel: string; lat: Lat }) {
  const t = useHostTheme();
  const tc = useTileColor();
  const xAxis = (label: string): Axis => ({ lo: 16, hi: 4096, log: true, ticks: [16, 64, 256, 1024, 4096], fmt: fmtSize, label });
  const marker = (s: Scales) => <line x1={s.sx(props.M)} x2={s.sx(props.M)} y1={s.y1} y2={s.y0} stroke={t.accent.primary} strokeWidth={1} opacity={0.6} />;
  const line = (r: { tile: TileSpec; res: L2Result[] }, f: (x: L2Result) => number, s: Scales) => (
    <polyline
      key={r.tile.id}
      points={poly(r.res.map((x, i) => [L2_M[i], f(x)] as [number, number]), s)}
      fill="none"
      stroke={tc(r.tile.color)}
      strokeWidth={r.tile.id === props.sel ? 2.6 : 1.3}
      opacity={r.tile.id === props.sel ? 1 : 0.7}
    />
  );
  const sel = props.sweeps.find((x) => x.tile.id === props.sel);
  return (
    <Grid columns={2} gap={24} align="start">
      <Stack gap={6}>
        <Text weight="medium">L2 hit rate against M</Text>
        <Plot x={xAxis("M (N, K as selected)")} y={{ lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: pct, label: "L2 hit rate" }} width={380} height={230} aria="L2 hit rate against M for each tile">
          {(s) => (
            <g>
              {sel && (
                <polyline
                  points={poly(sel.res.map((x, i) => [L2_M[i], x.hCrit] as [number, number]), s)}
                  fill="none"
                  stroke={t.category.red}
                  strokeWidth={1.3}
                  strokeDasharray="5 3"
                />
              )}
              {props.sweeps.map((r) => line(r, (x) => x.h, s))}
              {marker(s)}
            </g>
          )}
        </Plot>
        <Text size="small" tone="tertiary">
          Dashed red: h* for the selected tile&apos;s CU count. Above it, that tile&apos;s windows cannot pull 6.3 TB/s of misses.
        </Text>
      </Stack>
      <Stack gap={6}>
        <Text weight="medium">HBM bandwidth achieved against M</Text>
        <Plot x={xAxis("M (N, K as selected)")} y={{ lo: 0, hi: 8, ticks: [0, 2, 4, 6, 8], label: "HBM reads in the main loop (TB/s)" }} width={380} height={230} aria="HBM bandwidth achieved against M for each tile">
          {(s) => (
            <g>
              <line x1={s.x0} x2={s.x1} y1={s.sy(6.3)} y2={s.sy(6.3)} stroke={t.text.primary} strokeDasharray="4 3" />
              <text x={s.x1 - 4} y={s.sy(6.3) - 4} fontSize={9} textAnchor="end" fill={t.text.secondary}>
                6.3 TB/s measured
              </text>
              {props.sweeps.map((r) => line(r, (x) => x.hbm, s))}
              {marker(s)}
            </g>
          )}
        </Plot>
        <Text size="small" tone="tertiary">
          A memory-bound shape is only memory bound in practice if its line reaches 6.3 TB/s. Below it, the window, the MFMAs or a lack of
          CUs binds instead.
        </Text>
      </Stack>
    </Grid>
  );
}

function tileName(r: L2Result) {
  return `${r.tile.label}${r.tile.assumed ? " *" : ""}`;
}
function dash(bytes: number) {
  return bytes > 0 ? fmtMB(bytes) : "-";
}

function TrafficTable(props: { rows: L2Result[] }) {
  const minHbm = Math.min(...props.rows.map((r) => r.hbmBytes));
  return (
    <Table
      headers={[
        "Tile",
        "S",
        "L2 hit",
        "A + B loads (MB)",
        "of which miss to HBM",
        "Partial writes",
        "Partial reads",
        "Output writes",
        "Partial per unit (KB)",
        "All requests (MB)",
        "HBM reads + writes (MB)",
      ]}
      columnAlign={["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"]}
      rowTone={props.rows.map((r) => (r.hbmBytes <= minHbm * 1.0001 ? "success" : undefined))}
      rows={props.rows.map((r) => [
        tileName(r),
        `${r.S}`,
        pct(r.h),
        fmtMB(r.X),
        fmtMB(r.misses),
        dash(r.partialWrite),
        dash(r.partialWrite),
        fmtMB(r.outWrite),
        r.partialPerUnit > 0 ? `${r.partialPerUnit / 1024}` : "-",
        fmtMB(r.allBytes),
        fmtMB(r.hbmBytes),
      ])}
    />
  );
}

function TimeTable(props: { rows: L2Result[] }) {
  const best = Math.min(...props.rows.map((r) => r.t));
  return (
    <Table
      headers={[
        "Tile",
        "S",
        "CUs",
        "h*",
        "Main loop binds",
        "Main-loop HBM (TB/s)",
        "Main loop (us)",
        "Partials + output (us)",
        "Total (us)",
        "Average HBM (TB/s)",
        "TFLOPS",
      ]}
      columnAlign={["left", "right", "right", "right", "left", "right", "right", "right", "right", "right", "right"]}
      rowTone={props.rows.map((r) => (r.t <= best * 1.0001 ? "success" : undefined))}
      rows={props.rows.map((r) => [
        tileName(r),
        `${r.S}`,
        `${r.cus}`,
        pct(r.hCrit),
        CAP_TEXT[r.bind],
        r.hbm.toFixed(2),
        fmtUs(r.tLoop / 1000),
        fmtUs(r.tPost / 1000),
        fmtUs(r.t / 1000),
        (r.hbmBytes / r.t / 1000).toFixed(2),
        fmtTF(r.tflops),
      ])}
    />
  );
}

function SliderRow(props: { label: string; value: number; onChange: (v: number) => void }) {
  const t = useHostTheme();
  return (
    <Stack gap={4}>
      <Row justify="space-between">
        <Text size="small" tone="secondary">
          {props.label}
        </Text>
        <Text size="small" weight="medium">
          {props.value} ns
        </Text>
      </Row>
      <input
        type="range"
        min={80}
        max={1200}
        step={10}
        value={props.value}
        onChange={(e: { target: HTMLInputElement }) => props.onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: t.accent.primary }}
      />
    </Stack>
  );
}

function L2Section(props: { M: number; N: number; K: number; ridge: Ridge }) {
  const { M, N, K } = props;
  const [sMode, setSMode] = useState<SMode>("auto");
  const [lHit, setLHit] = useState(550);
  const [lMiss, setLMiss] = useState(550);
  const [sel, setSel] = useState("128x128");
  const lat: Lat = { hit: lHit, miss: lMiss };
  const rows = TILES.map((tile) => l2Model(M, N, K, tile, sMode, lat));
  const sweeps = TILES.map((tile) => ({ tile, res: L2_M.map((m) => l2Model(m, N, K, tile, sMode, lat)) }));
  const path = sweeps.find((x) => x.tile.id === sel)?.res ?? [];
  const a = ai(M, N, K);
  const memBound = a < ridgeOf(props.ridge);
  const big = rows[0];
  const small = rows[rows.length - 1];
  const setLat = (h: number, m: number) => {
    setLHit(h);
    setLMiss(m);
  };
  return (
    <Stack gap={18}>
      <Stack gap={6}>
        <H2>
          When memory bound: does a higher L2 hit rate run into the TCP window? ({fmtSize(M)} x {fmtSize(N)} x {fmtSize(K)})
        </H2>
        <Text tone="secondary">
          {memBound
            ? `This shape is memory bound (AI ${fmtNum(a)} against a ridge of ${Math.round(ridgeOf(props.ridge))}). `
            : `This shape is compute bound (AI ${fmtNum(a)}); select M = ${anyMemText()} or less above to see the memory-bound case. `}
          A CU can have at most 32 KB of loads in flight, hits and misses alike, so it can pull at most 32 KB / L_mean. Only misses reach HBM,
          so the busy CUs can pull at most CUs x 32 KB x (1 - h) / L_mean from it. A higher hit rate makes the window less useful
          for HBM: at 256 CUs and L = {lMiss === lHit ? `${lHit} ns` : `${lHit} / ${lMiss} ns`}, HBM cannot be saturated above h* ={" "}
          {pct(hCrit(256, lat))}.
        </Text>
      </Stack>

      <Grid columns="minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.2fr)" gap={20} align="start">
        <Stack gap={10}>
          <PillGroup<SMode> label="Split-K" options={[1, "auto"]} value={sMode} onChange={setSMode} fmt={(v) => (v === "auto" ? "auto: fill 256 CUs" : "S = 1")} />
          <PillGroup label="Highlighted tile" options={TILES.map((x) => x.id)} value={sel} onChange={setSel} />
        </Stack>
        <Stack gap={10}>
          <SliderRow label="Round trip of an L2 hit" value={lHit} onChange={setLHit} />
          <SliderRow label="Round trip of an L2 miss" value={lMiss} onChange={setLMiss} />
        </Stack>
        <Stack gap={6}>
          <Row gap={4} wrap>
            <Pill size="sm" active={lHit === 550 && lMiss === 550} onClick={() => setLat(550, 550)}>
              single L 550 (repo model, cold)
            </Pill>
            <Pill size="sm" active={lHit === 385 && lMiss === 385} onClick={() => setLat(385, 385)}>
              single L 385 (warm MALL)
            </Pill>
            <Pill size="sm" active={lHit === 450 && lMiss === 950} onClick={() => setLat(450, 950)}>
              hit 450 / miss 950
            </Pill>
          </Row>
          <Text size="small" tone="tertiary">
            v9_tile measured a mean of about 550 ns from issue at about 80% hits, with 130 to 280 ns of it between the TCP and L2, and
            found that under load the hit rate barely changes it: queueing at the TCP dominates. 450 / 950 keeps the same 550 ns mean at 80%
            hits but lets hits free their slot sooner.
          </Text>
        </Stack>
      </Grid>

      <Grid columns="minmax(0, 1fr) minmax(0, 1fr)" gap={24} align="start">
        <Stack gap={8}>
          <Text weight="medium">Where each tile sits: hit rate against busy CUs</Text>
          <PhaseChart rows={rows} path={path} sel={sel} lat={lat} M={M} onPick={setSel} />
        </Stack>
        <Callout tone={small.bind === "window" ? "warning" : "info"} title="What the picture says">
          <Stack gap={6}>
            <Text>
              Big tiles: {big.tile.id} at S = {big.S} keeps {big.cus} of 256 CUs busy, hit rate {pct(big.h)}. It is bound by{" "}
              {CAP_TEXT[big.bind]}, at {big.hbm.toFixed(2)} TB/s of HBM.
              {big.bind === "hbm"
                ? " Few hits, so the window has room to spare and HBM binds, as expected."
                : big.cus < NUM_CUS
                  ? " With this few CUs the windows cannot pull enough, even though most loads miss."
                  : ""}
            </Text>
            <Text>
              Small tiles: {small.tile.id} reaches {pct(small.h)} hits on {small.cus} CUs, against h* = {pct(small.hCrit)}. It is bound by{" "}
              {CAP_TEXT[small.bind]}, at {small.hbm.toFixed(2)} TB/s.
            </Text>
            <Text>
              When M fits in one tile row, each weight byte is used by exactly one tile, so B never hits. Every hit is a re-read of the A
              strip that the tile next door already fetched. Shrinking BN multiplies those re-reads: L2 requests grow from{" "}
              {fmtMB(big.X)} to {fmtMB(small.X)} MB, while HBM reads go from {fmtMB(big.misses)} to {fmtMB(small.misses)} MB (A + B once is{" "}
              {fmtMB(small.Uin)} MB). The hit rate rises, but the hits save no HBM traffic. They only take window slots.
            </Text>
            <Text>
              So yes: once h passes h* = 1 - 6.3 TB/s x L / (CUs x 32 KB), which is 1 - 106 / CUs at a single 550 ns, the window binds
              before HBM does. Split-K raises the CU count while BN stays large, which keeps h low; that is why it beats a small tile on
              memory-bound shapes.
            </Text>
          </Stack>
        </Callout>
      </Grid>

      <Stack gap={8}>
        <Text weight="medium">Per-CU bandwidth: which ceiling each tile hits first</Text>
        <Ladder rows={rows} sel={sel} />
      </Stack>

      <L2Sweeps sweeps={sweeps} M={M} sel={sel} lat={lat} />

      <Stack gap={8}>
        <Text weight="medium">
          Bytes moved by every tile on {fmtSize(M)} x {fmtSize(N)} x {fmtSize(K)} (MB unless marked)
        </Text>
        <TrafficTable rows={rows} />
        <Text size="small" tone="tertiary">
          A + B loads are what the units request through the TCP in the main loop, hits and misses. The misses are the part that reaches
          HBM: each XCD fetches the strips its concurrent block needs, so they exceed A + B once ({fmtMB(rows[0]?.Uin ?? 0)} MB). With S
          &gt; 1 every K-slice writes an fp32 partial of its tile (BM x BN x 4 bytes, the per-unit column), S x M x N x 4 bytes in all,
          and the reduction reads every one back, so partial reads equal partial writes. The output is the bf16 C, 2 x M x N bytes. All
          requests = A + B loads + partial writes + partial reads + output. HBM reads + writes = misses + partial writes + partial reads +
          output. Green: the fewest HBM bytes.
        </Text>
      </Stack>

      <Stack gap={8}>
        <Text weight="medium">Where the time goes</Text>
        <TimeTable rows={rows} />
        <Text size="small" tone="tertiary">
          Main loop = A + B loads / (CUs x the per-CU rate the binding ceiling allows), so a window-bound loop runs below 6.3 TB/s of HBM.
          Partials + output = (partial writes + partial reads + output) / 6.3 TB/s, after the main loop and not overlapped. That is
          pessimistic if they stay in the MALL or L2. Average HBM = HBM reads + writes / total time. Green: the fastest tile. Rows past M
          are masked and load nothing, but the MFMAs still run the padded tile. * clock assumed (2.2 GHz). The model is the per-CU one
          from shape-bound-intuition, with the hit-weighted round trip; it ignores wave tails within each wave and CU drift.
        </Text>
      </Stack>
    </Stack>
  );
}

function anyMemText() {
  return `${2 ** Math.floor(Math.log2(BAND_LO))}`;
}

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------
export default function MSweepAI() {
  const [M, setM] = useState(256);
  const [N, setN] = useState(4096);
  const [K, setK] = useState(4096);
  const [mul, setMul] = useState(10);
  const [ridgeId, setRidgeId] = useState("t128");
  const ridge = RIDGES.find((r) => r.id === ridgeId) ?? RIDGES[0];
  const R = ridgeOf(ridge);
  const P = plateau(N, K);
  const mc = crossM(N, K, R);
  const lastMem = [...M_POW2].reverse().find((m) => ai(m, N, K) < BAND_LO);
  const firstCompute = M_POW2.find((m) => ai(m, N, K) > BAND_HI);
  const anyMem = 2 ** Math.floor(Math.log2(BAND_LO));

  return (
    <Stack gap={22} style={{ padding: 4 }}>
      <Stack gap={6}>
        <H1>When does a bf16 GEMM stop being memory bound? Sweeping M on MI355X</H1>
        <Text tone="secondary">
          For bf16 in and out, AI = 1 / (1/M + 1/N + 1/K): a third of the harmonic mean of the three dimensions, so the smallest one
          sets it and AI is always below min(M, N, K). The MI355X ridge is {Math.round(BAND_LO)} to {Math.round(BAND_HI)} FLOP/B
          depending on the clock and HBM bandwidth you assume, so any M of {anyMem} or less is memory bound whatever N and K are.
        </Text>
      </Stack>

      <Row gap={36} align="end" wrap>
        <Stat value={lastMem ? `M ≤ ${fmtSize(lastMem)}` : "none"} label="memory bound under every ridge" tone="info" />
        <Stat
          value={firstCompute ? `M ≥ ${fmtSize(firstCompute)}` : "never"}
          label="compute bound under every ridge"
          tone={firstCompute ? "success" : "danger"}
        />
        <Stat value={mc === null ? "never" : `M = ${fmtNum(mc)}`} label={`crosses the selected ridge (${Math.round(R)})`} />
        <Stat value={fmtNum(P)} label="knee M and plateau AI: NK/(N+K)" />
      </Row>

      <Grid columns="minmax(0, 1.4fr) minmax(0, 1fr) minmax(0, 1.1fr)" gap={20} align="start">
        <Stack gap={10}>
          <PillGroup label="N" options={N_OPTS} value={N} onChange={setN} fmt={fmtSize} />
          <PillGroup label="K" options={K_OPTS} value={K} onChange={setK} fmt={fmtSize} />
        </Stack>
        <Stack gap={10}>
          <PillGroup label="Compare with N x" options={MUL_OPTS} value={mul} onChange={setMul} fmt={(v) => (v === 1 ? "off" : `x${v}`)} />
          <PillGroup label="M (selected)" options={M_POW2} value={M} onChange={setM} fmt={fmtSize} />
        </Stack>
        <Stack gap={4}>
          <Text size="small" tone="secondary">
            Ridge preset (peak / HBM bandwidth)
          </Text>
          <Select
            value={ridgeId}
            onChange={setRidgeId}
            options={RIDGES.map((r) => ({ value: r.id, label: `${r.label}: ${Math.round(ridgeOf(r))} FLOP/B` }))}
          />
          <Text size="small" tone="tertiary">
            256 CUs x 4096 bf16 FLOP/clk. Peak {fmtTF(peakOf(ridge))} TFLOPS at {ridge.clock} GHz, {ridge.bw} TB/s.
          </Text>
        </Stack>
      </Grid>

      <Grid columns="minmax(0, 1.7fr) minmax(0, 1fr)" gap={24} align="start">
        <Stack gap={8}>
          <H2>Arithmetic intensity against M</H2>
          <AISweep N={N} K={K} mul={mul} ridge={ridge} M={M} onPickM={setM} />
        </Stack>
        <Readout M={M} N={N} K={K} mul={mul} ridge={ridge} />
      </Grid>

      <Grid columns="minmax(0, 1.1fr) minmax(0, 1fr)" gap={24} align="start">
        <Stack gap={8}>
          <H2>Regime map at K = {fmtSize(K)}</H2>
          <RegimeGrid
            K={K}
            ridge={ridge}
            M={M}
            N={N}
            onPick={(m, n) => {
              setM(m);
              setN(n);
            }}
          />
        </Stack>
        <Stack gap={8}>
          <H2>What that intensity buys: the roofline ceiling</H2>
          <TflopsSweep N={N} K={K} mul={mul} ridge={ridge} M={M} onPickM={setM} />
        </Stack>
      </Grid>

      <Grid columns="minmax(0, 1.2fr) minmax(0, 1fr)" gap={24} align="start">
        <Stack gap={8}>
          <H2>How big must N = K be before a given M is compute bound?</H2>
          <MinNKTable ridge={ridge} />
        </Stack>
        <Callout tone="info" title="Rules of thumb for MI355X, bf16">
          <Stack gap={6}>
            <Text>
              M of {anyMem} or less is always memory bound: AI is below M, and M is below every ridge. N and K cannot rescue it.
            </Text>
            <Text>
              M = 512 is the transition: compute bound only if N and K are both large (at least {fmtNum(minNK(512, BAND_LO) ?? 0)} to{" "}
              {fmtNum(minNK(512, BAND_HI) ?? 0)}, depending on the ridge).
            </Text>
            <Text>
              If all three dimensions are at least 3 x ridge (about {fmtNum(3 * R)} here), AI is at least the ridge. 1024 and up is
              compute bound for typical layer sizes; 2048 and up almost always.
            </Text>
            <Text>
              Past the knee, M = NK/(N+K), M stops mattering and N, K set the plateau. Even infinite M needs N = K above 2 x ridge
              ({fmtNum(2 * R)} here).
            </Text>
            <Text>
              Scaling N scales the weight bytes with it. Left of the knee the weights dominate the bytes, so N x 10 leaves AI almost
              unchanged; right of the knee it raises the plateau.
            </Text>
          </Stack>
        </Callout>
      </Grid>

      <L2Section M={M} N={N} K={K} ridge={ridge} />

      <Text size="small" tone="tertiary">
        Source: analytical model only, no GPU measurements. Peak = 256 CUs x 4096 bf16 FLOP/clk x clock; clocks 1.71 and 2.11 GHz are
        v9_tile under full load, 2.4 GHz is spec; HBM 6.3 TB/s is the measured streaming read (experiments/hbm_bandwidth_xcd), 8 TB/s
        is spec. Traffic is compulsory (A, B and C each cross HBM once), which is the best case: real kernels re-read A and B strips
        unless L2 and the MALL catch them (docs/roofline.md).
      </Text>
    </Stack>
  );
}
