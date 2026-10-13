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
  useMemo,
  useState,
} from "cursor/canvas";

// ---------------------------------------------------------------------------------------------
// Constants. Per-CU values match tile-tcp-intuition; chip ceilings from experiments/hbm_bandwidth_xcd.
// ---------------------------------------------------------------------------------------------
const NUM_CUS = 256;
const XCDS = 8;
const FLOP_PER_CLK_CU = 4096;
const MFMA_CYCLES = 16; // one v_mfma_f32_16x16x32 (bf16)
const TCP_BYTES_PER_CLK = 64;
const L2_BYTES_PER_CLK_XCD = 2048;
const Q_BYTES = 32 * 1024;
const HBM_MEAS = 6300; // GB/s = B/ns, measured streaming read, all 8 XCDs
const HBM_SPEC = 8000;
const PEAK_CLOCK = 2.4;
const PEAK_SPEC = (NUM_CUS * FLOP_PER_CLK_CU * PEAK_CLOCK) / 1000; // TFLOPS
const L_COLD = 550;
const L_WARM = 385;

type ColorKey = "blue" | "green" | "orange" | "purple" | "accent";
type TileSpec = { id: string; label: string; bm: number; bn: number; bk: number; clock: number; color: ColorKey; assumed?: boolean };

// Clocks: measured under full load in v9_tile (WORKLOG); 64x64 is not measured.
const TILES: TileSpec[] = [
  { id: "256x256", label: "256x256x64", bm: 256, bn: 256, bk: 64, clock: 1.71, color: "blue" },
  { id: "256x128", label: "256x128x64", bm: 256, bn: 128, bk: 64, clock: 1.92, color: "green" },
  { id: "128x128", label: "128x128x64", bm: 128, bn: 128, bk: 64, clock: 2.11, color: "orange" },
  { id: "64x64", label: "64x64x64", bm: 64, bn: 64, bk: 64, clock: 2.2, color: "purple", assumed: true },
];

// v9_beyond_hotloop, 256x256x64, M = K = 4096, cold rotating buffers (wave_quant.csv): [N, TFLOPS]
const MEASURED_WQ: [number, number][] = [
  [512, 242.0], [768, 361.2], [1024, 476.7], [1280, 589.8], [1536, 693.8], [1792, 794.5], [2048, 885.5], [2304, 1028.0],
  [2560, 1141.4], [2816, 1210.5], [3072, 1262.0], [3328, 1288.3], [3584, 1301.5], [3840, 1335.2], [4096, 1372.5],
  [4352, 923.2], [4608, 974.1], [4864, 1021.6], [5120, 1069.7], [5376, 1116.1], [5632, 1146.4], [5888, 1185.4],
  [6144, 1215.0], [6400, 1245.9], [6656, 1267.1], [6912, 1318.7], [7168, 1332.5], [7424, 1341.4], [7680, 1372.9],
  [7936, 1396.4], [8192, 1408.7], [8448, 1107.5], [8704, 1139.2], [8960, 1171.2], [9216, 1205.8], [9472, 1235.8],
  [9728, 1268.4], [9984, 1294.3], [10240, 1318.2], [10496, 1331.7], [10752, 1354.6], [11008, 1362.0], [11264, 1398.8],
  [11520, 1394.7], [11776, 1417.0], [12032, 1435.1], [12288, 1445.3], [12544, 1193.8], [12800, 1221.3], [13056, 1241.7],
  [13312, 1266.6], [13568, 1293.1], [13824, 1317.5], [14080, 1343.5], [14336, 1361.1], [14592, 1370.0], [14848, 1389.2],
  [15104, 1391.9], [15360, 1406.8], [15616, 1416.7], [15872, 1427.5], [16128, 1439.1], [16384, 1436.4],
];

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------
type HitMode = "block" | "ideal" | "manual";
type GmMode = number | "auto";
type Env = { L: number; hitMode: HitMode; hit: number; gm: GmMode };
type SMode = number | "auto";
type Limiter = "window" | "tcp" | "l2" | "hbm";

/** Problem intensity: 2MNK FLOP over the bytes it must move at least once, 2(MK + NK + MN). */
function intensity(M: number, N: number, K: number) {
  return (M * N * K) / (M * K + N * K + M * N);
}

/**
 * v9's GROUP_SIZE_M rule (group_size_m in v9_tile/matmul_kernel.py): the GM that minimises the A + B
 * strips an XCD's p concurrent tiles need, GM*BM + ceil(p/GM)*BN. Ties go to the smaller GM.
 */
function autoGm(tile: TileSpec, tm: number, p: number) {
  let best = 1;
  let cost = Infinity;
  for (const gm of [1, 2, 4, 8, 16, 32]) {
    if (gm > Math.max(1, tm)) break;
    const c = gm * tile.bm + Math.ceil(p / gm) * tile.bn;
    if (c < cost) {
      cost = c;
      best = gm;
    }
  }
  return best;
}

/**
 * L2 hit rate when one XCD runs `units` units at once. v9 gives each XCD a contiguous run of the grouped
 * tile order, so the units it runs together are consecutive: groups of up to `tiles` tiles at the same
 * K-slice (slices of one tile read different K ranges and share nothing). A group of g tiles is a
 * gm x gn block of the output grid; per k-step it requests g x (BM + BN) x BK x 2 bytes but needs only
 * its gm A strips and gn B strips from beyond the L2.
 */
function blockHit(tile: TileSpec, tm: number, tn: number, tiles: number, units: number, gm: number) {
  const g = Math.max(1, Math.min(units, tiles));
  let bgm: number;
  let bgn: number;
  if (g >= tiles) {
    bgm = tm;
    bgn = tn;
  } else {
    bgm = Math.min(gm, tm, g);
    bgn = Math.min(tn, Math.ceil(g / bgm));
  }
  const h = Math.max(0, 1 - (bgm * tile.bm + bgn * tile.bn) / (g * (tile.bm + tile.bn)));
  return { h, gm: bgm, gn: bgn, g, units };
}

function perCU(tile: TileSpec, cus: number, h: number, L: number) {
  const cycles = (tile.bm / 32) * (tile.bn / 32) * (tile.bk / 32) * MFMA_CYCLES;
  const C = cycles / tile.clock;
  const B = (tile.bm + tile.bn) * tile.bk * 2;
  const need = B / C;
  const caps: Record<Limiter, number> = {
    window: Q_BYTES / L,
    tcp: TCP_BYTES_PER_CLK * tile.clock,
    l2: (L2_BYTES_PER_CLK_XCD * XCDS * tile.clock) / cus,
    hbm: h >= 1 ? Infinity : HBM_MEAS / (cus * (1 - h)),
  };
  let limiter: Limiter = "window";
  for (const k of ["tcp", "l2", "hbm"] as Limiter[]) if (caps[k] < caps[limiter]) limiter = k;
  const e = Math.min(1, caps[limiter] / need);
  return { cycles, C, B, need, caps, limiter, e };
}

function splitFor(M: number, N: number, K: number, tile: TileSpec, s: SMode) {
  const tiles = Math.ceil(M / tile.bm) * Math.ceil(N / tile.bn);
  const ks = Math.ceil(K / tile.bk);
  const S = s === "auto" ? Math.max(1, Math.min(ks, Math.floor(NUM_CUS / tiles))) : Math.max(1, Math.min(ks, s));
  return { tiles, ks, S };
}

function model(M: number, N: number, K: number, tile: TileSpec, s: SMode, env: Env) {
  const tm = Math.ceil(M / tile.bm);
  const tn = Math.ceil(N / tile.bn);
  const { tiles, ks, S } = splitFor(M, N, K, tile, s);
  const kPer = Math.ceil(ks / S);
  const T = tiles * S;
  const fullW = Math.floor(T / NUM_CUS);
  const rem = T % NUM_CUS;
  const W = fullW + (rem > 0 ? 1 : 0);
  const cus = Math.min(T, NUM_CUS);
  const B = (tile.bm + tile.bn) * tile.bk * 2;
  const X = T * kPer * B; // L2 -> CU bytes
  const Uin = 2 * (M * K + N * K);
  const hMax = Math.max(0, 1 - Uin / X);
  const unitsFirst = Math.ceil(cus / XCDS); // units one XCD runs at once in the first wave
  const gm = env.gm === "auto" ? autoGm(tile, tm, Math.min(unitsFirst, tiles)) : env.gm;
  const bFull = blockHit(tile, tm, tn, tiles, Math.ceil(Math.min(T, NUM_CUS) / XCDS), gm);
  const bRem = rem > 0 ? blockHit(tile, tm, tn, tiles, Math.ceil(rem / XCDS), gm) : bFull;
  const pickH = (blockH: number) =>
    env.hitMode === "block" ? Math.min(blockH, hMax) : env.hitMode === "ideal" ? hMax : Math.min(env.hit, hMax);
  const hFull = pickH(bFull.h);
  const hRem = pickH(bRem.h);
  const h = 1 - (fullW * NUM_CUS * (1 - hFull) + rem * (1 - hRem)) / T; // every unit requests the same bytes
  const block = fullW > 0 ? bFull : bRem;
  const hFirst = fullW > 0 ? hFull : hRem;
  const pFull = perCU(tile, NUM_CUS, hFull, env.L);
  const pRem = rem > 0 ? perCU(tile, rem, hRem, env.L) : pFull;
  const pFirst = fullW > 0 ? pFull : pRem;
  const unit = kPer * pFull.C; // ns per unit at full MFMA speed
  const tMain = (fullW * unit) / pFull.e + (rem > 0 ? unit / pRem.e : 0);
  const partialBytes = S > 1 ? 2 * S * M * N * 4 : 0; // fp32 partials written, then read back by the reduction
  const outBytes = 2 * M * N;
  const t = tMain + (partialBytes + outBytes) / HBM_MEAS;
  const flops = 2 * M * N * K;
  const tflops = flops / t / 1000;
  const hbmBytes = X * (1 - h) + partialBytes + outBytes;
  const ai = intensity(M, N, K);
  const peak = (NUM_CUS * FLOP_PER_CLK_CU * tile.clock) / 1000;
  const ridge = (peak * 1000) / HBM_MEAS;
  const ideal = Math.min(peak, (ai * HBM_MEAS) / 1000);
  const idealSpec = Math.min(PEAK_SPEC, (ai * HBM_SPEC) / 1000);
  const feedCap = Math.min(pFirst.need, pFirst.caps.window, pFirst.caps.tcp, pFirst.caps.l2);
  const hReq = Math.max(0, 1 - HBM_MEAS / (cus * feedCap));
  // Little's law for the chip: holding 6.3 TB/s for a round trip L needs HBM_MEAS x L bytes in flight. Each
  // CU's window holds at most Q, of which a share 1 - h is misses (L is one average for hits and misses).
  const missInFlight = cus * Q_BYTES * (1 - hFirst);
  const missNeeded = HBM_MEAS * env.L;
  const cusToSaturate = hFirst >= 1 ? Infinity : missNeeded / (Q_BYTES * (1 - hFirst));
  const factors = {
    fill: cus / NUM_CUS,
    tail: T / (W * cus),
    eff: (W * unit) / tMain,
    pad: (M * N * K) / (T * kPer * tile.bm * tile.bn * tile.bk),
    over: tMain / t,
  };
  return {
    tile, M, N, K, tm, tn, tiles, ks, S, kPer, T, fullW, rem, W, cus, B, X, Uin, hMax, h, hFirst, gm, block, pFirst, pFull, pRem,
    tMain, t, partialBytes, outBytes, flops, tflops, hbmTBs: hbmBytes / t / 1000, l2TBs: X / t / 1000,
    ai, peak, ridge, ideal, idealSpec, hReq, missInFlight, missNeeded, cusToSaturate, factors,
  };
}
type Result = ReturnType<typeof model>;

function ridgeN(M: number, K: number, ridge: number): number | null {
  if (intensity(M, 1e8, K) < ridge) return null;
  let lo = 1;
  let hi = 1e8;
  for (let i = 0; i < 80; i++) {
    const mid = Math.sqrt(lo * hi);
    if (intensity(M, mid, K) < ridge) lo = mid;
    else hi = mid;
  }
  return hi;
}

const LIMITER_BOUND: Record<Limiter, string> = {
  window: "L1-window bound",
  tcp: "L1-rate bound",
  l2: "L2 bound",
  hbm: "HBM (memory) bound",
};

function limiterText(r: Result) {
  const p = r.pFirst;
  const need = p.need.toFixed(0);
  switch (p.limiter) {
    case "window":
      return `the 32 KB L1 window delivers ${p.caps.window.toFixed(0)} of the ${need} GB/s each CU needs`;
    case "tcp":
      return `the L1 processes 64 B/clk = ${p.caps.tcp.toFixed(0)} of the ${need} GB/s each CU needs`;
    case "l2":
      return `each CU's L2 share is ${p.caps.l2.toFixed(0)} of the ${need} GB/s it needs`;
    case "hbm":
      return `HBM gives each CU ${p.caps.hbm.toFixed(0)} of ${need} GB/s (${r.cus} CUs x ${pct(1 - r.h)} misses share 6.3 TB/s)`;
  }
}

function binding(r: Result) {
  const f = r.factors;
  const cands = [
    { key: "fill", v: f.fill, bound: "parallelism bound", label: `parallelism: only ${r.cus} of 256 CUs have work` },
    { key: "tail", v: f.tail, bound: "parallelism bound", label: `wave tail: ${r.W} waves, the last has ${r.rem} of 256 units` },
    { key: "eff", v: f.eff, bound: LIMITER_BOUND[r.pFirst.limiter], label: `per-CU feed: ${limiterText(r)}` },
    { key: "pad", v: f.pad, bound: "padding bound", label: "padding: the tiles overhang the shape" },
    {
      key: "over",
      v: f.over,
      bound: r.S > 1 ? "split-K overhead bound" : "output-write bound",
      label: r.S > 1 ? `split-K partials: ${mb(r.partialBytes)} MB through HBM after the main loop` : "writing the output",
    },
  ];
  const worst = cands.reduce((a, c) => (c.v < a.v ? c : a), cands[0]);
  if (worst.v >= 0.9) return { key: "none", v: 1, bound: "MFMA (compute) bound", label: "nothing major: the MFMAs are busy at least 90% of the time" };
  return worst;
}

// ---------------------------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------------------------
function pct(x: number) {
  return `${Math.round(x * 100)}%`;
}
function fmtSize(v: number) {
  return v >= 1024 && v % 1024 === 0 ? `${v / 1024}K` : `${Math.round(v)}`;
}
function fmtTF(v: number) {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}
function mb(bytes: number) {
  const v = bytes / 1e6;
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}
function niceTicks(max: number, n = 5): number[] {
  const raw = max / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

// ---------------------------------------------------------------------------------------------
// Plot scaffolding (raw SVG: log axes)
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
  const x0 = 52;
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

function useColor() {
  const t = useHostTheme();
  return (c: ColorKey) => (c === "accent" ? t.accent.primary : t.category[c]);
}

function LineSwatch(props: { color: string; dashed?: boolean; width?: number }) {
  return (
    <svg width={20} height={8}>
      <line x1={0} x2={20} y1={4} y2={4} stroke={props.color} strokeWidth={props.width ?? 2} strokeDasharray={props.dashed ? "4 3" : undefined} />
    </svg>
  );
}

function LegendItem(props: { swatch: El; label: string; onClick?: () => void; active?: boolean }) {
  return (
    <div style={{ cursor: props.onClick ? "pointer" : undefined }} onClick={props.onClick}>
      <Row gap={6} align="center">
        {props.swatch}
        <Text size="small" tone={props.active ? undefined : "secondary"} weight={props.active ? "medium" : undefined}>
          {props.label}
        </Text>
      </Row>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// Controls helpers
// ---------------------------------------------------------------------------------------------
function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  const t = useHostTheme();
  return (
    <Stack gap={4}>
      <Row justify="space-between">
        <Text size="small" tone="secondary">
          {props.label}
        </Text>
        <Text size="small" weight="medium">
          {props.display}
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

// ---------------------------------------------------------------------------------------------
// Headline: TFLOPS against N
// ---------------------------------------------------------------------------------------------
const N_GRID = Array.from({ length: 512 }, (_, i) => (i + 1) * 64);
const N_TICKS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const TF_TICKS = [1, 3, 10, 30, 100, 300, 1000, 3000];

type SweepLine = { tile: TileSpec; current: boolean; res: Result[] };
type Sweep = { lines: SweepLine[]; alt: Result[]; altMode: SMode };

function NSweepChart(props: { sweep: Sweep; r: Result; s: SMode; onPickN: (n: number) => void; onPickTile: (id: string) => void }) {
  const t = useHostTheme();
  const color = useColor();
  const { sweep, r } = props;
  const M = r.M;
  const K = r.K;
  const cur = sweep.lines.find((l) => l.current);
  const roofMeas: [number, number][] = N_GRID.map((n) => [n, Math.min(r.peak, (intensity(M, n, K) * HBM_MEAS) / 1000)]);
  const roofSpec: [number, number][] = N_GRID.map((n) => [n, Math.min(PEAK_SPEC, (intensity(M, n, K) * HBM_SPEC) / 1000)]);
  const rn = ridgeN(M, K, r.ridge);
  const showMeasured = M === 4096 && K === 4096;
  const altLabel = sweep.altMode === "auto" ? "S = auto (fill 256 CUs)" : "S = 1";
  const sLabel = props.s === "auto" ? "S = auto" : `S = ${props.s}`;
  return (
    <Stack gap={8}>
      <Plot
        x={{ lo: 64, hi: 32768, log: true, ticks: N_TICKS, fmt: fmtSize, label: "N (log scale; click to set N)" }}
        y={{ lo: 1, hi: 4000, log: true, ticks: TF_TICKS, label: "TFLOPS (log scale)" }}
        height={340}
        width={760}
        aria="Modelled TFLOPS against N for each tile, with the rooflines"
        onPick={props.onPickN}
      >
        {(s) => (
          <g>
            {rn !== null && rn > 64 && (
              <g>
                <rect x={s.x0} y={s.y1} width={s.sx(Math.min(rn, 32768)) - s.x0} height={s.y0 - s.y1} fill={t.fill.quaternary} />
                <text x={s.x0 + 6} y={s.y1 + 14} fontSize={9.5} fill={t.text.tertiary}>
                  problem memory bound
                </text>
              </g>
            )}
            {rn !== null && rn <= 32768 && (
              <g>
                <line x1={s.sx(rn)} x2={s.sx(rn)} y1={s.y1} y2={s.y0} stroke={t.text.secondary} strokeDasharray="3 3" />
                <text x={s.sx(rn) + 5} y={s.y1 + 14} fontSize={9.5} fill={t.text.secondary}>
                  problem compute bound from N = {Math.round(rn)}
                </text>
              </g>
            )}
            <polyline points={poly(roofSpec, s)} fill="none" stroke={t.text.tertiary} strokeWidth={1.2} strokeDasharray="6 4" />
            <polyline points={poly(roofMeas, s)} fill="none" stroke={t.text.primary} strokeWidth={1.6} />
            {sweep.lines
              .filter((l) => !l.current)
              .map((l) => (
                <polyline
                  key={l.tile.id}
                  points={poly(l.res.map((x) => [x.N, x.tflops] as [number, number]), s)}
                  fill="none"
                  stroke={color(l.tile.color)}
                  strokeWidth={1.3}
                  opacity={0.65}
                />
              ))}
            {cur && (
              <g>
                <polyline
                  points={poly(sweep.alt.map((x) => [x.N, x.tflops] as [number, number]), s)}
                  fill="none"
                  stroke={color(cur.tile.color)}
                  strokeWidth={1.6}
                  strokeDasharray="5 3"
                />
                <polyline
                  points={poly(cur.res.map((x) => [x.N, x.tflops] as [number, number]), s)}
                  fill="none"
                  stroke={color(cur.tile.color)}
                  strokeWidth={2.6}
                />
              </g>
            )}
            {showMeasured &&
              MEASURED_WQ.map(([n, v]) => <circle key={n} cx={s.sx(n)} cy={s.sy(v)} r={2.6} fill="none" stroke={t.text.primary} strokeWidth={1} />)}
            {r.N >= 64 && r.N <= 32768 && (
              <g pointerEvents="none">
                <line x1={s.sx(r.N)} x2={s.sx(r.N)} y1={s.y1} y2={s.y0} stroke={t.accent.primary} strokeWidth={1} />
                <circle cx={s.sx(r.N)} cy={s.sy(r.tflops)} r={5} fill={t.accent.primary} stroke={t.bg.editor} strokeWidth={1.5} />
                <circle cx={s.sx(r.N)} cy={s.sy(r.ideal)} r={4.5} fill={t.bg.editor} stroke={t.text.primary} strokeWidth={1.5} />
                <text
                  x={s.sx(r.N) + (r.N > 8192 ? -8 : 8)}
                  y={s.sy(r.tflops) + 4}
                  fontSize={10}
                  textAnchor={r.N > 8192 ? "end" : "start"}
                  fill={t.text.primary}
                >
                  N = {r.N}: {fmtTF(r.tflops)} TFLOPS ({pct(r.tflops / r.ideal)} of the roofline)
                </text>
              </g>
            )}
          </g>
        )}
      </Plot>
      <Row gap={16} wrap>
        {sweep.lines.map((l) => (
          <LegendItem
            key={l.tile.id}
            swatch={<LineSwatch color={color(l.tile.color)} width={l.current ? 3 : 1.5} />}
            label={`${l.tile.label}${l.tile.assumed ? " (clock assumed)" : ""}, ${sLabel}`}
            active={l.current}
            onClick={l.tile.id === "custom" ? undefined : () => props.onPickTile(l.tile.id)}
          />
        ))}
        {cur && <LegendItem swatch={<LineSwatch color={color(cur.tile.color)} dashed />} label={`${cur.tile.label}, ${altLabel}`} />}
        <LegendItem swatch={<LineSwatch color={t.text.primary} width={1.6} />} label={`measured roofline: ${fmtTF(r.peak)} TFLOPS at ${r.tile.clock.toFixed(2)} GHz, 6.3 TB/s`} />
        <LegendItem swatch={<LineSwatch color={t.text.tertiary} dashed width={1.2} />} label="spec roofline: 2.5 PF at 2.4 GHz, 8 TB/s" />
        {showMeasured && (
          <LegendItem
            swatch={
              <svg width={10} height={10}>
                <circle cx={5} cy={5} r={3} fill="none" stroke={t.text.primary} />
              </svg>
            }
            label="measured: v9_beyond_hotloop 256x256x64, M = K = 4096"
          />
        )}
      </Row>
      <Text size="small" tone="tertiary">
        Model for M = {M}, K = {K}, bf16 in and out. Each solid line is one tile at {sLabel}; click a legend entry to select that tile.
        Filled dot: what the current tile reaches at N = {r.N}; hollow dot: the measured roofline there. The steps are wave
        quantization: a new wave starts every time the units pass a multiple of 256.
        {showMeasured
          ? " The measured points sit about 20% under the model on full waves (it assumes 100% MFMA busy at the preset clock) and above it on partial waves (with few CUs busy the clock rises)."
          : ""}
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Small multiples on the same N axis
// ---------------------------------------------------------------------------------------------
function SmallMultiples({ sweep, r }: { sweep: Sweep; r: Result }) {
  const t = useHostTheme();
  const color = useColor();
  const cur = sweep.lines.find((l) => l.current) ?? sweep.lines[0];
  const xAxis = (label: string): Axis => ({ lo: 64, hi: 32768, log: true, ticks: [64, 256, 1024, 4096, 16384], fmt: fmtSize, label });
  const marker = (s: Scales) =>
    r.N >= 64 && r.N <= 32768 ? <line x1={s.sx(r.N)} x2={s.sx(r.N)} y1={s.y1} y2={s.y0} stroke={t.accent.primary} strokeWidth={1} /> : <g />;
  const short: [number, number][] = [];
  cur.res.forEach((x, i) => {
    if (x.hReq > x.hFirst + 1e-9) short.push([x.N, (cur.res[i + 1] ?? x).N]);
  });
  const mbMax = Math.max(4, ...sweep.lines.map((l) => Math.max(...l.res.map((x) => x.missInFlight / 1e6))), (r.missNeeded / 1e6) * 1.2);
  const mbTicks = niceTicks(mbMax, 4);
  return (
    <Grid columns={2} gap={24} align="start">
      <Stack gap={6}>
        <Text weight="medium">CUs with work, averaged over waves</Text>
        <Plot
          x={xAxis("N")}
          y={{ lo: 0, hi: 260, ticks: [0, 64, 128, 192, 256], label: "CUs busy (T / waves)" }}
          width={380}
          height={220}
          aria="Average CUs with work against N for each tile"
        >
          {(s) => (
            <g>
              {sweep.lines.map((l) => (
                <polyline
                  key={l.tile.id}
                  points={poly(l.res.map((x) => [x.N, x.T / x.W] as [number, number]), s)}
                  fill="none"
                  stroke={color(l.tile.color)}
                  strokeWidth={l.current ? 2.4 : 1.2}
                  opacity={l.current ? 1 : 0.65}
                />
              ))}
              {marker(s)}
            </g>
          )}
        </Plot>
        <Text size="small" tone="tertiary">
          Units T = tiles x S. One wave is 256 units; below that, CUs sit idle. Now: {r.T} units, {r.cus} CUs, {r.W} wave
          {r.W === 1 ? "" : "s"}.
        </Text>
      </Stack>
      <Stack gap={6}>
        <Text weight="medium">L2 hit rate: needed against modelled ({cur.tile.label})</Text>
        <Plot
          x={xAxis("N")}
          y={{ lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: pct, label: "L2 hit rate, first wave" }}
          width={380}
          height={220}
          aria="Required, modelled and ideal L2 hit rate against N for the current tile"
        >
          {(s) => (
            <g>
              {short.map(([a, b]) => (
                <rect key={a} x={s.sx(a)} y={s.y1} width={Math.max(1, s.sx(b) - s.sx(a))} height={s.y0 - s.y1} fill={t.category.red} opacity={0.12} />
              ))}
              <polyline
                points={poly(cur.res.map((x) => [x.N, x.hMax] as [number, number]), s)}
                fill="none"
                stroke={t.text.tertiary}
                strokeWidth={1.3}
                strokeDasharray="5 3"
              />
              <polyline
                points={poly(cur.res.map((x) => [x.N, x.hFirst] as [number, number]), s)}
                fill="none"
                stroke={t.text.secondary}
                strokeWidth={1.8}
              />
              <polyline
                points={poly(cur.res.map((x) => [x.N, x.hReq] as [number, number]), s)}
                fill="none"
                stroke={color(cur.tile.color)}
                strokeWidth={2.2}
              />
              {marker(s)}
            </g>
          )}
        </Plot>
        <Row gap={12} wrap>
          <LegendItem swatch={<LineSwatch color={color(cur.tile.color)} />} label="needed so HBM does not bind" />
          <LegendItem swatch={<LineSwatch color={t.text.secondary} />} label="modelled (the hit-rate setting)" />
          <LegendItem swatch={<LineSwatch color={t.text.tertiary} dashed width={1.3} />} label="ideal: each input byte from HBM once" />
        </Row>
        <Text size="small" tone="tertiary">
          Red: at the modelled hit rate, HBM binds. Fewer CUs pull less in total, so they need a lower hit rate; the per-CU window caps
          the demand. Now: need {pct(r.hReq)}, modelled {pct(r.hFirst)}, ideal {pct(r.hMax)}.
        </Text>
      </Stack>
      <Stack gap={6}>
        <Text weight="medium">HBM traffic achieved</Text>
        <Plot
          x={xAxis("N")}
          y={{ lo: 0, hi: 8, ticks: [0, 2, 4, 6, 8], label: "HBM TB/s (misses + partials + output)" }}
          width={380}
          height={220}
          aria="Modelled HBM bandwidth against N for each tile"
        >
          {(s) => (
            <g>
              <line x1={s.x0} x2={s.x1} y1={s.sy(6.3)} y2={s.sy(6.3)} stroke={t.text.primary} strokeDasharray="4 3" />
              <text x={s.x1 - 4} y={s.sy(6.3) - 4} fontSize={9} textAnchor="end" fill={t.text.secondary}>
                6.3 TB/s measured
              </text>
              {sweep.lines.map((l) => (
                <polyline
                  key={l.tile.id}
                  points={poly(l.res.map((x) => [x.N, x.hbmTBs] as [number, number]), s)}
                  fill="none"
                  stroke={color(l.tile.color)}
                  strokeWidth={l.current ? 2.4 : 1.2}
                  opacity={l.current ? 1 : 0.65}
                />
              ))}
              {marker(s)}
            </g>
          )}
        </Plot>
        <Text size="small" tone="tertiary">
          A memory-bound problem is only memory bound in practice if this reaches 6.3 TB/s. Now: {r.hbmTBs.toFixed(2)} TB/s (
          {pct(r.hbmTBs / 6.3)}), of which split-K partials {mb(r.partialBytes)} MB.
        </Text>
      </Stack>
      <Stack gap={6}>
        <Text weight="medium">Little&apos;s law for the chip: HBM bytes in flight</Text>
        <Plot
          x={xAxis("N")}
          y={{ lo: 0, hi: mbTicks[mbTicks.length - 1], ticks: mbTicks, fmt: (v) => v.toFixed(v < 10 && v % 1 ? 1 : 0), label: "MB of misses in flight" }}
          width={380}
          height={220}
          aria="Most HBM bytes the busy CUs' windows can hold in flight against what 6.3 TB/s needs, against N"
        >
          {(s) => (
            <g>
              <line x1={s.x0} x2={s.x1} y1={s.sy(r.missNeeded / 1e6)} y2={s.sy(r.missNeeded / 1e6)} stroke={t.text.primary} strokeDasharray="4 3" />
              <text x={s.x1 - 4} y={s.sy(r.missNeeded / 1e6) - 4} fontSize={9} textAnchor="end" fill={t.text.secondary}>
                6.3 TB/s x L = {(r.missNeeded / 1e6).toFixed(2)} MB
              </text>
              {sweep.lines.map((l) => (
                <polyline
                  key={l.tile.id}
                  points={poly(l.res.map((x) => [x.N, x.missInFlight / 1e6] as [number, number]), s)}
                  fill="none"
                  stroke={color(l.tile.color)}
                  strokeWidth={l.current ? 2.4 : 1.2}
                  opacity={l.current ? 1 : 0.65}
                />
              ))}
              {marker(s)}
            </g>
          )}
        </Plot>
        <Text size="small" tone="tertiary">
          Lines: busy CUs x 32 KB x (1 - h), the most miss bytes the windows can have in flight. Under the dashed line HBM cannot reach
          6.3 TB/s however memory bound the problem is. L2 hits take window slots too, so a high hit rate leaves fewer for misses. Now:{" "}
          {(r.missInFlight / 1e6).toFixed(2)} MB from {r.cus} CUs; at this hit rate it takes{" "}
          {Number.isFinite(r.cusToSaturate) ? Math.ceil(r.cusToSaturate) : "infinitely many"} CUs to saturate HBM.
        </Text>
      </Stack>
    </Grid>
  );
}

// ---------------------------------------------------------------------------------------------
// Output grid and the CU map
// ---------------------------------------------------------------------------------------------
function GridDiagram({ r }: { r: Result }) {
  const t = useHostTheme();
  const boxW = 300;
  const boxH = 200;
  let sc = Math.min(boxW / r.N, boxH / r.M);
  let w = r.N * sc;
  let h = r.M * sc;
  const toScale = w >= 8 && h >= 8;
  if (w < 8) w = 8;
  if (h < 8) h = 8;
  const depth = Math.min(r.S, 4) - 1;
  const off = 5;
  const ox = 30;
  const oy = 20 + depth * off;
  const cells = r.tm * r.tn;
  const g = Math.max(1, Math.ceil(Math.sqrt(cells / 2048)));
  const cw = (w * Math.min(1, (r.tn * r.tile.bn) / r.N)) / r.tn;
  const ch = (h * Math.min(1, (r.tm * r.tile.bm) / r.M)) / r.tm;
  const rects: El[] = [];
  for (let i = 0; i < r.tm; i += g) {
    for (let j = 0; j < r.tn; j += g) {
      const idx = i * r.tn + j;
      const wave = Math.floor((idx * r.S) / NUM_CUS);
      const op = r.W > 1 ? 0.85 - (0.6 * wave) / (r.W - 1) : 0.85;
      rects.push(
        <rect
          key={`${i}-${j}`}
          x={ox + j * cw}
          y={oy + i * ch}
          width={Math.min(g, r.tn - j) * cw}
          height={Math.min(g, r.tm - i) * ch}
          fill={t.accent.primary}
          opacity={op}
          stroke={cw * g > 5 && ch * g > 5 ? t.bg.editor : "none"}
          strokeWidth={0.8}
        />,
      );
    }
  }
  const lastFull = r.rem === 0 ? 1 : r.rem / NUM_CUS;
  return (
    <Stack gap={8}>
      <svg width="100%" viewBox={`0 0 ${boxW + 60} ${boxH + 60 + depth * off}`} role="img" aria-label="Output matrix split into tiles, shaded by wave">
        {Array.from({ length: depth }, (_, d) => depth - d).map((d) => (
          <rect key={d} x={ox + d * off} y={oy - d * off} width={w} height={h} fill={t.bg.editor} stroke={t.stroke.secondary} />
        ))}
        {rects}
        <rect x={ox} y={oy} width={w} height={h} fill="none" stroke={t.text.secondary} />
        <rect
          x={ox}
          y={oy}
          width={Math.max(3, r.block.gn * cw)}
          height={Math.max(3, r.block.gm * ch)}
          fill="none"
          stroke={t.text.primary}
          strokeWidth={2}
          strokeDasharray="4 2"
        />
        <text x={ox + w / 2} y={oy - 6 - depth * off} fontSize={10} textAnchor="middle" fill={t.text.secondary}>
          N = {r.N} ({r.tn} tile{r.tn === 1 ? "" : "s"} of {r.tile.bn})
        </text>
        <text
          x={ox - 8}
          y={oy + h / 2}
          fontSize={10}
          textAnchor="middle"
          fill={t.text.secondary}
          transform={`rotate(-90 ${ox - 8} ${oy + h / 2})`}
        >
          M = {r.M} ({r.tm} of {r.tile.bm})
        </text>
        {r.S > 1 && (
          <text x={ox + w + depth * off + 6} y={oy - depth * off + 10} fontSize={9.5} fill={t.text.secondary}>
            x S = {r.S} K-slices
          </text>
        )}
      </svg>
      <Text size="small" tone="tertiary">
        {r.tiles} tile{r.tiles === 1 ? "" : "s"} x S = {r.S} = {r.T} units on 256 CUs: {r.W} wave{r.W === 1 ? "" : "s"}, the last{" "}
        {pct(lastFull)} full. Darker = earlier wave (drawn row-major; the kernel orders tiles in GM-row groups). Dashed outline: the{" "}
        {r.block.gm} x {r.block.gn} block of tiles one XCD runs at once (GM = {r.gm}), which shares its {r.block.gm} A strips and{" "}
        {r.block.gn} B strips through that XCD&apos;s L2: hit rate {pct(Math.min(r.block.h, r.hMax))}.
        {toScale ? "" : " Not to scale: one side is stretched so it stays visible."}
        {g > 1 ? ` Drawn in blocks of ${g}x${g} tiles.` : ""}
      </Text>
    </Stack>
  );
}

function CuMap({ r }: { r: Result }) {
  const t = useHostTheme();
  const cell = 10;
  const gap = 2;
  const left = 46;
  const busy = r.fullW > 0 ? NUM_CUS : r.rem;
  return (
    <Stack gap={8}>
      <svg width="100%" viewBox={`0 0 ${left + 32 * (cell + gap) + 4} ${8 * (cell + gap) + 6}`} role="img" aria-label="Which of the 256 CUs have work in the first wave">
        {Array.from({ length: 8 }, (_, x) => (
          <g key={x}>
            <text x={left - 6} y={x * (cell + gap) + cell - 1} fontSize={8.5} textAnchor="end" fill={t.text.tertiary}>
              XCD {x}
            </text>
            {Array.from({ length: 32 }, (_, c) => {
              const u = c * 8 + x;
              return (
                <rect
                  key={c}
                  x={left + c * (cell + gap)}
                  y={x * (cell + gap)}
                  width={cell}
                  height={cell}
                  rx={1.5}
                  fill={u < busy ? t.accent.primary : t.fill.tertiary}
                />
              );
            })}
          </g>
        ))}
      </svg>
      <Text size="small" tone="tertiary">
        First wave, units dealt round-robin over the 8 XCDs: {busy} of 256 CUs busy. Each busy CU is fed at{" "}
        {pct(r.pFirst.e)} of what its MFMAs need; HBM sees {r.cus} CUs x {pct(1 - r.h)} misses.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Classic roofline with the losses drawn as a vertical chain
// ---------------------------------------------------------------------------------------------
function RooflineChart({ r, others }: { r: Result; others: Result[] }) {
  const t = useHostTheme();
  const color = useColor();
  const aiGrid: number[] = [];
  for (let v = 1; v <= 8192; v *= 1.1) aiGrid.push(v);
  const meas: [number, number][] = aiGrid.map((a) => [a, Math.min(r.peak, (a * HBM_MEAS) / 1000)]);
  const spec: [number, number][] = aiGrid.map((a) => [a, Math.min(PEAK_SPEC, (a * HBM_SPEC) / 1000)]);
  const f = r.factors;
  const chain = [
    { label: "CU fill", f: f.fill },
    { label: "wave tail", f: f.tail },
    { label: "per-CU feed", f: f.eff },
    { label: "padding + writes", f: f.pad * f.over },
  ];
  let v = r.peak;
  const segs = chain.map((c) => {
    const a = v;
    v *= c.f;
    return { ...c, a, b: v };
  });
  return (
    <Stack gap={8}>
      <Plot
        x={{ lo: 1, hi: 8192, log: true, ticks: [1, 4, 16, 64, 256, 1024, 4096], label: "problem intensity: 2MNK / bytes moved at least once (FLOP/B)" }}
        y={{ lo: 1, hi: 4000, log: true, ticks: TF_TICKS, label: "TFLOPS (log scale)" }}
        width={560}
        height={320}
        aria="Roofline with ideal and achieved points for the current shape"
      >
        {(s) => {
          const x = s.sx(r.ai);
          return (
            <g>
              <polyline points={poly(spec, s)} fill="none" stroke={t.text.tertiary} strokeWidth={1.2} strokeDasharray="6 4" />
              <polyline points={poly(meas, s)} fill="none" stroke={t.text.primary} strokeWidth={1.6} />
              <line x1={s.sx(r.ridge)} x2={s.sx(r.ridge)} y1={s.sy(r.peak)} y2={s.y0} stroke={t.stroke.secondary} strokeDasharray="3 3" />
              <text x={s.sx(r.ridge) + 4} y={s.y0 - 6} fontSize={9} fill={t.text.tertiary}>
                ridge {Math.round(r.ridge)}
              </text>
              {segs.map((g, i) =>
                g.f < 0.98 ? (
                  <g key={g.label}>
                    <line x1={x} x2={x} y1={s.sy(g.a)} y2={s.sy(g.b)} stroke={t.category.red} strokeWidth={2.5} opacity={0.75} />
                    <line x1={x - 4} x2={x + 4} y1={s.sy(g.b)} y2={s.sy(g.b)} stroke={t.category.red} strokeWidth={1.5} />
                    {s.sy(g.b) - s.sy(g.a) > 11 && (
                      <text
                        x={r.ai > 1500 ? x - 8 : x + 8}
                        y={(s.sy(g.a) + s.sy(g.b)) / 2 + 3}
                        fontSize={9.5}
                        textAnchor={r.ai > 1500 ? "end" : "start"}
                        fill={t.text.secondary}
                      >
                        {i + 1}. {g.label} x{g.f.toFixed(2)}
                      </text>
                    )}
                  </g>
                ) : null,
              )}
              {others.map((o) => (
                <circle key={o.tile.id} cx={x} cy={s.sy(o.tflops)} r={3.5} fill={color(o.tile.color)} opacity={0.8} />
              ))}
              <circle cx={x} cy={s.sy(r.ideal)} r={5} fill={t.bg.editor} stroke={t.text.primary} strokeWidth={1.6} />
              <circle cx={x} cy={s.sy(r.tflops)} r={5.5} fill={t.accent.primary} stroke={t.bg.editor} strokeWidth={1.5} />
              <text
                x={r.ai > 1500 ? x - 10 : x + 10}
                y={s.sy(r.tflops) + 14}
                fontSize={10}
                textAnchor={r.ai > 1500 ? "end" : "start"}
                fill={t.text.primary}
              >
                achieved {fmtTF(r.tflops)} TFLOPS
              </text>
              <text
                x={r.ai > 1500 ? x - 10 : x + 10}
                y={s.sy(r.ideal) - 8}
                fontSize={10}
                textAnchor={r.ai > 1500 ? "end" : "start"}
                fill={t.text.primary}
              >
                ideal {fmtTF(r.ideal)} ({r.ai < r.ridge ? "memory" : "compute"} bound)
              </text>
            </g>
          );
        }}
      </Plot>
      <Row gap={14} wrap>
        <LegendItem swatch={<LineSwatch color={t.text.primary} width={1.6} />} label="measured roofline" />
        <LegendItem swatch={<LineSwatch color={t.text.tertiary} dashed width={1.2} />} label="spec roofline" />
        <LegendItem swatch={<LineSwatch color={t.category.red} width={2.5} />} label="losses from the compute peak, in order" />
        <LegendItem
          swatch={
            <svg width={10} height={10}>
              <circle cx={5} cy={5} r={3.5} fill={t.text.secondary} />
            </svg>
          }
          label="other tiles, same S mode"
        />
      </Row>
      <Text size="small" tone="tertiary">
        Every tile of a shape sits at the same intensity: the roofline sees only the problem. The tile decides how far below the line
        the kernel lands. The red chain starts at the compute peak, so for a memory-bound shape it passes the ideal point on its
        way down.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Waterfall of the losses (log scale, so each factor is a length)
// ---------------------------------------------------------------------------------------------
function Waterfall({ r }: { r: Result }) {
  const t = useHostTheme();
  const f = r.factors;
  const lim = r.pFirst.limiter;
  const limShort = { window: "32 KB L1 window", tcp: "L1 rate 64 B/clk", l2: "L2 share", hbm: "HBM share" }[lim];
  const rows = [
    { label: `Measured peak: 256 CUs x 4096 FLOP/clk x ${r.tile.clock.toFixed(2)} GHz`, f: 1 },
    { label: `CU fill: ${r.cus} of 256 CUs have work`, f: f.fill },
    { label: `Wave tail: ${r.T} units in ${r.W} wave${r.W === 1 ? "" : "s"}`, f: f.tail },
    { label: `Per-CU feed: ${f.eff < 0.995 ? limShort : "MFMAs fully fed"}`, f: f.eff },
    { label: `Padding: ${r.tm} x ${r.tn} tiles, ${r.kPer} k-steps per unit`, f: f.pad },
    { label: r.S > 1 ? `Split-K partials (${mb(r.partialBytes)} MB) + output` : `Output writes (${mb(r.outBytes)} MB)`, f: f.over },
  ];
  let v = r.peak;
  const vals = rows.map((row) => {
    const a = v;
    v *= row.f;
    return { ...row, a, b: v };
  });
  const lo = Math.min(1, r.tflops / 2);
  const W = 720;
  const left = 300;
  const right = 120;
  const rowH = 26;
  const H = (rows.length + 1) * rowH + 34;
  const ax: Axis = { lo, hi: r.peak * 1.05, log: true, ticks: [], label: "" };
  const sx = (x: number) => left + Math.max(0, frac(ax, x)) * (W - left - right);
  const ticks = [1, 10, 100, 1000].filter((x) => x >= lo && x <= r.peak * 1.05);
  return (
    <Stack gap={8}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Waterfall of losses from peak to achieved TFLOPS">
        {ticks.map((x) => (
          <g key={x}>
            <line x1={sx(x)} x2={sx(x)} y1={4} y2={H - 26} stroke={t.stroke.tertiary} />
            <text x={sx(x)} y={H - 14} fontSize={9} textAnchor="middle" fill={t.text.tertiary}>
              {x}
            </text>
          </g>
        ))}
        <text x={left + (W - left - right) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill={t.text.secondary}>
          TFLOPS (log scale: each factor is a length)
        </text>
        {vals.map((row, i) => {
          const y = 6 + i * rowH;
          return (
            <g key={row.label}>
              <text x={left - 8} y={y + 13} fontSize={10} textAnchor="end" fill={row.f < 0.9 ? t.text.primary : t.text.secondary}>
                {row.label}
              </text>
              <rect x={left} y={y + 3} width={sx(row.b) - left} height={rowH - 10} fill={t.fill.secondary} />
              {row.f < 0.999 && (
                <rect x={sx(row.b)} y={y + 3} width={sx(row.a) - sx(row.b)} height={rowH - 10} fill={t.category.red} opacity={0.3} />
              )}
              <text x={sx(row.a) + 6} y={y + 13} fontSize={10} fill={t.text.primary}>
                {i === 0 ? `${fmtTF(row.b)}` : `x${row.f.toFixed(2)}  ${fmtTF(row.b)}`}
              </text>
            </g>
          );
        })}
        {(() => {
          const y = 6 + rows.length * rowH;
          return (
            <g>
              <text x={left - 8} y={y + 13} fontSize={10.5} textAnchor="end" fill={t.text.primary} fontWeight={600}>
                Achieved
              </text>
              <rect x={left} y={y + 3} width={sx(r.tflops) - left} height={rowH - 10} fill={t.accent.primary} />
              <text x={sx(r.tflops) + 6} y={y + 13} fontSize={10.5} fill={t.text.primary}>
                {fmtTF(r.tflops)} TFLOPS
              </text>
            </g>
          );
        })()}
        <line x1={sx(r.ideal)} x2={sx(r.ideal)} y1={2} y2={H - 26} stroke={t.text.primary} strokeDasharray="4 3" />
        <text x={sx(r.ideal) - 4} y={H - 30} fontSize={9} textAnchor="end" fill={t.text.secondary}>
          roofline {fmtTF(r.ideal)}
        </text>
      </svg>
      <Text size="small" tone="tertiary">
        Achieved = peak x CU fill x wave tail x per-CU feed x padding x (main loop / total time). Per-CU feed is the existing canvas's
        ceiling, min(window, L1 rate, L2 share, HBM share) / (B/C), with the CU count set by the shape. The red part of each row is
        what that factor costs.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// TFLOPS against the split-K factor
// ---------------------------------------------------------------------------------------------
const S_GRID = Array.from({ length: 64 }, (_, i) => i + 1);

function SChart({ lines, r, env, onPickS }: { lines: TileSpec[]; r: Result; env: Env; onPickS: (s: number) => void }) {
  const t = useHostTheme();
  const color = useColor();
  const data = useMemo(
    () =>
      lines.map((tile) => {
        const res = S_GRID.map((S) => model(r.M, r.N, r.K, tile, S, env));
        const best = res.reduce((a, x) => (x.tflops > a.tflops ? x : a), res[0]);
        return { tile, res, best };
      }),
    [lines, r.M, r.N, r.K, env.L, env.hitMode, env.hit, env.gm],
  );
  const maxV = Math.max(...data.map((d) => d.best.tflops), r.ideal * 0.2);
  const yTicks = niceTicks(maxV * 1.1);
  const yHi = yTicks[yTicks.length - 1];
  return (
    <Stack gap={8}>
      <Plot
        x={{ lo: 1, hi: 64, log: true, ticks: [1, 2, 4, 8, 16, 32, 64], label: "split-K factor S (log scale; click to set S)" }}
        y={{ lo: 0, hi: yHi, ticks: yTicks, label: "TFLOPS" }}
        width={560}
        height={300}
        aria="Modelled TFLOPS against split-K factor for each tile on the current shape"
        onPick={(v) => onPickS(Math.max(1, Math.min(64, Math.round(v))))}
      >
        {(s) => (
          <g>
            {r.ideal <= yHi && (
              <g>
                <line x1={s.x0} x2={s.x1} y1={s.sy(r.ideal)} y2={s.sy(r.ideal)} stroke={t.text.primary} strokeDasharray="4 3" />
                <text x={s.x1 - 4} y={s.sy(r.ideal) - 4} fontSize={9} textAnchor="end" fill={t.text.secondary}>
                  roofline {fmtTF(r.ideal)}
                </text>
              </g>
            )}
            {data.map((d) => {
              const isCur = d.tile.id === r.tile.id;
              return (
                <g key={d.tile.id}>
                  <polyline
                    points={poly(d.res.map((x, i) => [S_GRID[i], x.tflops] as [number, number]), s)}
                    fill="none"
                    stroke={color(d.tile.color)}
                    strokeWidth={isCur ? 2.4 : 1.3}
                    opacity={isCur ? 1 : 0.65}
                  />
                  <circle cx={s.sx(d.best.S)} cy={s.sy(d.best.tflops)} r={isCur ? 4.5 : 3} fill={color(d.tile.color)} />
                </g>
              );
            })}
            <line x1={s.sx(r.S)} x2={s.sx(r.S)} y1={s.y1} y2={s.y0} stroke={t.accent.primary} strokeWidth={1} />
          </g>
        )}
      </Plot>
      <Row gap={14} wrap>
        {data.map((d) => (
          <LegendItem
            key={d.tile.id}
            swatch={<LineSwatch color={color(d.tile.color)} width={d.tile.id === r.tile.id ? 3 : 1.5} />}
            label={`${d.tile.label}: best S = ${d.best.S}, ${fmtTF(d.best.tflops)} TFLOPS`}
            active={d.tile.id === r.tile.id}
          />
        ))}
      </Row>
      <Text size="small" tone="tertiary">
        Raising S multiplies the units (more CUs, shorter K per unit) but writes S fp32 copies of the output and reads them back:
        2 x S x M x N x 4 bytes through HBM. Dot: each tile's best S. The curve flattens once the units fill 256 CUs and falls
        once the partials cost more than the extra CUs save. Vertical line: the current S = {r.S}.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------------------------
// Every tile on this shape
// ---------------------------------------------------------------------------------------------
function TileTable({ lines, M, N, K, env }: { lines: TileSpec[]; M: number; N: number; K: number; env: Env }) {
  const rows: Result[] = [];
  for (const tile of lines) {
    const one = model(M, N, K, tile, 1, env);
    const auto = model(M, N, K, tile, "auto", env);
    let best = one;
    for (const S of S_GRID) {
      const x = model(M, N, K, tile, S, env);
      if (x.tflops > best.tflops) best = x;
    }
    rows.push(one);
    if (auto.S !== 1) rows.push(auto);
    if (best.S !== 1 && best.S !== auto.S) rows.push(best);
  }
  const top = Math.max(...rows.map((x) => x.tflops));
  return (
    <Table
      headers={["Tile", "S", "Units", "CUs", "Per-CU feed", "Hit rate needed / modelled", "HBM TB/s", "TFLOPS", "Of roofline", "Binds first"]}
      columnAlign={["left", "right", "right", "right", "right", "right", "right", "right", "right", "left"]}
      rowTone={rows.map((x) => (x.tflops >= top * 0.999 ? "success" : x.tflops / x.ideal < 0.2 ? "danger" : undefined))}
      rows={rows.map((x) => [
        x.tile.label,
        `${x.S}`,
        `${x.T}`,
        `${x.cus}${x.W > 1 ? ` x ${x.W} waves` : ""}`,
        pct(x.pFirst.e),
        `${pct(x.hReq)} / ${pct(x.hFirst)}`,
        x.hbmTBs.toFixed(2),
        fmtTF(x.tflops),
        pct(x.tflops / x.ideal),
        binding(x).bound,
      ])}
    />
  );
}

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------
type Scenario = { id: string; label: string; M: number; N: number; K: number; tile: string; s: SMode };
const SCENARIOS: Scenario[] = [
  { id: "a", label: "M=1024 N=512: 256x256", M: 1024, N: 512, K: 8192, tile: "256x256", s: 1 },
  { id: "b", label: "same, 128x128", M: 1024, N: 512, K: 8192, tile: "128x128", s: 1 },
  { id: "c", label: "same, 128x128 + split-K", M: 1024, N: 512, K: 8192, tile: "128x128", s: "auto" },
  { id: "d", label: "M=1024 N=128: memory bound", M: 1024, N: 128, K: 8192, tile: "128x128", s: "auto" },
  { id: "e", label: "4096 cubed against measured", M: 4096, N: 4096, K: 4096, tile: "256x256", s: 1 },
  { id: "f", label: "4096x2048x8192: 256x128 (v9 README)", M: 4096, N: 2048, K: 8192, tile: "256x128", s: 1 },
  { id: "g", label: "M=64 N=8192: decode-like", M: 64, N: 8192, K: 8192, tile: "64x64", s: 1 },
];

const M_OPTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
const GM_OPTS: GmMode[] = ["auto", 1, 2, 4, 8, 16, 32];
const N_OPTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const K_OPTS = [1024, 2048, 4096, 8192, 16384, 32768];
const S_OPTS: SMode[] = [1, 2, 4, 8, 16, 32, 64, "auto"];

export default function ShapeBoundIntuition() {
  const t = useHostTheme();
  const [M, setM] = useState(1024);
  const [N, setN] = useState(512);
  const [K, setK] = useState(8192);
  const [bm, setBm] = useState(256);
  const [bn, setBn] = useState(256);
  const [bk, setBk] = useState(64);
  const [clock, setClock] = useState(1.71);
  const [s, setS] = useState<SMode>(1);
  const [L, setL] = useState(L_COLD);
  const [hitMode, setHitMode] = useState<HitMode>("block");
  const [hit, setHit] = useState(0.8);
  const [gm, setGm] = useState<GmMode>("auto");

  const env: Env = { L, hitMode, hit, gm };
  const ref = TILES.find((x) => x.bm === bm && x.bn === bn && x.bk === bk);
  const cur: TileSpec = ref
    ? { ...ref, clock }
    : { id: "custom", label: `${bm}x${bn}x${bk} (custom)`, bm, bn, bk, clock, color: "accent" };
  const lineTiles: TileSpec[] = useMemo(() => {
    const out = TILES.map((x) => (x.id === cur.id ? cur : x));
    if (cur.id === "custom") out.push(cur);
    return out;
  }, [bm, bn, bk, clock]);
  const r = model(M, N, K, cur, s, env);
  const b = binding(r);
  const altMode: SMode = s === "auto" ? 1 : "auto";

  const sweep: Sweep = useMemo(
    () => ({
      lines: lineTiles.map((tile) => ({ tile, current: tile.id === cur.id, res: N_GRID.map((n) => model(M, n, K, tile, s, env)) })),
      alt: N_GRID.map((n) => model(M, n, K, cur, altMode, env)),
      altMode,
    }),
    [lineTiles, M, K, s, L, hitMode, hit, gm],
  );
  const others = lineTiles.filter((x) => x.id !== cur.id).map((x) => model(M, N, K, x, s, env));

  const pickTile = (id: string) => {
    const x = TILES.find((y) => y.id === id);
    if (!x) return;
    setBm(x.bm);
    setBn(x.bn);
    setBk(x.bk);
    setClock(x.clock);
  };
  const applyScenario = (sc: Scenario) => {
    setM(sc.M);
    setN(sc.N);
    setK(sc.K);
    pickTile(sc.tile);
    setS(sc.s);
  };
  const ratio = r.tflops / r.ideal;
  const problemMem = r.ai < r.ridge;
  const rn = ridgeN(M, K, r.ridge);
  const fill256 = Math.ceil(NUM_CUS / Math.ceil(M / bm)) * bn;

  // Worked comparison on the current shape: the current tile at S = 1 against the best tile and S
  const cmpOne = model(M, N, K, cur, 1, env);
  let cmpBest = cmpOne;
  for (const tile of lineTiles)
    for (const S of S_GRID) {
      const x = model(M, N, K, tile, S, env);
      if (x.tflops > cmpBest.tflops) cmpBest = x;
    }

  return (
    <Stack gap={22} style={{ padding: 4 }}>
      <Stack gap={6}>
        <H1>From shape to bound: what the problem allows and what a tile reaches</H1>
        <Text tone="secondary">
          The roofline says whether a GEMM shape is compute or memory bound. It assumes every CU has work and every byte crosses HBM
          once. A tile decides how many CUs get work, how fast each must be fed, and how much traffic reaches L2 and HBM. Pick a
          shape and a tile and see which ceiling the kernel actually hits.
        </Text>
        <Text size="small" tone="tertiary">
          Model per CU from tile-tcp-intuition (16 cycles per MFMA, 32 KB window, L = 550 ns cold / 385 ns warm, 64 B/clk L1). Chip
          ceilings from experiments/hbm_bandwidth_xcd: L2 2 KB/clk per XCD, HBM 6.3 TB/s measured (8 TB/s spec). Split-K partials go
          through HBM both ways (experiments/l2_memsys).
        </Text>
      </Stack>

      <Row gap={6} wrap align="center">
        <Text size="small" tone="secondary">
          Scenarios:
        </Text>
        {SCENARIOS.map((sc) => (
          <Pill
            key={sc.id}
            size="sm"
            active={sc.M === M && sc.N === N && sc.K === K && sc.s === s && ref?.id === sc.tile}
            onClick={() => applyScenario(sc)}
          >
            {sc.label}
          </Pill>
        ))}
      </Row>

      <Row gap={36} align="end" wrap>
        <Stat
          value={problemMem ? "memory bound" : "compute bound"}
          label={`the problem: ${Math.round(r.ai)} FLOP/B against a ridge of ${Math.round(r.ridge)}`}
        />
        <Stat value={`${fmtTF(r.ideal)} TFLOPS`} label="roofline (measured HBM, preset clock)" />
        <Stat
          value={`${fmtTF(r.tflops)} TFLOPS`}
          label={`this tile reaches: ${pct(ratio)} of the roofline`}
          tone={ratio >= 0.8 ? "success" : ratio >= 0.4 ? "warning" : "danger"}
        />
        <Stat value={`${r.hbmTBs.toFixed(2)} TB/s`} label={`HBM traffic (${pct(r.hbmTBs / 6.3)} of 6.3)`} />
        <Stat value={`${r.cus} / 256`} label={`CUs with work, ${r.W} wave${r.W === 1 ? "" : "s"}`} />
      </Row>
      <Text weight="medium">
        This kernel is {b.bound}. Binds first: {b.label}.
      </Text>

      <Grid columns="minmax(0, 1.2fr) minmax(0, 1.1fr) minmax(0, 0.9fr) minmax(0, 1fr)" gap={20} align="start">
        <Stack gap={10}>
          <PillGroup label="M" options={M_OPTS} value={M} onChange={setM} fmt={fmtSize} />
          <PillGroup label="N" options={N_OPTS} value={N} onChange={setN} fmt={fmtSize} />
          <Slider
            label="N, fine (multiples of 64)"
            value={Math.log2(N)}
            min={6}
            max={15}
            step={1 / 32}
            display={`${N}`}
            onChange={(v) => setN(Math.max(64, Math.round(Math.pow(2, v) / 64) * 64))}
          />
          <PillGroup label="K" options={K_OPTS} value={K} onChange={setK} fmt={fmtSize} />
        </Stack>
        <Stack gap={10}>
          <Stack gap={4}>
            <Text size="small" tone="secondary">
              Tile preset (sets the measured clock)
            </Text>
            <Select
              value={ref ? ref.id : "custom"}
              onChange={pickTile}
              options={[
                ...TILES.map((x) => ({ value: x.id, label: `${x.label} @ ${x.clock} GHz${x.assumed ? " (assumed)" : ""}` })),
                { value: "custom", label: "custom", disabled: true },
              ]}
            />
          </Stack>
          <Row gap={14} wrap>
            <PillGroup label="BM" options={[64, 128, 256]} value={bm} onChange={setBm} />
            <PillGroup label="BN" options={[64, 128, 256]} value={bn} onChange={setBn} />
            <PillGroup label="BK" options={[64, 128]} value={bk} onChange={setBk} />
          </Row>
          <Slider label="Clock" value={clock} min={1.5} max={2.4} step={0.01} display={`${clock.toFixed(2)} GHz`} onChange={setClock} />
        </Stack>
        <Stack gap={10}>
          <PillGroup<SMode> label="Split-K S" options={S_OPTS} value={s} onChange={setS} fmt={(v) => (v === "auto" ? "auto" : `${v}`)} />
          <Text size="small" tone="tertiary">
            auto = enough K-slices to fill 256 CUs. Now S = {r.S}: {r.tiles} tile{r.tiles === 1 ? "" : "s"} x {r.S} = {r.T} units,{" "}
            {r.kPer} k-step{r.kPer === 1 ? "" : "s"} of {bk} each.
          </Text>
        </Stack>
        <Stack gap={10}>
          <Slider label="Round trip L" value={L} min={150} max={1200} step={5} display={`${L} ns`} onChange={setL} />
          <Row gap={6}>
            <Pill size="sm" active={L === L_COLD} onClick={() => setL(L_COLD)}>
              cold 550
            </Pill>
            <Pill size="sm" active={L === L_WARM} onClick={() => setL(L_WARM)}>
              warm 385
            </Pill>
          </Row>
          <Text size="small" tone="tertiary">
            Inputs A + B = {mb(r.Uin)} MB:{" "}
            {r.Uin <= 256e6
              ? "they fit in the 256 MB MALL, so a repeated call can run warm."
              : "more than the 256 MB MALL, so expect cold."}
          </Text>
          <Stack gap={4}>
            <Text size="small" tone="secondary">
              L2 hit rate
            </Text>
            <Select
              value={hitMode}
              onChange={(v: string) => setHitMode(v as HitMode)}
              options={[
                { value: "block", label: "Concurrent block per XCD (realistic)" },
                { value: "ideal", label: "Ideal: each input byte from HBM once" },
                { value: "manual", label: "Manual" },
              ]}
            />
          </Stack>
          {hitMode === "block" && (
            <Stack gap={4}>
              <PillGroup<GmMode> label="GROUP_SIZE_M" options={GM_OPTS} value={gm} onChange={setGm} fmt={(v) => `${v}`} />
              <Text size="small" tone="tertiary">
                Each XCD runs {r.block.units} unit{r.block.units === 1 ? "" : "s"} at once
                {r.block.g < r.block.units ? ` (${Math.ceil(r.block.units / r.block.g)} K-slices of the same ${r.block.g} tiles)` : ""}: a{" "}
                {r.block.gm} x {r.block.gn} block at GM = {r.gm}, hit rate {pct(r.hFirst)}.
              </Text>
            </Stack>
          )}
          {hitMode === "manual" && (
            <Slider
              label="L2 hit rate (capped at the possible)"
              value={hit}
              min={0}
              max={1}
              step={0.01}
              display={`${pct(hit)}${hit > r.hMax ? ` (capped at ${pct(r.hMax)})` : ""}`}
              onChange={setHit}
            />
          )}
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>What N does M = {M} need, and what does each tile get there?</H2>
        <Text size="small" tone="secondary">
          {rn === null
            ? `With M = ${M} the problem never crosses the ridge, however large N gets: its intensity tops out at MK / (M + K) = ${Math.round((M * K) / (M + K))} FLOP/B.`
            : `The problem crosses the ridge at N = ${Math.round(rn)}. To fill 256 CUs at S = 1, a ${bm}x${bn} tile needs N = ${fill256} (${Math.ceil(M / bm)} tile rows x ${Math.ceil(NUM_CUS / Math.ceil(M / bm))} columns).`}{" "}
          Between those two points the problem is compute bound but the tile leaves CUs idle.
        </Text>
        <NSweepChart sweep={sweep} r={r} s={s} onPickN={(v) => setN(Math.max(64, Math.min(32768, Math.round(v / 64) * 64)))} onPickTile={pickTile} />
      </Stack>

      <SmallMultiples sweep={sweep} r={r} />

      <Grid columns="minmax(0, 1fr) minmax(0, 1.25fr)" gap={24} align="start">
        <Card>
          <CardHeader trailing={`${r.T} units, ${r.W} wave${r.W === 1 ? "" : "s"}`}>What the grid looks like</CardHeader>
          <CardBody>
            <Stack gap={16}>
              <GridDiagram r={r} />
              <CuMap r={r} />
            </Stack>
          </CardBody>
        </Card>
        <Stack gap={8}>
          <H2>Where the kernel lands under the roofline</H2>
          <RooflineChart r={r} others={others} />
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>From peak to achieved, factor by factor</H2>
        <Waterfall r={r} />
      </Stack>

      <Grid columns="minmax(0, 1fr) minmax(0, 1fr)" gap={24} align="start">
        <Stack gap={8}>
          <H2>Split-K: more CUs against partial traffic</H2>
          <SChart lines={lineTiles} r={r} env={env} onPickS={(v) => setS(v)} />
        </Stack>
        <Stack gap={12}>
          <Callout tone="info" title="A compute-bound tile on a memory-bound shape">
            <Text>
              It is possible; it just leaves the GPU idle. On {M}x{N}x{K}, {cmpOne.tile.label} at S = 1 makes {cmpOne.T} unit
              {cmpOne.T === 1 ? "" : "s"}, so {cmpOne.cus} of 256 CUs work. Each is fed at {pct(cmpOne.pFirst.e)}, and the chip
              reaches {fmtTF(cmpOne.tflops)} TFLOPS ({pct(cmpOne.tflops / cmpOne.ideal)} of the roofline) and {cmpOne.hbmTBs.toFixed(2)}{" "}
              TB/s of HBM: neither bound, {binding(cmpOne).bound}. The best the model finds is {cmpBest.tile.label} at S = {cmpBest.S}:{" "}
              {fmtTF(cmpBest.tflops)} TFLOPS ({pct(cmpBest.tflops / cmpBest.ideal)}), {binding(cmpBest).bound}. Smaller tiles add CUs
              but each CU needs more bandwidth (lower tile intensity) and the L2 must hit more often; split-K adds CUs at the same
              intensity but pays for its partials in HBM traffic.
            </Text>
          </Callout>
          <Callout tone="neutral" title="What compute bound and memory bound mean here">
            <Text>
              They describe the problem: its intensity, 2MNK FLOP over the bytes it must move at least once, against the ridge,
              peak FLOP/s over HBM bandwidth. They are ceilings, not outcomes. A kernel reaches the lower one only if its tile clears
              the ceilings in between: enough units to occupy 256 CUs, a per-CU demand B/C that the 32 KB window and the L1 can feed,
              an L2 hit rate high enough for each CU's share of HBM, and split-K partials that cost less than the CUs they buy. A
              memory-bound problem adds one more: enough miss bytes in flight across the chip, busy CUs x 32 KB x (1 - h) against
              6.3 TB/s x L ({(r.missNeeded / 1e6).toFixed(1)} MB here). Name a kernel by the ceiling it actually hits: this one is{" "}
              {b.bound}.
            </Text>
          </Callout>
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Every reference tile on {M}x{N}x{K}</H2>
        <TileTable lines={lineTiles} M={M} N={N} K={K} env={env} />
        <Text size="small" tone="tertiary">
          Each tile at S = 1, at S = auto, and at its best S (1-64) where those differ. Green: the fastest row; red: under 20% of the
          roofline. Per-CU feed is the MFMA busy fraction the window, L1 rate, L2 share and HBM share allow.
        </Text>
      </Stack>

      <Text size="small" tone={t.kind === "light" ? "tertiary" : "quaternary"}>
        Assumptions. Units = ceil(M/BM) x ceil(N/BN) x S, dealt 256 at a time; a unit runs ceil(ceil(K/BK)/S) k-steps of C = (BM/32)(BN/32)(BK/32)
        x 16 cycles / clock, slowed by per-CU feed e = min(1, min(Q/L, 64 B/clk, 16 KB/clk / CUs, 6.3 TB/s / (CUs x misses)) / (B/C)).
        L is one average round trip and does not change with the hit rate or with HBM load (misses really hold slots longer than
        hits, and queues grow near saturation). Concurrent-block hit rate: each XCD runs ceil(CUs / 8) consecutive units of v9&apos;s
        grouped order, a gm x gn block that needs gm A and gn B strips per k-step from beyond its L2, so h = 1 - (gm BM + gn BN) /
        (units (BM + BN)), with K-slices of one tile sharing nothing; this matches the measured 79-83% for 256x256, 256x128 and
        128x128. It assumes the block&apos;s tiles stay in step: CU drift on long shapes lowers it (79% to 74% for a persistent kernel on
        16384x57344x8192, experiments/cu_drift_l2_reuse). Ideal hit rate assumes each input byte crosses HBM once. Split-K partials (2 x S x
        M x N x 4 B) and the bf16 output (2MN B) cross HBM after the main loop, not overlapped. Clocks are measured under full load
        per tile; with few CUs busy the real clock is higher. No prologue, epilogue, launch cost, LDS buffer limit or XCD
        placement effects.
      </Text>
    </Stack>
  );
}
