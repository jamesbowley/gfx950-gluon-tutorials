"""The default Stream-K plan given the tile, the default tile, and the pruned tuning
space, as they would go into AITER next to pick_gemm_tile (gemm_config_utils.py),
with the constants in the gluon DEFAULT.json under "STREAMK".

    from candidates import streamk_config, streamk_default, streamk_candidates
    rule = json.load(open("results/streamk_rule.json"))
    config = streamk_default(M, N, K, (256, 256, 64), 256, rule)  # plan at a given tile
    config = streamk_config(M, N, K, 256, rule)                  # tile and plan
    configs = streamk_candidates(M, N, K, (256, 256, 64), 256, rule)

Notation as in the kernel's docstring: P programs, T tiles, S = T % P tail tiles,
m = P // S programs per tail tile, pairs = ceil(K / (2 BLOCK_K)) K-pairs per tile,
cap = min(m, pairs).

Plan (streamk_default), the same constants for every tile and regime:
- dp when S = 0 or cap < 2;
- one_tile owner when m <= m_ow, on n = sqrt(pairs / c_ow) programs (its n - 1 serial
  partial reads against pairs / n of MAC each), if pairs >= pairs_ow (dp below it);
- else one_tile reduce-scatter on n = sqrt(pairs m / b) programs (pairs / n of MAC
  each against the partials all S n programs store and read at once), if it saves
  at least `gain` pairs of the tail's time, pairs (1 - 1 / n) (dp otherwise).
n is clamped to [2, cap] and rounded to the nearest listed split (OWNER_N, RS_N).

Tile (streamk_pick_tile): the lowest cost over every spec, each spec's work over its
measured full-wave TFLOPS per wave of P workgroups, the tail priced by the plan: a
split tail 1/n of a wave plus split_ksteps / k, a data-parallel partial wave
tail_wave_cost (below_tail_wave_cost below one wave) plus its fill. Ties within 1% go
to the earlier (larger) spec.

Tuning space (streamk_candidates): dp; rs at half, once and twice the plan's n; owner
at every n up to 4 where m <= 8. two_tile, spread and the l2 exchange are left out
(with_extras adds them).
"""

import math

OWNER_N = (2, 3, 4, 6, 8, 12, 16)
RS_N = (2, 4, 6, 8, 12, 16, 32, 64)
GROUP_SIZE_M = 4
# Every spec the kernel runs, largest first: BLOCK_K 64, and 128 where it fits LDS.
SPECS = ((256, 256, 64), (256, 128, 64), (128, 256, 64), (128, 128, 64), (128, 128, 128), (256, 64, 64),
         (64, 256, 64), (128, 64, 64), (128, 64, 128), (64, 128, 64), (64, 128, 128), (64, 64, 64), (64, 64, 128))


def _nearest(n, allowed, cap):
    ok = [a for a in allowed if a <= cap]
    return min(ok, key=lambda a: (abs(math.log(min(n, cap) / a)), a)) if ok else None


def _shape(M, N, K, tile, num_programs):
    bm, bn, bk = tile
    T = -(-M // bm) * -(-N // bn)
    k = -(-K // (2 * bk)) * 2
    S = T % num_programs
    return T, S, k, k // 2


def _config(tile, policy="dp", fixup="owner", split_n=None, exchange="wt"):
    bm, bn, bk = tile
    return {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": GROUP_SIZE_M,
            "STREAMK_POLICY": policy, "STREAMK_FIXUP": fixup, "SPLIT_N": split_n,
            "STREAMK_EXCHANGE": exchange}


def _rs_n(S, pairs, num_programs, c):
    """(the plan's reduce-scatter n before rounding, the cap)."""
    m = num_programs // S
    return max(2.0, math.sqrt(pairs * m / c["b"])), min(m, pairs)


def streamk_default(M, N, K, tile, num_programs, rule):
    """The plan's complete config for a shape at a tile (BLOCK_M, BLOCK_N, BLOCK_K)."""
    c = rule["formula"]
    T, S, k, pairs = _shape(M, N, K, tile, num_programs)
    if S == 0:
        return _config(tile)
    m = num_programs // S
    cap = min(m, pairs)
    if cap < 2:
        return _config(tile)
    if m <= c["m_ow"]:
        if pairs < c["pairs_ow"]:
            return _config(tile)
        n = _nearest(max(2.0, math.sqrt(pairs / c["c_ow"])), OWNER_N, cap)
        return _config(tile, "one_tile", "owner", n) if n else _config(tile)
    n = _nearest(_rs_n(S, pairs, num_programs, c)[0], RS_N, cap)
    if not n or pairs * (1 - 1 / n) < c["gain"]:
        return _config(tile)
    return _config(tile, "one_tile", "rs", n)


def spec_name(tile):
    return "x".join(str(x) for x in tile)


def streamk_tile_cost(M, N, K, tile, num_programs, rule):
    """Waves of the spec's work, priced as the plan runs its tail, over its TFLOPS."""
    t = rule["tile"]
    T, S, k, _ = _shape(M, N, K, tile, num_programs)
    W = T // num_programs
    cfg = streamk_default(M, N, K, tile, num_programs, rule)
    if S == 0:
        waves = W
    elif cfg["STREAMK_POLICY"] == "one_tile":
        waves = W + 1 / cfg["SPLIT_N"] + t["split_ksteps"] / k
    else:
        tail = t["below_tail_wave_cost"] if T < num_programs else t["tail_wave_cost"]
        waves = W + tail + (1 - tail) * S / num_programs
    return waves * tile[0] * tile[1] * tile[2] * k / t["tflops"][spec_name(tile)]


def streamk_pick_tile(M, N, K, num_programs, rule, specs=SPECS, valid=None):
    """(BLOCK_M, BLOCK_N, BLOCK_K) of the lowest Stream-K-aware cost; valid(tile)
    filters the specs that can run the shape (unsupported_reason in AITER)."""
    specs = [s for s in specs if spec_name(s) in rule["tile"]["tflops"] and (valid is None or valid(s))]
    costs = [streamk_tile_cost(M, N, K, s, num_programs, rule) for s in specs]
    best = min(costs)
    return next(s for s, c in zip(specs, costs) if c <= 1.01 * best)


def streamk_config(M, N, K, num_programs, rule, valid=None):
    """The complete default config of a shape: the tile, then its Stream-K plan."""
    tile = streamk_pick_tile(M, N, K, num_programs, rule, valid=valid)
    return streamk_default(M, N, K, tile, num_programs, rule)


def streamk_candidates(M, N, K, tile, num_programs, rule, with_extras=False):
    """The pruned Stream-K tuning space of a shape at a tile, default first."""
    T, S, k, pairs = _shape(M, N, K, tile, num_programs)
    out = [streamk_default(M, N, K, tile, num_programs, rule), _config(tile)]
    if S and min(num_programs // S, pairs) >= 2:
        n, cap = _rs_n(S, pairs, num_programs, rule["formula"])
        for f in (0.5, 1, 2):
            m = _nearest(n * f, RS_N, cap)
            if m:
                out.append(_config(tile, "one_tile", "rs", m))
        if num_programs // S <= 8:
            out += [_config(tile, "one_tile", "owner", m) for m in OWNER_N if m <= min(cap, 4)]
        if with_extras:
            out += [_config(tile, "two_tile"), _config(tile, "spread")]
            out += [_config(tile, "one_tile", "rs", m, "l2") for m in (2, 4, 8) if m <= cap]
    seen, unique = set(), []
    for cfg in out:
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def kernel_name(config):
    """A tuned CSV row's kernelName: tile, policy and ending; SPLIT_N goes in splitK.
    gluon_compute_bound_256x256x64_dp, ..._one_tile_rs, ..._one_tile_rs_l2."""
    name = (f"gluon_compute_bound_{config['BLOCK_M']}x{config['BLOCK_N']}x{config['BLOCK_K']}"
            f"_{config['STREAMK_POLICY']}")
    if config["STREAMK_POLICY"] == "one_tile":
        name += f"_{config['STREAMK_FIXUP']}"
    if config["STREAMK_EXCHANGE"] == "l2":
        name += "_l2"
    return name


def parse_kernel_name(name, split_k):
    """The config of a tuned row (kernelName, splitK)."""
    parts = name.split("_")
    assert parts[:3] == ["gluon", "compute", "bound"], name
    bm, bn, bk = (int(x) for x in parts[3].split("x"))
    rest = parts[4:]
    exchange = "l2" if rest[-1] == "l2" else "wt"
    rest = rest[:-1] if exchange == "l2" else rest
    if rest[:2] == ["one", "tile"]:
        return _config((bm, bn, bk), "one_tile", rest[2], int(split_k), exchange)
    policy = "_".join(rest)
    return _config((bm, bn, bk), policy)
