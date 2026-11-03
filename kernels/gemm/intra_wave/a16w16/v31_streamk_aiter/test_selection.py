"""CPU-only tests for v31's offline plans and derived launch constants."""

import json
import os
import sys
import unittest

import torch

from v31_streamk_aiter.hardware import (
    estimate_tile_lds_bytes,
    validate_tile_lds,
)
from v31_streamk_aiter.heuristics import (
    AUTO_MIN_TILES,
    DEFAULT_TILE,
    auto_tile,
    resolve_config as resolve_heuristic_config,
    select_plan,
    tile_count,
)
from v31_streamk_aiter.selection import (
    DeviceConfig,
    StreamKConfig,
    TileSpec,
    resolve_config,
    try_resolve_config,
)


GFX950 = DeviceConfig("gfx950", 256, 8, 160 * 1024)


def selected_launch(M, N, K, **kwargs):
    config = select_plan(M, N, K, **kwargs)
    return config, resolve_config(
        M,
        N,
        K,
        config=config,
        device=kwargs["device"],
    )


def exact_config(
    *,
    block_m=64,
    block_n=64,
    block_k=64,
    group_size_m=4,
    policy="dp",
    fixup="owner",
    split_n=None,
    exchange="wt",
):
    return StreamKConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_size_m=group_size_m,
        policy=policy,
        fixup=fixup,
        split_n=split_n,
        exchange=exchange,
    )


class SelectionTests(unittest.TestCase):
    def test_canonical_capped_split(self):
        config, launch = selected_launch(
            4352,
            4096,
            8192,
            device=GFX950,
            tile=TileSpec(256, 256, 64),
            policy="capped",
        )
        self.assertEqual(
            (
                launch["STREAMK_POLICY"],
                launch["STREAMK_FIXUP"],
                launch["RESOLVED_SPLIT_N"],
            ),
            ("one_tile", "rs", 8),
        )
        self.assertEqual(
            (launch["STREAMK_TILES"], launch["STREAMK_PROGRAMS"]),
            (16, 128),
        )
        self.assertIn("GROUP_SIZE_M", config)

    def test_capped_remainder_idles_instead_of_uneven_split(self):
        # 5 x 13 = 65 tiles, 32 K-steps. V29 used all 256 programs;
        # v31 uses a uniform three-way split.
        _, launch = selected_launch(
            320,
            832,
            2048,
            device=GFX950,
            tile=TileSpec(64, 64, 64),
            policy="capped",
        )
        self.assertEqual(
            (
                launch["STREAMK_POLICY"],
                launch["STREAMK_FIXUP"],
                launch["RESOLVED_SPLIT_N"],
            ),
            ("one_tile", "owner", 3),
        )
        self.assertEqual(launch["STREAMK_PROGRAMS"], 195)
        self.assertEqual(launch["GRID"], 195)

    def test_end_to_end_uses_full_grid(self):
        tile = TileSpec(64, 64, 64)
        _, two = selected_launch(
            1088, 1024, 512, device=GFX950, tile=tile, policy="two_tile"
        )
        _, spread = selected_launch(
            1024, 1792, 512, device=GFX950, tile=tile, policy="spread"
        )
        self.assertEqual(two["STREAMK_PROGRAMS"], 256)
        self.assertEqual(spread["STREAMK_PROGRAMS"], 256)

    def test_dp_grid_is_at_most_the_tile_count(self):
        config = exact_config(policy="dp").to_dict()
        below = resolve_config(192, 704, 512, config=config, device=GFX950)
        above = resolve_config(1088, 1024, 512, config=config, device=GFX950)
        self.assertEqual((below["TOTAL_TILES"], below["GRID"]), (33, 33))
        self.assertEqual((above["TOTAL_TILES"], above["GRID"]), (272, 256))
        self.assertFalse(below["COMPACT"])

    def test_l2_compact_grid_is_whole_tiles_per_xcd(self):
        def l2_launch(M, N, K, split_n):
            config = exact_config(
                policy="one_tile", fixup="rs", split_n=split_n, exchange="l2"
            ).to_dict()
            return resolve_config(M, N, K, config=config, device=GFX950)

        for split_n, grid in ((4, 32), (8, 64)):
            with self.subTest(split_n=split_n):
                launch = l2_launch(128, 128, 2048, split_n)
                self.assertEqual(launch["STREAMK_TILES"], 4)
                self.assertTrue(launch["COMPACT"])
                self.assertEqual(launch["GRID"], grid)
        # 33 tiles: five per XCD.
        launch = l2_launch(192, 704, 512, 4)
        self.assertEqual((launch["COMPACT"], launch["GRID"]), (True, 160))
        # One tile at split 32 fills XCD 0, so the grid is the full wave.
        launch = l2_launch(64, 64, 8192, 32)
        self.assertEqual((launch["COMPACT"], launch["GRID"]), (False, 256))

    def test_rs_reads_with_the_idle_programs(self):
        def launch(M, N, K, split_n, fixup="rs", exchange="wt", block=128):
            config = exact_config(
                block_m=block,
                block_n=block,
                block_k=128 if block == 128 else 64,
                policy="one_tile",
                fixup=fixup,
                split_n=split_n,
                exchange=exchange,
            ).to_dict()
            return resolve_config(M, N, K, config=config, device=GFX950)

        # 17x16 128x128 tiles, S = 16: R = min(16 blocks, 256 / 16) = 16.
        rs = launch(2176, 2048, 8192, 2)
        self.assertEqual(rs["STREAMK_TILES"], 16)
        self.assertEqual((rs["RS_R"], rs["RS_E"], rs["BAR_STRIDE"]), (16, 14, 16))
        self.assertEqual((rs["RS_NB"], rs["RS_U"]), (1, 16))
        self.assertEqual((rs["COMPACT"], rs["GRID"]), (False, 256))
        # S n = 256: no idle programs, the tile's own readers, slots t n.
        full = launch(2176, 2048, 8192, 16)
        self.assertEqual(
            (full["RS_R"], full["RS_E"], full["BAR_STRIDE"]), (16, 0, 16)
        )
        # A non-power-of-two split that fills the grid keeps its 2 own readers.
        three = launch(128 * 85, 128 * 3 + 128, 8192, 3)
        self.assertEqual(three["STREAMK_TILES"], 84)
        self.assertEqual((three["RS_R"], three["RS_E"]), (2, 0))
        # 64x64 tiles have 4 blocks, so 4 readers at most.
        small = launch(1088, 1024, 512, 4, block=64)
        self.assertEqual((small["RS_R"], small["RS_E"], small["BAR_STRIDE"]), (4, 0, 16))
        # A compact launch adds the extra readers: 33 tiles, R = 4, E = 2.
        compact = launch(384, 1408, 1024, 2)
        self.assertEqual(compact["STREAMK_TILES"], 33)
        self.assertEqual((compact["RS_R"], compact["RS_E"]), (4, 2))
        self.assertEqual((compact["COMPACT"], compact["GRID"]), (True, 132))
        # Owner and L2 keep the own-readers rule and no extra readers.
        for other in (
            launch(2176, 2048, 8192, 2, fixup="owner"),
            launch(2176, 2048, 8192, 2, exchange="l2"),
        ):
            self.assertEqual(
                (other["RS_R"], other["RS_E"], other["BAR_STRIDE"]), (2, 0, 0)
            )

    def test_spread_requires_more_than_half_a_wave(self):
        tile = TileSpec(64, 64, 64)
        device = DeviceConfig("test", 8, 1, 160 * 1024)
        for tail_tiles in (3, 4):
            with self.subTest(tail_tiles=tail_tiles):
                with self.assertRaisesRegex(ValueError, "more than half"):
                    select_plan(
                        64,
                        tail_tiles * 64,
                        512,
                        device=device,
                        tile=tile,
                        policy="spread",
                    )

        _, launch = selected_launch(
            64, 5 * 64, 512, device=device, tile=tile, policy="spread"
        )
        self.assertEqual(
            (launch["STREAMK_TILES"], launch["STREAMK_PROGRAMS"]),
            (5, 8),
        )

        config = exact_config(
            policy="spread",
        )
        self.assertIsNone(
            try_resolve_config(
                64,
                4 * 64,
                512,
                config=config,
                device=device,
            )
        )

    def test_explicit_split_is_never_silently_capped(self):
        with self.assertRaisesRegex(ValueError, "exceeds this shape's maximum 4"):
            select_plan(
                1024,
                1024,
                7168,
                device=GFX950,
                tile=TileSpec(128, 128, 128),
                policy="one_tile",
                split_n=8,
                fixup="auto",
            )
        launch = select_plan(
            1024,
            1024,
            7168,
            device=GFX950,
            tile=TileSpec(128, 128, 128),
            policy="one_tile",
            split_n=4,
            fixup="auto",
        )
        self.assertEqual(
            (launch["SPLIT_N"], launch["STREAMK_FIXUP"]),
            (4, "owner"),
        )

    def test_omitted_split_is_derived(self):
        launch = select_plan(
            1024,
            1024,
            7168,
            device=GFX950,
            tile=TileSpec(128, 128, 128),
            policy="one_tile",
        )
        self.assertEqual(
            (launch["SPLIT_N"], launch["STREAMK_FIXUP"]),
            (4, "owner"),
        )

    def test_small_tile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "excludes small tiles"):
            TileSpec(16, 256, 128)
        with self.assertRaisesRegex(TypeError, "integers"):
            TileSpec(64.0, 64, 64)

    def test_tile_is_integer_only_and_exposes_block_geometry(self):
        tile = TileSpec(256, 128, 64)
        self.assertEqual(tile.spec, "256x128x64")
        self.assertEqual(tile.blocks, 32)
        self.assertEqual(tile.quadrant_blocks, 8)
        with self.assertRaisesRegex(TypeError, "TileSpec"):
            select_plan(
                1024,
                1024,
                4096,
                device=GFX950,
                tile="256x128x64",
            )

    def test_auto_halves_exact_half_wave_only_at_k_threshold(self):
        M, N = 2048, 4096
        self.assertEqual(tile_count(M, N, DEFAULT_TILE), AUTO_MIN_TILES)
        self.assertEqual(auto_tile(M, N, 4096, GFX950), TileSpec(128, 256, 64))
        self.assertEqual(auto_tile(M, N, 4097, GFX950), DEFAULT_TILE)

    def test_auto_tile_count_is_independent_of_streamk_split(self):
        tile = auto_tile(2048, 4096, 4096, GFX950)
        config = select_plan(
            2048,
            4096,
            4096,
            device=GFX950,
            tile=None,
            policy="capped",
        )
        self.assertEqual(
            (config["BLOCK_M"], config["BLOCK_N"], config["BLOCK_K"]),
            (tile.block_m, tile.block_n, tile.block_k),
        )
        self.assertEqual(tile_count(2048, 4096, tile), GFX950.num_programs)

    def test_topology_override_changes_xcd_local_split(self):
        tile = TileSpec(64, 64, 64)
        smaller = DeviceConfig("test", 128, 4, 160 * 1024)
        with self.assertRaisesRegex(ValueError, "shape's maximum 10"):
            select_plan(
                3 * 64,
                4 * 64,
                2048,
                device=smaller,
                tile=tile,
                policy="one_tile",
                split_n=16,
                fixup="rs",
                exchange="l2",
            )
        full_launch = select_plan(
            3 * 64,
            4 * 64,
            2048,
            device=GFX950,
            tile=tile,
            policy="one_tile",
            split_n=16,
            fixup="rs",
            exchange="l2",
        )
        self.assertEqual(full_launch["SPLIT_N"], 16)

    def test_tuner_prunes_invalid_candidate_without_exception(self):
        config = exact_config(
            policy="one_tile",
            fixup="owner",
            split_n=4,
        )
        self.assertIsNone(
            try_resolve_config(
                16 * 64,
                16 * 64,
                2048,
                config=config,
                device=GFX950,
            )
        )
        with self.assertRaisesRegex(ValueError, "requires leftover tiles"):
            resolve_config(
                16 * 64,
                16 * 64,
                2048,
                config=config,
                device=GFX950,
            )
        raw = config.to_dict()
        self.assertIsNotNone(
            try_resolve_config(
                1024,
                1024,
                2048,
                config={**raw, "STREAMK_POLICY": "dp", "SPLIT_N": None},
                device=GFX950,
            )
        )
        self.assertIsNone(
            try_resolve_config(
                1024,
                1024,
                2048,
                config={**raw, "BLOCK_M": 32},
                device=GFX950,
            )
        )

    def test_exact_config_rejects_heuristic_values_and_missing_fields(self):
        raw = exact_config().to_dict()
        with self.assertRaisesRegex(ValueError, "invalid Stream-K policy"):
            resolve_config(
                1024,
                1024,
                2048,
                config={**raw, "STREAMK_POLICY": "capped"},
                device=GFX950,
            )
        with self.assertRaisesRegex(ValueError, "invalid Stream-K fixup"):
            resolve_config(
                1024,
                1024,
                2048,
                config={**raw, "STREAMK_FIXUP": "auto"},
                device=GFX950,
            )
        without_group = {key: value for key, value in raw.items() if key != "GROUP_SIZE_M"}
        with self.assertRaisesRegex(ValueError, "group_size_m"):
            resolve_config(
                1024,
                1024,
                2048,
                config=without_group,
                device=GFX950,
            )

    def test_capped_can_resolve_to_dp(self):
        config = resolve_heuristic_config(
            16 * 64,
            16 * 64,
            2048,
            config={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            device=GFX950,
        )
        launch = resolve_config(
            16 * 64,
            16 * 64,
            2048,
            config=config,
            device=GFX950,
        )
        self.assertEqual(launch["STREAMK_POLICY"], "dp")
        self.assertEqual(launch["STREAMK_PROGRAMS"], 0)

    def test_config_round_trip_records_exact_launch(self):
        candidate = exact_config(
            block_m=128,
            block_n=128,
            block_k=128,
            group_size_m=8,
            policy="one_tile",
            fixup="owner",
            split_n=4,
        )
        self.assertEqual(StreamKConfig.from_dict(candidate.to_dict()), candidate)
        launch = resolve_config(
            1024,
            1024,
            7168,
            config=candidate,
            device=GFX950,
        )
        self.assertEqual(
            {
                key: launch[key]
                for key in (
                    "BLOCK_M",
                    "BLOCK_N",
                    "BLOCK_K",
                    "GROUP_SIZE_M",
                    "STREAMK_POLICY",
                    "STREAMK_FIXUP",
                    "SPLIT_N",
                    "STREAMK_EXCHANGE",
                )
            },
            candidate.to_dict(),
        )

    def test_aiter_weight_transpose_is_a_zero_copy_view(self):
        weights = torch.empty((11, 32), dtype=torch.bfloat16)
        b = weights.transpose(0, 1)
        self.assertEqual((b.shape, b.stride()), ((32, 11), (1, 32)))
        self.assertEqual(b.data_ptr(), weights.data_ptr())

    def test_exact_config_is_resolved_for_each_shape(self):
        config = exact_config().to_dict()
        square = resolve_config(1024, 1024, 2048, config=config, device=GFX950)
        wide = resolve_config(1024, 2048, 2048, config=config, device=GFX950)
        self.assertNotEqual(square["TOTAL_TILES"], wide["TOTAL_TILES"])

    def test_lds_estimate_is_dtype_and_device_dependent(self):
        tile = TileSpec(256, 256, 64)
        required = estimate_tile_lds_bytes(tile, torch.bfloat16)
        self.assertEqual(required, 152064)
        self.assertEqual(validate_tile_lds(tile, torch.float16, GFX950), required)
        too_small = DeviceConfig("test", 256, 8, required - 1)
        with self.assertRaisesRegex(ValueError, "more than"):
            validate_tile_lds(tile, torch.bfloat16, too_small)
        with self.assertRaisesRegex(ValueError, "supports fp16 and bf16"):
            estimate_tile_lds_bytes(tile, torch.float32)



def v9_pids(pid, M, N, BM, BN, grid_mn, num_xcds, group_size_m):
    """v9_ref._get_pids in Python."""
    num_pid_m = -(-M // BM)
    num_pid_n = -(-N // BN)
    if num_xcds != 1:
        pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
        tall_xcds = grid_mn % num_xcds or num_xcds
        xcd, local_pid = pid % num_xcds, pid // num_xcds
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return group_m(pid, num_pid_m, num_pid_n, group_size_m)


def group_m(tile, num_pid_m, num_pid_n, group_size_m):
    if group_size_m == 1:
        return tile // num_pid_n, tile % num_pid_n
    in_group = group_size_m * num_pid_n
    first = tile // in_group * group_size_m
    size = min(num_pid_m - first, group_size_m)
    return first + (tile % in_group) % size, (tile % in_group) // size


def v31_pids(vpid, M, N, BM, BN, grid_mn, num_xcds, group_size_m):
    """matmul_kernel.vpid_tile_coords (ragged_transpose, group_m_tile_coords)."""
    tile = vpid
    if num_xcds != 1:
        part, total = vpid % num_xcds, grid_mn
        tile = part * (total // num_xcds) + min(part, total % num_xcds) + vpid // num_xcds
    return group_m(tile, -(-M // BM), -(-N // BN), group_size_m)


class KernelLaunchTests(unittest.TestCase):
    """The kernel module's own launch constants against the offline resolver."""

    def kernel_launch(self, M, N, K, config):
        from v31_streamk_aiter.matmul_kernel import _launch_constants

        if isinstance(config, StreamKConfig):
            config = config.to_dict()
        return _launch_constants(M, N, K, config, GFX950.num_programs, GFX950.num_xcds)[1]

    def test_kernel_constants_match_offline_resolver(self):
        keys = (
            "TOTAL_TILES", "STREAMK_TILES", "STREAMK_PROGRAMS", "RESOLVED_SPLIT_N",
            "END_TO_END", "L2", "COMPACT", "GRID", "RS_NB", "RS_U", "RS_R", "RS_E",
            "BAR_STRIDE",
        )
        checked = 0
        for tile in (TileSpec(256, 256, 64), TileSpec(128, 128, 128), TileSpec(64, 128, 64)):
            for tiles_m, tiles_n in ((3, 11), (16, 16), (17, 16), (16, 28), (33, 9), (2, 2)):
                M, N = tiles_m * tile.block_m - 8, tiles_n * tile.block_n
                for K in (1024, 4096, 8192 - 16):
                    for options in (
                        dict(policy="dp"),
                        dict(policy="capped"),
                        dict(policy="one_tile", split_n=2, fixup="owner"),
                        dict(policy="one_tile", split_n=4, fixup="rs"),
                        dict(policy="one_tile", split_n=4, fixup="rs", exchange="l2"),
                        dict(policy="two_tile"),
                        dict(policy="spread"),
                    ):
                        try:
                            config = select_plan(M, N, K, device=GFX950, tile=tile, **options)
                        except ValueError:
                            continue
                        offline = resolve_config(M, N, K, config=config, device=GFX950)
                        kernel = self.kernel_launch(M, N, K, config)
                        self.assertEqual(
                            {k: offline[k] for k in keys}, {k: kernel[k] for k in keys}
                        )
                        checked += 1
        self.assertGreater(checked, 100)

    def test_tile_remap_matches_v9(self):
        for bm, bn in ((256, 256), (128, 64)):
            for M, N in ((3 * bm, 11 * bn), (16 * bm, 16 * bn - 8), (7 * bm - 1, 5 * bn)):
                grid = -(-M // bm) * -(-N // bn)
                for group_size_m in (1, 4, 8):
                    for pid in range(grid):
                        self.assertEqual(
                            v9_pids(pid, M, N, bm, bn, grid, 8, group_size_m),
                            v31_pids(pid, M, N, bm, bn, grid, 8, group_size_m),
                        )


EXPERIMENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..",
                          "experiments", "streamk_v31_rule")


def recorded_pairs():
    """Every (spec, M, N, K) of the rule sweeps' training and held-out results."""
    import csv
    import glob

    pairs = set()
    for name in ("grid", "tilesweep", "gap", "formula", "heldout", "joint_heldout"):
        for path in glob.glob(os.path.join(EXPERIMENT, "results", f"{name}_r1.csv")):
            with open(path) as f:
                pairs.update((r["tile"], int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(f))
    return sorted(pairs)


class DefaultPlanTests(unittest.TestCase):
    """The default plan of a tile-only config (matmul_kernel.streamk_plan)."""

    def setUp(self):
        from v31_streamk_aiter import matmul_kernel

        self.mk = matmul_kernel

    def plan(self, M, N, K, tile):
        return self.mk.streamk_plan(M, N, K, tile, GFX950.num_programs)

    def test_constants_match_the_fitted_rule(self):
        path = os.path.join(EXPERIMENT, "results", "streamk_rule.json")
        if not os.path.exists(path):
            self.skipTest("experiment results not present")
        with open(path) as f:
            self.assertEqual(json.load(f)["formula"], self.mk.STREAMK_PLAN)

    def test_plan_matches_the_experiment_on_every_recorded_pair(self):
        pairs = recorded_pairs()
        if not pairs:
            self.skipTest("experiment results not present")
        sys.path.insert(0, EXPERIMENT)
        import candidates

        rule = {"formula": self.mk.STREAMK_PLAN}
        for spec, M, N, K in pairs:
            tile = tuple(int(x) for x in spec.split("x"))
            want = candidates.streamk_default(M, N, K, tile, GFX950.num_programs, rule)
            got = self.plan(M, N, K, tile)
            self.assertEqual(got, (want["STREAMK_POLICY"], want["STREAMK_FIXUP"], want["SPLIT_N"]),
                             (spec, M, N, K))
        self.assertGreater(len(pairs), 10000)

    def test_edge_cases(self):
        P = GFX950.num_programs
        tile = (256, 256, 64)
        # S == 0: whole waves stay data-parallel.
        self.assertEqual(self.plan(16 * 256, 16 * 256, 8192, tile), ("dp", "owner", None))
        # cap < 2: more than P / 2 tail tiles, or K of one pair.
        self.assertEqual(self.plan(16 * 256, 25 * 256, 8192, tile)[0], "dp")
        self.assertEqual(self.plan(17 * 256, 16 * 256, 128, tile)[0], "dp")
        # S = 16 (m = 16), K = 8192 (64 pairs): reduce-scatter on sqrt(64 * 16 / 6) -> 12.
        self.assertEqual(self.plan(17 * 256, 16 * 256, 8192, tile), ("one_tile", "rs", 12))
        # Short K: 8 pairs save less than 6 of the tail -> dp.
        self.assertEqual(self.plan(17 * 256, 16 * 256, 512, tile)[0], "dp")
        # S = 64 (m = 4): reduce-scatter, n capped at 4; S = 128 (m = 2), 64 pairs: the
        # owner, its sqrt(64 / 1.5) capped at the 2 programs per tail tile.
        self.assertEqual(self.plan(4096, 5120, 8192, tile), ("one_tile", "rs", 4))
        self.assertEqual(self.plan(4096, 6144, 8192, tile), ("one_tile", "owner", 2))
        # m <= 3 but under 10 pairs: dp.
        self.assertEqual(self.plan(4096, 6144, 1024, tile)[0], "dp")
        # BLOCK_K 128 halves the pairs of the same K; a ragged K rounds up to whole pairs.
        self.assertEqual(self.plan(17 * 128, 16 * 128, 8192, (128, 128, 128)),
                         self.plan(17 * 128, 16 * 128, 8192 - 48, (128, 128, 128)))
        for M, N, K, t in ((17 * 256, 16 * 256, 8192, tile), (3 * 128, 11 * 128, 4096, (128, 128, 128)),
                           (4096, 5120, 8192, tile)):
            policy, fixup, n = self.plan(M, N, K, t)
            if policy == "one_tile":
                T = -(-M // t[0]) * -(-N // t[1])
                self.assertLessEqual(n, min(P // (T % P), -(-K // (2 * t[2]))))

    def test_completion_rules(self):
        mk = self.mk
        P, X = GFX950.num_programs, GFX950.num_xcds
        tile_only = {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}
        full = mk.complete_config(tile_only, 17 * 256, 16 * 256, 8192, P)
        self.assertEqual(full["GROUP_SIZE_M"], 4)
        self.assertEqual((full["STREAMK_POLICY"], full["STREAMK_FIXUP"], full["SPLIT_N"], full["STREAMK_EXCHANGE"]),
                         ("one_tile", "rs", 12, "wt"))
        tuned = dict(full, STREAMK_FIXUP="owner", SPLIT_N=4, GROUP_SIZE_M=8)
        self.assertEqual(mk.complete_config(tuned, 17 * 256, 16 * 256, 8192, P), tuned)
        with self.assertRaises(ValueError):
            mk.complete_config(dict(tile_only, STREAMK_POLICY="dp"), 4096, 4096, 8192, P)
        with self.assertRaises(ValueError):
            mk.complete_config({"BLOCK_M": 256, "BLOCK_N": 256}, 4096, 4096, 8192, P)
        with self.assertRaises(ValueError):
            mk.complete_config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, 4096, 4096, 8192, P)

    def test_completed_configs_launch(self):
        mk = self.mk
        P, X = GFX950.num_programs, GFX950.num_xcds
        checked = 0
        for spec, M, N, K in recorded_pairs()[::7]:
            bm, bn, bk = (int(x) for x in spec.split("x"))
            if -(-K // (2 * bk)) * 2 * bk < 4 * bk:
                continue
            cfg = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}
            mk._launch_constants(M, N, K, cfg, P, X)
            checked += 1
        if not checked:
            self.skipTest("experiment results not present")


if __name__ == "__main__":
    unittest.main()
