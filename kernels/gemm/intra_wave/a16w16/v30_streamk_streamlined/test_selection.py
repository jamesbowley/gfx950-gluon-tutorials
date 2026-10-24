"""CPU-only tests for v30's semantic plans and derived launch constants."""

import unittest

import torch

from v30_streamk_streamlined.hardware import (
    estimate_tile_lds_bytes,
    validate_tile_lds,
)
from v30_streamk_streamlined.heuristics import (
    AUTO_MIN_TILES,
    DEFAULT_TILE,
    auto_tile,
    resolve_config as resolve_heuristic_config,
    select_plan,
    tile_count,
)
from v30_streamk_streamlined.selection import (
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
        # v30 uses a uniform three-way split.
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


if __name__ == "__main__":
    unittest.main()
