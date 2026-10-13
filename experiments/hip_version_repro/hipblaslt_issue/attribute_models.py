#!/usr/bin/env python3
"""Which model a BF16 GEMM shape comes from, using AITER's per-model shape lists.

The model is only recorded in the filename of aiter/configs/model_configs/<tag>_bf16_*gemm*.csv.
A tag is "exact" when the same (M, N, K, outdtype) row is in that model's list, and "layer" when
only the weight shape (N, K) matches: N and K identify the linear layer, M is the token count.
Shapes found only in the base aiter/configs/bf16_tuned_gemm.csv have no model recorded.

    python attribute_models.py M N K bfloat16|float32
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict

CONFIGS = "/home/jbowley/repos/aiter/aiter/configs"
NAMES = {
    "dsv3": "DeepSeek-V3",
    "dsv4": "DeepSeek-V4",
    "dsv41": "DeepSeek-V4.1",
    "flux2": "FLUX.2",
    "glm47": "GLM-4.7",
    "glm5": "GLM-5",
    "glm53": "GLM-5.3",
    "glm53_flash": "GLM-5.3-Flash",
    "gptoss": "gpt-oss",
    "kimi": "Kimi",
    "kimik2": "Kimi-K2",
    "kimik3": "Kimi-K3",
    "llama405B": "Llama 405B",
    "llama70B": "Llama 70B",
    "minimax_m3_eagle": "MiniMax-M3 (EAGLE)",
    "minimax_m3_mxfp4": "MiniMax-M3 (MXFP4)",
    "q3vl": "Qwen3-VL",
    "qwen32B": "Qwen 32B",
    "qwen3_5_397b": "Qwen3.5-397B",
    "qwen3_8_2p4t_a95b": "qwen3_8_2p4t_a95b",
    "qwen3_8b": "Qwen3-8B",
}
OUT = {"torch.bfloat16": "bfloat16", "torch.float32": "float32"}


def _tag(path):
    base = os.path.basename(path)
    m = re.match(r"(.+?)_(bf16_(tuned|untuned)_gemm|untuned_gemm_bf16)\.csv$", base)
    return m[1] if m else None


def _load():
    exact, layer = defaultdict(set), defaultdict(set)
    for path in glob.glob(os.path.join(CONFIGS, "model_configs", "*.csv")):
        tag = _tag(path)
        if tag is None:
            continue
        for r in csv.DictReader(open(path, newline="")):
            if r.get("dtype") != "torch.bfloat16":
                continue
            M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
            exact[(M, N, K, OUT.get(r["outdtype"], r["outdtype"]))].add(tag)
            layer[(N, K)].add(tag)
    base = set()
    for r in csv.DictReader(open(os.path.join(CONFIGS, "bf16_tuned_gemm.csv"), newline="")):
        if r["dtype"] == "torch.bfloat16":
            base.add((int(r["M"]), int(r["N"]), int(r["K"]), OUT.get(r["outdtype"], r["outdtype"])))
    return exact, layer, base


_EXACT, _LAYER, _BASE = _load()


def attribute(M, N, K, out):
    """[(model name, 'exact'|'layer')], exact matches first; [] when no model file lists it."""
    ex = _EXACT.get((M, N, K, out), set())
    ly = _LAYER.get((N, K), set()) - ex
    return [(NAMES.get(t, t), "exact") for t in sorted(ex)] + [(NAMES.get(t, t), "layer") for t in sorted(ly)]


def describe(M, N, K, out):
    tags = attribute(M, N, K, out)
    if tags:
        return "; ".join(f"{n} ({kind})" for n, kind in tags)
    if (M, N, K, out) in _BASE:
        return "AITER tuned list (model not recorded)"
    return "not in AITER lists"


if __name__ == "__main__":
    M, N, K = (int(v) for v in sys.argv[1:4])
    print(describe(M, N, K, sys.argv[4]))
