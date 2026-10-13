#!/usr/bin/env python3
"""Tutorial v20 (stream-K, auto policy) vs AITER's live tuned_gemm pick on the 630 shapes.

    ./run_rocm724_v20.sh python run_v20_vs_aiter.py sweep [--tag T] [--jit-tag T0] [--rounds 2] [--only 0,5,17] [--out F]
    python run_v20_vs_aiter.py report [--tag T] [--baseline results/details.csv]

--tag T writes results/details_T.csv, summary_T.md, envinfo_T.json and workers_T.log, loads v20
from v20_snapshot_T/ and builds AITER's modules in jit/rocm724_T/. Without a tag, the files are
untagged and AITER uses the Sep 29 build in hip_version_repro/jit/rocm724.

The method of hip_version_repro/sweep.py (Sep 29), on the same stack (torch 2.10.0+rocm7.2.4,
hipBLASLt 1.2.2) and the same GPU (HIP device 0, PCI bus 0x75 = 117). Per shape from
supported_status_scoped.csv: x (M, K) and w (N, K) bf16 from torch.manual_seed(0), output
dtype from the CSV (no shape has bias).

- pick: aiter.tuned_gemm.gemm_a16w16, whatever the AITER worktree (llirSched_integration, or
  its two commits rebased onto a newer main) dispatches; the commit is recorded. Its libtype
  and kernel come from get_GEMM_A16W16_config; the kernels it launches are recorded with the
  torch profiler.
- v9: AITER's gemm_a16w16(backend="gluon", kernel_type="compute_bound"), the Sep 29 column.
- v20: the tutorial's v20_streamk_reduce_scatter matmul(x, w.T, c) with its auto rule
  (STREAMK_POLICY=auto, STREAMK_FIXUP unset, all 256 programs, v9 tile order). The chosen policy,
  fixup and leftover tiles S are recorded. Where auto does not pick data-parallel,
  STREAMK_POLICY=dp is timed as well, as a diagnostic column only.
- v20_capped: STREAMK_POLICY=capped, where capped does not pick data-parallel (otherwise it is
  the same kernel as v20 or v20_dp). v20_two_tile: STREAMK_POLICY=two_tile, where auto did not
  pick it, there is a full wave to borrow and 256 // S >= 7. Both are inputs to
  experiments/streamk_capped_vs_auto's rule scoring.

Every output is checked against an fp32 F.linear reference cast to the output dtype
(allclose atol=1e-1, rtol=1e-2). v20 is checked again after timing, and its stream-K flags
and barrier counts must be back at 0. Timing is triton.testing.do_bench_cudagraph,
`--rounds` interleaved rounds, median.

AITER's llirSched plugin is opt-in per kernel; the tutorial's (LLVM_PASS_PLUGIN_PATH) and its
amdgcnas hook apply to every Triton kernel in the process. So the two run in separate
persistent worker processes, driven over JSON lines, one at a time on the GPU. v20 is loaded
from v20_snapshot[_T]/matmul_kernel.py, copied from the tutorial when the sweep first starts,
so edits to the live kernel during a run do not mix into the results.

Rows are appended to the details CSV as each shape finishes, and finished shapes are skipped,
so the run can be resumed.
"""

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
# --kernel: the stream-K kernel timed as streamk_wip, (directory, Gluon kernel, the directory and
# Gluon kernel of the v9 it hands one-wave data-parallel launches to). Its columns keep the v20_*
# names whichever kernel it is.
KERNELS = {
    "v20": ("v20_streamk_reduce_scatter", "v20_streamk_reduce_scatter", "v9_beyond_hotloop", "v9_beyond_hotloop"),
    "v24": ("v24_streamk_combined", "v24_streamk_combined", "v9_beyond_hotloop", "v9_beyond_hotloop"),
    "v26": ("v26_streamk_ragged_k", "v26_streamk_ragged_k", "v9_ragged", "v9_ragged"),
}
V20_LIVE = os.path.join(A16, KERNELS[os.environ.get("V20CMP_KERNEL", "v20")][0], "matmul_kernel.py")
# The v9 it hands off to is the live one (not snapshotted).
V9_LIVE = os.path.join(A16, KERNELS[os.environ.get("V20CMP_KERNEL", "v20")][2], "matmul_kernel.py")
SHAPES_CSV = os.path.join(HERE, "supported_status_scoped.csv")
RESULTS = os.path.join(HERE, "results")
HVR = os.path.join(REPO, "experiments", "hip_version_repro")
SEP29_DETAILS = os.path.join(HVR, "details_rocm724.csv")

AITER = "/home/jbowley/repos/aiter_llirsched_wt"
TRITON_PY = "/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python"
# HIP device (HIP_VISIBLE_DEVICES, set by run_rocm724_v20.sh from V20CMP_GPU) -> its PCI bus. HIP
# numbers the devices in a different order from rocm-smi: HIP 0 is rocm-smi GPU 3.
GPUS = {"0": 0x75, "3": 0x15}
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "0")
GPU_PCI_BUS = GPUS.get(GPU)
GPU_BDF = f"0000:{GPU_PCI_BUS or 0:02x}:00.0"

NUM_CUS = 256
TIMEOUT_S = 600
TAG = "@@JSON "
STREAMK_ENV = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_NUM_PROGRAMS", "STREAMK_BALANCE_XCDS",
               "PERSISTENT_TILE_ORDER")
PLUGIN_ENV = ("LLVM_PASS_PLUGIN_PATH", "TRITON_AMDGCNAS_PLUGIN", "AITER_LLIR_SCHED")


def run_paths(tag, jit_tag=None):
    """Per-run files. The untagged run (AITER at llirSched_integration) keeps the Sep 29 JIT build;
    a tagged run gets its own v20 snapshot and AITER JIT build, or jit_tag's build."""
    sfx = f"_{tag}" if tag else ""
    jit_tag = tag if jit_tag is None else jit_tag
    return dict(
        details=os.path.join(RESULTS, f"details{sfx}.csv"),
        summary=os.path.join(RESULTS, f"summary{sfx}.md"),
        envinfo=os.path.join(RESULTS, f"envinfo{sfx}.json"),
        log=os.path.join(RESULTS, f"workers{sfx}.log"),
        snapshot=os.path.join(HERE, f"v20_snapshot{sfx}", "matmul_kernel.py"),
        jit=os.path.join(HERE, "jit", f"rocm724_{jit_tag}") if jit_tag else os.path.join(HVR, "jit", "rocm724"),
    )


def geometry(M, N, K):
    tiles = (M // 256) * (N // 256)
    return dict(tiles=tiles, waves=tiles // NUM_CUS, S=tiles % NUM_CUS, k_steps=K // 64)


def load_shapes():
    with open(SHAPES_CSV, newline="") as f:
        return list(csv.DictReader(f))


def key(M, N, K, out):
    return (int(M), int(N), int(K), out)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------- workers


def send(obj):
    sys.stdout.write(TAG + json.dumps(obj) + "\n")
    sys.stdout.flush()


def check(out, ref):
    import torch

    out = out.float()
    return bool(torch.allclose(out, ref, atol=1e-1, rtol=1e-2)), float((out - ref).abs().max().item())


def device_info():
    import torch

    assert GPU_PCI_BUS is not None, f"HIP device {GPU} is not in GPUS"
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    p = torch.cuda.get_device_properties(0)
    assert p.pci_bus_id == GPU_PCI_BUS, f"HIP device {GPU} is on PCI bus {p.pci_bus_id}, expected {GPU_PCI_BUS}"
    return dict(torch=torch.__version__, hip=torch.version.hip, hip_device=GPU, gpu_pci_bus=p.pci_bus_id,
                gpu=p.gcnArchName)


PAD_BYTES = 256


def pad512(rows, cols, dtype):
    """An uninitialised (rows, cols) view whose row stride is PAD_BYTES longer when cols * itemsize
    is a multiple of 512 bytes (a multiple of 512 bytes causes Tagram channel hotspotting on
    MI300), and otherwise a fresh contiguous tensor."""
    import torch

    item = torch.empty((), dtype=dtype).element_size()
    pad = PAD_BYTES // item if cols * item % 512 == 0 else 0
    return torch.empty((rows, cols + pad), dtype=dtype, device="cuda")[:, :cols]


def storage_bytes(*ts):
    return sum(t.untyped_storage().nbytes() for t in ts)


class Inputs:
    """x, w, reference and output buffer for one shape, built identically in both workers.

    Layout "" is contiguous; with V20CMP_PAD, layout "_pad" holds the same x and w (and an output
    buffer) in pad512 buffers."""

    def __init__(self, s):
        import torch
        import torch.nn.functional as F

        self.M, self.N, self.K = int(s["M"]), int(s["N"]), int(s["K"])
        self.otype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[s["out"]]
        torch.manual_seed(0)
        self.x = torch.randn((self.M, self.K), dtype=torch.bfloat16, device="cuda")
        self.w = torch.randn((self.N, self.K), dtype=torch.bfloat16, device="cuda")
        self.ref = F.linear(self.x.float(), self.w.float()).to(self.otype).float()
        self.y = torch.empty((self.M, self.N), dtype=self.otype, device="cuda")
        self.layouts = ("", "_pad") if os.environ.get("V20CMP_PAD") else ("",)

        def padded_set(fill):
            x = pad512(self.M, self.K, torch.bfloat16)
            w = pad512(self.N, self.K, torch.bfloat16)
            for t, src in ((x, self.x), (w, self.w)):
                if fill:
                    t.copy_(src)
                else:
                    t.normal_()
            return x, w, pad512(self.M, self.N, self.otype)

        # --rotating-mb: extra (x, w, y) sets per layout so consecutive launches never read the same
        # inputs; at least 2, since one set can exceed the budget. Only set 0 is checked against ref.
        self.sets = {"": [(self.x, self.w, self.y)]}
        if "_pad" in self.layouts:
            self.sets["_pad"] = [padded_set(True)]
        mb = int(os.environ.get("V20CMP_ROTATE_MB", "0"))
        if mb:
            for lay, sets in self.sets.items():
                n = max(2, -(-(mb << 20) // storage_bytes(*sets[0])))
                for _ in range(n - 1):
                    sets.append((torch.randn_like(self.x), torch.randn_like(self.w), torch.empty_like(self.y))
                                if lay == "" else padded_set(False))
        self.copies = self.sets[""]
        self.used, self.rings = set(), []

    def strides(self, lay):
        x, w, y = self.sets[lay][0]
        return f"x {x.stride(0)}, w {w.stride(0)}, y {y.stride(0)}"

    def rotating(self, fn, lay="", keep_outputs=False):
        """A no-argument callable running fn(x, w, y) on the next set of layout lay. With
        keep_outputs, the last len(sets) outputs stay alive, so an fn that allocates its output
        inside a captured graph cannot be handed the same block on every call."""
        sets = self.sets[lay]
        if len(sets) == 1:
            return lambda: fn(*sets[0])
        import collections

        state = dict(i=0)
        ring = collections.deque(maxlen=len(sets))
        self.rings.append(ring)

        def call():
            i = state["i"] % len(sets)
            state["i"] += 1
            self.used.add(i)
            out = fn(*sets[i])
            if keep_outputs:
                ring.append(out)
            return out

        return call


def bench_all(fns, names, inp):
    """do_bench_cudagraph each callable; with rotation, also how many copies each one touched."""
    import triton

    res = {}
    for name in names:
        inp.used.clear()
        res[name] = triton.testing.do_bench_cudagraph(fns[name])
        if len(inp.copies) > 1:
            res[f"_used_{name}"] = len(inp.used)
        for ring in inp.rings:
            ring.clear()
    return res


def hbl_call(otype):
    """The call AITER's torch libtype makes (as hip_version_repro/hbl_all.py)."""
    import torch
    import torch.nn.functional as F

    if otype == torch.float32:
        return lambda x, w, y: torch.mm(x, w.t(), out_dtype=torch.float32)
    return lambda x, w, y: F.linear(x, w)


def hbl_out_call(otype):
    """hipBLASLt through torch, writing into y, so a padded y passes its row stride as ldc."""
    import torch

    if otype == torch.float32:
        return lambda x, w, y: torch.mm(x, w.t(), out_dtype=torch.float32, out=y)
    return lambda x, w, y: torch.mm(x, w.t(), out=y)


def hbl_fields(names):
    """Tile, stream-K token and GSU of hipBLASLt's first kernel, from its solution name."""
    import re

    first = names[0] if names else ""
    mt = re.search(r"_MT(\d+)x(\d+)x(\d+)_", first)
    sk = re.search(r"_(SK\d+)_", first)
    gsu = re.search(r"_GSU(\d+)_", first)
    return dict(hbl_tile=f"{mt.group(1)}x{mt.group(2)}x{mt.group(3)}" if mt else "",
                hbl_sk=sk.group(1) if sk else "", hbl_gsu=gsu.group(1) if gsu else "",
                hbl_nkernels=len(names))


class AiterWorker:
    def __init__(self):
        import aiter.tuned_gemm as tg
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx
        from aiter.tuned_gemm import get_GEMM_A16W16_config_

        self.tg = tg
        gfx, cu = get_gfx(), get_cu_num()
        self.tuned_keys = {
            (k[2], k[3], k[4], bool(k[5]), k[6], k[7])
            for k in get_GEMM_A16W16_config_()
            if k[0] == gfx and k[1] == cu and not k[8] and not k[9]
        }
        self.inp = None

    def info(self):
        import ctypes

        import torch
        import triton

        import aiter
        from aiter.ops.triton import llir_sched

        info = device_info()
        so = llir_sched.plugin_path()
        assert so and llir_sched.compile_options(), "AITER llirSched plugin not available"
        assert self.tg._opus_launch is not None, "OPUS unavailable: opus rows would silently run torch"
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        path = next(l.split()[-1] for l in open(f"/proc/{os.getpid()}/maps") if "libhipblaslt.so" in l)
        lib = ctypes.CDLL(path)
        h, v, buf = ctypes.c_void_p(), ctypes.c_int(), ctypes.create_string_buffer(256)
        lib.hipblasLtCreate(ctypes.byref(h))
        lib.hipblasLtGetVersion(h, ctypes.byref(v))
        lib.hipblasLtGetGitRevision(h, buf)
        aiter_dir = os.path.dirname(aiter.__file__)
        commit = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", aiter_dir, "log", "-1", "--format=%h %s"],
            capture_output=True, text=True,
        ).stdout.strip()
        base = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", aiter_dir, "log", "-1", "--format=%h %cs %s", "HEAD~2"],
            capture_output=True, text=True,
        ).stdout.strip()
        return info | dict(
            hipblaslt=f"{v.value} ({buf.value.decode()})",
            hipblaslt_path=path,
            triton=f"{triton.__version__} @ {os.path.dirname(triton.__file__)}",
            aiter=aiter_dir,
            aiter_commit=commit,
            aiter_base=base,
            aiter_jit_dir=os.environ.get("AITER_JIT_DIR", ""),
            llir_sched=so,
        )

    def setup(self, s, first=True):
        """Fresh inputs for s; the previous round's stay allocated until these exist, so a
        --realloc round never reuses their buffers. first: also profile what each call launches."""
        import torch
        from torch.profiler import ProfilerActivity, profile

        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as gluon_gemm
        from aiter.tuned_gemm import get_GEMM_A16W16_config

        old = self.inp  # noqa: F841 - held until the new inputs are allocated
        self.inp = inp = Inputs(s)
        old = None
        x, otype = inp.x, inp.otype
        cfg = get_GEMM_A16W16_config(M=inp.M, N=inp.N, K=inp.K, bias=False, dtype=str(x.dtype),
                                     otype=str(otype), scaleAB=False, bpreshuffle=False)
        kernel = cfg.get("kernelName")
        kernel = "" if kernel is None or kernel != kernel else str(kernel)
        tuned = (inp.M, inp.N, inp.K, False, str(x.dtype), str(otype)) in self.tuned_keys
        # raw: name -> (fn(x, w, y), layout); fns: the timed callables, rotating over that layout's
        # sets. pick and hbl allocate their output, so the rotation keeps their recent outputs alive.
        # On the padded layout hipBLASLt writes into the padded C through torch.mm(out=).
        base = {
            "pick": lambda x, w, y: self.tg.gemm_a16w16(x, w, None, otype),
            "v9": lambda x, w, y: gluon_gemm(x, w, None, otype, y, backend="gluon", kernel_type="compute_bound"),
        }
        if os.environ.get("V20CMP_HBL"):
            base["hbl"] = hbl_call(otype)
        self.raw = {}
        for n, f in base.items():
            for lay in inp.layouts:
                self.raw[n + lay] = (hbl_out_call(otype) if (n, lay) == ("hbl", "_pad") else f, lay)
        self.fns = {n: inp.rotating(f, lay, keep_outputs=n in ("pick", "pick_pad", "hbl"))
                    for n, (f, lay) in self.raw.items()}
        res = dict(aiter_pick=cfg["libtype"], aiter_kernel=kernel, src="tuned" if tuned else "untuned",
                   rot_copies=len(inp.copies), notes=[])
        if "_pad" in inp.layouts:
            res.update(rot_copies_pad=len(inp.sets["_pad"]), pad_strides=inp.strides("_pad"))
        self.ok_fns = []
        for name, (fn, lay) in self.raw.items():
            try:
                res[f"{name}_ok"], res[f"{name}_maxerr"] = check(fn(*inp.sets[lay][0]), inp.ref)
                self.ok_fns.append(name)
            except Exception as e:  # noqa: BLE001 - record and keep going
                res["notes"].append(f"{name}: {type(e).__name__}: {e}"[:300])
                res[f"{name}_ok"], res[f"{name}_maxerr"] = False, float("nan")
        torch.cuda.synchronize()
        if not first:
            return res

        def launched(name):
            fn, lay = self.raw[name]
            names = []
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn(*inp.sets[lay][0])
                torch.cuda.synchronize()
            for e in sorted((e for e in prof.events() if e.device_type.name == "CUDA"),
                            key=lambda e: e.time_range.start):
                if e.name not in names:
                    names.append(e.name)
            return names

        names = launched("pick") if "pick" in self.ok_fns else []
        res["pick_kernels_launched"] = " | ".join(names)
        res["hbl_kernels"] = " | ".join(k for k in names if "Cijk_" in k)
        if "hbl" in self.raw:
            hbl = [k for k in launched("hbl") if "Cijk_" in k] if "hbl" in self.ok_fns else []
            res["hbl_kernels_torch"] = " | ".join(hbl)
            res.update(hbl_fields(hbl))
        if "_pad" in inp.layouts:
            # A padded call that launches a kernel its contiguous call does not (other than a different
            # hipBLASLt solution) is copying its inputs, e.g. FlyDSL's a.contiguous().
            for n in base:
                if f"{n}_pad" not in self.ok_fns or n not in self.ok_fns:
                    continue
                plain = names if n == "pick" else launched(n)
                padded = launched(f"{n}_pad")
                extra = [k for k in padded if k not in plain and "Cijk_" not in k]
                res[f"{n}_pad_copy"] = bool(extra)
                res[f"{n}_pad_extra_kernels"] = " | ".join(extra)
                if n == "pick":
                    res["pick_pad_kernels_launched"] = " | ".join(padded)
                if n == "hbl":
                    res.update({f"{k}_pad": v for k, v in hbl_fields([k for k in padded if "Cijk_" in k]).items()})
        return res

    def time(self):
        return bench_all(self.fns, self.ok_fns, self.inp)

    def free(self):
        import torch

        self.inp = self.fns = self.raw = None
        torch.cuda.empty_cache()
        return {}


def split_label(policy, fixup, programs, g):
    """The configuration a rule picks, with the split the kernel actually runs: tile-aligned one-tile
    gives each leftover tile min(P // S, pairs) programs, the first P % S tiles one more."""
    if policy != "one_tile":
        return {"dp": "data-parallel", "spread": "spread", "two_tile": "two-tile"}[policy]
    S, pairs = g["S"], g["k_steps"] // 2
    snp = programs or NUM_CUS
    n_lo, n_hi = min(snp // S, pairs), min(snp // S + 1, pairs)
    n = f"{n_lo}-{n_hi}" if n_hi > n_lo and snp % S else f"{n_lo}"
    return f"one-tile {'reduce-scatter' if fixup == 'rs' else 'owner'}, n = {n}"


def pick_label(policy, fixup, programs, S):
    """A host rule's pick, e.g. "one-tile rs, n = 8"; programs is S * n, or None for all 256."""
    if policy == "one_tile":
        return f"one-tile {fixup}" + (f", n = {programs // S}" if programs else "")
    return {"dp": "data-parallel", "spread": "spread", "two_tile": "two-tile"}[policy]


class V20Worker:
    def __init__(self):
        import importlib.util

        for k in STREAMK_ENV:
            os.environ.pop(k, None)
        sys.path.insert(0, A16)
        import bench  # noqa: F401  (llirSched + amdgcnas hooks from the environment, utils on path)

        self.snapshot = os.environ["V20CMP_SNAPSHOT"]
        spec = importlib.util.spec_from_file_location("v20_snapshot_matmul_kernel", self.snapshot)
        self.mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mk)
        with open(self.snapshot) as f:
            self.v9_same_dtype_only = "c.dtype == a.dtype" in f.read()
        self.sched = os.environ.get("V20CMP_V20_SCHED", "tutorial")
        if self.sched == "aiter":
            # AITER's llirSched copy schedules only kernels that opt in; opt in the tutorial's v20
            # and the v9 it hands off to, as AITER's own v9 does.
            from aiter.ops.triton import llir_sched

            self.llir_opts = llir_sched.compile_options()
            assert self.llir_opts, "AITER llirSched plugin not available"
            spec = KERNELS[os.environ.get("V20CMP_KERNEL", "v20")]
            kernels = [getattr(self.mk, spec[1]), getattr(sys.modules[self.mk.v9_matmul.__module__], spec[3])]
            for k in kernels:
                run = k.run
                k.run = lambda *a, _run=run, **kw: _run(*a, **{**self.llir_opts, **kw})
        self.inp = None

    def info(self):
        import triton

        from triton import knobs

        info = device_info()
        if self.sched == "aiter":
            assert knobs.runtime.add_stages_inspection_hook is None, "amdgcnas hook installed"
            assert not os.environ.get("TRITON_AMDGCNAS_PLUGIN")
        else:
            assert knobs.runtime.add_stages_inspection_hook is not None, "amdgcnas hook not installed"
        assert os.environ.get("LLVM_PASS_PLUGIN_PATH"), "llirSched plugin not set"
        return info | dict(
            triton=f"{triton.__version__} @ {os.path.dirname(triton.__file__)}",
            v20_sched=self.sched,
            llir_plugin=os.environ["LLVM_PASS_PLUGIN_PATH"],
            llir_opts=str(getattr(self, "llir_opts", "")),
            amdgcnas=os.environ.get("TRITON_AMDGCNAS_PLUGIN"),
            v20_snapshot=self.snapshot,
            v20_snapshot_sha256=sha256(self.snapshot),
            v20_p_slot_pad_elems=int(getattr(getattr(self.mk, "P_SLOT_PAD", None), "value", 0)),
        )

    def _run(self, policy):
        """fn(x, w, y) running v20 with STREAMK_POLICY=policy (None: the kernel's default)."""

        def fn(x, w, y):
            if policy is None:
                os.environ.pop("STREAMK_POLICY", None)
            else:
                os.environ["STREAMK_POLICY"] = policy
            return self.mk.matmul(x, w.T, y)

        return fn

    def setup(self, s, first=True):
        """As AiterWorker.setup; first also records the kernels v20 launches."""
        import torch
        from torch.profiler import ProfilerActivity, profile

        old = self.inp  # noqa: F841 - held until the new inputs are allocated
        self.inp = inp = Inputs(s)
        if os.environ.get("V20CMP_REALLOC_WS"):
            # A fresh stream-K workspace (P, locks, tile_sync) too, at new addresses.
            old_ws = dict(self.mk._WORKSPACE)  # noqa: F841 - held until the new one is allocated
            self.mk._WORKSPACE.clear()
            self.mk.streamk_workspace(inp.x.device, NUM_CUS, 256, 256)
            old_ws = None
        old = None
        g = geometry(inp.M, inp.N, inp.K)
        policy, fixup = self.mk.streamk_policy(g["tiles"], g["k_steps"], 256, 256)
        c_policy, c_fixup, c_programs = self.mk.capped_policy(g["tiles"], g["k_steps"], 256)
        if os.environ.get("V20CMP_HEADLINE") == "capped":
            # v20 is capped on every shape; dp is a diagnostic where capped splits, and the old
            # persistent path where matmul now hands a one-wave dp launch to v9. auto and two_tile
            # are not timed (capped never picks two-tile).
            delegated = c_policy == "dp" and g["tiles"] <= NUM_CUS and hasattr(self.mk, "v9_matmul")
            # Snapshots from WORKLOG entries 11-12 hand off only when C has A's dtype (bf16 here).
            if self.v9_same_dtype_only:
                delegated = delegated and inp.otype == torch.bfloat16
            self.raw = {"v20": self._run("capped")}
            if c_policy != "dp" or delegated:
                self.raw["v20_dp"] = self._run("dp")
            res = dict(v20_policy=c_policy, v20_fixup="" if c_policy == "dp" else c_fixup,
                       v20_label=split_label(c_policy, c_fixup, c_programs, g), v20_v9_delegated=delegated,
                       notes=[])
        else:
            self.raw = {"v20": self._run("auto")}
            if policy != "dp":
                self.raw["v20_dp"] = self._run("dp")
            # capped's dp is the same kernel as v20 or v20_dp. two_tile is timed only where a hybrid
            # rule could pick it: a full wave to borrow and every tile split at least 7 ways.
            if c_policy != "dp":
                self.raw["v20_capped"] = self._run("capped")
            if policy != "two_tile" and g["tiles"] >= NUM_CUS and 0 < g["S"] and NUM_CUS // g["S"] >= 7:
                self.raw["v20_two_tile"] = self._run("two_tile")
            res = dict(v20_policy=policy, v20_fixup="" if policy == "dp" else fixup, v20_capped_policy=c_policy,
                       v20_capped_label=pick_label(c_policy, c_fixup, c_programs, g["S"]), notes=[])
        # --rule-policy NAME: the kernel module's NAME_policy returns (policy, fixup, programs) like
        # capped_policy; its dp is the same kernel as v20 or v20_dp.
        rule = os.environ.get("V20CMP_RULE_POLICY")
        if rule:
            r_policy, r_fixup, r_programs = getattr(self.mk, f"{rule}_policy")(g["tiles"], g["k_steps"], 256)
            if r_policy != "dp":
                self.raw["v20_rule"] = self._run(rule)
            res.update(v20_rule_policy=r_policy, v20_rule_label=pick_label(r_policy, r_fixup, r_programs, g["S"]))
        self.raw = {n + lay: (f, lay) for n, f in self.raw.items() for lay in inp.layouts}
        self.fns = {n: inp.rotating(f, lay) for n, (f, lay) in self.raw.items()}
        res["rot_copies"] = len(inp.copies)
        self.ok_fns = []
        for name, (fn, lay) in self.raw.items():
            try:
                res[f"{name}_ok"], res[f"{name}_maxerr"] = check(fn(*inp.sets[lay][0]), inp.ref)
                self.ok_fns.append(name)
            except Exception as e:  # noqa: BLE001
                res["notes"].append(f"{name}: {type(e).__name__}: {e}"[:300])
                res[f"{name}_ok"], res[f"{name}_maxerr"] = False, float("nan")
        torch.cuda.synchronize()
        if first and "v20" in self.ok_fns:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                self.raw["v20"][0](*inp.sets[""][0])
                torch.cuda.synchronize()
            res["v20_kernels_launched"] = " | ".join(sorted({e.name for e in prof.events()
                                                             if e.device_type.name == "CUDA"}))
        return res

    def time(self):
        return bench_all(self.fns, self.ok_fns, self.inp)

    def post(self):
        """Check v20 (and capped) again after the graph replays, and that the flags and counts are re-armed."""
        import torch

        res = {}
        for name in ("v20", "v20_capped", "v20_rule", "v20_pad"):
            if name in self.ok_fns:
                fn, lay = self.raw[name]
                x, w, y = self.inp.sets[lay][0]
                y.fill_(float("nan"))
                res[f"{name}_post_ok"], _ = check(fn(x, w, y), self.inp.ref)
        torch.cuda.synchronize()
        locks, tile_sync = self.mk.streamk_workspace(self.inp.x.device, 256, 256, 256)[1:3]
        if tile_sync.numel() == 5 * 256:
            # v24: the barriers' (count, generation) pairs, the L2 epoch flags (which hold epochs),
            # then the atomic ending's zero counts, which must all be back at 0.
            left = int((tile_sync[:512:2] != 0).sum()) + int((tile_sync[768:] != 0).sum())
        else:
            left = int((tile_sync[0::2] != 0).sum())
        res["v20_flags_left"] = int((locks != 0).sum()) + left
        return res

    def free(self):
        import torch

        self.inp = self.fns = self.raw = None
        torch.cuda.empty_cache()
        return {}


def worker_main(role):
    w = AiterWorker() if role == "aiter" else V20Worker()
    send({"ready": True})
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["cmd"] == "info":
                resp = w.info()
            elif req["cmd"] == "setup":
                resp = w.setup(req["shape"], req.get("first", True))
            elif req["cmd"] == "exit":
                break
            else:
                resp = getattr(w, req["cmd"])()
            send({"ok": True, "data": resp})
        except Exception as e:  # noqa: BLE001
            import traceback

            send({"ok": False, "error": f"{type(e).__name__}: {e}"[:500], "tb": traceback.format_exc()[-2000:]})


# ---------------------------------------------------------------- driver


class Timeout(Exception):
    pass


class WorkerError(Exception):
    pass


class Worker:
    def __init__(self, role, log, paths):
        self.role, self.log, self.paths = role, log, paths
        self.start()

    def env(self):
        env = {k: v for k, v in os.environ.items() if k not in STREAMK_ENV + PLUGIN_ENV}
        if self.role == "aiter":
            env.update(
                PYTHONPATH=f"{HVR}/shim:{TRITON_PY}:{AITER}",
                AITER_JIT_DIR=self.paths["jit"],
                TRITON_CACHE_DIR=f"{HERE}/cache/aiter",
            )
        elif os.environ.get("V20CMP_V20_SCHED") == "aiter":
            # The AITER worker's loader and AITER's opt-in llirSched copy, no amdgcnas hook; the
            # worker opts the tutorial kernels in.
            env.update(
                PYTHONPATH=f"{HVR}/shim:{TRITON_PY}:{AITER}",
                TRITON_CACHE_DIR=f"{HERE}/cache/v20_aiter_sched",
                V20CMP_SNAPSHOT=self.paths["snapshot"],
            )
        else:
            env.update(
                PYTHONPATH=f"{REPO}/scripts/triton_deepbind_shim:{TRITON_PY}",
                LLVM_PASS_PLUGIN_PATH=f"{REPO}/plugins/llir_scheduler/libLlirSched.so",
                TRITON_AMDGCNAS_PLUGIN="1",
                TRITON_CACHE_DIR=f"{HERE}/cache/v20",
                V20CMP_SNAPSHOT=self.paths["snapshot"],
            )
        return env

    def start(self):
        self.p = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "worker", self.role],
            cwd=A16 if self.role == "v20" else HERE, env=self.env(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, start_new_session=True,
        )
        self.q = queue.Queue()
        threading.Thread(target=self._read, args=(self.p.stdout, self.q), daemon=True).start()
        self._wait()

    @staticmethod
    def _read(stream, q):
        for line in stream:
            if line.startswith(TAG):
                q.put(json.loads(line[len(TAG):]))
        q.put(None)

    def _wait(self, timeout=TIMEOUT_S):
        try:
            msg = self.q.get(timeout=timeout)
        except queue.Empty:
            raise Timeout(f"{self.role}: no reply after {timeout} s")
        if msg is None:
            raise WorkerError(f"{self.role}: worker exited (rc={self.p.poll()})")
        return msg

    def call(self, cmd, **kw):
        self.p.stdin.write(json.dumps(dict(cmd=cmd, **kw)) + "\n")
        self.p.stdin.flush()
        msg = self._wait()
        if not msg["ok"]:
            self.log.write(f"[{self.role}] {cmd} failed:\n{msg.get('tb', '')}\n")
            self.log.flush()
            raise WorkerError(f"{self.role} {cmd}: {msg['error']}")
        return msg["data"]

    def kill(self):
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.p.wait()

    def restart(self):
        self.kill()
        self.start()


def wait_idle():
    """The benchmark GPU must read 0% busy on 5 consecutive 2 s samples; wait, never switch GPU."""
    path = f"/sys/bus/pci/devices/{GPU_BDF}/gpu_busy_percent"
    n = 0
    while n < 5:
        with open(path) as f:
            u = int(f.read().strip())
        if u == 0:
            n += 1
        else:
            n = 0
            print(f"GPU {GPU_BDF} busy ({u}%), waiting", flush=True)
        time.sleep(2)
    print(f"GPU {GPU_BDF} idle at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)


def tflops(M, N, K, ms):
    return round(2.0 * M * N * K / statistics.median(ms) * 1e-9, 1) if ms else float("nan")


def run_shape(s, rounds, workers):
    M, N, K = int(s["M"]), int(s["N"]), int(s["K"])
    g = geometry(M, N, K)
    shape = dict(M=M, N=N, K=K, out=s["out"])
    row = dict(M=M, N=N, K=K, bias=s["bias"], out=s["out"], scope=s["scope"], band=s["band"], **g)
    rule = bool(os.environ.get("V20CMP_RULE_POLICY"))
    capped = os.environ.get("V20CMP_HEADLINE") == "capped"
    rotate = bool(int(os.environ.get("V20CMP_ROTATE_MB", "0")))
    pad = bool(os.environ.get("V20CMP_PAD"))
    realloc = bool(os.environ.get("V20CMP_REALLOC"))
    notes, res, used = [], {}, {}
    ms = {n: [] for n in ("pick", "v9", "v20", "v20_dp", "v20_capped", "v20_two_tile")
          + (("v20_rule",) if rule else ()) + (("hbl",) if capped else ())
          + (PAD_NAMES if pad else ())}
    roles = [role for role in ("aiter", "v20") if role in workers]
    # A worker that failed is restarted without this shape's inputs, so it sits out the rest of it.
    dead = set()

    def guarded(role, cmd, **kw):
        if role in dead:
            return None
        try:
            return workers[role].call(cmd, **kw)
        except (Timeout, WorkerError) as e:
            notes.append(str(e))
            dead.add(role)
            workers[role].restart()
            return None

    def merge(r):
        """Round 0's metadata; a check holds only if it held in every round, max error is the worst."""
        notes.extend(n for n in r.pop("notes") if n not in notes)
        for k, v in r.items():
            if k not in res:
                res[k] = v
            elif k.endswith("_ok"):
                res[k] = bool(res[k]) and bool(v)
            elif k.endswith("_maxerr"):
                res[k] = v if v != v or res[k] != res[k] else max(res[k], v)

    # --realloc: every round gets fresh inputs (new buffers), so a timing that depends on where the
    # buffers landed shows up as spread between rounds instead of being repeated in all of them.
    for rnd in range(rounds):
        if rnd == 0 or realloc:
            for role in roles:
                r = guarded(role, "setup", shape=shape, first=rnd == 0)
                if r:
                    merge(r)
        for role in roles:
            for name, x in (guarded(role, "time") or {}).items():
                if name.startswith("_used_"):
                    used[name[6:]] = min(used.get(name[6:], x), x)
                elif name in ms:
                    ms[name].append(x)
    post = guarded("v20", "post")
    for role in roles:
        guarded(role, "free")

    row.update(
        src=res.get("src", ""),
        aiter_pick=res.get("aiter_pick", ""),
        aiter_kernel=res.get("aiter_kernel", ""),
        aiter_pick_tflops=tflops(M, N, K, ms["pick"]),
        v9_tflops=tflops(M, N, K, ms["v9"]),
        v20_tflops=tflops(M, N, K, ms["v20"]),
        v20_policy=res.get("v20_policy", ""),
        v20_fixup=res.get("v20_fixup", ""),
        **(dict(v20_label=res.get("v20_label", ""), v20_snapshot=os.environ.get("V20CMP_SNAPSHOT_SHA", ""),
                hbl_tflops=tflops(M, N, K, ms["hbl"]), hbl_tile=res.get("hbl_tile", ""),
                hbl_sk=res.get("hbl_sk", ""), hbl_gsu=res.get("hbl_gsu", ""),
                hbl_nkernels=res.get("hbl_nkernels", ""), hbl_ok=res.get("hbl_ok", False),
                hbl_maxerr=f"{res.get('hbl_maxerr', float('nan')):.3g}",
                hbl_kernels_torch=res.get("hbl_kernels_torch", "")) if capped else {}),
        **(dict(rot_copies=res.get("rot_copies", ""),
                rot_used_min=min(used.values()) if used else "") if rotate else {}),
        v20_dp_tflops=tflops(M, N, K, ms["v20_dp"]),
        v20_capped_policy=res.get("v20_capped_policy", ""),
        v20_capped_label=res.get("v20_capped_label", ""),
        v20_capped_tflops=tflops(M, N, K, ms["v20_capped"]),
        v20_two_tile_tflops=tflops(M, N, K, ms["v20_two_tile"]),
        **(dict(v20_rule_policy=res.get("v20_rule_policy", ""), v20_rule_label=res.get("v20_rule_label", ""),
                v20_rule_tflops=tflops(M, N, K, ms["v20_rule"]), v20_rule_ok=res.get("v20_rule_ok", ""),
                v20_rule_post_ok=(post or {}).get("v20_rule_post_ok", "")) if rule else {}),
        pick_ok=res.get("pick_ok", False), v9_ok=res.get("v9_ok", False),
        v20_ok=res.get("v20_ok", False), v20_dp_ok=res.get("v20_dp_ok", ""),
        v20_capped_ok=res.get("v20_capped_ok", ""), v20_two_tile_ok=res.get("v20_two_tile_ok", ""),
        v20_post_ok=(post or {}).get("v20_post_ok", False),
        v20_capped_post_ok=(post or {}).get("v20_capped_post_ok", ""),
        v20_flags_left=(post or {}).get("v20_flags_left", ""),
        pick_maxerr=f"{res.get('pick_maxerr', float('nan')):.3g}",
        v9_maxerr=f"{res.get('v9_maxerr', float('nan')):.3g}",
        v20_maxerr=f"{res.get('v20_maxerr', float('nan')):.3g}",
        hbl_kernels=res.get("hbl_kernels", "") if res.get("aiter_pick") in ("torch", "hipblaslt") else "",
        pick_kernels_launched=res.get("pick_kernels_launched", ""),
        **(dict(v20_v9_delegated=res.get("v20_v9_delegated", ""),
                v20_kernels_launched=res.get("v20_kernels_launched", "")) if capped else {}),
        **(pad_columns(M, N, K, ms, res, post) if pad else {}),
        **{f"{n}_ms": " ".join(f"{x:.4f}" for x in ms[n]) for n in ms},
        **{f"{n}_spread": spread(ms[n]) for n in ms},
        note="; ".join(notes),
    )
    return row


PAD_NAMES = ("pick_pad", "v9_pad", "hbl_pad", "v20_pad", "v20_dp_pad")


def spread(ms):
    """(slowest - fastest) / median round, in percent."""
    return round((max(ms) - min(ms)) / statistics.median(ms) * 100, 2) if len(ms) > 1 else float("nan")


def pad_columns(M, N, K, ms, res, post):
    """The padded layout's timings and checks (--pad512)."""
    cols = dict(pad_strides=res.get("pad_strides", ""), rot_copies_pad=res.get("rot_copies_pad", ""))
    for n in PAD_NAMES:
        cols[f"{n}_tflops"] = tflops(M, N, K, ms[n])
        cols[f"{n}_ok"] = res.get(f"{n}_ok", "")
        cols[f"{n}_maxerr"] = f"{res.get(f'{n}_maxerr', float('nan')):.3g}"
    for n in ("pick", "v9", "hbl"):
        cols[f"{n}_pad_copy"] = res.get(f"{n}_pad_copy", "")
        cols[f"{n}_pad_extra_kernels"] = res.get(f"{n}_pad_extra_kernels", "")
    cols.update(pick_pad_kernels_launched=res.get("pick_pad_kernels_launched", ""),
                hbl_tile_pad=res.get("hbl_tile_pad", ""), hbl_nkernels_pad=res.get("hbl_nkernels_pad", ""),
                v20_pad_post_ok=(post or {}).get("v20_pad_post_ok", ""))
    return cols


def sweep(args):
    global V20_LIVE, V9_LIVE
    assert args.v20_only or os.environ.get("V20CMP_ENV") == "rocm724", "run through run_rocm724_v20.sh"
    os.environ["V20CMP_KERNEL"] = args.kernel
    V20_LIVE = os.path.join(A16, KERNELS[args.kernel][0], "matmul_kernel.py")
    V9_LIVE = os.path.join(A16, KERNELS[args.kernel][2], "matmul_kernel.py")
    os.makedirs(RESULTS, exist_ok=True)
    paths = run_paths(args.tag, args.jit_tag)
    snap = paths["snapshot"]
    if not os.path.exists(snap):
        src = run_paths(args.snapshot_from)["snapshot"] if args.snapshot_from else V20_LIVE
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        shutil.copy2(src, snap)
        print(f"v20 snapshot taken from {src}", flush=True)
    if sha256(snap) != sha256(V20_LIVE):
        print(f"note: the live v20 kernel differs from the snapshot ({sha256(snap)}); "
              "the sweep uses the snapshot", flush=True)

    if args.rule_policy:
        os.environ["V20CMP_RULE_POLICY"] = args.rule_policy
    if args.headline == "capped":
        os.environ["V20CMP_HEADLINE"] = "capped"
        os.environ["V20CMP_HBL"] = "1"
    if args.rotating_mb:
        os.environ["V20CMP_ROTATE_MB"] = str(args.rotating_mb)
    if args.pad512:
        assert args.headline == "capped", "--pad512 times the capped headline's kernels"
        os.environ["V20CMP_PAD"] = "1"
    if args.realloc:
        os.environ["V20CMP_REALLOC"] = "1"
    if args.realloc_workspace:
        assert args.realloc, "--realloc-workspace needs --realloc"
        os.environ["V20CMP_REALLOC_WS"] = "1"
    os.environ["STREAMK_P_PAD"] = str(args.p_pad)
    os.environ["V20CMP_V20_SCHED"] = args.v20_sched
    os.environ["V20CMP_SNAPSHOT_SHA"] = sha256(snap)
    wait_idle()
    log = open(paths["log"], "a")
    workers = {role: Worker(role, log, paths) for role in (("v20",) if args.v20_only else ("aiter", "v20"))}
    info = {role: w.call("info") for role, w in workers.items()}
    worklog = os.path.join(os.path.dirname(V20_LIVE), "WORKLOG.md")
    with open(worklog) as f:
        headings = [l.strip() for l in f if l.startswith("## ")]
    info["run"] = dict(headline=args.headline, rotating_mb=args.rotating_mb, rounds=args.rounds,
                       v20_snapshot_sha256=sha256(snap), v20_snapshot_from=args.snapshot_from or V20_LIVE,
                       v20_worklog_latest=headings[-1] if headings else "",
                       v20_live_sha256_at_start=sha256(V20_LIVE),
                       v9_live_sha256_at_start=sha256(V9_LIVE), pad512=args.pad512,
                       pad_bytes=PAD_BYTES if args.pad512 else 0, realloc=args.realloc,
                       realloc_workspace=args.realloc_workspace, streamk_p_pad=args.p_pad,
                       v20_sched=args.v20_sched, kernel=args.kernel, kernel_name=KERNELS[args.kernel][1],
                       v9_handoff=KERNELS[args.kernel][3], hip_device=GPU)
    for role, i in info.items():
        print(f"{role}: " + " ".join(f"{k}={v}" for k, v in i.items()), flush=True)
    with open(paths["envinfo"], "w") as f:
        json.dump(info, f, indent=2)

    out = args.out or paths["details"]
    done = {key(r["M"], r["N"], r["K"], r["out"]) for r in read_rows(out)}
    shapes = load_shapes()
    only = {int(i) for i in args.only.split(",")} if args.only else None
    t0 = time.time()
    try:
        for i, s in enumerate(shapes):
            if key(s["M"], s["N"], s["K"], s["out"]) in done or (only is not None and i not in only):
                continue
            row = run_shape(s, args.rounds, workers)
            append_row(out, row)
            flags = "".join(
                f" {n.upper()}-WRONG" for n in (("v20",) if args.v20_only else ("pick", "v9", "v20"))
                if str(row[f"{n}_ok"]) != "True"
            )
            flags += "".join(f" {n.upper()}-WRONG" for n in ("v20_capped", "v20_two_tile", "v20_rule")
                             if str(row.get(f"{n}_ok")) == "False")
            if (str(row["v20_post_ok"]) != "True" or str(row["v20_capped_post_ok"]) == "False"
                    or str(row.get("v20_rule_post_ok")) == "False" or str(row["v20_flags_left"]) != "0"):
                flags += (f" V20-POST(ok={row['v20_post_ok']},capped={row['v20_capped_post_ok']},"
                          f"rule={row.get('v20_rule_post_ok', '')},flags={row['v20_flags_left']})")
            dp = "" if row["v20_dp_tflops"] != row["v20_dp_tflops"] else f", dp {row['v20_dp_tflops']:.1f}"
            if args.pad512:
                flags += "".join(f" {n.upper()}-WRONG" for n in PAD_NAMES if str(row[f"{n}_ok"]) == "False")
                flags += "".join(f" {n.upper()}-PAD-COPY" for n in ("pick", "v9", "hbl")
                                 if str(row[f"{n}_pad_copy"]) == "True")
                if str(row["v20_pad_post_ok"]) == "False":
                    flags += " V20-PAD-POST"
            extra = "".join(f" {n} {row[f'v20_{n}_tflops']:.1f}" for n in ("capped", "two_tile", "rule")
                            if row.get(f"v20_{n}_tflops", float("nan")) == row.get(f"v20_{n}_tflops", float("nan")))
            if args.rule_policy:
                extra += f" rule: {row['v20_rule_label']}"
            if args.headline == "capped":
                flags += "" if str(row["hbl_ok"]) == "True" else " HBL-WRONG"
                rot = f" copies {row['rot_copies']} (used >= {row['rot_used_min']})" if args.rotating_mb else ""
                deleg = " ->v9" if str(row["v20_v9_delegated"]) == "True" else ""
                padded = (f" | pad: pick {row['pick_pad_tflops']:.1f} hbl {row['hbl_pad_tflops']:.1f} "
                          f"v9 {row['v9_pad_tflops']:.1f} v20 {row['v20_pad_tflops']:.1f} "
                          f"(spread v20 {row['v20_spread']:.1f}%/{row['v20_pad_spread']:.1f}%)") if args.pad512 else ""
                print(
                    f"[{i + 1}/{len(shapes)} {time.time() - t0:6.0f}s] {row['M']}x{row['N']}x{row['K']} "
                    f"{row['out']:8s} S={row['S']:3d} pick={row['aiter_pick']:6s} {row['aiter_pick_tflops']:7.1f} "
                    f"hbl={row['hbl_tflops']:7.1f} ({row['hbl_tile']}) v9={row['v9_tflops']:7.1f} "
                    f"v20={row['v20_tflops']:7.1f} ({row['v20_label']}{deleg}{dp}){rot}{padded}"
                    f"{flags}{' ' + row['note'] if row['note'] else ''}",
                    flush=True,
                )
                continue
            print(
                f"[{i + 1}/{len(shapes)} {time.time() - t0:6.0f}s] {row['M']}x{row['N']}x{row['K']} {row['out']:8s} "
                f"S={row['S']:3d} pick={row['aiter_pick']:6s} {row['aiter_pick_tflops']:7.1f} "
                f"v9={row['v9_tflops']:7.1f} v20={row['v20_tflops']:7.1f} ({row['v20_policy']} {row['v20_fixup']}{dp})"
                f" capped: {row['v20_capped_label']}{extra}"
                f"{flags}{' ' + row['note'] if row['note'] else ''}",
                flush=True,
            )
    finally:
        for w in workers.values():
            w.kill()
    print("DONE", flush=True)


# ---------------------------------------------------------------- report


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def pct(a, b):
    return (a / b - 1.0) * 100.0 if a == a and b == b and b > 0 else float("nan")


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def stats_line(label, gaps):
    gaps = [g for g in gaps if g == g]
    if not gaps:
        return f"| {label} | 0 | - | - | - | - | - |"
    wins = sum(g > 0 for g in gaps)
    ties = sum(abs(g) <= 1.0 for g in gaps)
    return (f"| {label} | {len(gaps)} | {wins} | {ties} | {statistics.median(gaps):+.1f}% | "
            f"{quantile(gaps, 0.1):+.1f}% | {quantile(gaps, 0.9):+.1f}% |")


STATS_HEAD = ("| | shapes | v20 faster | within 1% | median | p10 | p90 |\n"
              "|---|---:|---:|---:|---:|---:|---:|")


def waves_bucket(r):
    w, s = int(r["waves"]), int(r["S"])
    if w == 0:
        return "under 1 wave"
    if w == 1:
        return "1 to 2 waves"
    if w < 4:
        return "2 to 4 waves"
    if w < 8:
        return "4 to 8 waves"
    return "8 waves or more"


def policy_label(r):
    p, f = r["v20_policy"], r["v20_fixup"]
    return {"dp": "data-parallel", "two_tile": f"two-tile ({f})",
            "one_tile": "one-tile reduce-scatter" if f == "rs" else "one-tile owner"}.get(p, p or "?")


def drift(rows, base_rows, title, v9_col, changed_table=True):
    """This run against an earlier one: same picks, speed drift, and every shape whose pick changed."""
    base = {key(r["M"], r["N"], r["K"], r["out"]): r for r in base_rows}
    pairs = [(r, base[key(r["M"], r["N"], r["K"], r["out"])]) for r in rows
             if key(r["M"], r["N"], r["K"], r["out"]) in base]
    if not pairs:
        return []
    same = [(r, o) for r, o in pairs if (o["aiter_pick"], o["aiter_kernel"]) == (r["aiter_pick"], r["aiter_kernel"])]
    changed = [(r, o) for r, o in pairs if (r, o) not in same]

    def spread(label, xs):
        xs = [x for x in xs if x == x]
        if not xs:
            return []
        return [f"- {label}: median {statistics.median(xs):+.1f}%, p10 {quantile(xs, 0.1):+.1f}%, "
                f"p90 {quantile(xs, 0.9):+.1f}%."]

    L = [f"## Against {title}", "",
         f"- AITER's library and kernel are the same on {len(same)} of {len(pairs)} shapes and differ on {len(changed)}."]
    L += spread("AITER's pick now against then, same kernel",
                [pct(fnum(r["aiter_pick_tflops"]), fnum(o["aiter_pick_tflops"])) for r, o in same])
    L += spread("AITER's pick now against then, changed kernel",
                [pct(fnum(r["aiter_pick_tflops"]), fnum(o["aiter_pick_tflops"])) for r, o in changed])
    L += spread("v9 now against then", [pct(fnum(r["v9_tflops"]), fnum(o[v9_col])) for r, o in pairs])
    if "v20_tflops" in pairs[0][1]:
        L += spread("v20 now against then", [pct(fnum(r["v20_tflops"]), fnum(o["v20_tflops"])) for r, o in pairs])
        then = [pct(fnum(o["v20_tflops"]), fnum(o["aiter_pick_tflops"])) for _, o in pairs]
        now = [pct(fnum(r["v20_tflops"]), fnum(r["aiter_pick_tflops"])) for r, _ in pairs]
        L += [f"- v20 faster than AITER's pick: {sum(g > 0 for g in now)} shapes now (median {statistics.median(now):+.1f}%), "
              f"{sum(g > 0 for g in then)} then (median {statistics.median(then):+.1f}%)."]
    L += [""]
    if changed_table and changed:
        L += [f"### Shapes where AITER's pick changed ({len(changed)})", "",
              "| shape | out | pick then | TFLOPS then | pick now | TFLOPS now | change | v20 now | v20 vs AITER now |",
              "|---|---|---|---:|---|---:|---:|---:|---:|"]
        for r, o in sorted(changed, key=lambda p: pct(fnum(p[0]["aiter_pick_tflops"]), fnum(p[1]["aiter_pick_tflops"]))):
            L.append(
                f"| {r['M']}x{r['N']}x{r['K']} | {r['out']} | {o['aiter_pick']} `{o['aiter_kernel'][:60]}` | "
                f"{fnum(o['aiter_pick_tflops']):.0f} | {r['aiter_pick']} `{r['aiter_kernel'][:60]}` | "
                f"{fnum(r['aiter_pick_tflops']):.0f} | {pct(fnum(r['aiter_pick_tflops']), fnum(o['aiter_pick_tflops'])):+.1f}% | "
                f"{fnum(r['v20_tflops']):.0f} | {pct(fnum(r['v20_tflops']), fnum(r['aiter_pick_tflops'])):+.1f}% |")
        L += [""]
    return L


def aiter_tile(r):
    """MxN tile of AITER's pick, from its kernel name (hipBLASLt: the MT of its first kernel)."""
    import re

    lib = r["aiter_pick"]
    if lib in ("torch", "hipblaslt"):
        m = re.search(r"MT(\d+)x(\d+)x\d+", r["hbl_kernels"])
    elif lib == "flydsl":
        m = re.search(r"_t(\d+)x(\d+)x\d+", r["aiter_kernel"])
    elif lib == "opus":
        m = re.search(r"opus_gemm(?:_[a-z_]+)?_\d+x(\d+)x(\d+)x\d+", r["aiter_kernel"])
    elif lib == "asm":
        m = re.search(r"(\d+)x(\d+)", r["aiter_kernel"])
    else:
        m = None
    return f"{m.group(1)}x{m.group(2)}" if m else "?"


def grid_bucket(r):
    t = int(r["tiles"])
    return "Under 256 (less than one wave)" if t < 256 else ("256 to 1023" if t < 1024 else "1024 or more")


GRID_ORDER = ("Under 256 (less than one wave)", "256 to 1023", "1024 or more")


def config_group(r):
    p, f = r["v20_policy"], r["v20_fixup"]
    if p == "one_tile":
        return f"One-tile {'reduce-scatter' if f == 'rs' else 'owner'}, K {'< 2048' if int(r['K']) < 2048 else '>= 2048'}"
    return {"dp": "Data-parallel", "spread": "Spread"}.get(p, p)


def report_capped(args, rows, info):
    """Report for a --headline capped run: streamk_wip (v20 capped) against AITER's pick, v9 and
    hipBLASLt (torch), in the ticket's layout, plus the configuration and hipBLASLt tile per shape."""
    a, v, run = info.get("aiter", {}), info.get("v20", {}), info.get("run", {})
    for r in rows:
        r["t_pick"], r["t_v9"], r["t_v20"], r["t_hbl"] = (
            fnum(r["aiter_pick_tflops"]), fnum(r["v9_tflops"]), fnum(r["v20_tflops"]), fnum(r["hbl_tflops"]))
        r["t_dp"] = fnum(r["v20_dp_tflops"])
        r["a_tile"] = aiter_tile(r)
    rot = run.get("rotating_mb", 0)

    def counts(gaps):
        gaps = [g for g in gaps if g == g]
        if not gaps:
            return "0 | 0 | 0 | -"
        return (f"{sum(g > 1 for g in gaps)} | {sum(abs(g) <= 1 for g in gaps)} | {sum(g < -1 for g in gaps)} | "
                f"{statistics.median(gaps):+.1f}%")

    def two_way(title, keyf, order, other_col, other_name, base_cols=(("t_v9", "v9"), ("t_v20", "streamk_wip"))):
        head = " | ".join(f"{n}: faster | {n}: within 1% | {n}: slower | {n}: median" for _, n in base_cols)
        out = [f"**{title}**", "", f"| | Shapes | {head} |",
               "|---|---:|" + "---:|---:|---:|---:|" * len(base_cols)]
        groups = {}
        for r in rows:
            groups.setdefault(keyf(r), []).append(r)
        for k in ["All shapes"] + [k for k in order if k in groups] + sorted(k for k in groups if k not in order):
            rs = rows if k == "All shapes" else groups.get(k, [])
            if not rs:
                continue
            cells = " | ".join(counts([pct(r[c], r[other_col]) for r in rs]) for c, _ in base_cols)
            out.append(f"| {k} | {len(rs)} | {cells} |")
        return out + ["", f"Each pair of columns compares that kernel with {other_name}; positive medians mean it is faster.", ""]

    kname = run.get("kernel_name", "v20_streamk_reduce_scatter")
    L = [f"# streamk_wip ({run.get('kernel', 'v20')}, capped) vs AITER, v9 and hipBLASLt on the 630 shapes"
         f"{f' ({rot} MB rotating buffer)' if rot else ''}", ""]
    L += [
        "Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from "
        "`torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with "
        f"{run.get('rounds', 2)} interleaved rounds and the median taken.",
        "",
        f"- Stack: torch {a.get('torch', '?')}, hipBLASLt {a.get('hipblaslt', '?')}, Triton {a.get('triton', '?')}.",
        f"- GPU: HIP device {a.get('hip_device', '0')}, PCI bus {a.get('gpu_pci_bus', '?')}, {a.get('gpu', '?')}.",
        f"- AITER: {a.get('aiter_commit', '?')}, on top of {a.get('aiter_base', '?')}.",
        f"- streamk_wip: tutorial `{kname}`, snapshot sha256 {run.get('v20_snapshot_sha256', '?')} "
        f"(newest WORKLOG entry: \"{run.get('v20_worklog_latest', '?').lstrip('# ')}\"), `STREAMK_POLICY=capped` (the "
        "kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in "
        "`v20_label`.",
        "- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and "
        "`torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first "
        "`Cijk_` kernel it launches.",
        ("- Cache state: each callable rotates over max(2, ceil(" + str(rot) + " MB / set)) copies of x, w and C "
         "inside the captured graph, so no launch reads the inputs of the previous one. Outputs that a callable "
         "allocates itself (AITER's pick, hipBLASLt) are kept alive in a ring so they rotate too.") if rot else
        "- Cache state: back to back on the same buffers (inputs stay cached between launches).",
        "- \"Faster\" and \"slower\" mean a gap of more than 1%; \"within 1%\" holds the rest.",
        "",
    ]
    if rot:
        cop = [int(r["rot_copies"]) for r in rows if r.get("rot_copies")]
        short = [r for r in rows if r.get("rot_used_min") and r.get("rot_copies")
                 and int(float(r["rot_used_min"])) < int(r["rot_copies"])]

        def footprint_mb(r):
            M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
            ob = 4 if r["out"] == "float32" else 2
            return int(float(r["rot_used_min"])) * ((M * K + N * K) * 2 + M * N * ob) / 2**20

        fp = sorted(((footprint_mb(r), r) for r in rows if r.get("rot_used_min")), key=lambda p: p[0])
        small = [r for f, r in fp if f < 512]
        L += [f"Copies per shape: median {statistics.median(cop):.0f}, minimum {min(cop)}; {sum(c == 2 for c in cop)} "
              f"shapes at the minimum of 2.",
              f"`do_bench_cudagraph` sizes its graph from a host-side estimate, so on {len(short)} small shapes a timed "
              "graph touched fewer copies than it had. What matters is the data a graph cycles through: copies touched "
              f"x set size is at least {fp[0][0]:.0f} MB on every shape (median {statistics.median([f for f, _ in fp]):.0f} MB), "
              f"against a 256 MB last-level cache; {len(small)} shapes are under 512 MB"
              + (f" ({', '.join(f'{r['M']}x{r['N']}x{r['K']}' for r in small[:8])})" if small else "") + ".", ""]

    L += two_way("Against AITER's pick, by the configuration capped chose", config_group,
                 ("Data-parallel", "One-tile reduce-scatter, K >= 2048", "One-tile owner, K >= 2048",
                  "One-tile reduce-scatter, K < 2048", "One-tile owner, K < 2048", "Spread"),
                 "t_pick", "AITER's pick")
    L += two_way("Against AITER's pick, by the tile AITER uses",
                 lambda r: "AITER uses a 256x256 tile" if r["a_tile"] == "256x256" else "AITER uses another tile",
                 ("AITER uses a 256x256 tile", "AITER uses another tile"), "t_pick", "AITER's pick")
    L += two_way("Against AITER's pick, by v9 grid size (number of 256x256 output tiles)", grid_bucket, GRID_ORDER,
                 "t_pick", "AITER's pick")
    L += two_way("Against hipBLASLt (torch), by v9 grid size", grid_bucket, GRID_ORDER, "t_hbl", "hipBLASLt",
                 base_cols=(("t_pick", "AITER's pick"), ("t_v9", "v9"), ("t_v20", "streamk_wip")))

    tiles = {}
    for r in rows:
        tiles.setdefault(r["hbl_tile"] or "?", []).append(r)
    L += ["**hipBLASLt's tile choice**", "",
          "| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | "
          "slower | median |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t, rs in sorted(tiles.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {t} | {len(rs)} | {statistics.median([r['t_hbl'] for r in rs]):.0f} | "
                 f"{counts([pct(r['t_v20'], r['t_hbl']) for r in rs])} |")
    sk = {}
    for r in rows:
        sk[(r["hbl_sk"] or "none", r["hbl_gsu"] or "-", r["hbl_nkernels"])] = sk.get(
            (r["hbl_sk"] or "none", r["hbl_gsu"] or "-", r["hbl_nkernels"]), 0) + 1
    L += ["", "Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): "
          + ", ".join(f"{k[0]} GSU{k[1]} x{k[2]}: {n}" for k, n in sorted(sk.items(), key=lambda kv: -kv[1]))
          + ". `SK3` marks a kernel built with stream-K support, not that this launch split K.", ""]

    L += two_way("streamk_wip against v9, by configuration", config_group,
                 ("Data-parallel", "One-tile reduce-scatter, K >= 2048", "One-tile owner, K >= 2048",
                  "One-tile reduce-scatter, K < 2048", "One-tile owner, K < 2048", "Spread"),
                 "t_v9", "v9", base_cols=(("t_v20", "streamk_wip"),))
    split = [r for r in rows if r["v20_policy"] != "dp"]
    if split:
        groups = {}
        for r in split:
            groups.setdefault(config_group(r), []).append(pct(r["t_v20"], r["t_dp"]))
        L += ["**Diagnostic: capped's split against forcing data-parallel**", "",
              "| Configuration | Shapes | split faster | within 1% | split slower | median |", "|---|---:|---:|---:|---:|---:|"]
        L += [f"| {k} | {len(g)} | {counts(g)} |" for k, g in sorted(groups.items())]
        L += [""]

    sel = ["512x2304x16384", "512x5120x25600", "512x7168x35840", "8192x2304x16384", "4096x5120x25600",
           "4096x4608x8192", "1024x3072x512", "2048x2048x512", "4096x5120x640", "8192x2048x512"]
    by = {f"{r['M']}x{r['N']}x{r['K']}": r for r in rows}
    L += ["**Selected shapes (TFLOPS)**", "",
          "| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | "
          "streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |",
          "|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|"]
    for s in sel:
        r = by.get(s)
        if r:
            L.append(f"| {s} | {r['tiles']} | {r['aiter_pick']} | {r['t_pick']:.0f} | {r['t_hbl']:.0f} | {r['hbl_tile']} | "
                     f"{r['t_v9']:.0f} | {r['t_v20']:.0f} | {r['v20_label']} | {pct(r['t_v20'], r['t_v9']):+.0f}% | "
                     f"{pct(r['t_v20'], r['t_pick']):+.0f}% | {pct(r['t_v20'], r['t_hbl']):+.0f}% |")
    L += [""]

    if args.baseline and os.path.exists(args.baseline) and os.path.abspath(args.baseline) != os.path.abspath(args.details):
        base = {key(r["M"], r["N"], r["K"], r["out"]): r for r in read_rows(args.baseline)}
        pairs = [(r, base[key(r["M"], r["N"], r["K"], r["out"])]) for r in rows
                 if key(r["M"], r["N"], r["K"], r["out"]) in base]
        cols = [("aiter_pick_tflops", "AITER's pick"), ("v9_tflops", "v9"), ("v20_tflops", "v20")]
        if pairs and "hbl_tflops" in pairs[0][1]:
            cols.append(("hbl_tflops", "hipBLASLt"))
        L += [f"**Against `{os.path.relpath(args.baseline, HERE)}`** (this run / that run - 1, median; "
              "for v20 the baseline may use a different policy or kernel snapshot)", "",
              "| | Shapes | " + " | ".join(n for _, n in cols) + " |", "|---|---:|" + "---:|" * len(cols)]
        groups = {}
        for r, o in pairs:
            for k in ("All shapes", grid_bucket(r), "K < 2048" if int(r["K"]) < 2048 else "K >= 2048"):
                groups.setdefault(k, []).append((r, o))
        for k in ("All shapes",) + GRID_ORDER + ("K < 2048", "K >= 2048"):
            ps = groups.get(k, [])
            if not ps:
                continue
            cells = []
            for c, _ in cols:
                g = [x for x in (pct(fnum(r[c]), fnum(o[c])) for r, o in ps) if x == x]
                cells.append(f"{statistics.median(g):+.1f}%" if g else "-")
            L.append(f"| {k} | {len(ps)} | " + " | ".join(cells) + " |")
        same = sum(1 for r, o in pairs if (r["aiter_pick"], r["aiter_kernel"]) == (o["aiter_pick"], o["aiter_kernel"]))
        L += ["", f"AITER's library and kernel are the same on {same} of {len(pairs)} shapes.", ""]

    bad = [r for r in rows if any(str(r.get(c)) != "True" for c in ("pick_ok", "v9_ok", "v20_ok", "hbl_ok", "v20_post_ok"))
           or str(r.get("v20_dp_ok")) == "False" or str(r["v20_flags_left"]) != "0" or r["note"]]
    L += ["**Wrong outputs, failures and notes**", ""]
    if not bad:
        L += ["None.", ""]
    else:
        L += ["| Shape | out | pick ok (max err) | v9 ok | hipBLASLt ok | streamk_wip ok | after timing | flags left | note |",
              "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['M']}x{r['N']}x{r['K']} | {r['out']} | {r['pick_ok']} ({r['pick_maxerr']}) | {r['v9_ok']} | "
              f"{r['hbl_ok']} | {r['v20_ok']} ({r['v20_maxerr']}) | {r['v20_post_ok']} | {r['v20_flags_left']} | "
              f"{r['note'][:150]} |" for r in bad]
        L += [""]
        pick_only = [r for r in bad if str(r["pick_ok"]) != "True" and all(
            str(r.get(c)) == "True" for c in ("v9_ok", "v20_ok", "hbl_ok", "v20_post_ok")) and str(r["v20_flags_left"]) == "0"
            and not r["note"]]
        L += [f"{len(pick_only)} of these are AITER's pick alone outside the tolerance; every other kernel passed.", ""]

    L += ["## Appendix: every shape", "",
          "| Shape | out | tiles | streamk_wip config | streamk_wip | v9 | AITER pick | AITER | hipBLASLt | hipBLASLt tile |",
          "|---|---|---:|---|---:|---:|---|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['M']}x{r['N']}x{r['K']} | {r['out']} | {r['tiles']} | {r['v20_label']} | {r['t_v20']:.1f} | "
                 f"{r['t_v9']:.1f} | {r['aiter_pick']} | {r['t_pick']:.1f} | {r['t_hbl']:.1f} | {r['hbl_tile']} |")
    L += [""]
    with open(args.summary, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.summary}")


def report(args):
    paths = run_paths(args.tag)
    args.details = args.details or paths["details"]
    args.summary = args.summary or paths["summary"]
    rows = read_rows(args.details)
    info = {}
    if os.path.exists(paths["envinfo"]):
        with open(paths["envinfo"]) as f:
            info = json.load(f)
    if rows and "v20_pad_tflops" in rows[0]:
        from pad_report import report_pad

        return report_pad(args, rows, info)
    if rows and "v20_label" in rows[0]:
        return report_capped(args, rows, info)
    for r in rows:
        r["gap"] = pct(fnum(r["v20_tflops"]), fnum(r["aiter_pick_tflops"]))
        r["gap_v9"] = pct(fnum(r["v20_tflops"]), fnum(r["v9_tflops"]))
        r["gap_dp"] = pct(fnum(r["v20_tflops"]), fnum(r["v20_dp_tflops"]))

    # A --v20-only run has no AITER or v9 timings; its report keeps only the v20 sections.
    timed = [r for r in rows if r["gap"] == r["gap"]]
    aiter_run = bool(timed)
    a, v = info.get("aiter", {}), info.get("v20", {})
    if not aiter_run:
        L = ["# v20 stream-K policies on the 630 shapes (v20 only)", "",
             "Shapes: `supported_status_scoped.csv`, timed as in the AITER comparison (`do_bench_cudagraph`, 2 "
             "interleaved rounds, median) but without the AITER worker (`sweep --v20-only`).", "",
             f"- Stack: torch {v.get('torch', '?')}, Triton {v.get('triton', '?')}.",
             f"- GPU: HIP device 0, PCI bus {v.get('gpu_pci_bus', '?')}, {v.get('gpu', '?')}.",
             f"- v20: `v20_streamk_reduce_scatter` (snapshot sha256 {v.get('v20_snapshot_sha256', '?')}), v9 tile order.",
             ""]
    else:
        L = ["# v20 stream-K (auto) vs AITER's pick on the 630 shapes", ""]
    L += [] if not aiter_run else [
        "Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). "
        "Method as in `hip_version_repro/sweep.py` (Sep 29): inputs from `torch.manual_seed(0)`, every output "
        "checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 2 interleaved rounds and the "
        "median taken.",
        "",
        f"- Stack: torch {a.get('torch', '?')}, hipBLASLt {a.get('hipblaslt', '?')}, Triton {a.get('triton', '?')}.",
        f"- GPU: HIP device 0, PCI bus {a.get('gpu_pci_bus', '?')} (the Sep 29 GPU), {a.get('gpu', '?')}.",
        f"- AITER: `{a.get('aiter', '?')}` at {a.get('aiter_commit', '2a1c2087c (llirSched_integration)')}, "
        f"on top of {a.get('aiter_base', '8253efc40 (main, Sep 28)')}; its llirSched plugin `{a.get('llir_sched', '?')}`.",
        f"- v20: the tutorial's `v20_streamk_reduce_scatter` (snapshot sha256 {v.get('v20_snapshot_sha256', '?')}), "
        "auto policy and fixup, all 256 programs, v9 tile order, with the tutorial's llirSched plugin and amdgcnas hook. "
        "It writes C in C's dtype, so the fp32-output shapes are like for like.",
        "- Gaps are v20 against the other kernel: positive means v20 is faster. "
        "\"Within 1%\" counts shapes inside run-to-run noise.",
        "",
    ]

    checked = ("pick", "v9", "v20") if aiter_run else ("v20",)
    bad = [r for r in rows if any(str(r[f"{n}_ok"]) != "True" for n in checked)
           or any(str(r.get(c)) == "False" for c in ("v20_capped_ok", "v20_two_tile_ok", "v20_capped_post_ok",
                                                      "v20_rule_ok", "v20_rule_post_ok"))
           or str(r["v20_post_ok"]) != "True" or str(r["v20_flags_left"]) != "0" or r["note"]]

    def section(title, keyf, order=None):
        groups = {}
        for r in rows:
            groups.setdefault(keyf(r), []).append(r["gap"])
        keys = order or sorted(groups)
        out = [f"### By {title}", "", STATS_HEAD]
        out += [stats_line(k, groups[k]) for k in keys if k in groups]
        return out + [""]

    if aiter_run:
        L += [f"Measured {len(rows)} shapes; {len(timed)} have both v20 and AITER timings.", ""]
        L += ["## v20 against AITER's pick", "", STATS_HEAD, stats_line("all shapes", [r["gap"] for r in rows]), ""]
        L += section("the policy auto chose", policy_label,
                     ["data-parallel", "two-tile (rs)", "two-tile (owner)", "one-tile reduce-scatter", "one-tile owner"])
        L += section("AITER's pick", lambda r: r["aiter_pick"] or "?")
        L += section("scope (as in the CSV)", lambda r: r["scope"])
        L += section("output tiles at 256x256", waves_bucket,
                     ["under 1 wave", "1 to 2 waves", "2 to 4 waves", "4 to 8 waves", "8 waves or more"])
        L += section("K", lambda r: "K < 2048" if int(r["K"]) < 2048 else "K >= 2048", ["K < 2048", "K >= 2048"])
        L += section("leftover tiles S", lambda r: "S = 0" if int(r["S"]) == 0 else
                     ("S > 128" if int(r["S"]) > 128 else "1 <= S <= 128"), ["S = 0", "1 <= S <= 128", "S > 128"])

        L += ["## v20 against v9 (stream-K and persistence gain)", "",
              STATS_HEAD.replace("v20 faster", "v20 faster than v9")]
        groups = {}
        for r in rows:
            groups.setdefault(policy_label(r), []).append(r["gap_v9"])
        L += [stats_line("all shapes", [r["gap_v9"] for r in rows])]
        L += [stats_line(k, groups[k]) for k in
              ["data-parallel", "two-tile (rs)", "two-tile (owner)", "one-tile reduce-scatter", "one-tile owner"]
              if k in groups]
        L += [""]
    else:
        L += [f"Measured {len(rows)} shapes.", ""]

    split = [r for r in rows if r["v20_policy"] != "dp"]
    L += ["## Diagnostic: auto's stream-K choice against forcing data-parallel", "",
          f"On the {len(split)} shapes where auto picks a stream-K path. Positive means auto's choice is faster "
          "than v20 `dp`; this column never counts as v20's result.", "",
          STATS_HEAD.replace("v20 faster", "auto faster")]
    groups = {}
    for r in split:
        groups.setdefault((policy_label(r), "K < 2048" if int(r["K"]) < 2048 else "K >= 2048"), []).append(r["gap_dp"])
    L += [stats_line("all", [r["gap_dp"] for r in split])]
    L += [stats_line(f"{p}, {k}", groups[(p, k)]) for p, k in sorted(groups)]
    L += [""]

    if rows and "v20_capped_tflops" in rows[0]:
        for r in rows:
            dp_t = fnum(r["v20_tflops"]) if r["v20_policy"] == "dp" else fnum(r["v20_dp_tflops"])
            r["capped_t"] = dp_t if r["v20_capped_policy"] == "dp" else fnum(r["v20_capped_tflops"])
            r["gap_capped"] = pct(r["capped_t"], fnum(r["v20_tflops"]))
        L += ["## v20 capped against v20 auto", "",
              "`STREAMK_POLICY=capped` against the default rule. Where capped picks data-parallel its time is the "
              "data-parallel column (the same kernel). Positive means capped is faster.", "",
              STATS_HEAD.replace("v20 faster", "capped faster")]
        L += [stats_line("all shapes", [r["gap_capped"] for r in rows])]
        groups = {}
        for r in rows:
            groups.setdefault(waves_bucket(r), []).append(r["gap_capped"])
        L += [stats_line(b, groups[b]) for b in
              ("under 1 wave", "1 to 2 waves", "2 to 4 waves", "4 to 8 waves", "8 waves or more") if b in groups]
        L += [""]
        groups = {}
        for r in rows:
            groups.setdefault((policy_label(r), r["v20_capped_label"].split(",")[0]), []).append(r["gap_capped"])
        L += ["### By auto's pick and capped's pick", "", STATS_HEAD.replace("v20 faster", "capped faster")]
        L += [stats_line(f"auto {a}, capped {c}", groups[(a, c)]) for a, c in sorted(groups)]
        L += [""]

    def table(title, rs):
        out = [f"### {title}", "",
               "| shape | out | tiles | S | k-steps | AITER pick | AITER | v9 | v20 | v20 policy | v20 dp | v20 vs AITER |",
               "|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|"]
        for r in rs:
            out.append(
                f"| {r['M']}x{r['N']}x{r['K']} | {r['out']} | {r['tiles']} | {r['S']} | {r['k_steps']} | "
                f"{r['aiter_pick']} | {fnum(r['aiter_pick_tflops']):.0f} | {fnum(r['v9_tflops']):.0f} | "
                f"{fnum(r['v20_tflops']):.0f} | {policy_label(r)} | "
                f"{'-' if r['v20_policy'] == 'dp' else format(fnum(r['v20_dp_tflops']), '.0f')} | {r['gap']:+.1f}% |")
        return out + [""]

    if aiter_run:
        ranked = sorted(timed, key=lambda r: r["gap"])
        L += ["## Largest gaps (TFLOPS)", ""]
        L += table("AITER furthest ahead", ranked[:20])
        L += table("v20 furthest ahead", ranked[::-1][:20])

        if args.baseline and os.path.abspath(args.baseline) != os.path.abspath(args.details):
            L += drift(rows, read_rows(args.baseline), f"`{os.path.relpath(args.baseline, HERE)}`", "v9_tflops")
        L += drift(rows, read_rows(SEP29_DETAILS), "the Sep 29 sweep (`hip_version_repro/details_rocm724.csv`)",
                   "gluon_tflops", changed_table=False)

    L += ["## Wrong outputs, failures and notes", ""]
    if not bad:
        L += ["None: every " + ("pick, v9 and " if aiter_run else "") + "v20 output matched the reference, v20 "
              "matched again after timing, and its flags and barrier counts were all back at 0.", ""]
    else:
        L += ["| shape | out | pick ok | v9 ok | v20 ok | v20 after timing | v20 flags left | note |",
              "|---|---|---|---|---|---|---|---|"]
        for r in bad:
            L.append(f"| {r['M']}x{r['N']}x{r['K']} | {r['out']} | {r['pick_ok']} ({r['pick_maxerr']}) | "
                     f"{r['v9_ok']} | {r['v20_ok']} ({r['v20_maxerr']}) | {r['v20_post_ok']} | "
                     f"{r['v20_flags_left']} | {r['note'][:200]} |")
        L += [""]

    with open(args.summary, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.summary}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--tag", default="", help="suffix for this run's files, snapshot and AITER JIT build")
    s.add_argument("--jit-tag", default=None,
                   help="reuse that tagged run's AITER JIT build (it must be the same AITER commit)")
    s.add_argument("--rule-policy", default=None,
                   help="also time STREAMK_POLICY=NAME as column v20_rule (the kernel needs NAME_policy)")
    s.add_argument("--v20-only", action="store_true",
                   help="time v20 only, without the AITER worker, on whatever stack runs this script "
                        "(no run_rocm724_v20.sh); the AITER and v9 columns stay empty")
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--only", default=None, help="comma-separated 0-based shape indices")
    s.add_argument("--out", default=None, help="details CSV (default results/details[_TAG].csv)")
    s.add_argument("--headline", choices=["auto", "capped"], default="auto",
                   help="capped: v20 is STREAMK_POLICY=capped on every shape (dp as a diagnostic where it "
                        "splits; no auto or two_tile), and hipBLASLt through torch is timed as column hbl")
    s.add_argument("--rotating-mb", type=int, default=0,
                   help="rotate every callable over max(2, ceil(N MB / set)) copies of x, w and y, so "
                        "consecutive launches never read the same inputs")
    s.add_argument("--snapshot-from", default=None,
                   help="take the v20 snapshot from that tagged run instead of the live kernel")
    s.add_argument("--pad512", action="store_true",
                   help="also time every kernel on a padded layout: each row stride (x, w, C) that is a "
                        f"multiple of 512 bytes gets {PAD_BYTES} more; columns *_pad")
    s.add_argument("--realloc", action="store_true",
                   help="fresh inputs (new buffers) for every round, so placement-dependent timings show "
                        "up as spread between rounds (*_spread)")
    s.add_argument("--realloc-workspace", action="store_true",
                   help="with --realloc, also a fresh stream-K workspace (P, locks, tile_sync) every round")
    s.add_argument("--v20-sched", choices=["tutorial", "aiter"], default="tutorial",
                   help="tutorial: the v20 worker uses the tutorial's llirSched plugin and amdgcnas hook; "
                        "aiter: AITER's opt-in llirSched copy (as AITER's own v9), no amdgcnas hook")
    s.add_argument("--p-pad", type=int, default=0,
                   help="STREAMK_P_PAD: bytes of padding after each program's partial slot in P")
    s.add_argument("--kernel", choices=sorted(KERNELS), default="v20",
                   help="the tutorial stream-K kernel timed as streamk_wip (columns v20_*)")
    w = sub.add_parser("worker")
    w.add_argument("role", choices=["aiter", "v20"])
    r = sub.add_parser("report")
    r.add_argument("--tag", default="")
    r.add_argument("--details", default=None, help="default results/details[_TAG].csv")
    r.add_argument("--summary", default=None, help="default results/summary[_TAG].md")
    r.add_argument("--baseline", default=os.path.join(RESULTS, "details.csv"),
                   help="an earlier details CSV to compare with (default the untagged run)")
    args = p.parse_args()
    if args.mode == "worker":
        worker_main(args.role)
    elif args.mode == "sweep":
        sweep(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
