##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################
"""Litmus tests of the gfx950 memory model: what each cache bit combination and fence delivers.

litmus_kernel. 256 workgroups of 2 waves, one workgroup per CU (a 128 KiB LDS allocation keeps a
second one off). Each writer/reader pair owns LINES 128-byte lines; lane l of the acting wave
touches dword 0 of line l. Roles, by PLACE:
- 0, same workgroup: wave 0 of every program writes, wave 1 of the same program reads;
- 1, same XCD: local slot pid // 8 even writes, pid + 8 (same XCD, another CU) reads;
- 2, other XCD: even pids write, pid + 1 (the next XCD) reads.
Waves without a role run the same instructions on a private dummy region, so no wave ever
branches on its role and the code is identical for every placement.

Timeline, after a relaxed atomic grid barrier at ts (each wave's own s_memrealtime):
1. prime (reader): read every line with PRIME bits (X0), drain; this leaves the old line in the
   reader's L1 / L2 where those bits allocate.
2. at ts + t_write (+ a per-pair jitter), writer programs only: the writer stores the launch's epoch with STORE bits,
   then WFENCE, then optional pollution stores (pol_iters x 64 lines, default bits, writer
   programs only), then a drain.
3. SYNC 0: at ts + t_read the reader applies RFENCE and reads. Between the write and the read
   no memory operation runs, so staleness can only come from the caches.
   SYNC 1: the writer sets flag[pair] = epoch with an atomic (FSEM_W, FSCOPE) after WFENCE; the
   reader spins on it (relaxed, FSCOPE), optionally acquires once (FSEM_R), applies RFENCE and
   reads at once. A bounded spin: a timeout is recorded, never a hang.
   SYNC 2: a workgroup barrier (s_barrier) after WFENCE; PLACE 0 only.
4. read (reader): an empty timer bracket, the timed read of line 0 with LOAD bits, two more empty
   brackets, then every line with LOAD bits (X1).
All data accesses are inline assembly with explicit cache bits, so each cell's encoding is
exactly the one named.

PHASES (runtime bits): 1 prime, 2 write, 4 read, 8 writer-side read of its lines at prime time
(used for the reference latencies); the two-kernel control runs prime | write in one launch and
read in the next.
"""

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

NUM_PROGRAMS = 256
NUM_XCDS = 8
NW = 2
LINES = 64
LINE = 128
NPAIRS = 256
REC = 32
LDS_BYTES = 128 * 1024  # of 160 KiB per CU: a second workgroup cannot fit

_NW = gl.constexpr(NW)
_LINES = gl.constexpr(LINES)
_REC = gl.constexpr(REC)
_LDS = gl.constexpr(LDS_BYTES)

# record fields (int64 per pair)
FIELDS = ["w_xcc", "w_hw", "r_xcc", "r_hw", "w_ts", "r_ts", "r_primed", "w_start", "w_done", "r_start", "r_done",
          "timed_val", "mt", "rt", "e1", "e2", "e3", "timeout", "mt0", "rt0", "mt1", "rt1", "w_pid", "r_pid", "spins"]
FIELD = {n: i for i, n in enumerate(FIELDS)}
(F_W_XCC, F_W_HW, F_R_XCC, F_R_HW, F_W_TS, F_R_TS, F_R_PRIMED, F_W_START, F_W_DONE, F_R_START, F_R_DONE, F_TIMED_VAL,
 F_MT, F_RT, F_E1, F_E2, F_E3, F_TIMEOUT, F_MT0, F_RT0, F_MT1, F_RT1, F_W_PID, F_R_PID,
 F_SPINS) = [gl.constexpr(i) for i in range(len(FIELDS))]

BITS = {"none": "", "sc0nt": "sc0 nt", "sc1": "sc1", "sc0sc1": "sc0 sc1", "sc0": "sc0", "nt": "nt"}
WFENCES = {"none": 0, "wg": 1, "agent": 2, "agent_cc": 3}
RFENCES = {"none": 0, "agent": 1, "inv_sc0": 2, "agent_cc": 3, "agent_wait": 4, "inv_sc0sc1": 5}


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [_NW], [0]))


@gluon.jit
def _asm(code: gl.constexpr):
    gl.inline_asm_elementwise(code + "\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def _now():
    """s_memrealtime as a wave-uniform scalar (no memory wait, no barrier)."""
    return gl.max(gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one()],
                                            dtype=gl.int64, is_pure=False, pack=1), axis=0)


@gluon.jit
def _clocks():
    """(s_memrealtime, s_memtime) sampled together."""
    rt, mt = gl.inline_asm_elementwise("s_memrealtime $0\ns_memtime $1\ns_waitcnt lgkmcnt(0)", "=s,=s,v,~{memory}",
                                       [_one()], dtype=(gl.int64, gl.int64), is_pure=False, pack=1)
    return gl.max(rt, axis=0), gl.max(mt, axis=0)


@gluon.jit
def _getreg(REG: gl.constexpr):
    return gl.max(gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(" + REG + ")", "=s,v", [_one()],
                                            dtype=gl.int32, is_pure=True, pack=1), axis=0)


@gluon.jit
def _wait_until(deadline):
    while _now() < deadline:
        pass


@gluon.jit
def _ld(addr, BITS: gl.constexpr):
    """One dword per lane, explicit cache bits, completed before the value is returned."""
    return gl.inline_asm_elementwise("global_load_dword $0, $1, off " + BITS + "\ns_waitcnt vmcnt(0)",
                                     "=&v,v,~{memory}", [addr], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _st(addr, val, BITS: gl.constexpr):
    """One dword per lane, explicit cache bits, not waited for."""
    gl.inline_asm_elementwise("global_store_dword $1, $2, off " + BITS + "\nv_mov_b32 $0, 0", "=v,v,v,~{memory}",
                              [addr, val], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _timed_ld(addr, BITS: gl.constexpr, LOAD: gl.constexpr):
    """The timer bracket: drain, s_memrealtime + s_memtime, the load (LOAD) or nothing, wait for it,
    s_memtime + s_memrealtime. Returns (value, s_memtime cycles, s_memrealtime ticks)."""
    if LOAD:
        body: gl.constexpr = "global_load_dword $0, $5, off " + BITS + "\ns_waitcnt vmcnt(0)\n"
    else:
        body: gl.constexpr = "v_mov_b32 $0, 0\n"
    v, mt0, mt1, rt0, rt1 = gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n"
        "s_memrealtime $3\n"
        "s_memtime $1\n"
        "s_waitcnt lgkmcnt(0)\n" + body + "s_memtime $2\n"
        "s_memrealtime $4\n"
        "s_waitcnt lgkmcnt(0)", "=&v,=&s,=&s,=&s,=&s,v,~{memory}", [addr],
        dtype=(gl.int32, gl.int64, gl.int64, gl.int64, gl.int64), is_pure=False, pack=1)
    return v, mt1 - mt0, rt1 - rt0


@gluon.jit
def _clocked(code: gl.constexpr):
    """s_memtime cycles for `code` followed by s_waitcnt vmcnt(0), after a drain."""
    t0, t1 = gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0) lgkmcnt(0)\ns_memtime $0\ns_waitcnt lgkmcnt(0)\n" + code +
        "\ns_waitcnt vmcnt(0)\ns_memtime $1\ns_waitcnt lgkmcnt(0)", "=&s,=&s,v,~{memory}", [_one()],
        dtype=(gl.int64, gl.int64), is_pure=False, pack=1)
    return gl.max(t1 - t0, axis=0)


@gluon.jit
def _put(rec, field: gl.constexpr, val, mask):
    """Store a per-wave value into a record field from lane 0 of the waves selected by mask."""
    gl.store(rec + field, val.to(gl.int64), mask=mask)


@gluon.jit
def _wfence(WFENCE: gl.constexpr, scratch, m):
    if WFENCE == 1:
        _asm("s_waitcnt vmcnt(0)")  # workgroup release (non-TG-split): wait for my own stores
    elif WFENCE == 2:
        _asm("buffer_wbl2 sc1\ns_waitcnt vmcnt(0)")  # agent release, as LLVM emits it for gfx942/950
    elif WFENCE == 3:
        gl.atomic_xchg(scratch, 0, mask=m, sem="release", scope="gpu")  # the compiler's own agent release


@gluon.jit
def _rfence(RFENCE: gl.constexpr, scratch, m):
    if RFENCE == 1:
        _asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1")  # agent acquire, as LLVM emits it
    elif RFENCE == 2:
        _asm("s_waitcnt vmcnt(0)\nbuffer_inv sc0")  # a NOP outside TG-split mode (ISA Table 52): control
    elif RFENCE == 3:
        gl.atomic_add(scratch, 0, mask=m, sem="acquire", scope="gpu")  # the compiler's own agent acquire
    elif RFENCE == 4:
        _asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1\ns_waitcnt vmcnt(0)")  # agent acquire, completed before the read
    elif RFENCE == 5:
        _asm("s_waitcnt vmcnt(0)\nbuffer_inv sc0 sc1")  # the documented cross-agent read fence


@gluon.jit
def litmus_kernel(
    data_addr,
    dummy_addr,
    pol_addr,
    flag_ptr,
    scratch_ptr,
    cnt_ptr,
    rec_ptr,
    x0_ptr,
    x1_ptr,
    epoch,
    place,
    phases,
    t_write,
    t_read,
    t_timeout,
    jitter,
    pol_iters,
    pol_first,
    rgap,
    third_xcc,
    t_third,
    prog_rec_ptr,
    STORE: gl.constexpr,
    LOAD: gl.constexpr,
    PRIME: gl.constexpr,
    WFENCE: gl.constexpr,
    RFENCE: gl.constexpr,
    SYNC: gl.constexpr,
    FSCOPE: gl.constexpr,
    FSEM_W: gl.constexpr,
    FSEM_R: gl.constexpr,
    THIRD: gl.constexpr,
):
    L: gl.constexpr = gl.BlockedLayout([1], [64], [_NW], [0])
    lane = gl.arange(0, 64 * _NW, L)
    wave = lane // 64
    ln = lane % 64
    pid = gl.program_id(0)

    lds = gl.allocate_shared_memory(gl.int8, [_LDS], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    lds.slice(0, 64 * _NW).store(gl.zeros([64 * _NW], gl.int8, L))

    if place == 0:
        pair = pid
        prog_w = pid >= 0
        prog_r = pid >= 0
        wave_r = 1
    elif place == 1:
        slot = pid // 8
        pair = (slot // 2) * 8 + pid % 8
        prog_w = slot % 2 == 0
        prog_r = slot % 2 == 1
        wave_r = 0
    else:
        pair = pid // 2
        prog_w = pid % 2 == 0
        prog_r = pid % 2 == 1
        wave_r = 0
    w_lane = prog_w & (wave == 0)
    r_lane = prog_r & (wave == wave_r)
    w0 = w_lane & (ln == 0)
    r0 = r_lane & (ln == 0)

    tgt = data_addr + ((pair * _LINES + ln) * 128).to(gl.int64)
    dmy = dummy_addr + (((pid * _NW + wave) * _LINES + ln) * 128).to(gl.int64)
    w_addr = gl.where(w_lane, tgt, dmy)
    r_addr = gl.where(r_lane, tgt, dmy)
    tgt0 = data_addr + (pair * _LINES * 128).to(gl.int64) + lane.to(gl.int64) * 0
    dmy0 = dummy_addr + ((pid * _NW + wave) * _LINES * 128).to(gl.int64)
    t_addr = gl.where(r_lane, tgt0, dmy0)
    scratch = scratch_ptr + pid * _NW + wave

    xcc = _getreg("HW_REG_XCC_ID") & 15
    hw = _getreg("HW_REG_HW_ID")

    # Grid barrier on its own line: relaxed device-scope atomics, no cache maintenance.
    _asm("s_waitcnt vmcnt(0)\ns_barrier")
    gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    while gl.atomic_add(cnt_ptr, 0, sem="relaxed", scope="gpu") < 256:
        pass
    _asm("s_nop 0")
    ts, mts = _clocks()

    # Every stamp stays in registers until the end, so nothing but the experiment's own accesses
    # touches memory between the grid barrier and the final read.
    zero = lane * 0
    x0 = zero
    primed = ts
    w_start = ts
    w_done = ts
    timeout = 0
    spins = 0
    r_start = ts
    rt1 = ts
    mt1 = mts
    tv = zero
    mt = zero.to(gl.int64)
    rt = zero.to(gl.int64)
    e1 = zero.to(gl.int64)
    e2 = zero.to(gl.int64)
    e3 = zero.to(gl.int64)
    x1 = zero

    if phases & 8:
        _ld(w_addr, PRIME)  # the writer reads its lines first (references: puts them in the LLC / its L2)
    if phases & 1:
        x0 = _ld(r_addr, PRIME)
        primed = _now()

    # Only writer programs run the write phase (in PLACE 0 that is every program, and wave 1 stores to
    # its dummy region), so a flag reader is already spinning when the flag is set.
    if (phases & 2 != 0) & prog_w:
        jit = (epoch * 7919 + pair * 104729) % jitter
        _wait_until(ts + t_write + jit)
        w_start = _now()
        if pol_first:
            # Stress for the ordering test: queue the pollution stores ahead of the data store.
            for i in range(pol_iters):
                pa = pol_addr + (((pid * pol_iters + i) * 64 + ln) * 128).to(gl.int64)
                _st(pa, epoch + zero, "")
        _st(w_addr, epoch + zero, STORE)
        _wfence(WFENCE, scratch, ln == 0)
        if SYNC == 1:
            gl.atomic_xchg(flag_ptr + pair, epoch, sem=FSEM_W, scope=FSCOPE)
        if SYNC == 2:
            _asm("s_barrier")
        if pol_first == 0:
            # Pollution on the writer's XCD only: default stores that push dirty lines out of its L2.
            for i in range(pol_iters):
                pa = pol_addr + (((pid * pol_iters + i) * 64 + ln) * 128).to(gl.int64)
                _st(pa, epoch + zero, "")
        _asm("s_waitcnt vmcnt(0)")
        w_done = _now()

    third_t = ts * 0
    if THIRD != 0:
        # A third party: every program on XCD third_xcc issues one cache operation at ts + t_third,
        # whatever its role, and times it (s_memtime cycles, op plus its s_waitcnt vmcnt(0)).
        if xcc == third_xcc:
            _wait_until(ts + t_third)
            if THIRD == 1:
                third_t = _clocked("buffer_inv sc1")
            elif THIRD == 2:
                third_t = _clocked("buffer_wbl2 sc1")
            else:
                third_t = _clocked("buffer_wbl2 sc1\ns_waitcnt vmcnt(0)\nbuffer_inv sc1")

    if phases & 4:
        if SYNC == 0:
            if rgap > 0:
                # The reader's fence rgap ticks before the read instead of just before it.
                _wait_until(ts + t_read - rgap)
                if prog_r:
                    _rfence(RFENCE, scratch, ln == 0)
            _wait_until(ts + t_read)
        elif SYNC == 1:
            if prog_r:
                deadline = ts + t_timeout
                f = gl.atomic_add(flag_ptr + pair, 0, sem="relaxed", scope=FSCOPE)
                while (f != epoch) & (_now() < deadline):
                    f = gl.atomic_add(flag_ptr + pair, 0, sem="relaxed", scope=FSCOPE)
                    spins += 1
                if FSEM_R == "acquire":
                    gl.atomic_add(flag_ptr + pair, 0, sem="acquire", scope=FSCOPE)
                timeout = (f != epoch).to(gl.int32)
        _asm("s_nop 0")
        if prog_r & (rgap == 0):
            _rfence(RFENCE, scratch, ln == 0)
        r_start = _now()
        _, e1, _ = _timed_ld(t_addr, LOAD, False)
        tv, mt, rt = _timed_ld(t_addr, LOAD, True)
        _, e2, _ = _timed_ld(t_addr, LOAD, False)
        _, e3, _ = _timed_ld(t_addr, LOAD, False)
        x1 = _ld(r_addr, LOAD)
        rt1, mt1 = _clocks()

    gl.store(prog_rec_ptr + pid * 4 + zero, third_t + zero, mask=lane == 0)
    gl.store(prog_rec_ptr + pid * 4 + 1 + zero, xcc + zero, mask=lane == 0)
    rec = rec_ptr + pair * _REC + zero
    _put(rec, F_W_XCC, xcc + zero, w0)
    _put(rec, F_W_HW, hw + zero, w0)
    _put(rec, F_R_XCC, xcc + zero, r0)
    _put(rec, F_R_HW, hw + zero, r0)
    _put(rec, F_W_PID, pid + zero, w0)
    _put(rec, F_R_PID, pid + zero, r0)
    _put(rec, F_W_TS, ts + zero, w0)
    _put(rec, F_R_TS, ts + zero, r0)
    _put(rec, F_RT0, ts + zero, r0)
    _put(rec, F_MT0, mts + zero, r0)
    if phases & 1:
        gl.store(x0_ptr + pair * _LINES + ln, x0, mask=r_lane)
        _put(rec, F_R_PRIMED, primed + zero, r0)
    if phases & 2:
        _put(rec, F_W_START, w_start + zero, w0)
        _put(rec, F_W_DONE, w_done + zero, w0)
    if phases & 4:
        gl.store(x1_ptr + pair * _LINES + ln, x1, mask=r_lane)
        _put(rec, F_TIMEOUT, timeout + zero, r0)
        _put(rec, F_SPINS, spins + zero, r0)
        _put(rec, F_R_START, r_start + zero, r0)
        _put(rec, F_R_DONE, rt1 + zero, r0)
        _put(rec, F_RT1, rt1 + zero, r0)
        _put(rec, F_MT1, mt1 + zero, r0)
        _put(rec, F_TIMED_VAL, tv, r0)
        _put(rec, F_MT, mt, r0)
        _put(rec, F_RT, rt, r0)
        _put(rec, F_E1, e1, r0)
        _put(rec, F_E2, e2, r0)
        _put(rec, F_E3, e3, r0)

    keep = gl.sum(lds.slice(0, 64 * _NW).load(L).to(gl.int32), axis=0)
    gl.store(scratch_ptr + 1024 + pid + lane * 0, keep + lane * 0, mask=lane == 0)


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------

MIB = 2**20


class Workspace:

    def __init__(self, device="cuda", pol_mib=160):
        i32 = torch.int32
        self.data = torch.zeros(NPAIRS * LINES * LINE // 4, device=device, dtype=i32)
        self.dummy = torch.zeros(NUM_PROGRAMS * NW * LINES * LINE // 4, device=device, dtype=i32)
        self.pol = torch.zeros(pol_mib * MIB // 4, device=device, dtype=i32)
        self.flags = torch.zeros(NPAIRS * 32, device=device, dtype=i32)  # one line per flag
        self.scratch = torch.zeros(4096, device=device, dtype=i32)
        self.cnt = torch.zeros(32, device=device, dtype=i32)
        self.rec = torch.zeros(NPAIRS * REC, device=device, dtype=torch.int64)
        self.x0 = torch.zeros(NPAIRS * LINES, device=device, dtype=i32)
        self.x1 = torch.zeros(NPAIRS * LINES, device=device, dtype=i32)
        self.prog_rec = torch.zeros(NUM_PROGRAMS * 4, device=device, dtype=torch.int64)
        self.flush_buf = torch.ones(1024 * MIB // 4, device=device, dtype=torch.float32)
        self.epoch = 1000

    def flush(self):
        """Read 1 GiB: evicts L2 and the 256 MB LLC, leaves both clean."""
        self.flush_buf.sum()


def npairs(place):
    return NPAIRS if place == 0 else NPAIRS // 2


def launch(ws, store="none", load="none", prime=None, wfence="none", rfence="none", sync=0, place=2, phases=7,
           t_write_us=20.0, t_read_us=70.0, t_timeout_us=2000.0, jitter_us=0.0, pol_iters=0, pol_first=False,
           rgap_us=0.0, fscope="gpu",
           fsem_w="relaxed", fsem_r="relaxed", third=0, third_xcc=0, t_third_us=40.0, new_epoch=True):
    """One launch. Bits are BITS keys; prime defaults to the load bits. third: 0 off, 1 buffer_inv sc1,
    2 buffer_wbl2 sc1, 3 both, issued by every program on XCD third_xcc at ts + t_third_us."""
    if new_epoch:
        ws.epoch += 1
    ws.cnt.zero_()
    assert pol_iters * NUM_PROGRAMS * 64 * LINE <= ws.pol.numel() * 4
    return litmus_kernel[(NUM_PROGRAMS, )](
        ws.data.data_ptr(), ws.dummy.data_ptr(), ws.pol.data_ptr(), ws.flags, ws.scratch, ws.cnt, ws.rec, ws.x0,
        ws.x1, ws.epoch, place, phases, int(t_write_us * 100), int(t_read_us * 100), int(t_timeout_us * 100),
        max(1, int(jitter_us * 100)), pol_iters, int(pol_first), int(rgap_us * 100), third_xcc, int(t_third_us * 100), ws.prog_rec, STORE=BITS[store],
        LOAD=BITS[load], PRIME=BITS[prime if prime is not None else load], WFENCE=WFENCES[wfence],
        RFENCE=RFENCES[rfence], SYNC=sync, FSCOPE=fscope, FSEM_W=fsem_w, FSEM_R=fsem_r, THIRD=third, num_warps=NW)


def results(ws, place):
    """(rec, x0, x1) of the active pairs as CPU tensors: (P, REC) int64, (P, LINES) int32 x2."""
    P = npairs(place)
    return (ws.rec.view(NPAIRS, REC)[:P].cpu(), ws.x0.view(NPAIRS, LINES)[:P].cpu(),
            ws.x1.view(NPAIRS, LINES)[:P].cpu())


def cu_key(xcc, hw):
    """XCC, SE, SH and CU from HW_REG_XCC_ID / HW_REG_HW_ID."""
    return xcc * 256 + ((hw >> 13) & 7) * 32 + ((hw >> 12) & 1) * 16 + ((hw >> 8) & 15)
