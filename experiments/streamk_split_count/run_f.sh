#!/bin/bash
# Detached driver for sweep F (v24) on GPU 0. It keeps running after the terminal, SSH or the
# editor disconnects (setsid + nohup), allows one run at a time (flock) and resumes where it
# stopped (sweep.py skips the cases it already has).
#
#   run_f.sh start [stages]   wait until GPU 0 is idle, then each stage in turn (default: f):
#                             f       sweep F (sweep.py f), then sweep.py f_report
#                             noise   sweep F's second round (sweep.py f_noise)
#                             aiter   the held-out AITER test (../streamk_aiter_test/test.py run, then report)
#   run_f.sh status    stage, progress, the last log lines
#   run_f.sh stop      kill the driver and everything it started
#   run_f.sh log       follow results/run_f.log
#
# Timing needs the GPU to itself: before the sweep the driver waits until GPU 0 has shown under
# IDLE_USE % use and under IDLE_VRAM % memory for IDLE_SECONDS, and before every later chunk of CHUNK
# cases for CHUNK_IDLE_SECONDS (another job that arrives mid-sweep holds the next chunk).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RES=$HERE/results
mkdir -p "$RES"
LOG=$RES/run_f.log

export PYTHONPATH=$REPO/scripts/triton_deepbind_shim:/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python
export LLVM_PASS_PLUGIN_PATH=$REPO/plugins/llir_scheduler/libLlirSched.so
unset TRITON_AMDGCNAS_PLUGIN
export HIP_VISIBLE_DEVICES=0
PY=${PYTHON:-$(command -v python3)}
# HIP and rocm-smi number the GPUs differently in this container (HIP 0 is rocm-smi 3 here), so
# watch the rocm-smi index whose PCI bus is HIP device 0's.
smi_index() {
    local bus
    bus=$("$PY" -c "import torch; print(torch.cuda.get_device_properties(0).pci_bus_id)" 2>/dev/null | tail -1)
    rocm-smi --showbus 2>/dev/null | "$PY" -c "
import re, sys
for line in sys.stdin:
    m = re.search(r'GPU\[(\d+)\].*PCI Bus: [0-9a-fA-F]+:([0-9a-fA-F]+):', line)
    if m and int(m.group(2), 16) == int('$bus'):
        print(m.group(1))
        break"
}
GPU=$(smi_index)
IDLE_USE=${IDLE_USE:-10}
IDLE_VRAM=${IDLE_VRAM:-10}
IDLE_SECONDS=${IDLE_SECONDS:-600}
CHUNK_IDLE_SECONDS=${CHUNK_IDLE_SECONDS:-60}
CHUNK=${CHUNK:-100}

alive() {
    [ -f "$RES/run_f.pid" ] && kill -0 "$(cat "$RES/run_f.pid")" 2>/dev/null
}

descendants() {
    local c
    for c in $(ps -o pid= --ppid "$1"); do
        echo "$c"
        descendants "$c"
    done
}

gpu_idle() {
    local use vram
    use=$(rocm-smi -d "$GPU" --showuse 2>/dev/null | sed -n 's/.*GPU use (%): \([0-9]*\).*/\1/p' | head -1)
    vram=$(rocm-smi -d "$GPU" --showmemuse 2>/dev/null | sed -n 's/.*VRAM%): \([0-9]*\).*/\1/p' | head -1)
    [ -n "$use" ] && [ -n "$vram" ] && [ "$use" -lt "$IDLE_USE" ] && [ "$vram" -lt "$IDLE_VRAM" ]
}

wait_idle() {
    local need=$1 quiet=0 said=0
    while [ "$quiet" -lt "$need" ]; do
        if gpu_idle; then
            quiet=$((quiet + 10))
        else
            quiet=0
            if [ "$said" -eq 0 ]; then
                echo "=== $(date -u +%FT%TZ) GPU $GPU busy, waiting for ${need} s idle"
                said=1
            fi
        fi
        sleep 10
    done
    [ "$said" -eq 1 ] && echo "=== $(date -u +%FT%TZ) GPU $GPU idle"
    return 0
}

case "${1:-}" in
    start)
        if ! flock -n "$RES/run_f.lock" true; then
            echo "already running (pid $(cat "$RES/run_f.pid" 2>/dev/null)); see: $0 status"
            exit 1
        fi
        shift
        setsid nohup "$0" _run "${@:-f}" >> "$LOG" 2>&1 < /dev/null &
        sleep 1
        echo "started (pid $(cat "$RES/run_f.pid" 2>/dev/null)); log: $LOG"
        ;;
    _run)
        exec 9> "$RES/run_f.lock"
        flock -n 9 || { echo "another run holds $RES/run_f.lock"; exit 1; }
        echo $$ > "$RES/run_f.pid"
        [ -n "$GPU" ] || { echo "cannot find HIP device 0 in rocm-smi"; exit 1; }
        shift
        echo "=== $(date -u +%FT%TZ) start $* on $(hostname), python $PY, HIP device 0 = rocm-smi GPU $GPU"
        need=$IDLE_SECONDS
        for stage in "$@"; do
            case "$stage" in
                f) count="sweep.py todo f"; step="sweep.py f"; report="sweep.py f_report" ;;
                noise) count="sweep.py todo f_noise"; step="sweep.py f_noise"; report="" ;;
                aiter) count="../streamk_aiter_test/test.py todo"; step="../streamk_aiter_test/test.py run"
                       report="../streamk_aiter_test/test.py report" ;;
                *) echo "unknown stage $stage"; exit 1 ;;
            esac
            while :; do
                todo=$(cd "$HERE" && "$PY" -W ignore $count | tail -1)
                [ "$todo" = "0" ] && break
                echo "$stage" > "$RES/run_f.stage"
                wait_idle "$need"
                need=$CHUNK_IDLE_SECONDS
                echo "=== $(date -u +%FT%TZ) $stage: $todo cases to run; next $CHUNK"
                (cd "$HERE" && SWEEP_MAX_CASES=$CHUNK "$PY" -u -W ignore $step) || {
                    echo "$stage failed (exit $?)" > "$RES/run_f.stage"; exit 1; }
            done
            if [ -n "$report" ]; then
                echo "$stage report" > "$RES/run_f.stage"
                (cd "$HERE" && "$PY" -W ignore $report > /dev/null)
            fi
            echo "=== $(date -u +%FT%TZ) $stage done"
        done
        echo "done" > "$RES/run_f.stage"
        echo "=== $(date -u +%FT%TZ) done"
        ;;
    status)
        if alive; then echo "running (pid $(cat "$RES/run_f.pid"))"; else echo "not running"; fi
        echo "stage: $(cat "$RES/run_f.stage" 2>/dev/null)"
        echo "rows: sweep F $(cat "$RES/sweep_f.jsonl" 2>/dev/null | wc -l), round 2 $(cat "$RES/sweep_f_round2.jsonl" \
            2>/dev/null | wc -l), AITER test $(cat "$HERE/../streamk_aiter_test/results/test.jsonl" 2>/dev/null | wc -l)"
        tail -n 3 "$LOG" 2>/dev/null
        ;;
    stop)
        if ! alive; then echo "not running"; exit 0; fi
        pid=$(cat "$RES/run_f.pid")
        tree="$(descendants "$pid") $pid"
        kill $tree 2>/dev/null
        sleep 2
        kill -9 $tree 2>/dev/null
        echo "stopped $pid"
        ;;
    log)
        tail -f "$LOG"
        ;;
    *)
        sed -n '2,12p' "$0"
        exit 1
        ;;
esac
