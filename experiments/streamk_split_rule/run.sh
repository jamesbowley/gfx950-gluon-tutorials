#!/bin/bash
# Detached driver for the split-rule sweep. It keeps running after the terminal, SSH or the
# editor disconnects (setsid + nohup), allows one run at a time (flock) and resumes where it
# stopped, because every stage skips the cases it already has.
#
#   run.sh start [main|retime|retime2|validate]   (retime2: second pass, refit, install, re-validate)
#                                  main: main, repeat, maps, fit; retime: the rows hit by the
#                                  slice change, then repeat, maps, fit; validate: off-grid, real-model
#   run.sh status                  stage, progress, ETA, the last log lines
#   run.sh stop                    kill the driver and everything it started
#   run.sh log                     follow results/run.log
#
# A container or host restart still kills it: run "run.sh start" again to resume.
# SPLIT_RULE_RESULTS (another results directory) and SPLIT_RULE_SHAPES (a comma-separated subset
# of shapes) are for smoke tests.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RES=${SPLIT_RULE_RESULTS:-$HERE/results}
export SPLIT_RULE_RESULTS=$RES
mkdir -p "$RES"

export PYTHONPATH=$REPO/scripts/triton_deepbind_shim:/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python
export LLVM_PASS_PLUGIN_PATH=$REPO/plugins/llir_scheduler/libLlirSched.so
export TRITON_AMDGCNAS_PLUGIN=1
export HIP_VISIBLE_DEVICES=0
PY=${PYTHON:-$(command -v python3)}

stages() {
    case "$1" in
        main) echo "main repeat maps fit" ;;
        retime) echo "retime repeat maps fit" ;;
        retime2) echo "retime2 repeat maps fit install validate realmodel" ;;
        validate) echo "validate realmodel" ;;
        *) echo "unknown pipeline $1" >&2; return 1 ;;
    esac
}

alive() {
    [ -f "$RES/run.pid" ] && kill -0 "$(cat "$RES/run.pid")" 2>/dev/null
}

descendants() {
    local c
    for c in $(ps -o pid= --ppid "$1"); do
        echo "$c"
        descendants "$c"
    done
}

case "${1:-}" in
    start)
        pipeline=${2:-main}
        stages "$pipeline" > /dev/null || exit 1
        if ! flock -n "$RES/run.lock" true; then
            echo "already running (pid $(cat "$RES/run.pid" 2>/dev/null)); see: $0 status"
            exit 1
        fi
        setsid nohup "$0" _run "$pipeline" >> "$RES/run.log" 2>&1 < /dev/null &
        sleep 1
        echo "started $pipeline (pid $(cat "$RES/run.pid" 2>/dev/null)); log: $RES/run.log"
        ;;
    _run)
        exec 9> "$RES/run.lock"
        flock -n 9 || { echo "another run holds $RES/run.lock"; exit 1; }
        echo $$ > "$RES/run.pid"
        echo "=== $(date -u +%FT%TZ) start $2 on $(hostname), python $PY"
        for s in $(stages "$2"); do
            echo "$s" > "$RES/stage"
            echo "=== $(date -u +%FT%TZ) stage $s"
            (cd "$HERE" && "$PY" -u sweep.py "$s")
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "$s failed (exit $rc)" > "$RES/stage"
                echo "=== $(date -u +%FT%TZ) stage $s failed (exit $rc)"
                exit $rc
            fi
        done
        echo "$2 done" > "$RES/stage"
        echo "=== $(date -u +%FT%TZ) $2 done"
        ;;
    status)
        if alive; then echo "running (pid $(cat "$RES/run.pid"))"; else echo "not running"; fi
        (cd "$HERE" && "$PY" sweep.py status)
        echo "--- last log lines"
        tail -n 4 "$RES/run.log" 2>/dev/null
        ;;
    stop)
        if ! alive; then echo "not running"; exit 0; fi
        pid=$(cat "$RES/run.pid")
        # The rocprof cases run in their own sessions, so kill the whole tree, not the group.
        tree="$(descendants "$pid") $pid"
        kill $tree 2>/dev/null
        sleep 2
        kill -9 $tree 2>/dev/null
        echo "stopped $pid"
        ;;
    log)
        tail -f "$RES/run.log"
        ;;
    *)
        sed -n '2,15p' "$0"
        exit 1
        ;;
esac
