# Hot-loop latency at a pinned shader clock (results/tcpL_report.md): 256x256 and 256x128 on 4096x4096x8192,
# cold inputs. Pin the clock from the host first (the container cannot write sysfs), e.g.
#     sudo rocm-smi -d 6 --setperfdeterminism 1400
# then, with LABEL naming the pin:
#     GPU=6 bash v9_tile/run_clock_pin.sh 1400
# ATT on launches 1 and 10, reduced by att_latency.py --loop and att_stalls.py; counter sets 0, 1, 5 and 7.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
unset TRITON_AMDGCNAS_PLUGIN
label=${1:?usage: run_clock_pin.sh LABEL}
POINTS=256x256:4096x4096x8192,256x128:4096x4096x8192
T=v9_tile/tmp/clockpin/$label
R=v9_tile/results/clockpin
mkdir -p "$T" "$R"
rocm-smi -d "$HIP_VISIBLE_DEVICES" --showclocks --showperflevel --showuniqueid > "$R/${label}_smi_before.txt" 2>&1
for launch in 1 10; do
    python v9_tile/sweep.py att --att-launch $launch --points $POINTS --rotating-mb 2048 --tmp "$T/l$launch" \
        --out "$R/${label}_l${launch}_att.csv"
    for tile in 256x256 256x128; do
        ui=$(ls -d "$T/l$launch/sweep_att/${tile}_4096x4096x8192"/ui_*)
        python ../../../../experiments/tcp_latency/att_latency.py "$ui" --name "${tile}_${label}_l${launch}" \
            --loop --plot --exact-stall 64 --out "$R"
    done
    python v9_tile/att_stalls.py "$T/l$launch/sweep_att/"*_4096x4096x8192 > "$R/${label}_l${launch}_stalls.md"
done
python v9_tile/sweep.py counters --sets 0,1,5,7 --points $POINTS --rotating-mb 2048 --tmp "$T/counters" \
    --out "$R/${label}_counters.csv"
rocm-smi -d "$HIP_VISIBLE_DEVICES" --showclocks --showperflevel > "$R/${label}_smi_after.txt" 2>&1
