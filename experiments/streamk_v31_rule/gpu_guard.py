"""Run a command on one GPU and stop it as soon as anyone else uses that GPU.

    python experiments/streamk_v31_rule/gpu_guard.py --smi-index 1 --log LOG -- CMD ...

rocm-smi lists the processes on a GPU by host PID, which this container cannot map
to its own, so the guard compares counts: a process on the GPU (rocm-smi
--showpidgpus) is foreign when there are more of them than processes of ours, the
command's process tree members with /dev/kfd open. The command does not start while
a foreign process is there; once it runs, a foreign process seen in `settle`
consecutive looks kills the command's process group and the guard exits with
status 4. Otherwise it exits with the command's status.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time


def log(path, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} guard: {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def gpu_pids(smi_index):
    """Host PIDs that rocm-smi --showpidgpus lists on GPU smi_index."""
    out = subprocess.run(
        ["rocm-smi", "--showpidgpus"], capture_output=True, text=True, timeout=120
    ).stdout
    pids, cur = [], None
    for line in out.splitlines():
        m = re.match(r"PID (\d+) is using", line.strip())
        if m:
            cur = m.group(1)
        elif cur and re.fullmatch(r"[\d\s]+", line.strip() or "x"):
            if str(smi_index) in line.split():
                pids.append(cur)
            cur = None
    return pids


def descendants(root):
    children = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(name))
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(children.get(pid, []))
    return out


def uses_kfd(pid):
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                if os.readlink(f"/proc/{pid}/fd/{fd}") == "/dev/kfd":
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def foreign_count(smi_index, root):
    ours = 0 if root is None else sum(uses_kfd(p) for p in descendants(root))
    return len(gpu_pids(smi_index)) - ours


def foreign_seen(smi_index, root, settle, wait):
    """True when every one of `settle` looks, `wait` s apart, sees a foreign process."""
    for i in range(settle):
        if foreign_count(smi_index, root) <= 0:
            return False
        if i + 1 < settle:
            time.sleep(wait)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smi-index", type=int, required=True, help="rocm-smi GPU index")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--settle", type=int, default=3)
    ap.add_argument("--log")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command")
    if foreign_seen(args.smi_index, None, args.settle, 2.0):
        log(args.log, f"GPU {args.smi_index} is in use by someone else; not starting {cmd}")
        sys.exit(4)
    log(args.log, f"start on rocm-smi GPU {args.smi_index}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, start_new_session=True)
    while True:
        try:
            status = proc.wait(timeout=args.interval)
            log(args.log, f"exit {status}: {' '.join(cmd)}")
            sys.exit(status)
        except subprocess.TimeoutExpired:
            pass
        if foreign_seen(args.smi_index, proc.pid, args.settle, 2.0):
            log(args.log, f"FOREIGN process on GPU {args.smi_index}; stopping {' '.join(cmd)}")
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            sys.exit(4)


if __name__ == "__main__":
    main()
