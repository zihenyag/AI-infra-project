"""将 S1/S2 分配到独立 V100；保持既有 S0 和三份测量源码不变。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import run_v100_experiment as runner


ASSIGNMENTS = {
    "S1": ("1", "GPU-e0793c56-a140-f368-abb5-9d265981da85", 30172),
    "S2": ("2", "GPU-bce0b2ef-65eb-c23e-491f-89e85889c17c", 30173),
}


def gpu_inventory():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    return [dict(zip(("index", "uuid", "name", "memory_total_mib", "memory_used_mib", "utilization_pct"),
                     (v.strip() for v in row))) for row in csv.reader(io.StringIO(result))]


def validate_gpu(inventory, busy, index, uuid):
    row = next((r for r in inventory if r["index"] == index and r["uuid"] == uuid), None)
    if row is None or row["name"] != "Tesla V100-SXM2-32GB":
        raise RuntimeError(f"指定显卡身份/型号不匹配：{index}/{uuid}")
    if uuid in busy or float(row["memory_used_mib"]) > 128 or float(row["utilization_pct"]) != 0:
        raise RuntimeError(f"指定显卡已占用，不启动重复任务：{row}")
    return index, uuid


def select_gpu(baseline):
    index, uuid, _ = ASSIGNMENTS[baseline]
    busy = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True, timeout=15)
    return validate_gpu(gpu_inventory(), busy, index, uuid)


def validate_port(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def start(existing_s0):
    formal = runner.FORMAL
    existing_s0 = existing_s0.resolve()
    if not existing_s0.is_relative_to(formal / "runs/submission/S0"):
        raise ValueError("S0目录必须属于本实验 formal/runs/submission/S0")
    s0 = json.loads((existing_s0 / "environment.json").read_text())
    # 派发期间持锁；运行期间另由每组worker锁防止双启动。
    with (formal / "parallel.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule_path = formal / "execution_schedule.json"
        if schedule_path.exists():
            raise RuntimeError("已存在并行派发记录；先检查状态，不自动重复启动")
        for baseline, (_, _, port) in ASSIGNMENTS.items():
            if list((formal / "runs/submission" / baseline).glob("*/environment.json")):
                raise RuntimeError(f"{baseline}已有正式运行记录；不自动重复启动")
            select_gpu(baseline)
            validate_port(port)
        inventory = gpu_inventory()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        job = formal / "jobs" / f"parallel-{stamp}"
        job.mkdir(parents=True)
        script = Path(__file__).resolve()
        schedule = {
            "mode": "same_model_different_gpus_parallel",
            "started_ns": time.time_ns(), "started_at_utc": stamp,
            "gpu_inventory": inventory, "job_dir": str(job),
            "wrapper_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "existing_s0_run": str(existing_s0), "existing_s0_gpu_uuid": s0["gpu_uuid"],
            "limitations": ["S0先独占后与S1/S2并行", "共享CPU与主机内存，非严格资源隔离对照"],
            "assignments": {b: {"gpu_index": i, "gpu_uuid": u, "port": p}
                            for b, (i, u, p) in ASSIGNMENTS.items()},
            "workers": {}, "status": "dispatching",
        }
        runner.write_json(schedule_path, schedule)
        try:
            for baseline in ASSIGNMENTS:
                command = [sys.executable, "-B", str(script), "worker", "--baseline", baseline,
                           "--job-dir", str(job)]
                with (job / f"{baseline}.log").open("x") as log:
                    process = subprocess.Popen(command, cwd=runner.ROOT, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=log, start_new_session=True)
                schedule["workers"][baseline] = {"pid": process.pid, "command": command}
                runner.write_json(schedule_path, schedule)
            schedule["status"] = "dispatched"
        except BaseException as exc:
            schedule.update(status="dispatch_failed", error=repr(exc))
            raise
        finally:
            runner.write_json(schedule_path, schedule)
        print(json.dumps(schedule, ensure_ascii=False), flush=True)


def worker(baseline, job):
    job = job.resolve()
    if not job.is_relative_to(runner.FORMAL / "jobs"):
        raise ValueError("worker目录必须属于 formal/jobs")
    with (runner.FORMAL / f"parallel-{baseline}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"baseline": baseline, "pid": os.getpid(), "started_ns": time.time_ns(),
                 "status": "running", "gpu_index": ASSIGNMENTS[baseline][0],
                 "gpu_uuid": ASSIGNMENTS[baseline][1], "port": ASSIGNMENTS[baseline][2]}
        runner.write_json(job / f"{baseline}.status.json", state)
        runner.PORT = ASSIGNMENTS[baseline][2]
        runner.idle_gpu = lambda: select_gpu(baseline)
        try:
            runner.run(baseline, False, None)
            state["status"] = "completed"
        except BaseException as exc:
            state.update(status="failed", error=repr(exc))
            raise
        finally:
            state["ended_ns"] = time.time_ns()
            runner.write_json(job / f"{baseline}.status.json", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "worker"])
    parser.add_argument("--existing-s0-run", type=Path)
    parser.add_argument("--baseline", choices=list(ASSIGNMENTS))
    parser.add_argument("--job-dir", type=Path)
    args = parser.parse_args()
    if args.action == "start":
        if not args.existing_s0_run:
            parser.error("start requires --existing-s0-run")
        start(args.existing_s0_run)
    else:
        if not args.baseline or not args.job_dir:
            parser.error("worker requires --baseline and --job-dir")
        worker(args.baseline, args.job_dir)


if __name__ == "__main__":
    main()
