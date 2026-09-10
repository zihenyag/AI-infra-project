"""V100 兼容复现入口。prepare 为 CPU 写入；run 会占用 GPU，均须单独批准。"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from smoke_v100 import LAUNCHER, ROOT, MODEL_ID, idle_gpu

PORT = 30171
MODEL = ROOT / "models/Qwen2.5-3B-Instruct"
FORMAL = ROOT / "formal"
SEED = 42


@dataclass(frozen=True)
class Point:
    name: str
    mode: str
    isl: int
    n: int
    groups: int
    qps: float
    osl: int = 128
    concurrency: int = 4


POINTS = [
    Point("random_2k_q1", "random", 2048, 100, 100, 1),
    Point("prefix_2k_q1", "prefix_repetition", 2048, 100, 10, 1),
    Point("pressure_4k_q1", "prefix_repetition", 4096, 128, 32, 1),
    Point("prefix_2k_q025", "prefix_repetition", 2048, 100, 10, .25),
    Point("prefix_2k_q05", "prefix_repetition", 2048, 100, 10, .5),
    Point("prefix_1k_q1", "prefix_repetition", 1024, 100, 10, 1),
    Point("prefix_4k_q1", "prefix_repetition", 4096, 100, 10, 1),
]
PILOT = Point("reload_gate", "controlled_prefix_repetition", 4096, 18, 11, 0,
              osl=16, concurrency=1)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def setup_env():
    env = os.environ.copy()
    for key, value in {
        "TMPDIR": ROOT / "tmp", "XDG_CACHE_HOME": ROOT / ".cache",
        "TRITON_CACHE_DIR": ROOT / ".cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": ROOT / ".cache/torchinductor",
        "TORCH_EXTENSIONS_DIR": ROOT / ".cache/torch_extensions",
        "CUDA_CACHE_PATH": ROOT / ".cache/cuda", "TORCH_HOME": ROOT / ".cache/torch",
        "HF_HOME": ROOT / ".cache/huggingface", "MODELSCOPE_CACHE": ROOT / "modelscope_cache",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
        "SGLANG_USE_MODELSCOPE": "true",
    }.items():
        env[key] = str(value)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    os.environ.update(env)
    return env


def input_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def arrivals(n, qps):
    rng = random.Random(SEED + 1)
    result = [0.0]
    for _ in range(n - 1):
        result.append(result[-1] + (rng.expovariate(qps) if qps else 0))
    return result


def prepare():
    setup_env()
    from transformers import AutoTokenizer
    import sglang.bench_serving as bench
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    FORMAL.mkdir(parents=True, exist_ok=True)
    manifest = {"model": MODEL_ID, "seed": SEED, "sglang": "0.4.6.post5",
                "kv_dtype": "fp16", "gpu_kv_bytes": 1008 * 1024**2,
                "host_kv_bytes": 8 * 1024**3, "points": []}
    for point in POINTS:
        path = FORMAL / "datasets" / f"{point.name}.json"
        if path.exists():
            dataset = json.loads(path.read_text())
            assert dataset["point"] == asdict(point)
        else:
            random.seed(SEED)
            # 上游缓存键没有 seed；显式定址到本实验目录，禁止写 ~/.cache。
            bench.get_gen_prefix_cache_path = lambda args, tok: (
                FORMAL / "generator_cache" / f"{point.isl}_{point.n}_{point.groups}_seed42.pkl")
            generated = bench.sample_generated_shared_prefix_requests(
                point.groups, point.n // point.groups, point.isl // 2,
                point.isl - point.isl // 2, point.osl, tokenizer, argparse.Namespace())
            rows = [{"text": item.prompt, "prompt_tokens": item.prompt_len}
                    for item in generated]
            dataset = {"point": asdict(point), "rows": rows,
                       "arrival_offsets_s": arrivals(point.n, point.qps),
                       "input_sha256": input_hash(rows)}
            write_json(path, dataset)
        manifest["points"].append({**asdict(point), "input_sha256": dataset["input_sha256"]})
    # 独立机制探针：一个热前缀多次命中后写入 CPU，再以 > GPU KV 池的不同输入驱逐。
    # 使用精确 token IDs，仅此探针不是上游正式负载；不混入性能统计。
    rng = random.Random(20260910)
    vocab = [v for v in tokenizer.get_vocab().values() if v not in tokenizer.all_special_ids]
    hot = rng.choices(vocab, k=2048)
    rows = [{"input_ids": hot + rng.choices(vocab, k=2048), "prompt_tokens": 4096}
            for _ in range(6)]
    rows += [{"input_ids": rng.choices(vocab, k=4096), "prompt_tokens": 4096}
             for _ in range(10)]
    rows += [{"input_ids": hot + rng.choices(vocab, k=2048), "prompt_tokens": 4096}
             for _ in range(2)]
    write_json(FORMAL / "datasets/reload_gate.json", {
        "point": asdict(PILOT), "rows": rows, "arrival_offsets_s": [0] * len(rows),
        "input_sha256": input_hash(rows),
    })
    write_json(FORMAL / "manifest.json", manifest)
    print(json.dumps({"prepared_points": len(POINTS),
                      "formal_requests_all_baselines": sum(p.n for p in POINTS) * 3,
                      "manifest": str(FORMAL / "manifest.json")}), flush=True)


def http(path, data=None, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


async def generate(session, row, point, rid, offered, semaphore):
    queued_at = time.perf_counter()
    async with semaphore:
        start = time.perf_counter()
        start_ns = time.time_ns()
        payload = {key: row[key] for key in ("text", "input_ids") if key in row}
        payload.update(rid=rid, stream=True,
                       sampling_params={"temperature": 0, "max_new_tokens": point.osl,
                                        "ignore_eos": True})
        chunks, last, last_count, meta, text = [], None, 0, {}, ""
        async with session.post(f"http://127.0.0.1:{PORT}/generate", json=payload) as response:
            response.raise_for_status()
            while True:
                line = await response.content.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                item = json.loads(data)
                meta = item.get("meta_info", meta)
                text = item.get("text", text)
                count = int(meta.get("completion_tokens", 0))
                if count > last_count:
                    now = time.perf_counter()
                    chunks.append({"time_s": now - start, "new_tokens": count - last_count})
                    last, last_count = now, count
        end = time.perf_counter()
        if last_count != point.osl or not chunks:
            raise RuntimeError(f"请求 {rid} 生成 {last_count}/{point.osl}: {meta}")
        if int(meta["prompt_tokens"]) != row["prompt_tokens"]:
            raise RuntimeError(f"请求 {rid} 输入 token 数不一致")
        # 一次 SSE 可能包含多个 token；保留分块原始数据，不伪称逐 token 计时。
        itls = []
        for previous, current in zip(chunks, chunks[1:]):
            dt = (current["time_s"] - previous["time_s"]) / current["new_tokens"]
            itls.extend([dt] * current["new_tokens"])
        return {"rid": rid, "start_ns": start_ns, "end_ns": time.time_ns(),
                "offered_offset_s": offered, "client_wait_s": start - queued_at,
                "ttft_s": chunks[0]["time_s"], "e2e_s": end - start,
                "decode_window_s": chunks[-1]["time_s"] - chunks[0]["time_s"],
                "itls_s": itls, "chunks": chunks, "meta_info": meta,
                "text": text, "success": True}


async def load_requests(point, dataset, output, warmup=False):
    import aiohttp
    semaphore = asyncio.Semaphore(point.concurrency)
    rows = dataset["rows"][:4] if warmup else dataset["rows"]
    offsets = [0] * len(rows) if warmup else dataset["arrival_offsets_s"]
    started, start_ns = time.perf_counter(), time.time_ns()
    results = []
    prefix = f"{'warmup' if warmup else point.name}-{start_ns}"
    with output.open("w", buffering=1) as handle:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
            async def worker(index):
                await asyncio.sleep(max(0, started + offsets[index] - time.perf_counter()))
                result = await generate(session, rows[index], point, f"{prefix}-{index}",
                                        offsets[index], semaphore)
                result["request_index"] = index
                result["offered_e2e_s"] = time.perf_counter() - started - offsets[index]
                result["offered_ttft_s"] = (result["start_ns"] - start_ns) / 1e9 - offsets[index] + result["ttft_s"]
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(rows):
                    print(f"{point.name} {'warmup' if warmup else 'measured'}: {len(results)}/{len(rows)}", flush=True)
            # gather 首次错误时立即失败，不自动重试；关闭 session 终止客户端任务。
            tasks = [asyncio.create_task(worker(i)) for i in range(len(rows))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    return {"start_ns": start_ns, "end_ns": time.time_ns(),
            "duration_s": time.perf_counter() - started, "completed": len(results)}


def event_rows(folder, start_ns, end_ns):
    rows = []
    for path in folder.glob("events.*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if start_ns <= row.get("start_ns", row["time_ns"]) <= end_ns:
                rows.append(row)
    return rows


def sample_system(index, process, output, stop):
    """采样只读状态；不因负载或文件暂时不增长而终止实验。"""
    next_status = time.monotonic()
    with output.open("w", buffering=1) as handle:
        while not stop.is_set():
            row = {"time_ns": time.time_ns(), "server_exit_code": process.poll()}
            try:
                result = subprocess.run([
                    "nvidia-smi", "-i", index,
                    "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=10, check=True)
                row["gpu"] = dict(zip(["memory_used_mib", "utilization_pct", "power_w", "temperature_c"],
                                      [float(x.strip()) for x in result.stdout.split(",")]))
                row["meminfo"] = {line.split(":")[0]: int(line.split()[1])
                                  for line in Path("/proc/meminfo").read_text().splitlines()
                                  if line.startswith(("MemAvailable:", "MemTotal:"))}
            except Exception as exc:
                row["sampling_error"] = repr(exc)
            handle.write(json.dumps(row) + "\n")
            if time.monotonic() >= next_status:
                print(f"MONITOR server_alive={process.poll() is None} gpu={row.get('gpu')}", flush=True)
                next_status = time.monotonic() + 30
            stop.wait(5)


def run_point(point, baseline, run_dir, events, pilot):
    dataset = json.loads((FORMAL / "datasets" / f"{point.name}.json").read_text())
    assert input_hash(dataset["rows"]) == dataset["input_sha256"]
    assert asdict(point) == dataset["point"]
    folder = run_dir / point.name
    folder.mkdir()
    print(f"START {baseline}/{point.name}", flush=True)
    asyncio.run(load_requests(point, dataset, folder / "warmup.jsonl", warmup=True))
    http("/flush_cache", {}, timeout=70)
    time.sleep(1)
    (folder / "metrics.before.prom").write_text(http("/metrics"))
    start_ns = time.time_ns()
    try:
        measurement = asyncio.run(asyncio.wait_for(
            load_requests(point, dataset, folder / "requests.jsonl"), timeout=1800))
        time.sleep(1)  # 后台传输/事件记录完成；不计入请求吞吐的测量窗口。
        end_ns = time.time_ns()
        observed = event_rows(events, start_ns, end_ns)
        write_json(folder / "events.json", observed)
        (folder / "metrics.after.prom").write_text(http("/metrics"))
        transfers = [e for e in observed if e["kind"] == "transfer"]
        counts = {direction: sum(e["bytes"] for e in transfers if e["direction"] == direction)
                  for direction in ("gpu_to_cpu", "cpu_to_gpu")}
        host_tokens = sum(e["tokens"] for e in observed if e["kind"] == "host_reuse")
        evictions = sum(e["tokens"] for e in observed if e["kind"] == "eviction")
        if pilot and baseline == "S2" and not (
                counts["gpu_to_cpu"] > 0 and counts["cpu_to_gpu"] > 0 and host_tokens > 0 and evictions > 0):
            raise RuntimeError(f"回载机制门槛未通过: {counts}, host={host_tokens}, evict={evictions}")
        result = {"status": "completed", "baseline": baseline, "point": asdict(point),
                  "measurement": measurement, "input_sha256": dataset["input_sha256"],
                  "transfer_bytes": counts, "host_reuse_tokens": host_tokens,
                  "evicted_tokens": evictions}
        write_json(folder / "result.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(folder / "result.json", {"status": "failed", "error": repr(exc)})
        raise


def server_command(launcher, baseline):
    command = [sys.executable, str(launcher), "--model-path", str(MODEL),
               "--host", "127.0.0.1", "--port", str(PORT), "--tp-size", "1",
               "--dtype", "half", "--kv-cache-dtype", "auto",
               "--attention-backend", "torch_native", "--sampling-backend", "pytorch",
               "--disable-cuda-graph", "--disable-overlap-schedule", "--disable-custom-all-reduce",
               "--chunked-prefill-size", "-1", "--max-total-tokens", "28672",
               "--max-running-requests", "4", "--context-length", "8192",
               "--mem-fraction-static", "0.65", "--page-size", "1",
               "--random-seed", "42", "--enable-metrics", "--stream-interval", "1"]
    if baseline == "S0":
        command += ["--disable-radix-cache"]
    if baseline == "S2":
        command += ["--enable-hierarchical-cache", "--hicache-size", "8",
                    "--hicache-write-policy", "write_through_selective"]
    return command


def run(baseline, pilot, only_points):
    env = setup_env()
    index, uuid = idle_gpu()
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", PORT))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = FORMAL / "runs" / ("pilot" if pilot else "submission") / baseline / stamp
    run_dir.mkdir(parents=True)
    events = run_dir / "events"
    events.mkdir()
    code_dir = Path(__file__).resolve().parent
    injection = f"sys.path.insert(0, {str(code_dir)!r})\nfrom v100_observer import install\ninstall()\n"
    launcher = run_dir / "launch.py"
    launcher.write_text(LAUNCHER.replace('if __name__ == "__main__":', injection + '\nif __name__ == "__main__":'))
    env.update(CUDA_VISIBLE_DEVICES=uuid, V100_EVENTS_DIR=str(events))
    command = server_command(launcher, baseline)
    from importlib.metadata import version
    provenance = {"command": command, "gpu_index": index, "gpu_uuid": uuid,
                  "versions": {p: version(p) for p in ("sglang", "torch", "triton", "transformers")},
                  "code_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), code_dir / "v100_observer.py", code_dir / "smoke_v100.py")}}
    write_json(run_dir / "environment.json", provenance)
    print(f"RUN_DIR={run_dir}", flush=True)
    process = None
    sampler, stop = None, threading.Event()
    with (run_dir / "server.log").open("w") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                       start_new_session=True)
            sampler = threading.Thread(target=sample_system,
                                       args=(index, process, run_dir / "system.jsonl", stop), daemon=True)
            sampler.start()
            deadline, progress = time.monotonic() + 600, time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"服务退出 {process.returncode}; {run_dir}/server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("服务启动超过 600 秒")
                try:
                    http("/health_generate", timeout=3)
                    break
                except Exception:
                    if time.monotonic() >= progress:
                        print("等待模型服务就绪", flush=True)
                        progress = time.monotonic() + 30
                    time.sleep(2)
            points = [PILOT] if pilot else [p for p in POINTS if not only_points or p.name in only_points]
            for point in points:
                run_point(point, baseline, run_dir, events, pilot)
            write_json(run_dir / "status.json", {"status": "completed", "points": [p.name for p in points]})
        except BaseException as exc:
            write_json(run_dir / "status.json", {"status": "failed", "error": repr(exc)})
            raise
        finally:
            stop.set()
            if sampler:
                sampler.join(timeout=15)
            if process:
                print("清理本次服务进程并释放 GPU", flush=True)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--baseline", choices=["S0", "S1", "S2"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--points", nargs="+", choices=[p.name for p in POINTS])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if not args.baseline:
            parser.error("run requires --baseline")
        run(args.baseline, args.pilot, args.points)


if __name__ == "__main__":
    main()
