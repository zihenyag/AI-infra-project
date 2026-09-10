"""交付版重跑入口：默认只打印计划，--execute 才使用GPU，不覆盖原始数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import run_v100_experiment as runner


def verify_model(model, provenance):
    expected = json.loads(provenance.read_text())["model_files"]
    for name, meta in expected.items():
        with (model / name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != meta["sha256"]:
            raise ValueError(f"模型指纹不匹配：{name}")


def exact_gpu(index):
    rows = subprocess.check_output([
        "nvidia-smi", "-i", index, "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15).strip().split(",")
    actual, uuid, name, used, util = [s.strip() for s in rows]
    busy = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True, timeout=15)
    if name != "Tesla V100-SXM2-32GB" or uuid in busy or float(used) > 128 or float(util) != 0:
        raise RuntimeError(f"指定卡不可用于此锁定配置：{rows}")
    return actual, uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True, help="全新的运行目录，全部新输出放在其中")
    parser.add_argument("--model", type=Path, required=True, help="已经下载的未量化Qwen2.5-3B目录")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "raw")
    parser.add_argument("--baseline", choices=["all", "S0", "S1", "S2"], default="all")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=30171)
    parser.add_argument("--pilot", action="store_true", help="只运行S2的18请求机制探针")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root, model, data = args.work_root.resolve(), args.model.resolve(), args.data_root.resolve()
    baselines = ["S2"] if args.pilot else (["S0", "S1", "S2"] if args.baseline == "all" else [args.baseline])
    plan = {"execute": args.execute, "work_root": str(root), "model": str(model),
            "baselines": baselines, "gpu_index": args.gpu, "port": args.port, "pilot": args.pilot,
            "execution": "single_gpu_sequential" if len(baselines) > 1 else "single_gpu",
            "note": "只改变路径/GPU/端口，不改测量函数；此为新的重跑，不是已交付实验的再次验证结果"}
    if root.exists():
        parser.error("work-root已存在：请选择新目录，禁止覆盖历史输出")
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if sys.version_info < (3, 11):
        parser.error("需要Python3.11+；GPU实测使用3.12.3")
    verify_model(model, data / "provenance/environment.json")
    exact_gpu(args.gpu)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", ".cache", "modelscope_cache"):
        (root / name).mkdir()
    runner.ROOT, runner.MODEL, runner.FORMAL, runner.PORT = root, model, root / "formal", args.port
    runner.idle_gpu = lambda: exact_gpu(args.gpu)
    shutil.copytree(data / "datasets", runner.FORMAL / "datasets")
    shutil.copy2(data / "manifest.json", runner.FORMAL / "manifest.json")
    runner.write_json(root / "rerun_plan.json", plan)
    for baseline in baselines:
        runner.run(baseline, args.pilot, None)


if __name__ == "__main__":
    main()
