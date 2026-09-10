"""在独立 V100 环境测试 Qwen2.5-3B；本脚本需要单独批准下载/占用 GPU。"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path("/root/beamforming/CV/v100-probe")
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PORT = 30170

# 在 spawn 出的 worker 中也应用相同的三个上游 native 实现。
# 不改模型数学、不替换缓存；仅绕过未编译 sm70 的融合算子。
LAUNCHER = '''
import os
import sys
import inspect
import textwrap
import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.rotary_embedding import RotaryEmbedding

RMSNorm.forward_cuda = RMSNorm.forward_native
SiluAndMul.forward_cuda = SiluAndMul.forward_native
RotaryEmbedding.forward_cuda = RotaryEmbedding.forward_native
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# 仅为经过基础算子验证的 sm70/native/FP16 配置放行旧版架构门槛。
# 保留真实 capability；其他 dtype、量化和后端仍遵守原版限制。
import sglang.srt.model_executor.model_runner as model_runner
from importlib.metadata import version
assert version("sglang") == "0.4.6.post5"
source = textwrap.dedent(inspect.getsource(model_runner.ModelRunner.load_model))
guard = 'raise RuntimeError("SGLang only supports sm75 and above.")'
replacement = """if not (
                    torch.cuda.get_device_capability() == (7, 0)
                    and self.server_args.attention_backend == "torch_native"
                    and self.server_args.sampling_backend == "pytorch"
                    and self.server_args.quantization is None
                    and self.server_args.kv_cache_dtype == "auto"
                    and self.server_args.disable_cuda_graph
                    and self.server_args.disable_overlap_schedule
                    and self.server_args.disable_custom_all_reduce
                ):
                    raise RuntimeError("Unsupported configuration for V100 probe")
                logger.warning("Experimental sm70 native/FP16 compatibility route")"""
assert source.count(guard) == 1, "Upstream architecture guard changed"
namespace = {}
exec(compile(source.replace(guard, replacement), "<v100-load-model>", "exec"),
     model_runner.__dict__, namespace)
model_runner.ModelRunner.load_model = namespace["load_model"]

if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    try:
        launch_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
'''


def idle_gpu() -> tuple[str, str]:
    rows = csv.reader(io.StringIO(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)))
    busy = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader",
    ], text=True)
    for row in rows:
        index, name, uuid, free, util = (part.strip() for part in row)
        if "V100" in name and uuid not in busy and int(free) > 30000 and int(util) == 0:
            return index, uuid
    raise RuntimeError("没有满足空闲条件的 V100")


def request(path: str, data: dict | None = None, timeout: float = 60) -> dict:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def run_baseline(baseline: str, model: Path, launcher: Path, env: dict) -> dict:
    index, uuid = idle_gpu()
    env = dict(env, CUDA_VISIBLE_DEVICES=uuid)
    with socket.socket() as sock:
        # 上一组服务退出后的 TIME_WAIT 不代表仍有服务监听。
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", PORT))
    command = [
        sys.executable, str(launcher), "--model-path", str(model),
        "--host", "127.0.0.1", "--port", str(PORT), "--tp-size", "1",
        "--dtype", "half", "--kv-cache-dtype", "auto",
        "--attention-backend", "torch_native", "--sampling-backend", "pytorch",
        "--disable-cuda-graph", "--disable-overlap-schedule",
        "--disable-custom-all-reduce", "--chunked-prefill-size", "-1",
        "--max-total-tokens", "28672", "--max-running-requests", "4",
        "--context-length", "8192", "--mem-fraction-static", "0.65",
        "--page-size", "1", "--enable-metrics",
    ]
    if baseline == "S0":
        command += ["--disable-radix-cache"]
    if baseline == "S2":
        command += ["--enable-hierarchical-cache", "--hicache-size", "8",
                    "--hicache-write-policy", "write_through"]
    result = {"baseline": baseline, "gpu_index": index, "gpu_uuid": uuid,
              "command": command, "status": "failed"}
    with (ROOT / "logs" / f"{baseline}.log").open("w") as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=log,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            next_progress = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"服务退出: {process.returncode}")
                try:
                    request("/get_model_info", timeout=2)
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= next_progress:
                        print(f"{baseline}: waiting for model service", flush=True)
                        next_progress = time.monotonic() + 30
                    time.sleep(2)
            else:
                raise TimeoutError("服务在 600 秒内未就绪")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": (
                    "Background notes: " + "This is a cache reuse test. " * 80
                    + "\nIgnore the notes. What is 1 + 1? Answer with just the number."
                )}
            ], tokenize=False, add_generation_prompt=True)
            responses = []
            for _ in range(2):
                responses.append(request("/generate", {
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                }, timeout=120))
            if not all("2" in item.get("text", "") for item in responses):
                raise RuntimeError(f"实际生成检查失败: {responses}")
            cached = responses[-1].get("meta_info", {}).get("cached_tokens")
            if baseline == "S0" and cached != 0:
                raise RuntimeError(f"S0 缓存关闭检查失败: cached_tokens={cached}")
            if baseline != "S0" and (cached is None or cached <= 0):
                raise RuntimeError(f"前缀复用检查失败: cached_tokens={cached}")
            result.update(status="passed", responses=responses)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    (ROOT / "logs" / f"{baseline}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    os.chdir(ROOT)
    _, gpu_uuid = idle_gpu()
    env = dict(os.environ)
    for name, value in {
        "TMPDIR": ROOT / "tmp", "XDG_CACHE_HOME": ROOT / ".cache",
        "TRITON_CACHE_DIR": ROOT / ".cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": ROOT / ".cache/torchinductor",
        "TORCH_EXTENSIONS_DIR": ROOT / ".cache/torch_extensions",
        "CUDA_CACHE_PATH": ROOT / ".cache/cuda", "TORCH_HOME": ROOT / ".cache/torch",
        "HF_HOME": ROOT / ".cache/huggingface", "MODELSCOPE_CACHE": ROOT / "modelscope_cache",
        "PYTHONDONTWRITEBYTECODE": "1", "SGLANG_USE_MODELSCOPE": "true",
    }.items():
        env[name] = str(value)
        os.environ[name] = str(value)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    launcher = ROOT / "launch_v100.py"
    launcher.write_text(LAUNCHER)
    # 先检查完整启动入口，避免缺少依赖时先下载数 GB 权重。
    check = subprocess.run(
        [sys.executable, str(launcher), "--help"],
        env=dict(env, CUDA_VISIBLE_DEVICES=gpu_uuid),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    (ROOT / "logs/import-check.log").write_text(check.stdout)
    if check.returncode:
        print(check.stdout[-8000:], flush=True)
        raise RuntimeError(f"SGLang 启动入口检查失败: {check.returncode}")
    from modelscope.hub.snapshot_download import snapshot_download
    model = ROOT / "models/Qwen2.5-3B-Instruct"
    print("Downloading Qwen2.5-3B-Instruct from ModelScope", flush=True)
    snapshot_download(MODEL_ID, local_dir=str(model),
                      cache_dir=str(ROOT / "modelscope_cache"))
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    results = []
    for baseline in ("S0", "S1", "S2"):
        result = run_baseline(baseline, model, launcher, env)
        results.append(result)
        if result["status"] != "passed":
            break
    summary = {"model": MODEL_ID, "sglang": "0.4.6.post5", "weight_dtype": "fp16",
               "kv_dtype": "fp16", "attention": "torch_native",
               "adaptations": ["RMSNorm.forward_native", "SiluAndMul.forward_native",
                               "RotaryEmbedding.forward_native",
                               "scoped sm70 native/FP16 architecture guard"],
               "results": results,
               "all_baselines_passed": len(results) == 3 and all(
                   item["status"] == "passed" for item in results)}
    (ROOT / "logs/smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_baselines_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
