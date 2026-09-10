
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

sys.path.insert(0, '/root/beamforming/CV/v100-probe')
from v100_observer import install
install()

if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    try:
        launch_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
