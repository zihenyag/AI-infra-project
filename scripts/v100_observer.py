"""锁定 SGLang 0.4.6.post5 的被动观测；不更改缓存命中/驱逐策略。"""
from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
import queue
import textwrap
import threading
import time

_pending = queue.Queue()
_lock = threading.Lock()
_handle = None
_started = False


def emit(kind, **fields):
    global _handle
    with _lock:
        if _handle is None:
            folder = Path(os.environ["V100_EVENTS_DIR"])
            folder.mkdir(parents=True, exist_ok=True)
            _handle = (folder / f"events.{os.getpid()}.jsonl").open("a", buffering=1)
        _handle.write(json.dumps({"kind": kind, "time_ns": time.time_ns(),
                                  "pid": os.getpid(), **fields}) + "\n")


def _drain():
    while True:
        start, end, fields = _pending.get()
        # 不在计算/传输线程添加 synchronize。后台等待事件完成。
        while not end.query():
            time.sleep(0.005)
        emit(**fields, cuda_elapsed_s=start.elapsed_time(end) / 1000)


def _events():
    global _started
    import torch
    if not _started:
        with _lock:
            if not _started:
                threading.Thread(target=_drain, daemon=True).start()
                _started = True
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def observed_copy(tensor, device, direction):
    """测量原有阻塞 .to() 调用；CUDA 时间包含流内等待，非纯 PCIe DMA。"""
    start, end = _events()
    start_ns = time.time_ns()
    start.record()
    t0 = time.perf_counter()
    result = tensor.to(device=device, non_blocking=False)
    wall = time.perf_counter() - t0
    end.record()
    _pending.put((start, end, dict(kind="transfer", direction=direction,
                                  bytes=tensor.numel() * tensor.element_size(),
                                  start_ns=start_ns, end_ns=time.time_ns(),
                                  host_call_s=wall)))
    return result


def replace_source(cls, name, module, old, new):
    source = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    if source.count(old) != 1:
        raise RuntimeError(f"上游源码不匹配: {cls.__name__}.{name}")
    namespace = {}
    exec(compile(source.replace(old, new), "<v100-observer>", "exec"),
         module.__dict__, namespace)
    setattr(cls, name, namespace[name])


def install():
    from importlib.metadata import version
    assert version("sglang") == "0.4.6.post5"
    import sglang.srt.mem_cache.memory_pool as pool
    import sglang.srt.managers.cache_controller as controller
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.managers.scheduler import Scheduler

    pool._v100_copy = controller._v100_copy = observed_copy
    replace_source(
        controller.HiCacheController, "write_aux_func", controller,
        "op_.data = self.mem_pool_device.get_flat_data(op_.device_indices).to(\n"
        "            self.mem_pool_host.device\n        )",
        "op_.data = _v100_copy(self.mem_pool_device.get_flat_data(op_.device_indices), "
        "self.mem_pool_host.device, 'gpu_to_cpu')",
    )
    replace_source(
        pool.MHATokenToKVPool, "transfer_per_layer", pool,
        "flat_data.to(device=self.device, non_blocking=False)",
        "_v100_copy(flat_data, self.device, 'cpu_to_gpu')",
    )

    original_host_init = pool.HostKVCache.__init__

    @functools.wraps(original_host_init)
    def host_init(self, device_pool, host_to_device_ratio, host_size,
                  pin_memory, device, page_size):
        if host_size != 8:
            raise ValueError("正式 V100 实验仅允许 8 GiB host KV")
        original_host_init(self, device_pool, host_to_device_ratio,
                           8 * 1024**3 / 1e9, pin_memory, device, page_size)
        emit("host_pool", tokens=self.size, bytes=self.size * self.size_per_token,
             requested_bytes=8 * 1024**3, dtype=str(self.dtype))

    pool.HostKVCache.__init__ = host_init

    def instrument_evict(cls):
        original = cls.evict

        @functools.wraps(original)
        def evict(self, *args, **kwargs):
            allocator = self.token_to_kv_pool_allocator
            before = allocator.available_size()
            result = original(self, *args, **kwargs)
            emit("eviction", tokens=max(0, allocator.available_size() - before))
            return result

        cls.evict = evict

    instrument_evict(RadixCache)
    instrument_evict(HiRadixCache)
    original_load = HiRadixCache.init_load_back

    @functools.wraps(original_load)
    def load(self, last_node, prefix_indices, mem_quota=None):
        before = len(prefix_indices)
        result = original_load(self, last_node, prefix_indices, mem_quota)
        added = len(result[1]) - before
        if added:
            emit("host_reuse", tokens=added)
        return result

    HiRadixCache.init_load_back = load
    original_batch = Scheduler.run_batch

    @functools.wraps(original_batch)
    def batch(self, batch):
        start, end = _events()
        allocator = self.token_to_kv_pool_allocator
        tree = self.tree_cache
        host = getattr(tree, "token_to_kv_pool_host", None)
        fields = dict(
            kind="batch", phase="decode" if batch.forward_mode.is_decode() else "prefill",
            request_ids=[r.rid for r in batch.reqs],
            batch_size=len(batch.reqs), start_ns=time.time_ns(),
            gpu_used_tokens=allocator.size - allocator.available_size(),
            gpu_total_tokens=allocator.size,
            host_used_tokens=host.size - host.available_size() if host else 0,
            host_total_tokens=host.size if host else 0,
            computed_prompt_tokens=sum(r.extend_input_len for r in batch.reqs)
            if not batch.forward_mode.is_decode() else 0,
        )
        start.record()
        t0 = time.perf_counter()
        result = original_batch(self, batch)
        fields.update(host_call_s=time.perf_counter() - t0, end_ns=time.time_ns())
        end.record()
        _pending.put((start, end, fields))
        return result

    Scheduler.run_batch = batch
