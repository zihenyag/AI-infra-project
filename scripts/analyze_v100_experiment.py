"""仅从保存的 V100 原始结果生成汇总和静态实验图，不连接服务器。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    rank = (len(values) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] * (hi - rank) + values[hi] * (rank - lo) if hi != lo else values[lo]


def distribution(values, prefix, scale=1):
    values = [float(v) * scale for v in values]
    return {f"{prefix}_{stat}": fn(values) if values else None for stat, fn in {
        "mean": statistics.fmean, "p50": lambda v: quantile(v, .5),
        "p95": lambda v: quantile(v, .95), "p99": lambda v: quantile(v, .99),
    }.items()}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(folder):
    result = json.loads((folder / "result.json").read_text())
    point = result["point"]
    requests = read_jsonl(folder / "requests.jsonl")
    assert len(requests) == point["n"] == result["measurement"]["completed"]
    assert len({r["rid"] for r in requests}) == len(requests)
    assert all(r["success"] and r["meta_info"]["completion_tokens"] == point["osl"] for r in requests)
    assert all(math.isfinite(r["e2e_s"]) and 0 <= r["ttft_s"] <= r["e2e_s"] for r in requests)
    events = json.loads((folder / "events.json").read_text())
    batches = [e for e in events if e["kind"] == "batch"]
    transfers = [e for e in events if e["kind"] == "transfer"]
    assert batches, "缺少批次观测，不得生成完整结果"
    request_ids = {r["rid"] for r in requests}
    assert all(set(e["request_ids"]) <= request_ids for e in batches), "测量窗口混入了其他请求"
    duration = result["measurement"]["duration_s"]
    prompts = sum(r["meta_info"]["prompt_tokens"] for r in requests)
    outputs = sum(r["meta_info"]["completion_tokens"] for r in requests)
    assert sum(e["batch_size"] for e in batches) == outputs, "批次输出token计数不完整"
    cached = sum(r["meta_info"]["cached_tokens"] for r in requests)
    assert 0 <= cached <= prompts
    if result["baseline"] == "S0":
        assert cached == 0, "S0 必须没有前缀复用"
    row = {"baseline": result["baseline"], "point": point["name"], "mode": point["mode"],
           "target_isl": point["isl"], "qps": point["qps"], "concurrency": point["concurrency"],
           "seed": 42, "requested": point["n"], "completed": len(requests), "duration_s": duration,
           "actual_isl_mean": prompts / len(requests), "output_tokens": outputs,
           "prompt_tokens": prompts, "cached_tokens": cached, "cache_hit_ratio": cached / prompts,
           "request_throughput": len(requests) / duration, "output_token_throughput": outputs / duration,
           "logical_token_throughput": (prompts + outputs) / duration,
           "host_reuse_tokens": result["host_reuse_tokens"], "evicted_tokens": result["evicted_tokens"],
           "input_sha256": result["input_sha256"], "source": str(folder)}
    for name in ("ttft_s", "e2e_s", "offered_ttft_s", "offered_e2e_s", "client_wait_s", "decode_window_s"):
        row.update(distribution([r[name] for r in requests], name.removesuffix("_s") + "_ms", 1000))
    row.update(distribution([v for r in requests for v in r["itls_s"]], "itl_ms", 1000))
    for phase in ("prefill", "decode"):
        selected = [e for e in batches if e["phase"] == phase]
        row.update(distribution([e["cuda_elapsed_s"] for e in selected], f"batch_{phase}_cuda_ms", 1000))
        row[f"batch_{phase}_cuda_total_s"] = sum(e["cuda_elapsed_s"] for e in selected)
    computed = sum(e["computed_prompt_tokens"] for e in batches)
    row["computed_prompt_tokens"] = computed
    row["cached_compute_accounting_difference"] = computed - (prompts - cached)
    assert row["cached_compute_accounting_difference"] == 0, "计算/缓存token会计不一致"
    row["gpu_reuse_tokens"] = cached - row["host_reuse_tokens"]
    assert row["gpu_reuse_tokens"] >= 0
    row["gpu_reuse_ratio_of_prompt"] = row["gpu_reuse_tokens"] / prompts
    row["host_reuse_ratio_of_prompt"] = row["host_reuse_tokens"] / prompts
    row["external_query_tokens"] = None  # 未独立测量host查询分母。
    row["evicted_tokens_per_s"] = row["evicted_tokens"] / duration
    row["physical_token_throughput"] = (computed + outputs) / duration
    row["prefill_token_throughput"] = computed / row["batch_prefill_cuda_total_s"] if row["batch_prefill_cuda_total_s"] else None
    for tier in ("gpu", "host"):
        ratios = [e[f"{tier}_used_tokens"] / e[f"{tier}_total_tokens"] for e in batches if e[f"{tier}_total_tokens"]]
        row[f"{tier}_kv_usage_peak"] = max(ratios) if ratios else (0 if tier == "host" and result["baseline"] != "S2" else None)
        row[f"{tier}_kv_usage_batch_sample_mean"] = statistics.fmean(ratios) if ratios else None
    for direction in ("gpu_to_cpu", "cpu_to_gpu"):
        selected = [e for e in transfers if e["direction"] == direction]
        num_bytes = sum(e["bytes"] for e in selected)
        assert num_bytes == result["transfer_bytes"][direction], "原始事件与汇总传输字节不一致"
        row[f"{direction}_bytes"] = num_bytes
        row[f"{direction}_operations"] = len(selected)
        for timing in ("host_call_s", "cuda_elapsed_s"):
            elapsed = sum(e[timing] for e in selected)
            row[f"{direction}_{timing}"] = elapsed
            row[f"{direction}_GBps_by_{timing}"] = num_bytes / elapsed / 1e9 if elapsed else None
        row.update(distribution([e["bytes"] for e in selected], f"{direction}_size_bytes"))
    row["host_transfer_accounting_difference_bytes"] = (
        row["cpu_to_gpu_bytes"] - row["host_reuse_tokens"] * 36864)
    assert row["host_transfer_accounting_difference_bytes"] == 0, "回载token与逐层传输字节不一致"
    # 1秒滑动窗口峰值，使用 SSE 块抵达时刻；不是设备瞬时吞吐。
    times = sorted((r["start_ns"] / 1e9 + c["time_s"], c["new_tokens"])
                   for r in requests for c in r["chunks"])
    peak = current = left = 0
    for right, (stamp, count) in enumerate(times):
        current += count
        while stamp - times[left][0] >= 1:
            current -= times[left][1]
            left += 1
        peak = max(peak, current)
    row["peak_output_tokens_per_1s"] = peak
    return row, requests


def execution_checks(provenance, schedule):
    """不同卡可收齐兼容数据，但不能因此宣称同卡/资源隔离实验。"""
    uuids = {p["gpu_uuid"] for p in provenance}
    matched_gpu = len(uuids) <= 1
    inventory = {r["uuid"]: r for r in schedule.get("gpu_inventory", [])}
    known = bool(uuids) and uuids <= inventory.keys()
    matched_gpu_model = known and len({(inventory[u]["name"], inventory[u]["memory_total_mib"])
                                      for u in uuids}) == 1
    expected = {b: a["gpu_uuid"] for b, a in schedule.get("assignments", {}).items()}
    if "existing_s0_gpu_uuid" in schedule:
        expected["S0"] = schedule["existing_s0_gpu_uuid"]
    mapped = bool(provenance) and all(expected.get(p["baseline"]) == p["gpu_uuid"] for p in provenance)
    parallel = schedule.get("mode") == "same_model_different_gpus_parallel"
    valid = (matched_gpu_model and mapped) if parallel else matched_gpu
    return {"matched_gpu": matched_gpu, "matched_gpu_model": matched_gpu_model if known else None,
            "gpu_assignment_verified": valid,
            "execution_mode": schedule.get("mode", "single_gpu_sequential" if matched_gpu else "unverified_multi_gpu"),
            "resource_isolated_comparison": False if parallel else None}


def collect(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for point in manifest["points"]:
        dataset = json.loads((root / "datasets" / f"{point['name']}.json").read_text())
        digest = hashlib.sha256(json.dumps(dataset["rows"], sort_keys=True).encode()).hexdigest()
        assert digest == dataset["input_sha256"] == point["input_sha256"], "数据集SHA256校验失败"
        assert len(dataset["rows"]) == len(dataset["arrival_offsets_s"]) == point["n"]
        assert dataset["point"] == {k: v for k, v in point.items() if k != "input_sha256"}
    expected = {(baseline, p["name"]): p for baseline in ("S0", "S1", "S2") for p in manifest["points"]}
    selected, failed = {}, []
    for path in sorted((root / "runs/submission").glob("*/*/*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] != "completed":
            failed.append(str(path))
            continue
        key = (result["baseline"], result["point"]["name"])
        if key not in expected:
            continue
        assert expected[key]["input_sha256"] == result["input_sha256"]
        # 同一测点如有显式重跑，取第一份完整成功结果；不挑选最优性能。
        selected.setdefault(key, path.parent)
    schedule_path = root / "execution_schedule.json"
    schedule = json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
    rows, raw = [], {}
    provenance = []
    for key, folder in sorted(selected.items()):
        env = json.loads((folder.parent / "environment.json").read_text())
        provenance.append({**env, "baseline": key[0]})
        row, requests = summarize(folder)
        dataset = json.loads((root / "datasets" / f"{key[1]}.json").read_text())
        assert sorted(r["request_index"] for r in requests) == list(range(len(dataset["rows"])))
        for request in requests:
            index = request["request_index"]
            assert request["offered_offset_s"] == dataset["arrival_offsets_s"][index]
            assert request["meta_info"]["prompt_tokens"] == dataset["rows"][index]["prompt_tokens"]
        row["source"] = str(folder.relative_to(root))
        row.update(gpu_uuid=env["gpu_uuid"], gpu_index=env["gpu_index"])
        dispatch = schedule.get("started_ns")
        measured = json.loads((folder / "result.json").read_text())["measurement"]
        row["parallel_timing_phase"] = (
            "not_recorded" if dispatch is None else
            "before_dispatch" if measured["end_ns"] < dispatch else
            "after_dispatch" if measured["start_ns"] >= dispatch else "crosses_dispatch")
        rows.append(row)
        raw[key] = requests
    report = {"completed_points": len(rows), "expected_points": len(expected),
              "complete": set(selected) == set(expected), "failed_attempts": failed,
              "missing": [list(key) for key in sorted(set(expected) - set(selected))],
              "strict_original_config_match": False,
              "matched_code": len({json.dumps(p["code_hashes"], sort_keys=True) for p in provenance}) <= 1,
              "matched_versions": len({json.dumps(p["versions"], sort_keys=True) for p in provenance}) <= 1,
              **execution_checks(provenance, schedule),
              "execution_schedule": schedule,
              "compatibility_deviations": ["unquantized Qwen2.5-3B", "FP16 KV", "SGLang 0.4.6.post5 native operators"],
              "rows": rows}
    report["complete"] = (report["complete"] and report["matched_code"] and report["matched_versions"]
                          and report["gpu_assignment_verified"])
    return report, raw


def plots(rows, raw, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("PingFang SC", "Heiti TC", "Noto Sans CJK SC", "Arial Unicode MS") if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#242424", "text.color": "#242424", "font.size": 11})
    # 硬性两色上限：S0中性灰，S1蓝，S2橙；标记和线型保证灰度可分。
    styles = {"S0": ("#616161", "o", "-"), "S1": ("#2774AE", "s", "--"), "S2": ("#D27B2A", "^", "-.")}
    output.mkdir(parents=True, exist_ok=True)
    lookup = {(r["baseline"], r["point"]): r for r in rows}
    for axis_name, names, x_key, label in [
        ("qps", ["prefix_2k_q025", "prefix_2k_q05", "prefix_2k_q1"], "qps", "计划到达率（QPS）"),
        ("context", ["prefix_1k_q1", "prefix_2k_q1", "prefix_4k_q1"], "actual_isl_mean", "实际平均输入长度（tokens）"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for ax, key, title in zip(axes, ["ttft_ms_p50", "itl_ms_p50"], ["TTFT 中位数", "ITL 中位数"]):
            for baseline, (color, marker, line) in styles.items():
                selected = [lookup[(baseline, name)] for name in names if (baseline, name) in lookup]
                if selected:
                    ax.plot([r[x_key] for r in selected], [r[key] for r in selected], color=color,
                            marker=marker, linestyle=line, linewidth=1.6, label=baseline)
            ax.set(xlabel=label, ylabel="毫秒", title=title, ylim=(0, None))
            ax.grid(axis="y", color="#e8e8e8")
            ax.legend(frameon=False)
        fig.suptitle(f"{'QPS' if axis_name == 'qps' else '输入长度'}与延迟：V100 兼容复现\n"
                     "共享前缀；每点100请求；OSL128；并发上限4；seed42；仅连接已测离散点", fontsize=12)
        fig.savefig(output / f"{axis_name}_latency.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, metric, title in zip(axes, ("ttft_s", "itls_s"), ("TTFT", "ITL")):
        series = []
        for baseline, (color, marker, line) in styles.items():
            requests = raw.get((baseline, "pressure_4k_q1"), [])
            values = sorted(v * 1000 for r in requests for v in (r[metric] if metric == "itls_s" else [r[metric]]))
            if values:
                series.append((values, color, line))
                ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post",
                        color=color, linestyle=line, linewidth=1.6, label=f"{baseline}，n={len(values)}")
        ax.set(xlabel=f"{title}（毫秒）", ylabel="经验累计比例", ylim=(0, 1), title=f"{title} 经验 CDF")
        ax.grid(color="#e8e8e8")
        ax.legend(frameon=False, fontsize=9)
        if metric == "itls_s" and series:
            # 完整横轴保留极端长间隔；插图放大主体，避免把差异挤成一条竖线。
            inset = ax.inset_axes([.38, .16, .57, .52])
            for values, color, line in series:
                inset.step(values, [(i + 1) / len(values) for i in range(len(values))],
                           where="post", color=color, linestyle=line, linewidth=1.1)
            lo = min(quantile(v, .01) for v, _, _ in series)
            hi = max(quantile(v, .99) for v, _, _ in series)
            inset.set(xlim=(lo * .95, hi * 1.05), ylim=(0, 1), title="ITL主体放大（毫秒）")
            inset.title.set_fontsize(9)
            inset.tick_params(labelsize=8)
            inset.grid(color="#e8e8e8")
    fig.suptitle("压力点延迟分布：128请求／组，32个共享前缀，QPS1\nITL样本按流式块分摊；同一次请求内的样本不独立", fontsize=12)
    fig.savefig(output / "pressure_latency_cdf.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    core = ["random_2k_q1", "prefix_2k_q1", "pressure_4k_q1"]
    for ax, metric, title, unit in zip(axes, ["cache_hit_ratio", "output_token_throughput"],
                                     ["前缀缓存命中比例", "输出 token 吞吐"], ["命中tokens／输入tokens", "tokens／秒"]):
        for i, (baseline, (color, _, _)) in enumerate(styles.items()):
            values = [lookup.get((baseline, name), {}).get(metric, float("nan")) for name in core]
            ax.bar([x + (i - 1) * .25 for x in range(3)], values, width=.23, color=color,
                   edgecolor="#333333", linewidth=.6, label=baseline, hatch=["", "//", ".."][i])
        ax.set_xticks(range(3), ["随机2K", "共享前缀2K", "压力4K"])
        ax.set(ylabel=unit, title=title, ylim=(0, None))
        ax.grid(axis="y", color="#e8e8e8")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    fig.suptitle("核心9点缓存与吞吐对照\n每组100／100／128请求；同输入、同到达序列；包含排队效应", fontsize=12)
    fig.savefig(output / "core_cache_throughput.png", dpi=180)
    plt.close(fig)


def write_report(report, output):
    def fmt(value, digits=2):
        return "未测量" if value is None else f"{value:.{digits}f}"

    rows = report["rows"]
    lookup = {(r["baseline"], r["point"]): r for r in rows}
    lines = [
        "# V100 Benchmark 1 兼容复现实验记录", "", "## Material Passport", "",
        "- Origin Skill: experiment-agent", "- Origin Mode: validate", "- Origin Date: 2026-09-10",
        "- Verification Status: ANALYZED", "- Version Label: v100_benchmark1_v1", "",
        "已从原始记录复算和校验；尚未进行第二次独立GPU重跑，不标为VERIFIED。", "",
        "## 完成状态", "",
        f"已完成 {report['completed_points']}/{report['expected_points']} 个正式测点。"
        + ("V100兼容测点数据已收齐；实验条件限制见下文。" if report["complete"] else "当前为阶段性结果，未完成项不得按0计入比较。"),
        "", "框架为SGLang 0.4.6.post5，未量化Qwen2.5-3B-Instruct，FP16权重/KV，native算子适配。"
        "这不是原文量化模型/BF16配置的严格复现。", "",
        f"同代码检查：{report['matched_code']}；同GPU检查：{report['matched_gpu']}。"
        "每个baseline复用相同输入SHA256和Poisson到达序列。", "",
        f"执行方式：{report.get('execution_mode', '未记录')}；"
        f"同型号/容量显卡检查：{report.get('matched_gpu_model', '未记录')}。",
        "使用三张同型号、同规格的Tesla V100-SXM2-32GB，每组独占一张GPU，三组并行运行，TP均为1。"
        "共享主机CPU、内存和I/O；S0较早启动，前3点先于并行派发完成。"
        if report.get("execution_mode") == "same_model_different_gpus_parallel" else
        "未记录并行派发；不推定已经验证主机资源独占。", "",
        "## 核心9点", "",
        "| 负载 | 配置 | 完成 | TTFT p50/p95（ms） | ITL p50（ms） | E2E p50（s） | 输出tokens/s | 命中比例 | H→D（MB） |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point, label in [("random_2k_q1", "随机2K"), ("prefix_2k_q1", "共享2K"), ("pressure_4k_q1", "压力4K")]:
        for baseline in ("S0", "S1", "S2"):
            row = lookup.get((baseline, point))
            if row:
                lines.append(f"| {label} | {baseline} | {row['completed']} | {fmt(row['ttft_ms_p50'])}/{fmt(row['ttft_ms_p95'])} | "
                             f"{fmt(row['itl_ms_p50'])} | {fmt(row['e2e_ms_p50']/1000)} | {fmt(row['output_token_throughput'])} | "
                             f"{row['cache_hit_ratio']:.1%} | {row['cpu_to_gpu_bytes']/1e6:.2f} |")
            else:
                lines.append(f"| {label} | {baseline} | 未完成 | — | — | — | — | — | — |")
    lines += ["", "## 分层传输", "",
              "以下为实际payload拷贝；H→D按层、D→H按打包批次记录，操作粒度不同。"
              "带宽用payload/阻塞拷贝调用墙钟时间计算，包含运行时与等待，非裸PCIe峰值。", "",
              "| S2测点 | D→H MB | H→D MB | D→H GB/s | H→D GB/s | host回载tokens | 驱逐tokens |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["baseline"] == "S2":
            lines.append(f"| {row['point']} | {row['gpu_to_cpu_bytes']/1e6:.2f} | {row['cpu_to_gpu_bytes']/1e6:.2f} | "
                         f"{fmt(row['gpu_to_cpu_GBps_by_host_call_s'])} | {fmt(row['cpu_to_gpu_GBps_by_host_call_s'])} | "
                         f"{row['host_reuse_tokens']} | {row['evicted_tokens']} |")
    lines += ["", "## 解释边界", "",
              "- 每点4次warmup后清空缓存，正式OSL128、并发上限4，单一seed42。",
              "- TTFT/E2E主列从实际HTTP发送起算；CSV同时保留客户端等待和从计划到达起算的offered延迟。",
              "- SSE块可能携带多token，ITL按块间隔分摊；同请求内样本不独立，不能做虚假的显著性检验。",
              "- 目标ISL来自官方生成器参数；实际输入重新分词后略有变化，CSV和长度图使用实测token数。",
              "- GPU/host缓存使用率来自每批次采样，均值不是严格时间加权均值。",
              "- prefill/decode CUDA事件计时为批次跨度，包含流内等待和CPU发射间隙，不是独立request或纯kernel耗时。",
              "- 未配置SLO，因此不计算goodput；纯设备队列时间、load stall、overlap ratio等未独立测量。",
              "- 没有跨seed重复、置信区间或独立重跑，不能声称普遍提升或统计显著。",
              "- S2使用默认write_through_selective。没有发生回载的测点照实报告0，不修改策略以获得正收益。", "",
              "## 原始记录与图表", "",
              "[完整汇总CSV](summary.csv) · [状态与完整数值JSON](summary.json)", ""]
    for name in ("qps_latency", "context_latency", "pressure_latency_cdf", "core_cache_throughput"):
        path = output / "figures" / f"{name}.png"
        if path.exists():
            lines += [f"![{name}](figures/{name}.png)", ""]
    lines += ["## 缺失与失败", "", "```json", json.dumps({"missing": report["missing"],
              "failed_attempts": report["failed_attempts"]}, ensure_ascii=False, indent=2), "```", ""]
    (output / "实验记录.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    report, raw = collect(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if report["rows"]:
        with (args.output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            writer.writerows(report["rows"])
    if args.plot:
        plots(report["rows"], raw, args.output / "figures")
    write_report(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
