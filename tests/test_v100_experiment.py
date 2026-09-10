"""V100 实验边界与协议的 CPU 单元测试，不导入 CUDA 或访问服务器。"""
import importlib.util
from pathlib import Path
import sys


def module():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("run_v100_experiment", scripts / "run_v100_experiment.py")
        result = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = result
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.remove(str(scripts))


def test_matrix_and_paired_arrivals():
    m = module()
    assert len(m.POINTS) == 7
    assert sum(p.n for p in m.POINTS) * 3 == 2184
    core = m.POINTS[:3]
    assert sum(p.n for p in core) * 3 == 984
    assert all(p.osl == 128 and p.concurrency == 4 for p in m.POINTS)
    a = m.arrivals(100, 1)
    assert a == m.arrivals(100, 1)
    assert a[0] == 0 and all(x < y for x, y in zip(a, a[1:]))
    assert m.arrivals(100, .25) == [4 * x for x in a]


def test_native_commands_and_default_policy():
    m = module()
    for baseline in ("S0", "S1", "S2"):
        cmd = m.server_command(Path("/tmp/launcher.py"), baseline)
        assert "bf16" not in cmd
        assert cmd[cmd.index("--max-total-tokens") + 1] == "28672"
        assert ("--disable-radix-cache" in cmd) == (baseline == "S0")
        assert ("--enable-hierarchical-cache" in cmd) == (baseline == "S2")
    assert "write_through_selective" in m.server_command(Path("/tmp/launcher.py"), "S2")


def test_hash_and_launcher_syntax():
    m = module()
    assert m.input_hash([{"text": "one"}]) != m.input_hash([{"text": "two"}])
    compile(m.LAUNCHER.replace('if __name__ == "__main__":',
                              'from v100_observer import install\ninstall()\nif __name__ == "__main__":'),
            "launcher", "exec")


def test_event_window_uses_operation_start(tmp_path):
    m = module()
    (tmp_path / "events.1.jsonl").write_text(
        '{"kind":"transfer","start_ns":20,"time_ns":100}\n'
        '{"kind":"host_reuse","time_ns":21}\n'
        '{"kind":"transfer","start_ns":5,"time_ns":25}\n')
    assert len(m.event_rows(tmp_path, 10, 30)) == 2


def test_analysis_quantiles_and_empty_measurements():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import analyze_v100_experiment as analysis
        assert analysis.quantile([1, 2, 3, 4], .5) == 2.5
        assert analysis.quantile([7], .99) == 7
        assert analysis.distribution([], "absent")["absent_mean"] is None
        assert analysis.distribution([.001], "latency_ms", 1000)["latency_ms_mean"] == 1
    finally:
        sys.path.remove(str(scripts))


def test_partial_report_does_not_claim_completion(tmp_path):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import analyze_v100_experiment as analysis
        analysis.write_report({"rows": [], "completed_points": 0, "expected_points": 21,
                               "complete": False, "matched_code": True, "matched_gpu": True,
                               "missing": [["S0", "random_2k_q1"]], "failed_attempts": []}, tmp_path)
        content = (tmp_path / "实验记录.md").read_text()
        assert "阶段性结果" in content
        assert "0/21" in content
        assert "兼容口径完整" not in content
    finally:
        sys.path.remove(str(scripts))


def test_parallel_assignment_and_busy_guard():
    import pytest
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import parallel_v100_experiments as parallel
        assert set(parallel.ASSIGNMENTS) == {"S1", "S2"}
        assert len({item[0] for item in parallel.ASSIGNMENTS.values()}) == 2
        assert len({item[2] for item in parallel.ASSIGNMENTS.values()}) == 2
        assert all(item[0] != "0" and item[2] != 30171 for item in parallel.ASSIGNMENTS.values())
        index, uuid, _ = parallel.ASSIGNMENTS["S1"]
        row = {"index": index, "uuid": uuid, "name": "Tesla V100-SXM2-32GB",
               "memory_used_mib": "0", "utilization_pct": "0"}
        assert parallel.validate_gpu([row], "", index, uuid) == (index, uuid)
        with pytest.raises(RuntimeError, match="已占用"):
            parallel.validate_gpu([row], uuid, index, uuid)
        with pytest.raises(RuntimeError, match="身份/型号"):
            parallel.validate_gpu([row], "", index, "GPU-wrong")
    finally:
        sys.path.remove(str(scripts))


def test_parallel_hardware_is_verified_without_claiming_same_gpu():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import analyze_v100_experiment as analysis
        schedule = {"mode": "same_model_different_gpus_parallel",
                    "existing_s0_gpu_uuid": "GPU-zero",
                    "assignments": {"S1": {"gpu_uuid": "GPU-one"}},
                    "gpu_inventory": [{"uuid": u, "name": "Tesla V100-SXM2-32GB",
                                       "memory_total_mib": "32768"} for u in ("GPU-zero", "GPU-one")]}
        provenance = [{"baseline": "S0", "gpu_uuid": "GPU-zero"},
                      {"baseline": "S1", "gpu_uuid": "GPU-one"}]
        checked = analysis.execution_checks(provenance, schedule)
        assert checked["gpu_assignment_verified"]
        assert checked["matched_gpu_model"]
        assert not checked["matched_gpu"]
        assert checked["resource_isolated_comparison"] is False
        schedule["gpu_inventory"][1]["name"] = "Different GPU"
        assert not analysis.execution_checks(provenance, schedule)["gpu_assignment_verified"]
        schedule["gpu_inventory"][1]["name"] = "Tesla V100-SXM2-32GB"
        provenance[1]["baseline"] = "S2"
        assert not analysis.execution_checks(provenance, schedule)["gpu_assignment_verified"]
        assert not analysis.execution_checks(provenance, {})["gpu_assignment_verified"]
    finally:
        sys.path.remove(str(scripts))
