"""交付入口的CPU检查，不连接服务器或使用GPU。"""
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_rerun_defaults_to_read_only_plan_and_rejects_existing_root(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/reproduce_v100.py'
    command = [sys.executable, '-B', str(script), '--work-root', str(tmp_path/'new'),
               '--model', str(tmp_path/'absent-model')]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert not json.loads(result.stdout)['execute']
    assert not (tmp_path/'new').exists()
    (tmp_path/'new').mkdir()
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0 and '禁止覆盖' in rejected.stderr


def test_analysis_rejects_changed_input_dataset(tmp_path):
    scripts = Path(__file__).parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    try:
        from analyze_v100_experiment import collect
        (tmp_path/'datasets').mkdir()
        (tmp_path/'manifest.json').write_text(json.dumps({'points':[{'name':'x','input_sha256':'incorrect'}]}))
        (tmp_path/'datasets/x.json').write_text(json.dumps({'rows':[{'text':'changed'}], 'input_sha256':'incorrect'}))
        with pytest.raises(AssertionError, match='SHA256'):
            collect(tmp_path)
    finally:
        sys.path.remove(str(scripts))
