"""不需要GPU：校验交付文件SHA256、测量源码、数据集和全部正式结果。"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_v100_experiment import collect


def verify(root):
    root = root.resolve()
    manifest = root / 'SHA256SUMS.json'
    checked = 0
    if not manifest.exists():
        raise ValueError('缺少SHA256SUMS.json')
    for name, digest in json.loads(manifest.read_text()).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'非法文件路径：{name}')
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest, name
        checked += 1
    report, _ = collect(root / 'raw')
    assert report['complete'] and not report['failed_attempts']
    for env in (root / 'raw/runs/submission').glob('*/*/environment.json'):
        saved = json.loads(env.read_text())
        status = json.loads((env.parent / 'status.json').read_text())
        assert status['status'] == 'completed' and len(status['points']) == 7
        for name, digest in saved['code_hashes'].items():
            with (root / 'scripts' / name).open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest, name
    expected = json.loads((root / 'results/summary.json').read_text())
    assert report == expected, '重新汇总与交付summary.json不一致'
    return {'status': 'passed', 'files_sha256_checked': checked,
            'points': report['completed_points'], 'requests': sum(r['completed'] for r in report['rows']),
            'measurement_code_matches': True, 'raw_reanalysis_matches': True,
            'gpu_rerun_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
