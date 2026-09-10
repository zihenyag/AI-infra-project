"""实验结束后补采环境、模型指纹和实际测量源码；不启动GPU服务。"""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path('/root/beamforming/CV/v100-probe')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    output = ROOT / 'formal/provenance'
    output.mkdir(exist_ok=False)
    versions = {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata['Name']}
    files = {}
    for path in sorted((ROOT / 'models/Qwen2.5-3B-Instruct').iterdir()):
        if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt'):
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    snapshot = {
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'collection_phase': 'post_run_read_only_inspection',
        'python': platform.python_version(), 'platform': platform.platform(),
        'model_id': 'Qwen/Qwen2.5-3B-Instruct', 'model_source': 'ModelScope',
        'download_revision': 'master; immutable upstream revision was not captured',
        'model_files': files, 'packages': dict(sorted(versions.items())),
        'environment_note': 'Independent venv; torch/triton borrowed read-only through v100_base.pth from existing SEDS site-packages. Not a fresh isolated reinstall test.',
    }
    commands = {
        'gpu': ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,power.limit', '--format=csv,noheader,nounits'],
        'topology': ['nvidia-smi', 'topo', '-m'], 'cpu': ['lscpu'], 'memory': ['free', '-b'],
    }
    for name, command in commands.items():
        run = subprocess.run(command, capture_output=True, text=True, timeout=30)
        snapshot[name] = {'command': command, 'returncode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}
    source = output / 'measurement_code'
    source.mkdir()
    for name in ('run_v100_experiment.py', 'v100_observer.py', 'smoke_v100.py', 'parallel_v100_experiments.py'):
        shutil.copy2(ROOT / name, source / name)
    upstream = metadata.distribution('sglang')
    snapshot['upstream_file_sha256'] = {}
    for relative in ('sglang/bench_serving.py', 'sglang/srt/mem_cache/radix_cache.py',
                     'sglang/srt/mem_cache/hiradix_cache.py', 'sglang/srt/mem_cache/memory_pool.py',
                     'sglang/srt/managers/cache_controller.py', 'sglang/srt/model_executor/model_runner.py'):
        path = Path(upstream.locate_file(relative))
        snapshot['upstream_file_sha256'][relative] = digest(path)
    (output / 'environment.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    (output / 'packages-snapshot.txt').write_text('\n'.join(f'{n}=={v}' for n,v in sorted(versions.items())) + '\n')
    print(json.dumps({'output': str(output), 'model_files': len(files), 'packages': len(versions)}))


if __name__ == '__main__':
    main()
