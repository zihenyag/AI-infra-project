"""封存指定交付目录：验代码/数据/文档链接，生成SHA256与ZIP并验CRC。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from verify_delivery import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve()
    archive=root.with_suffix('.zip')
    if archive.exists():
        parser.error('同名ZIP已存在，不自动覆盖')
    for path in sorted((root/'scripts').glob('*.py')):
        compile(path.read_text(),str(path),'exec')
    for path in [root/'README.md',root/'report/实验报告.md',*(root/'docs').glob('*.md')]:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('https://','http://','#')): continue
            resolved=(path.parent/target.split('#')[0]).resolve()
            assert resolved.is_relative_to(root) and resolved.exists(), (str(path),target)
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json']
    sums={}
    for p in files:
        assert not p.is_symlink()
        with p.open('rb') as handle:
            sums[str(p.relative_to(root))]=hashlib.file_digest(handle,'sha256').hexdigest()
    (root/'SHA256SUMS.json').write_text(json.dumps(sums,ensure_ascii=False,indent=2)+'\n')
    receipt=verify(root)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file(): z.write(p,Path(root.name)/p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'ZIP CRC失败'
        assert len(z.namelist()) == len(files)+1
    with archive.open('rb') as handle:
        archive_hash=hashlib.file_digest(handle,'sha256').hexdigest()
    receipt.update(archive=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=archive_hash,
                   total_file_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),zip_crc='passed')
    root.with_name(root.name+'-封包回执.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
