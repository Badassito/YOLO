"""Build and verify a complete source bundle outside the source repository."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--readme', type=Path)
    parser.add_argument('--wheel', type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT):
        parser.error('Generated releases belong in task Scratch, outside the repository')
    tree = ast.parse((ROOT / 'XTA/__init__.py').read_text(encoding='utf-8'))
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(getattr(target, 'id', '') == '__version__' for target in node.targets))
    launcher = f'GPT-6-Astra-Ultra_v{version}_SLURM.py'
    paths = [ROOT / name for name in (launcher, 'ARCHITECTURE.md', 'pyproject.toml', 'setup.py', 'MANIFEST.in', '.gitignore')]
    for directory in ('XTA', 'tools', 'native', 'tests'):
        paths.extend(path for path in (ROOT / directory).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts
                     and path.suffix in {'.py', '.json', '.md', '.c', '.h', '.sh', '.ps1'})
    payloads = {}
    for path in sorted(paths):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Source path resolves outside the repository: {path}')
        payloads[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    if args.readme:
        payloads['READ_ME_FIRST.txt'] = args.readme.read_bytes()
    manifest = {'version': version, 'launcher': launcher,
                'files': {name: digest(data) for name, data in sorted(payloads.items())}}
    payloads['RELEASE_MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    if args.wheel:
        with zipfile.ZipFile(args.wheel) as wheel:
            assert wheel.testzip() is None
            package = {name for name in wheel.namelist() if name.startswith('XTA/') and name.endswith(('.py', '.json', '.md'))}
            expected = {name for name in payloads if name.startswith('XTA/')}
            assert package == expected, (package - expected, expected - package)
            for name in expected:
                assert wheel.read(name) == payloads[name], name
            launchers = [name for name in wheel.namelist() if name.endswith('_SLURM.py')]
            assert len(launchers) == 1 and launchers[0].endswith('/' + launcher), launchers
            assert wheel.read(launchers[0]) == payloads[launcher]
            metadata = wheel.read(f'xta-{version}.dist-info/METADATA').decode()
            assert f'Version: {version}' in metadata.splitlines()
            assert not any(name.endswith(('.pyc', '.nbc', '.nbi')) for name in wheel.namelist())
        print(f'Wheel verified against {len(expected)} current package files.')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'XTA_v{version}_complete_source.zip'
    prefix = f'XTA_v{version}/'
    with tempfile.NamedTemporaryFile(dir=output, prefix='.release-', suffix='.zip', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, 'w') as target:
            for name, data in sorted(payloads.items()):
                info = zipfile.ZipInfo(prefix + name, date_time=(1980, 1, 1, 0, 0, 0))
                target.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(temporary) as source:
            assert source.testzip() is None
            assert set(source.namelist()) == {prefix + name for name in payloads}
            for name, data in payloads.items():
                assert source.read(prefix + name) == data, name
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    for artifact in (archive, *((args.wheel,) if args.wheel else ())):
        checksum = digest(artifact.read_bytes())
        artifact.with_suffix(artifact.suffix + '.sha256').write_text(f'{checksum}  {artifact.name}\n', encoding='ascii')
        print(f'{artifact.name}: {artifact.stat().st_size:,} bytes; SHA256 {checksum}')
    print(f'Verified {len(payloads)} source members.')


if __name__ == '__main__':
    main()
