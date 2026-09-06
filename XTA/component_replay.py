"""Persistent, checksummed inputs for isolated component-projection replay.

Capture is opt-in and bounded. It copies the immutable view-native CVOL before
its normal owner deletes it; already-projected NRRDs cannot recover this input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Mapping, Optional, Sequence
import uuid

from .geometry import ViewInfo, physical_view_name

_SCHEMA = 'xta-component-projection-replay.v1'
_FILES = ('meta.json', 'index.bin', 'chunks.bin')
_CHUNK_BYTES = 8 * 1024 * 1024
_LOCK = threading.Lock()
_CAPTURE = None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _descriptor_digest(value: dict) -> str:
    unsigned = {key: item for key, item in value.items() if key != 'descriptor_sha256'}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configure_component_replay_capture(
    root: Optional[Path],
    *,
    view_names: Sequence[str] = (),
    max_captures: int = 3,
    max_total_bytes: int = 4 * 1024**3,
    require_persistent: bool = True,
) -> None:
    """Enable bounded capture or disable it with None.

    Names/globs match either the exact runtime name or physical view name. Use
    three explicit selectors to obtain one costly transverse, sagittal and
    coronal view; a broad wildcard can otherwise fill the quota with one family.
    Completed captures already in this directory count against both quotas.
    ``require_persistent=False`` is intended for disposable local test fixtures.
    """
    global _CAPTURE
    if root is None:
        with _LOCK:
            _CAPTURE = None
        return
    root = Path(root).expanduser().resolve()
    if int(max_captures) <= 0 or int(max_total_bytes) <= 0:
        raise ValueError('Component replay capture limits must be positive')
    if require_persistent and os.name == 'posix':
        ephemeral = ('/tmp', '/var/tmp', '/dev/shm', '/run', '/mnt/localssd',
                     os.environ.get('TMPDIR', ''), os.environ.get('SLURM_TMPDIR', ''))
        if any(root.is_relative_to(Path(path).resolve()) for path in ephemeral if path):
            raise ValueError(f'Component replay capture must survive the job: {root}')
    root.mkdir(parents=True, exist_ok=True)
    count = used = 0
    captured_views = set()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        # Include abandoned private copies in byte accounting; never silently
        # delete evidence left by a killed controller.
        for name in _FILES:
            file = child/'input.cvol'/name
            if file.is_file():
                used += file.stat().st_size
        descriptor = child/'manifest.json'
        if descriptor.is_file():
            try:
                value = json.loads(descriptor.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if value.get('schema') == _SCHEMA:
                count += 1
                captured_views.add(str(value.get('view', {}).get('name', '')))
    with _LOCK:
        _CAPTURE = {'root': root, 'view_names': tuple(str(value) for value in view_names),
                    'max_captures': int(max_captures), 'max_total_bytes': int(max_total_bytes),
                    'count': count, 'bytes': used, 'captured_views': captured_views,
                    'completed': 0, 'skipped': 0}


def component_replay_capture_status() -> dict:
    with _LOCK:
        if _CAPTURE is None:
            return {'enabled': False}
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in _CAPTURE.items() if key != 'captured_views'} | {'enabled': True}


def capture_component_projection(
    component_store_path: Path,
    *,
    view: ViewInfo,
    out_shape_tyx: Sequence[int],
    added_voxels: int,
    layer_metadata: Mapping[str, object],
) -> Optional[Path]:
    """Copy one selected immutable component and atomically publish its descriptor."""
    with _LOCK:
        state = _CAPTURE
        if state is None or int(added_voxels) <= 0:
            return None
        selectors = state['view_names']
        if selectors and not any(fnmatchcase(name, pattern) for pattern in selectors
                                 for name in (str(view.name), physical_view_name(view))):
            return None
        if str(view.name) in state['captured_views']:
            return None
    source = Path(component_store_path).resolve()
    sizes = {name: (source/name).stat().st_size for name in _FILES}
    total_bytes = sum(sizes.values())
    output_shape = tuple(int(value) for value in out_shape_tyx)
    if len(output_shape) != 3 or min(output_shape) <= 0:
        raise ValueError('Component replay requires three positive output dimensions')
    metadata = json.loads(json.dumps(dict(layer_metadata), default=str))
    with _LOCK:
        if str(view.name) in state['captured_views']:
            return None
        if state['count'] >= state['max_captures'] or state['bytes'] + total_bytes > state['max_total_bytes']:
            state['skipped'] += 1
            return None
        state['count'] += 1
        state['bytes'] += total_bytes
        state['captured_views'].add(str(view.name))
    capture_id = f'{physical_view_name(view)}-{uuid.uuid4().hex[:12]}'
    capture_id = ''.join(char if char.isalnum() or char in '-_.' else '_' for char in capture_id)
    destination = state['root']/capture_id
    try:
        with tempfile.TemporaryDirectory(prefix='.capture-', dir=state['root']) as temporary:
            staging = Path(temporary)
            (staging/'input.cvol').mkdir()
            records = {}
            for name in _FILES:
                original = source/name
                before = original.stat()
                digest = hashlib.sha256()
                copied = 0
                with original.open('rb') as reader, (staging/'input.cvol'/name).open('wb') as writer:
                    while block := reader.read(_CHUNK_BYTES):
                        writer.write(block)
                        digest.update(block)
                        copied += len(block)
                after = original.stat()
                if copied != sizes[name] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f'Component store changed during immutable capture: {original}')
                records[f'input.cvol/{name}'] = {'bytes': copied, 'sha256': digest.hexdigest()}
            input_meta = json.loads((staging/'input.cvol'/'meta.json').read_text(encoding='utf-8'))
            descriptor = {'schema': _SCHEMA, 'capture_id': capture_id, 'captured_unix': time.time(),
                          'source_path_for_provenance_only': str(source), 'view': asdict(view),
                          'out_shape_tyx': output_shape, 'input_shape': input_meta['shape'],
                          'input_format': input_meta['format'], 'added_voxels': int(added_voxels),
                          'layer_metadata': metadata, 'files': records}
            descriptor['descriptor_sha256'] = _descriptor_digest(descriptor)
            (staging/'manifest.json').write_text(json.dumps(descriptor, indent=2)+'\n', encoding='utf-8')
            staging.rename(destination)
        with _LOCK:
            state['completed'] += 1
        print(f'Component projection replay captured: {destination} ({total_bytes / 1024**2:.1f} MiB)', flush=True)
        return destination
    except BaseException:
        with _LOCK:
            state['count'] -= 1
            state['bytes'] -= total_bytes
            state['captured_views'].discard(str(view.name))
        raise


@dataclass(frozen=True)
class ComponentProjectionReplay:
    root: Path
    source_path: Path
    view: ViewInfo
    output_shape: tuple[int, int, int]
    added_voxels: int
    layer_metadata: dict
    descriptor: dict


def load_component_replay(path: Path) -> ComponentProjectionReplay:
    """Validate the complete snapshot before interpreting geometry or reading masks."""
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    value = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if value.get('schema') != _SCHEMA or value.get('descriptor_sha256') != _descriptor_digest(value):
        raise ValueError('Component replay descriptor schema/checksum mismatch')
    expected = {f'input.cvol/{name}' for name in _FILES}
    if set(value['files']) != expected:
        raise ValueError('Component replay files must be exactly the three local CVOL files')
    for relative, record in value['files'].items():
        file = (root/relative).resolve()
        if not file.is_relative_to(root) or file.stat().st_size != int(record['bytes']) or _digest(file) != record['sha256']:
            raise ValueError(f'Component replay payload checksum mismatch: {relative}')
    view_fields = {field.name for field in fields(ViewInfo)}
    if set(value['view']) != view_fields:
        raise ValueError('Component replay ViewInfo fields differ from this runtime')
    view_data = dict(value['view'])
    view_data['azimuths_deg'] = tuple(float(angle) for angle in view_data['azimuths_deg'])
    view = ViewInfo(**view_data)
    output_shape = tuple(int(dimension) for dimension in value['out_shape_tyx'])
    if len(output_shape) != 3 or min(output_shape) <= 0:
        raise ValueError('Invalid replay output dimensions')
    input_meta = json.loads((root/'input.cvol'/'meta.json').read_text(encoding='utf-8'))
    if input_meta['shape'] != value['input_shape'] or input_meta['format'] != value['input_format']:
        raise ValueError('Component replay input geometry differs from descriptor')
    return ComponentProjectionReplay(root, root/'input.cvol', view, output_shape,
                                     int(value['added_voxels']), dict(value['layer_metadata']), value)
