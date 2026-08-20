"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np

GIB = 1024 ** 3

NRRD_SPACE = "left-posterior-superior"

SCRIPT_VERSION = '17.0.7'

SCRIPT_VERSION_COMPACT = '1707'

SCRIPT_BASENAME = f'GPT-5.6-Sol-Pro_v{SCRIPT_VERSION}_SLURM.py'

OUTPUT_NRRD_PREFIX = ''

LEGACY_OUTPUT_NRRD_PREFIX = 'HW_'

def variant_nrrd_stem(stem: object) -> str:
    """Return an unprefixed NRRD/manifest stem.

    v16.1.3 removes the HW_ output namespace. Repeated legacy prefixes are stripped so
    caller-supplied stems and resumed metadata cannot recreate the retired prefix.
    """
    raw = str(stem)
    while raw.startswith(LEGACY_OUTPUT_NRRD_PREFIX):
        raw = raw[len(LEGACY_OUTPUT_NRRD_PREFIX):]
    return raw

RADIAL_TEXTURE_VARIANT_LABEL = 'pure hardware-linear texture'

def _parse_angles(
    values: Sequence[str] | str | float | int | None,
) -> List[float]:
    """Accept comma-separated, whitespace-separated, or mixed angle values."""
    return _parse_float_list(values)

def resolve_tta_angles(
    values: Sequence[str] | str | float | int | None,
) -> List[float]:
    """Return finite, modulo-360, unique TTA angles in request order.

    v16.4.0 treats each returned value as a separate view variant. Equivalent rotations
    such as 0, 360, and -360 are rejected instead of scheduling duplicate work whose
    binary union cannot carry weighting semantics.
    """
    raw_angles = _parse_angles(values) or [0.0, 120.0, 240.0]
    resolved: List[float] = []
    seen: Dict[int, float] = {}
    for raw in raw_angles:
        angle = float(raw)
        if not math.isfinite(angle):
            raise ValueError(f'--angle values must be finite; got {raw!r}')
        normalized = float(angle % 360.0)
        if math.isclose(normalized, 360.0, rel_tol=0.0, abs_tol=1e-9) or math.isclose(
            normalized, 0.0, rel_tol=0.0, abs_tol=1e-9
        ):
            normalized = 0.0
        # Quantize only for duplicate detection; retain the normalized float itself.
        duplicate_key = int(round(normalized * 1_000_000_000.0))
        if duplicate_key in seen:
            raise ValueError(
                f'--angle contains equivalent rotations {seen[duplicate_key]:g} and {angle:g}; '
                'each TTA view variant must be geometrically unique modulo 360 degrees'
            )
        seen[duplicate_key] = angle
        resolved.append(normalized)
    return resolved

def _parse_token_list(values: Sequence[str] | str | None) -> List[str]:
    """Accept comma and/or whitespace separated string tokens."""
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else [str(v) for v in values]
    parts: List[str] = []
    for raw in raw_values:
        raw = str(raw).strip()
        if not raw:
            continue
        parts.extend([p for p in re.split(r"[,\s]+", raw) if p])
    return parts

SAVE_OPTION_TOKENS: Tuple[str, ...] = (
    'images',
    'labels',
    'binary',
    'low_quality',
    'nrrd',
    'voxel_volume',
    'high_quality',
    'summary',
)

@dataclass(frozen=True)
class SaveRequest:
    """Canonical ``--save`` selection plus embedded low-quality downbins."""

    options: Tuple[str, ...]
    low_quality_downbins: Tuple[str, ...] = ()

def resolve_save_request(values: Sequence[str] | str | None) -> SaveRequest:
    """Validate ``--save`` and parse ``low_quality[:DOWNBIN[,DOWNBIN...]]``.

    Commas normally separate output tokens.  Once a ``low_quality:`` token starts,
    following numeric comma/whitespace fields remain part of its downbin list until the
    next recognized output token.  This permits, for example, ``--save
    images,low_quality:0.5,1024 nrrd`` without reintroducing a second low-quality flag.
    """
    if values is None:
        return SaveRequest(())

    raw_values = [values] if isinstance(values, str) else [str(v) for v in values]
    pieces: List[str] = []
    for raw_value in raw_values:
        for whitespace_group in re.split(r"\s+", str(raw_value).strip()):
            if not whitespace_group:
                continue
            pieces.extend(part.strip() for part in whitespace_group.split(',') if part.strip())

    valid = set(SAVE_OPTION_TOKENS)
    resolved: List[str] = []
    low_quality_downbins: List[str] = []
    collecting_low_quality = False

    for raw in pieces:
        lowered = str(raw).strip().lower()
        if lowered.startswith('low_quality:'):
            option, payload = lowered.split(':', 1)
            if option != 'low_quality' or not payload:
                raise ValueError(
                    '--save low_quality uses low_quality[:LOW_QUALITY_DOWNBIN]; '
                    f'got {raw!r}'
                )
            if 'low_quality' not in resolved:
                resolved.append('low_quality')
            low_quality_downbins.append(payload)
            collecting_low_quality = True
            continue

        if lowered in valid:
            collecting_low_quality = False
            if lowered not in resolved:
                resolved.append(lowered)
            continue

        if collecting_low_quality:
            # Numeric validation and canonical rounding happen after source geometry is
            # known in ``resolve_low_quality_downbin_specs``.  Reject obviously nonnumeric
            # fields here so a misspelled output token cannot silently become a downbin.
            try:
                float(lowered)
            except Exception as exc:
                expected = ', '.join(SAVE_OPTION_TOKENS)
                raise ValueError(
                    f'--save values must be one or more of: {expected}; '
                    f'got {raw!r}'
                ) from exc
            low_quality_downbins.append(lowered)
            continue

        expected = ', '.join(SAVE_OPTION_TOKENS)
        raise ValueError(
            f'--save values must be one or more of: {expected}; got {raw!r}'
        )

    return SaveRequest(
        options=tuple(resolved),
        low_quality_downbins=tuple(low_quality_downbins),
    )

def resolve_save_options(values: Sequence[str] | str | None) -> List[str]:
    """Compatibility helper returning only canonical output option names."""
    return list(resolve_save_request(values).options)

@dataclass(frozen=True)
class PostprocessingRequest:
    """Resolved user-facing final-volume postprocessing selection."""

    keep_objects: int = 0
    enable_3d_void_fill: bool = False
    gaussian_smoothing_enabled: bool = False
    gaussian_sigma: float = 3.0
    gaussian_passes: int = 1

def resolve_postprocessing_options(
    values: Sequence[str] | str | None,
) -> PostprocessingRequest:
    """Resolve ``--postprocessing`` structured tokens without legacy flag aliases."""
    keep_objects = 0
    enable_3d_void_fill = False
    gaussian_enabled = False
    gaussian_sigma = 3.0
    gaussian_passes = 1
    seen: set[str] = set()

    for raw in _parse_token_list(values):
        slots = [part.strip() for part in str(raw).split(':')]
        token = slots[0].lower()
        if token not in {'keep_objects', '3d_void_fill', 'gaussian_smoothing'}:
            raise ValueError(
                '--postprocessing accepts keep_objects[:NUMBER_OF_OBJECTS], '
                '3d_void_fill, and gaussian_smoothing[:STANDARD_DEVIATION]'
                '[:SMOOTHING_PASSES]; '
                f'got {raw!r}'
            )
        if token in seen:
            raise ValueError(f'--postprocessing contains duplicate option {token!r}')
        seen.add(token)

        if token == 'keep_objects':
            if len(slots) > 2 or (len(slots) == 2 and not slots[1]):
                raise ValueError(
                    f'--postprocessing {raw!r} must use keep_objects[:NUMBER_OF_OBJECTS]'
                )
            try:
                keep_objects = int(slots[1]) if len(slots) == 2 else 1
            except Exception as exc:
                raise ValueError(
                    f'--postprocessing {raw!r} has a non-integer NUMBER_OF_OBJECTS'
                ) from exc
            if keep_objects < 1:
                raise ValueError(
                    '--postprocessing keep_objects requires NUMBER_OF_OBJECTS >= 1'
                )
            continue

        if token == '3d_void_fill':
            if len(slots) != 1:
                raise ValueError(
                    '--postprocessing 3d_void_fill does not accept parameters'
                )
            enable_3d_void_fill = True
            continue

        if len(slots) > 3:
            raise ValueError(
                f'--postprocessing {raw!r} must use '
                'gaussian_smoothing[:STANDARD_DEVIATION][:SMOOTHING_PASSES]'
            )
        # Empty optional slots retain their documented defaults, so
        # ``gaussian_smoothing::2`` requests two passes at sigma 3.
        try:
            gaussian_sigma = (
                float(slots[1]) if len(slots) >= 2 and slots[1] else 3.0
            )
            gaussian_passes = (
                int(slots[2]) if len(slots) >= 3 and slots[2] else 1
            )
        except Exception as exc:
            raise ValueError(
                f'--postprocessing {raw!r} has an invalid Gaussian parameter'
            ) from exc
        if not math.isfinite(gaussian_sigma) or gaussian_sigma <= 0.0:
            raise ValueError(
                '--postprocessing gaussian_smoothing requires STANDARD_DEVIATION > 0'
            )
        if gaussian_passes < 1:
            raise ValueError(
                '--postprocessing gaussian_smoothing requires SMOOTHING_PASSES >= 1'
            )
        gaussian_enabled = True

    return PostprocessingRequest(
        keep_objects=int(keep_objects),
        enable_3d_void_fill=bool(enable_3d_void_fill),
        gaussian_smoothing_enabled=bool(gaussian_enabled),
        gaussian_sigma=float(gaussian_sigma),
        gaussian_passes=int(gaussian_passes),
    )

CARTESIAN_VIEW_TOKENS: Tuple[str, ...] = ('transverse', 'sagittal', 'coronal')

RADIAL_VIEW_TOKENS: Tuple[str, ...] = (
    'transverse', 'sagittal', 'coronal',
    'tilted_transverse', 'tilted_sagittal', 'tilted_coronal',
)

TILT_DIRECTION_TOKENS: Tuple[str, ...] = ('vertical', 'horizontal', 'both')

@dataclass(frozen=True)
class RadialViewRequest:
    """One Radial target paired with an explicit spacing or the per-view auto default."""

    view: str
    azimuth_angle: Optional[float] = None  # None means auto/full coverage.

@dataclass(frozen=True)
class TiltedViewGroup:
    """One structured ``--enable_tilted`` group before signed variants are expanded."""

    views: Tuple[str, ...]
    tilt_angles: Tuple[float, ...]
    tilt_directions: Tuple[str, ...]  # Canonicalized to vertical/horizontal.

def _parse_float_list(values: Sequence[str] | str | float | int | None) -> List[float]:
    """Accept comma and/or whitespace separated floating-point lists."""
    if values is None:
        return []
    if isinstance(values, (float, int)):
        return [float(values)]
    raw_values = [values] if isinstance(values, str) else [str(v) for v in values]
    parts: List[str] = []
    for raw in raw_values:
        raw = str(raw).strip()
        if not raw:
            continue
        parts.extend([p for p in re.split(r"[,\s]+", raw) if p])
    return [float(p) for p in parts]

def _resolve_unique_view_tokens(
    values: Sequence[str] | str | None,
    *,
    valid: Sequence[str],
    flag_name: str,
) -> List[str]:
    raw = [str(v).strip().lower() for v in _parse_token_list(values)]
    valid_set = set(str(v) for v in valid)
    out: List[str] = []
    for token in raw:
        if token not in valid_set:
            expected = ', '.join(str(v) for v in valid)
            raise ValueError(f'{flag_name} values must be one of: {expected}; got {token!r}')
        if token in out:
            raise ValueError(f'{flag_name} contains duplicate value {token!r}')
        out.append(token)
    return out

def resolve_cartesian_views(values: Sequence[str] | str | None) -> List[str]:
    """Resolve flat ``--enable_cartesian`` values without adding an implicit view."""
    return _resolve_unique_view_tokens(
        values,
        valid=CARTESIAN_VIEW_TOKENS,
        flag_name='--enable_cartesian',
    )

def _structured_group_values(
    values: Sequence[str] | str | None,
    *,
    flag_name: str,
) -> List[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else [str(v) for v in values]
    groups: List[str] = []
    for raw in raw_values:
        token = str(raw).strip()
        if not token:
            raise ValueError(f'{flag_name} contains an empty structured group')
        groups.append(token)
    return groups

def _split_structured_group(
    raw_group: str,
    *,
    slot_count: int,
    flag_name: str,
) -> List[str]:
    slots = [part.strip() for part in str(raw_group).split(':')]
    if len(slots) > int(slot_count):
        raise ValueError(
            f'{flag_name} group {raw_group!r} has {len(slots)} slots; '
            f'expected at most {int(slot_count)}'
        )
    slots.extend([''] * (int(slot_count) - len(slots)))
    return slots

def _parse_comma_slot(slot: str) -> List[str]:
    return [part.strip() for part in str(slot).split(',') if part.strip()]

def _resolve_tilt_direction_slot(slot: str, *, group: str) -> Tuple[str, ...]:
    tokens = [token.lower() for token in _parse_comma_slot(slot)] if str(slot).strip() else ['both']
    out: List[str] = []
    for token in tokens:
        if token not in TILT_DIRECTION_TOKENS:
            expected = ', '.join(TILT_DIRECTION_TOKENS)
            raise ValueError(
                f'--enable_tilted group {group!r} has invalid TILT_DIRECTION {token!r}; '
                f'expected one or more of: {expected}'
            )
        expanded = ('vertical', 'horizontal') if token == 'both' else (token,)
        for direction in expanded:
            if direction not in out:
                out.append(direction)
    return tuple(out)

def _resolve_tilt_angle_slot(slot: str, *, group: str) -> Tuple[float, ...]:
    raw_values = _parse_comma_slot(slot) if str(slot).strip() else ['30']
    out: List[float] = []
    seen: set[float] = set()
    for raw in raw_values:
        try:
            angle = float(raw)
        except Exception as exc:
            raise ValueError(
                f'--enable_tilted group {group!r} has invalid TILT_ANGLE {raw!r}'
            ) from exc
        if not math.isfinite(angle) or not (0.0 < float(angle) <= 45.0):
            raise ValueError(
                f'--enable_tilted group {group!r} requires every TILT_ANGLE to be '
                f'greater than 0 and less than or equal to 45; got {raw!r}'
            )
        if float(angle) not in seen:
            seen.add(float(angle))
            out.append(float(angle))
    return tuple(out)

def resolve_tilted_view_groups(
    values: Sequence[str] | str | None,
) -> List[TiltedViewGroup]:
    """Resolve ``VIEW:TILT_ANGLE:TILT_DIRECTION`` groups.

    Spaces separate groups; commas select multiple values inside a slot. Empty angle and
    direction slots default to 30 degrees and both directions, respectively.
    """
    groups: List[TiltedViewGroup] = []
    seen_groups: set[Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[str, ...]]] = set()
    for raw_group in _structured_group_values(values, flag_name='--enable_tilted'):
        view_slot, angle_slot, direction_slot = _split_structured_group(
            raw_group,
            slot_count=3,
            flag_name='--enable_tilted',
        )
        views = tuple(_resolve_unique_view_tokens(
            _parse_comma_slot(view_slot),
            valid=CARTESIAN_VIEW_TOKENS,
            flag_name=f'--enable_tilted group {raw_group!r} VIEW',
        ))
        if not views:
            raise ValueError(
                f'--enable_tilted group {raw_group!r} must specify at least one VIEW'
            )
        angles = _resolve_tilt_angle_slot(angle_slot, group=raw_group)
        directions = _resolve_tilt_direction_slot(direction_slot, group=raw_group)
        key = (views, angles, directions)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        groups.append(TiltedViewGroup(
            views=views,
            tilt_angles=angles,
            tilt_directions=directions,
        ))
    return groups

def tilted_group_base_views(groups: Sequence[TiltedViewGroup]) -> List[str]:
    """Return unique Tilted base views in first-request order."""
    out: List[str] = []
    for group in groups:
        for view in group.views:
            if view not in out:
                out.append(str(view))
    return out

def resolve_radial_view_requests(
    values: Sequence[str] | str | None,
) -> List[RadialViewRequest]:
    """Resolve ``VIEWS:AZIMUTH_ANGLE`` groups into one unambiguous request per view."""
    out: List[RadialViewRequest] = []
    seen: set[str] = set()
    for raw_group in _structured_group_values(values, flag_name='--enable_radial'):
        view_slot, angle_slot = _split_structured_group(
            raw_group,
            slot_count=2,
            flag_name='--enable_radial',
        )
        views = _resolve_unique_view_tokens(
            _parse_comma_slot(view_slot),
            valid=RADIAL_VIEW_TOKENS,
            flag_name=f'--enable_radial group {raw_group!r} VIEWS',
        )
        if not views:
            raise ValueError(
                f'--enable_radial group {raw_group!r} must specify at least one VIEW'
            )
        angle: Optional[float]
        angle_token = str(angle_slot).strip().lower()
        if angle_token in {'', 'auto'}:
            angle = None
        else:
            if len(_parse_comma_slot(angle_slot)) != 1:
                raise ValueError(
                    f'--enable_radial group {raw_group!r} accepts one AZIMUTH_ANGLE '
                    'shared by every VIEW in that group'
                )
            try:
                angle = float(angle_slot)
            except Exception as exc:
                raise ValueError(
                    f'--enable_radial group {raw_group!r} has invalid AZIMUTH_ANGLE '
                    f'{angle_slot!r}'
                ) from exc
            if not math.isfinite(float(angle)) or float(angle) <= 0.0:
                raise ValueError(
                    f'--enable_radial group {raw_group!r} requires AZIMUTH_ANGLE '
                    f'to be greater than 0 or omitted/auto; got {angle_slot!r}'
                )
        for view in views:
            if view in seen:
                raise ValueError(
                    f'--enable_radial assigns {view!r} more than once; each Radial VIEW '
                    'must have exactly one paired AZIMUTH_ANGLE'
                )
            seen.add(view)
            out.append(RadialViewRequest(view=str(view), azimuth_angle=angle))
    return out

@dataclass(frozen=True)
class ChannelFormat:
    """Canonical model-input channel layout for one center slice."""

    token: str
    kind: str  # gray, rgb, custom
    channel_count: int
    stride: int
    offsets: Tuple[int, ...]

DEFAULT_CHANNEL_FORMAT = ChannelFormat(
    token='gray',
    kind='gray',
    channel_count=1,
    stride=1,
    offsets=(0,),
)

_CUSTOM_CHANNEL_FORMAT_RE = re.compile(r'^C([1-9]\d*)S([1-9]\d*)$', re.IGNORECASE)

def resolve_channel_format(value: str | ChannelFormat | None) -> ChannelFormat:
    """Validate and canonicalize the single ``--channel_format`` value."""
    if isinstance(value, ChannelFormat):
        return value
    raw = 'gray' if value is None else str(value).strip()
    lowered = raw.lower()
    if lowered in {'gray', 'grey'}:
        return DEFAULT_CHANNEL_FORMAT
    if lowered == 'rgb':
        return ChannelFormat('RGB', 'rgb', 3, 1, (0, 0, 0))

    match = _CUSTOM_CHANNEL_FORMAT_RE.fullmatch(raw)
    if match is None:
        raise ValueError(
            f'Unsupported --channel_format value {raw!r}; use gray/grey, RGB, '
            'or C{odd_channel_count}S{stride>=1}, e.g. C5S1'
        )
    channel_count = int(match.group(1))
    stride = int(match.group(2))
    if channel_count % 2 == 0:
        raise ValueError(
            f'--channel_format {raw!r} is invalid: C must be odd, got C={channel_count}'
        )
    if stride < 1:  # Defensive; the regex already excludes zero and negative values.
        raise ValueError(
            f'--channel_format {raw!r} is invalid: S must be an integer >= 1, got S={stride}'
        )
    half = channel_count // 2
    offsets = tuple(position * stride for position in range(-half, half + 1))
    return ChannelFormat(
        token=f'C{channel_count}S{stride}',
        kind='custom',
        channel_count=channel_count,
        stride=stride,
        offsets=offsets,
    )

QUANTIZE_ALIASES: Dict[str, int | str] = {
    '8': 8,
    '16': 16,
    '32': 32,
    'int8': 8,
    'fp16': 16,
    'fp32': 32,
    'w8a8': 8,
    'w16a16': 16,
    'w32a32': 32,
    'w8a16': 'w8a16',
    'w8a32': 'w8a32',
}

def resolve_quantize(value: object) -> int | str | None:
    """Canonicalize the unified Ultralytics inference-precision setting."""
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {'', 'none', 'null', 'default'}:
        return None
    resolved = QUANTIZE_ALIASES.get(token)
    if resolved is None:
        valid = ', '.join(repr(v) for v in QUANTIZE_ALIASES)
        raise ValueError(f'unsupported --quantize value {value!r}; expected one of {valid}, or none')
    return resolved

def _parse_quantize_arg(value: str) -> int | str | None:
    try:
        return resolve_quantize(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

def quantize_uses_fp16(value: object) -> bool:
    """True only for the FP16 inference scheme; exported INT8 backends own their binding dtype."""
    return resolve_quantize(value) == 16

def quantize_display(value: object) -> str:
    resolved = resolve_quantize(value)
    return 'auto' if resolved is None else str(resolved)

@dataclass(frozen=True)
class BackendModelSelection:
    """Tagged GPU/CPU model artifacts supplied through ``--model``."""

    gpu: Optional[str] = None
    cpu: Optional[str] = None

@dataclass(frozen=True)
class BackendDeviceSelection:
    """Logical CUDA devices plus the optional OpenVINO CPU backend."""

    gpu_devices: Tuple[str, ...] = ()
    cpu: bool = False

@dataclass(frozen=True)
class BackendPrecisionSelection:
    """Resolved runtime precision for each selected inference backend."""

    gpu: int | str | None = None
    cpu: str = 'auto'

@dataclass(frozen=True)
class BackendBatchSelection:
    """Static model batch requested independently for GPU and CPU."""

    gpu: int = 1
    cpu: int = 1

def _cli_value_tokens(values: Sequence[str] | str | None) -> List[str]:
    """Return argparse values without splitting a quoted model path containing spaces."""
    if values is None:
        return []
    if isinstance(values, str):
        try:
            return [str(token) for token in shlex.split(values) if str(token).strip()]
        except Exception:
            return [token for token in str(values).split() if token]
    return [str(value).strip() for value in values if str(value).strip()]

def resolve_backend_models(values: Sequence[str] | str | None) -> BackendModelSelection:
    """Resolve order-independent ``gpu:PATH`` and ``cpu:PATH`` model entries."""
    resolved: Dict[str, str] = {}
    for raw in _cli_value_tokens(values):
        backend, sep, payload = str(raw).partition(':')
        backend = backend.strip().lower()
        if not sep or backend not in {'gpu', 'cpu'} or not payload.strip():
            raise ValueError(
                '--model requires one or both tagged entries: gpu:/path/to/model '
                'cpu:/path/to/openvino'
            )
        if backend in resolved:
            raise ValueError(f'--model contains duplicate {backend}: entries')
        resolved[backend] = payload.strip()
    if not resolved:
        raise ValueError(
            '--model requires one or both tagged entries: gpu:/path/to/model '
            'cpu:/path/to/openvino'
        )
    return BackendModelSelection(gpu=resolved.get('gpu'), cpu=resolved.get('cpu'))

def _append_gpu_device_tokens(raw: str, output: List[str]) -> None:
    for token in re.split(r'[,\s]+', str(raw).strip().strip(',')):
        token = token.strip().strip(',')
        if not token:
            continue
        low = token.lower()
        if low.startswith('gpu:'):
            token = token.split(':', 1)[1]
            low = token.lower()
        if low.startswith('cuda:'):
            index = low.split(':', 1)[1]
        else:
            index = low
        if not index.isdigit():
            raise ValueError(
                f'--device GPU indexes must be non-negative integers; got {token!r}'
            )
        canonical = f'cuda:{int(index)}'
        if canonical not in output:
            output.append(canonical)

def resolve_backend_devices(values: Sequence[str] | str | None) -> BackendDeviceSelection:
    """Resolve ``GPU_INDEXES[:cpu]`` with both portions absent by default."""
    gpu_devices: List[str] = []
    cpu_enabled = False
    for raw_token in _cli_value_tokens(values):
        token = str(raw_token).strip()
        low = token.lower()
        if low in {'cpu', ':cpu'}:
            cpu_enabled = True
            continue
        if low.endswith(':cpu'):
            cpu_enabled = True
            token = token[:-4].rstrip(':,')
            if not token:
                continue
        _append_gpu_device_tokens(token, gpu_devices)
    if not gpu_devices and not cpu_enabled:
        raise ValueError(
            '--device must select at least one backend: GPU indexes (for example 0,1,2,3), '
            'cpu, or a hybrid value such as 0,1,2,3:cpu'
        )
    return BackendDeviceSelection(gpu_devices=tuple(gpu_devices), cpu=bool(cpu_enabled))

_CPU_PRECISION_ALIASES: Dict[str, str] = {
    'auto': 'auto', 'default': 'auto',
    'bf16': 'bf16', 'bfloat16': 'bf16',
    'fp32': 'fp32', 'f32': 'fp32', '32': 'fp32',
    'fp16': 'fp16', 'f16': 'fp16', '16': 'fp16',
    'int8': 'int8', 'i8': 'int8', '8': 'int8',
}

def _resolve_cpu_precision(value: object) -> str:
    token = str(value).strip().lower()
    resolved = _CPU_PRECISION_ALIASES.get(token)
    if resolved is None:
        expected = ', '.join(sorted(set(_CPU_PRECISION_ALIASES.values())))
        raise ValueError(f'unsupported CPU precision {value!r}; expected one of: {expected}')
    return resolved

def resolve_backend_precisions(
    values: Sequence[str] | str | None,
    devices: BackendDeviceSelection,
) -> BackendPrecisionSelection:
    """Resolve tagged ``gpu:... cpu:...`` precision values and one-backend shorthand."""
    tokens = _cli_value_tokens(values)
    tagged: Dict[str, str] = {}
    untagged: List[str] = []
    for raw in tokens:
        backend, sep, payload = str(raw).partition(':')
        if sep and backend.lower() in {'gpu', 'cpu'}:
            key = backend.lower()
            if key in tagged:
                raise ValueError(f'--quantize contains duplicate {key}: entries')
            if not payload.strip():
                raise ValueError(f'--quantize {key}: requires a precision value')
            tagged[key] = payload.strip()
        else:
            untagged.append(str(raw).strip())
    if tagged and untagged:
        raise ValueError('--quantize cannot mix tagged and untagged values')
    if 'gpu' in tagged and not devices.gpu_devices:
        raise ValueError('--quantize contains gpu: settings, but --device selected no GPU backend')
    if 'cpu' in tagged and not devices.cpu:
        raise ValueError('--quantize contains cpu: settings, but --device selected no CPU backend')
    if untagged:
        if len(untagged) != 1:
            raise ValueError('--quantize accepts one shorthand value or tagged gpu:/cpu: values')
        if devices.gpu_devices and devices.cpu:
            raise ValueError(
                'Hybrid --quantize values must be tagged, for example '
                '--quantize gpu:fp16 cpu:bf16'
            )
        tagged['gpu' if devices.gpu_devices else 'cpu'] = untagged[0]
    gpu_value: int | str | None = None
    if 'gpu' in tagged:
        gpu_token = tagged['gpu'].strip().lower()
        gpu_value = None if gpu_token in {'auto', 'default', 'none'} else resolve_quantize(gpu_token)
    cpu_value = _resolve_cpu_precision(tagged.get('cpu', 'auto'))
    return BackendPrecisionSelection(gpu=gpu_value, cpu=cpu_value)

def resolve_backend_batches(
    values: Sequence[str] | str | None,
    devices: BackendDeviceSelection,
) -> BackendBatchSelection:
    """Resolve tagged ``gpu:N cpu:N`` batch sizes and one-backend shorthand."""
    tokens = _cli_value_tokens(values)
    tagged: Dict[str, str] = {}
    untagged: List[str] = []
    for raw in tokens:
        backend, sep, payload = str(raw).partition(':')
        if sep and backend.lower() in {'gpu', 'cpu'}:
            key = backend.lower()
            if key in tagged:
                raise ValueError(f'--batch contains duplicate {key}: entries')
            tagged[key] = payload.strip()
        else:
            untagged.append(str(raw).strip())
    if tagged and untagged:
        raise ValueError('--batch cannot mix tagged and untagged values')
    if 'gpu' in tagged and not devices.gpu_devices:
        raise ValueError('--batch contains gpu: settings, but --device selected no GPU backend')
    if 'cpu' in tagged and not devices.cpu:
        raise ValueError('--batch contains cpu: settings, but --device selected no CPU backend')
    if untagged:
        if len(untagged) != 1:
            raise ValueError('--batch accepts one shorthand value or tagged gpu:/cpu: values')
        if devices.gpu_devices and devices.cpu:
            raise ValueError(
                'Hybrid --batch values must be tagged, for example --batch gpu:1 cpu:1'
            )
        tagged['gpu' if devices.gpu_devices else 'cpu'] = untagged[0]

    def _one(key: str) -> int:
        raw = tagged.get(key, '1')
        try:
            value = int(raw)
        except Exception as exc:
            raise ValueError(f'--batch {key}:{raw} is not a positive integer') from exc
        if value < 1:
            raise ValueError(f'--batch {key}:{raw} must be >= 1')
        return int(value)

    return BackendBatchSelection(gpu=_one('gpu'), cpu=_one('cpu'))

def resolve_auto_positive_int(value: object, *, flag_name: str) -> Optional[int]:
    token = str(value).strip().lower()
    if token in {'', 'auto', 'default', 'none'}:
        return None
    try:
        resolved = int(token)
    except Exception as exc:
        raise ValueError(f'{flag_name} must be auto or a positive integer; got {value!r}') from exc
    if resolved < 1:
        raise ValueError(f'{flag_name} must be >= 1; got {resolved}')
    return int(resolved)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO segmentation TTA for large cylindrical video volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )

    p.add_argument("--input", required=True, type=str, help="Input video path")
    p.add_argument("--output", default=None, type=str, help="Output directory (default ./{Filename}/)")
    p.add_argument(
        "--temp",
        default=None,
        type=str,
        help=(
            "Scratch/temp root location. The supplied root is created when needed and receives a "
            "unique {Filename}_{PID}_temp run directory. Omission defaults to <output>/temp; "
            "an explicit YOLO_TTA_SCRATCH_PREFER_SHM setting may select a roomy memory-backed root"
        ),
    )
    p.add_argument(
        "--device", nargs="+", default=None, type=str, metavar="GPU_INDEXES[:cpu]",
        help=(
            "Inference backends. GPU indexes and cpu both default to absent. Use 0,1,2,3 "
            "for GPU-only, cpu for CPU-only, or 0,1,2,3:cpu for hybrid inference. GPU "
            "indexes are torch logical indexes into CUDA_VISIBLE_DEVICES"
        ),
    )
    p.add_argument(
        "--model", required=True, nargs="+", type=str, metavar="{gpu,cpu}:PATH",
        help=(
            "Tagged model artifacts. Supply gpu:/path/to/engine, cpu:/path/to/openvino, "
            "or both. The CPU artifact must be an ordinary raw-head OpenVINO segmentation "
            "IR, not an end-to-end/NMS-embedded export. Hybrid inference requires both "
            "entries; CPU and GPU artifacts are not verified to originate from identical weights"
        ),
    )
    p.add_argument(
        "--channel_format",
        default="gray",
        type=str,
        metavar="{gray,RGB,CxSy}",
        help=(
            "Model-input channel layout. gray/grey uses center slice N as one channel; "
            "RGB triplicates N into three channels; C{odd}S{stride>=1}, e.g. C5S1, "
            "uses neighboring view slices in ascending offset order. Radial and Tilted "
            "Radial indices wrap; Cartesian and Tilted Cartesian indices edge-clamp. "
            "Only one value is accepted and every prediction remains assigned to N"
        ),
    )

    p.add_argument("--imgsz", default=2048, type=int, help="Square input size used for inference")
    p.add_argument(
        "--batch", nargs="+", default=None, type=str, metavar="[{gpu,cpu}:]N",
        help=(
            "Backend-specific static model batch. Use gpu:1 cpu:1 in hybrid mode; a "
            "single untagged value is accepted for a single-backend run. Each source pads "
            "its final batch by repeating the last real slice and discards synthetic results"
        ),
    )
    p.add_argument("--conf", default=0.15, type=float, help="Passed to YOLO predict")
    p.add_argument("--min_conf", default=0.30, type=float,
                   help="Remove prediction-set objects whose combined confidence is below this threshold. 0 disables the check")
    p.add_argument(
        "--quantize", nargs="+", default=None, type=str, metavar="[{gpu,cpu}:]PRECISION",
        help=(
            "Backend-specific execution precision. GPU accepts auto/fp16/fp32 and exported "
            "quantized schemes; CPU accepts auto/bf16/fp32/fp16/int8. Hybrid values must be "
            "tagged, for example gpu:fp16 cpu:bf16"
        ),
    )
    p.add_argument(
        "--cpu_instances", default="auto", type=str, metavar="auto|N",
        help="Persistent OpenVINO model processes; auto creates one process per populated socket",
    )
    p.add_argument(
        "--cpu_threads", default="auto", type=str, metavar="auto|N",
        help="Whole-job OpenVINO inference thread budget, divided across CPU instances",
    )
    p.add_argument(
        "--cpu_streams", default="auto", type=str, metavar="auto|N",
        help="OpenVINO execution streams per socket-local CPU instance",
    )
    p.add_argument(
        "--cpu_infer_requests", default="auto", type=str, metavar="auto|N",
        help="Concurrent asynchronous OpenVINO infer requests per CPU instance",
    )

    p.add_argument(
        "--angle",
        nargs="+",
        default=["0,120,240"],
        type=str,
        metavar="DEG",
        help=(
            "Rotation angles in degrees for augmentation. Comma-separated, whitespace-separated, "
            "and mixed forms are accepted"
        ),
    )
    p.add_argument("--min_radius", default=0.0, type=float,
                   help="Remove objects whose radius is smaller than this value, measured on the YOLO output masks "
                        "in each prediction set's own native 2D slice plane, before backprojection, independently "
                        "per active view. 0 disables the check")
    p.add_argument(
        "--enable_cartesian",
        nargs="+",
        default=None,
        type=str,
        metavar="VIEW",
        help=(
            "Enable one or more Cartesian views: transverse, sagittal, coronal. "
            "This is a flat multivalue flag, so comma-separated, whitespace-separated, "
            "and mixed forms are accepted. No Cartesian view is enabled by default"
        ),
    )
    p.add_argument(
        "--enable_radial",
        nargs="+",
        default=None,
        type=str,
        metavar="VIEWS[:AZIMUTH_ANGLE]",
        help=(
            "Enable one or more structured Radial groups. VIEWS accepts comma-separated "
            "transverse, sagittal, coronal, tilted_transverse, tilted_sagittal, and "
            "tilted_coronal values. AZIMUTH_ANGLE is one positive degree spacing shared by "
            "every view in its group; omission or 'auto' selects the largest per-view "
            "full-coverage spacing. Spaces separate groups. Upright Radial targets do not "
            "require their Cartesian base, while tilted_* targets expand across every enabled "
            "Tilted variant of that base and are skipped with a log when none exist"
        ),
    )
    p.add_argument(
        "--enable_tilted",
        nargs="+",
        default=None,
        type=str,
        metavar="VIEW[:TILT_ANGLE[:TILT_DIRECTION]]",
        help=(
            "Enable one or more structured Tilted groups. VIEW accepts comma-separated "
            "transverse, sagittal, and coronal values. TILT_ANGLE accepts comma-separated "
            "positive values <=45 and defaults to 30; each creates positive and negative "
            "variants. TILT_DIRECTION accepts vertical, horizontal, both, or a comma-separated "
            "combination and defaults to both. Spaces separate groups. A Cartesian base need "
            "not be enabled"
        ),
    )
    p.add_argument(
        "--enable_tile",
        nargs="+",
        default=None,
        type=str,
        metavar="TILE_SIZE:TILE_STRIDE",
        help=(
            "Enable one or more structured dense-tile groups. TILE_SIZE is the side length in "
            "parent-view source pixels represented by one (--imgsz,--imgsz) inference range; "
            "smaller values increase magnification. TILE_STRIDE is the positive parent-view "
            "source-pixel step between adjacent tiles and must be <= TILE_SIZE; smaller values "
            "increase overlap and recall. Both slots are required and spaces separate groups"
        ),
    )

    p.add_argument(
        "--save",
        nargs="+",
        default=None,
        type=str,
        metavar="OUTPUT",
        help=(
            "Save one or more output groups: images (active-view image sequences; channel "
            "formats with at least five channels use one grayscale page per channel in multi-page TIFF), labels "
            "(final YOLO segmentation labels), binary (final TIFF sequence plus FFV1 MKV), "
            "low_quality[:LOW_QUALITY_DOWNBIN] (one or more isotropic presentation resolutions; "
            "for example low_quality:0.5,1024), nrrd "
            "(single-layer Slicer decomposition plus manifest), voxel_volume (native-space "
            "white-voxel count for the summary), high_quality (native-resolution final overlay), "
            "and summary (summary text file). Comma-separated, whitespace-separated, and mixed "
            "forms are accepted. No output group is selected by default; existing output paths "
            "and filenames are retained"
        ),
    )
    p.add_argument(
        "--postprocessing",
        nargs="+",
        default=None,
        type=str,
        metavar="OPERATION[:PARAMETERS]",
        help=(
            "Enable one or more final-volume operations: keep_objects[:NUMBER_OF_OBJECTS] "
            "(default N=1), 3d_void_fill, and "
            "gaussian_smoothing[:STANDARD_DEVIATION][:SMOOTHING_PASSES] "
            "(defaults sigma=3 and passes=1). No postprocessing operation is enabled by default"
        ),
    )
    p.add_argument("--centerline_filter_passes", default=0, type=int,
                   help="Maximum centerline-guided post-union passes. Pass 0 is the untouched audit checkpoint; 0 disables filtering and its audit NRRDs")
    p.add_argument("--centerline_filter_backend", default="embedded", choices=["embedded", "off"],
                   help="Centerline backend. embedded uses the in-script SciPy EDT medial-ridge tracker; off disables filtering")
    p.add_argument("--centerline_auto_remove", action="store_true",
                   help="Opt in to removing whole unprotected 2D components that satisfy every centerline, temporal, and backend-reliability guard. Protected connected anatomy remains marker-only; this flag never subtracts a watershed partition")
    p.add_argument("--centerline_radius_factor", default=2.5, type=float, metavar="X",
                   help="Flag foreground that reaches the circle of radius X times the local EDT medial-ridge radius in the strict tangent-normal 2D plane")
    p.add_argument("--centerline_temporal_context", default=8, type=int,
                   help="Clean source slices required on each side before an unprotected 2D component may be removed. Centerline anomaly runs themselves have no duration cap")
    p.add_argument("--centerline_surface_max_dim", default=512, type=int,
                   help="Maximum axis of the block-max-pooled foreground crop supplied to the centerline backend; bounds extraction without striding away thin branches")
    p.add_argument("--centerline_surface_points", default=5000, type=int,
                   help="Centerline complexity budget. The embedded backend permits up to approximately 4x this many raw ridge samples before its global safety cap")
    p.add_argument("--centerline_timeout", default=900.0, type=float,
                   help="Seconds allowed for each isolated embedded-centerline attempt before preserving the current union and using safe pass-through behavior")
    p.add_argument("--interpolation_distance", default=15, type=int,
                   help="Maximum view-native slice/frame distance used to search for interpolation candidates. Radial interpolation wraps around frame order. 0 disables interpolation")
    p.add_argument("--interpolation_walk_back", default=3, type=int,
                   help="Additional source slices to bridge before the endpoint slice. 0 disables walk-back bridges")
    p.add_argument("--interpolation_candidates", default=1, type=int,
                   help="Accept up to the Nth nearest interpolation candidate per endpoint projection")
    p.add_argument("--interpolation_passes", default=1, type=int,
                   help="Run the interpolation process this many passes, treating the previous pass as real")
    p.add_argument("--interpolation_min_radius", default=3, type=float,
                   help="Reject a candidate connection if the bridge radius is equal to, or smaller than, this value. 0 disables the check")
    p.add_argument("--interpolation_search_angle", default=15.0, type=float,
                   help="Projection growth angle in degrees. Must be greater than -90 and less than 90")

    return p

