"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np
from ._deps import cv2

from .cuda_backend import gpu_worker_fused_preflight_specs


# Worker-local affinity state. Spawned accelerator workers initialize these values after
# applying their target-specific CPU placement; no parent-process state is inherited.
_GPU_WORKER_NUMA_PIN: Optional[set] = None
_GPU_WORKER_NUMA_FULL: Optional[set] = None

def _linux_cpu_feature_flags() -> set[str]:
    """Return the Linux CPU feature flags visible inside the current allocation."""
    flags: set[str] = set()
    try:
        for line in Path('/proc/cpuinfo').read_text(errors='replace').splitlines():
            key, sep, value = line.partition(':')
            if sep and key.strip().lower() in {'flags', 'features'}:
                flags.update(token.strip().lower() for token in value.split() if token.strip())
    except Exception:
        pass
    return flags

def _resolve_openvino_model_xml_path(model_path: str | Path) -> Path:
    """Resolve an OpenVINO IR directory/XML/BIN path without invoking Ultralytics."""
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() == '.xml':
            return path
        if path.suffix.lower() == '.bin' and path.with_suffix('.xml').is_file():
            return path.with_suffix('.xml')
        raise ValueError(
            f'OpenVINO CPU model must be an IR .xml file or an export directory; got {path}'
        )
    if not path.is_dir():
        raise FileNotFoundError(path)
    preferred = path / f'{path.name}.xml'
    if preferred.is_file():
        return preferred
    xml_files = sorted(candidate for candidate in path.glob('*.xml') if candidate.is_file())
    if len(xml_files) == 1:
        return xml_files[0]
    if not xml_files:
        raise FileNotFoundError(f'No OpenVINO IR .xml file exists under {path}')
    raise ValueError(
        f'OpenVINO export directory {path} contains multiple .xml files; pass one explicitly: '
        + ', '.join(str(candidate.name) for candidate in xml_files)
    )

def _openvino_partial_shape_values(port: object) -> Tuple[Optional[int], ...]:
    """Return static dimension values and ``None`` for dynamic dimensions."""
    partial_shape = getattr(port, 'partial_shape', None)
    if partial_shape is None:
        get_partial_shape = getattr(port, 'get_partial_shape', None)
        partial_shape = get_partial_shape() if callable(get_partial_shape) else None
    if partial_shape is None:
        try:
            return tuple(int(value) for value in getattr(port, 'shape'))
        except Exception:
            return ()
    values: List[Optional[int]] = []
    try:
        dimensions = list(partial_shape)
    except Exception:
        dimensions = []
    for dimension in dimensions:
        try:
            is_static = bool(getattr(dimension, 'is_static'))
        except Exception:
            is_static = False
        if is_static:
            try:
                values.append(int(dimension.get_length()))
                continue
            except Exception:
                try:
                    values.append(int(dimension))
                    continue
                except Exception:
                    pass
        values.append(None)
    return tuple(values)

def _openvino_port_name(port: object, fallback: str) -> str:
    for accessor in ('get_any_name',):
        fn = getattr(port, accessor, None)
        if callable(fn):
            try:
                name = str(fn())
                if name:
                    return name
            except Exception:
                pass
    try:
        name = str(getattr(port, 'any_name'))
        if name:
            return name
    except Exception:
        pass
    return str(fallback)

def _openvino_element_type_name(port: object) -> str:
    """Return a stable lower-case OpenVINO element-type token for one port."""
    element_type = None
    getter = getattr(port, 'get_element_type', None)
    if callable(getter):
        try:
            element_type = getter()
        except Exception:
            element_type = None
    if element_type is None:
        try:
            element_type = getattr(port, 'element_type')
        except Exception:
            element_type = None
    if element_type is None:
        return 'dynamic'
    type_name = getattr(element_type, 'get_type_name', None)
    if callable(type_name):
        try:
            token = str(type_name()).strip().lower()
            if token:
                return token
        except Exception:
            pass
    raw = str(element_type).strip().lower()
    match = re.search(r"['\"]([^'\"]+)['\"]", raw)
    return str(match.group(1) if match else raw).strip().lower()

def _openvino_export_class_count(model_xml: Path) -> Optional[int]:
    """Read Ultralytics export metadata when present; absence is not an error."""
    root = Path(model_xml).parent
    candidates = (
        root / 'metadata.yaml', root / 'metadata.yml', root / 'metadata.json',
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.suffix.lower() == '.json':
                metadata = json.loads(candidate.read_text())
            else:
                try:
                    import yaml  # type: ignore
                except Exception:
                    continue
                metadata = yaml.safe_load(candidate.read_text())
            names = metadata.get('names') if isinstance(metadata, dict) else None
            if isinstance(names, (dict, list, tuple)) and len(names) > 0:
                return int(len(names))
        except Exception:
            continue
    return None

def _openvino_model_has_int8_quantization(model: object, model_xml: Optional[Path] = None) -> bool:
    """Return True only when the supplied IR carries explicit INT8 quantization.

    OpenVINO's CPU capability list says what the processor/plugin *can* execute; it does
    not prove that a particular model was quantized. Ultralytics/NNCF INT8 IRs normally
    retain FakeQuantize (or explicit QuantizeLinear/DequantizeLinear) operations. Scan the
    graph first and use the XML only as a conservative compatibility fallback.
    """
    get_ops = getattr(model, 'get_ops', None)
    if callable(get_ops):
        try:
            for operation in get_ops():
                getter = getattr(operation, 'get_type_name', None)
                if callable(getter):
                    raw_name = getter()
                else:
                    raw_name = type(operation).__name__
                token = re.sub(r'[^a-z0-9]+', '', str(raw_name).lower())
                if token in {'fakequantize', 'quantizelinear', 'dequantizelinear'}:
                    return True
        except Exception:
            pass
    if model_xml is not None:
        try:
            # IR XML files are small relative to the weights and this check runs once per
            # socket worker. Match operation types, not incidental layer names.
            xml_text = Path(model_xml).read_text(encoding='utf-8', errors='ignore')
            if re.search(
                r"type\s*=\s*['\"](?:FakeQuantize|QuantizeLinear|DequantizeLinear)['\"]",
                xml_text, flags=re.IGNORECASE,
            ):
                return True
        except Exception:
            pass
    return False

def _normalize_openvino_segmentation_outputs(
    outputs: Sequence[np.ndarray],
    *,
    batch_size: int,
    expected_class_count: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return ``(head[B,C,A], proto[B,M,H,W], class_count)`` for a YOLO segment IR.

    Layout selection is solved jointly: a 4-D tensor is considered in BCHW and BHWC
    form, and a 3-D tensor in BCA and BAC form. The winning pair must satisfy the YOLO
    attribute identity ``C = 4 + classes + mask_channels``. This avoids mistaking a
    small spatial axis for the prototype-channel axis.
    """
    arrays = [np.asarray(value) for value in outputs]
    proto_options: List[np.ndarray] = []
    head_options: List[np.ndarray] = []
    for value in arrays:
        if int(value.shape[0]) != int(batch_size):
            continue
        if value.ndim == 4:
            if 1 <= int(value.shape[1]) <= 512:
                proto_options.append(value)
            if 1 <= int(value.shape[3]) <= 512:
                proto_options.append(np.moveaxis(value, -1, 1))
        elif value.ndim == 3:
            head_options.extend((value, np.swapaxes(value, 1, 2)))

    combinations: List[Tuple[Tuple[int, ...], np.ndarray, np.ndarray, int]] = []
    for proto_candidate in proto_options:
        mask_channels = int(proto_candidate.shape[1])
        proto_h = int(proto_candidate.shape[2])
        proto_w = int(proto_candidate.shape[3])
        if proto_h <= 1 or proto_w <= 1:
            continue
        for head_candidate in head_options:
            channels = int(head_candidate.shape[1])
            anchors = int(head_candidate.shape[2])
            class_count = int(channels) - 4 - int(mask_channels)
            if class_count < 1 or anchors < 1 or channels > anchors:
                continue
            if (
                expected_class_count is not None
                and int(class_count) != int(expected_class_count)
            ):
                continue
            # Prefer a known class count, then the smallest plausible class count and
            # prototype channel count, then the largest anchor/spatial domains.
            score = (
                0 if expected_class_count is not None else 1,
                int(class_count),
                0 if int(mask_channels) <= 256 else 1,
                int(mask_channels),
                -int(anchors),
                -int(proto_h * proto_w),
            )
            combinations.append((score, head_candidate, proto_candidate, int(class_count)))

    if not combinations:
        raise RuntimeError(
            'OpenVINO segmentation outputs do not contain a compatible raw YOLO head and '
            'mask-prototype pair; '
            f'expected_class_count={expected_class_count}, output shapes='
            f'{[tuple(int(v) for v in value.shape) for value in arrays]}. '
            'Ultralytics end-to-end/NMS-embedded OpenVINO exports are not supported by the '
            'v17 raw-head adapter; export the ordinary segmentation IR.'
        )
    _score, head, proto, class_count = min(combinations, key=lambda item: item[0])
    return (
        np.ascontiguousarray(head, dtype=np.float32),
        np.ascontiguousarray(proto, dtype=np.float32),
        int(class_count),
    )

def _openvino_cpu_payloads_from_outputs(
    outputs: Sequence[np.ndarray],
    *,
    batch_size: int,
    conf_threshold: float,
    out_size: int,
    expected_class_count: Optional[int] = None,
) -> List[CpuRetinaMaskPayload]:
    """Convert one raw OpenVINO YOLO-seg batch into CPU-retina payloads."""
    head, protos, class_count = _normalize_openvino_segmentation_outputs(
        outputs, batch_size=int(batch_size),
        expected_class_count=expected_class_count,
    )
    payloads: List[CpuRetinaMaskPayload] = []
    threshold = float(conf_threshold)
    mask_channels = int(protos.shape[1])
    for batch_index in range(int(batch_size)):
        head_i = np.asarray(head[int(batch_index)], dtype=np.float32)
        scores = head_i[4:4 + int(class_count), :]
        if int(class_count) == 1:
            confs_all = scores[0]
        else:
            confs_all = np.max(scores, axis=0)
        keep = np.flatnonzero(confs_all >= float(threshold))
        if keep.size <= 0:
            payloads.append(CpuRetinaMaskPayload(
                proto=np.ascontiguousarray(protos[int(batch_index)], dtype=np.float32),
                coeffs=np.zeros((0, mask_channels), dtype=np.float32),
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                confs=np.zeros((0,), dtype=np.float32),
                orig_shape=(int(out_size), int(out_size)),
                img_shape=(int(out_size), int(out_size)),
                frame_path='',
            ))
            continue
        selected = np.ascontiguousarray(head_i[:, keep], dtype=np.float32)
        xy = selected[0:2, :].T
        half_wh = selected[2:4, :].T * np.float32(0.5)
        boxes = np.concatenate((xy - half_wh, xy + half_wh), axis=1).astype(np.float32, copy=False)
        _clip_boxes_np(boxes, (int(out_size), int(out_size)))
        coeffs = selected[4 + int(class_count):4 + int(class_count) + mask_channels, :].T
        payloads.append(CpuRetinaMaskPayload(
            proto=np.ascontiguousarray(protos[int(batch_index)], dtype=np.float32),
            coeffs=np.ascontiguousarray(coeffs, dtype=np.float32),
            boxes_xyxy=np.ascontiguousarray(boxes, dtype=np.float32),
            confs=np.ascontiguousarray(confs_all[keep], dtype=np.float32),
            orig_shape=(int(out_size), int(out_size)),
            img_shape=(int(out_size), int(out_size)),
            frame_path='',
        ))
    return payloads

def _binary_slice_metadata_from_array(mask_volume: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute exact compact slice metadata for one task-local binary result window."""
    volume = np.asarray(mask_volume)
    if volume.ndim != 3:
        raise ValueError(f'binary metadata requires a 3-D array, got {volume.shape}')
    z_dim, plane_h, plane_w = (int(value) for value in volume.shape)
    slice_any = np.zeros((z_dim,), dtype=bool)
    slice_bboxes = np.zeros((z_dim, 4), dtype=np.int64)
    packed_rows = np.zeros((z_dim, int((plane_h + 7) // 8)), dtype=np.uint8)
    for z in range(z_dim):
        plane = np.asarray(volume[int(z)], dtype=bool)
        rows = np.any(plane, axis=1)
        packed_rows[int(z)] = np.packbits(rows)
        if not bool(np.any(rows)):
            continue
        cols = np.any(plane, axis=0)
        y_ids = np.flatnonzero(rows)
        x_ids = np.flatnonzero(cols)
        slice_any[int(z)] = True
        slice_bboxes[int(z)] = (
            int(y_ids[0]), int(y_ids[-1]) + 1,
            int(x_ids[0]), int(x_ids[-1]) + 1,
        )
    return {
        'slice_any': slice_any,
        'slice_bboxes': slice_bboxes,
        'slice_row_any': packed_rows,
        'slice_row_count': np.asarray([int(plane_h)], dtype=np.int64),
    }

class _OpenVinoCpuSegmenter:
    """One socket-local OpenVINO compiled model with a shallow async request pool."""

    def __init__(
        self,
        model_path: str,
        *,
        imgsz: int,
        batch: int,
        input_channels: int,
        requested_precision: str,
        inference_threads: int,
        physical_cores: int,
        streams: Optional[int],
        infer_requests: Optional[int],
    ) -> None:
        try:
            import openvino as ov  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                'OpenVINO CPU inference requires the openvino Python package. '
                'Install a current OpenVINO runtime in the SLURM environment.'
            ) from exc
        self.ov = ov
        self.model_xml = _resolve_openvino_model_xml_path(model_path)
        self.imgsz = int(imgsz)
        self.batch = max(1, int(batch))
        self.input_channels = max(1, int(input_channels))
        self.inference_threads = max(1, int(inference_threads))
        self.physical_cores = max(1, int(physical_cores))
        self.requested_precision = _resolve_cpu_precision(requested_precision)
        self.core = ov.Core()
        capabilities_raw: Optional[object] = None
        # Prefer the typed property on current OpenVINO, then retain the established
        # string key for older runtimes. Capability discovery is deliberately strict:
        # cpu:auto falls back to FP32 when OpenVINO cannot prove BF16 support, while an
        # explicit cpu:bf16 request fails instead of silently trusting CPU flags alone.
        try:
            device_properties = getattr(getattr(ov, 'properties', None), 'device', None)
            capability_property = getattr(device_properties, 'capabilities', None)
            if capability_property is not None:
                capabilities_raw = self.core.get_property('CPU', capability_property)
        except Exception:
            capabilities_raw = None
        if capabilities_raw is None:
            try:
                capabilities_raw = self.core.get_property('CPU', 'OPTIMIZATION_CAPABILITIES')
            except Exception:
                capabilities_raw = None
        if isinstance(capabilities_raw, str):
            capability_values = [capabilities_raw]
        else:
            try:
                capability_values = list(capabilities_raw) if capabilities_raw is not None else []
            except Exception:
                capability_values = []
        self.capabilities = {str(value).upper() for value in capability_values}
        cpu_flags = _linux_cpu_feature_flags()
        self.cpu_flags = frozenset(str(value) for value in cpu_flags)
        self.amx_tile_available = 'amx_tile' in cpu_flags
        self.amx_bf16_available = bool(
            self.amx_tile_available
            and 'amx_bf16' in cpu_flags
            and 'BF16' in self.capabilities
        )
        self.amx_int8_available = bool(
            self.amx_tile_available
            and 'amx_int8' in cpu_flags
            and 'INT8' in self.capabilities
        )

        model = self.core.read_model(str(self.model_xml))
        self.model_int8_quantized = bool(
            _openvino_model_has_int8_quantization(model, self.model_xml)
        )
        if self.requested_precision == 'auto':
            # A quantized export remains quantized regardless of the execution hint. For
            # ordinary floating-point IRs, prefer AMX BF16 and otherwise retain FP32.
            self.resolved_precision = (
                'int8'
                if self.model_int8_quantized
                else ('bf16' if self.amx_bf16_available else 'fp32')
            )
        else:
            self.resolved_precision = str(self.requested_precision)

        if self.model_int8_quantized and self.resolved_precision != 'int8':
            raise RuntimeError(
                f'cpu:{self.resolved_precision} was requested, but the supplied OpenVINO IR '
                'contains explicit INT8 quantization operations. A runtime precision hint cannot '
                'restore pre-quantization quality; use cpu:int8/auto or supply a floating-point IR.'
            )
        if self.resolved_precision == 'int8' and not self.model_int8_quantized:
            raise RuntimeError(
                'cpu:int8 was requested, but the supplied OpenVINO IR does not contain explicit '
                'INT8 quantization operations. INT8 is an export-time property; export a genuinely '
                'quantized OpenVINO model instead of asking the runtime to quantize an FP model.'
            )
        if self.resolved_precision == 'bf16' and not self.amx_bf16_available:
            raise RuntimeError(
                'cpu:bf16 was requested, but AMX_TILE + AMX_BF16 and OpenVINO BF16 capability '
                f'were not all visible (flags contain amx_tile={"amx_tile" in cpu_flags}, '
                f'amx_bf16={"amx_bf16" in cpu_flags}, capabilities={sorted(self.capabilities)}).'
            )
        if self.resolved_precision == 'fp16' and 'FP16' not in self.capabilities:
            raise RuntimeError(
                f'cpu:fp16 was requested, but OpenVINO CPU capabilities are {sorted(self.capabilities)}'
            )
        if self.resolved_precision == 'int8' and 'INT8' not in self.capabilities:
            raise RuntimeError(
                f'cpu:int8 was requested, but OpenVINO CPU capabilities are {sorted(self.capabilities)}'
            )
        if len(model.inputs) != 1:
            raise RuntimeError(
                f'OpenVINO CPU model must expose exactly one image input; got {len(model.inputs)}'
            )
        model_input = model.input(0)
        self.input_element_type = _openvino_element_type_name(model_input)
        self.expected_class_count = _openvino_export_class_count(self.model_xml)
        if self.input_element_type in {'i8', 'int8'}:
            raise RuntimeError(
                'The OpenVINO IR exposes an int8 image input, but its input zero-point/scale '
                'contract is not discoverable from the generic adapter. Use an Ultralytics '
                'OpenVINO export with a float or embedded-preprocessing uint8 input.'
            )
        original_shape = _openvino_partial_shape_values(model_input)
        if original_shape and len(original_shape) != 4:
            raise RuntimeError(
                f'OpenVINO CPU model input must be rank 4; got {original_shape}'
            )
        layout = 'NCHW'
        if len(original_shape) == 4:
            second = original_shape[1]
            last = original_shape[3]
            if second == self.input_channels:
                layout = 'NCHW'
            elif last == self.input_channels:
                layout = 'NHWC'
            elif second is not None and last is not None:
                raise ModelInputChannelMismatchError(
                    f'OpenVINO CPU model input shape {original_shape} does not match '
                    f'--channel_format C={self.input_channels}'
                )
        self.input_layout = layout
        target_shape = (
            (self.batch, self.input_channels, self.imgsz, self.imgsz)
            if layout == 'NCHW'
            else (self.batch, self.imgsz, self.imgsz, self.input_channels)
        )
        reshape_needed = not original_shape or any(value is None for value in original_shape)
        if original_shape and not reshape_needed:
            for actual, expected in zip(original_shape, target_shape):
                if int(actual) != int(expected):
                    raise RuntimeError(
                        f'OpenVINO CPU model has static input shape {original_shape}, but v17 requested '
                        f'batch={self.batch}, C={self.input_channels}, imgsz={self.imgsz} ({target_shape}).'
                    )
        elif reshape_needed:
            try:
                model.reshape({model_input: list(target_shape)})
            except Exception as exc:
                raise RuntimeError(
                    f'Unable to reshape dynamic OpenVINO input {original_shape} to {target_shape}'
                ) from exc

        config: Dict[str, object] = {
            'PERFORMANCE_HINT': 'THROUGHPUT',
            'INFERENCE_NUM_THREADS': int(self.inference_threads),
            'ENABLE_CPU_PINNING': True,
            'ENABLE_HYPER_THREADING': bool(self.inference_threads > self.physical_cores),
        }
        if streams is not None:
            config['NUM_STREAMS'] = int(streams)
        if infer_requests is not None:
            config['PERFORMANCE_HINT_NUM_REQUESTS'] = int(infer_requests)
        if self.resolved_precision in {'bf16', 'fp16', 'fp32'}:
            type_value = {
                'bf16': ov.Type.bf16,
                'fp16': ov.Type.f16,
                'fp32': ov.Type.f32,
            }[self.resolved_precision]
            config['INFERENCE_PRECISION_HINT'] = type_value
        # INT8 is an export-time property. Do not ask OpenVINO to quantize an FP model here.
        self.compile_config = dict(config)
        try:
            self.compiled_model = self.core.compile_model(model, 'CPU', config)
        except Exception as exc:
            raise RuntimeError(
                f'OpenVINO CPU compile failed for {self.model_xml} with config {config}: {exc}'
            ) from exc
        self.input_port = self.compiled_model.input(0)
        self.input_name = _openvino_port_name(self.input_port, 'images')
        self.output_ports = tuple(self.compiled_model.outputs)
        if not self.output_ports:
            raise RuntimeError('OpenVINO CPU model exposes no outputs')
        if infer_requests is None:
            try:
                request_count = int(
                    self.compiled_model.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
                )
            except Exception:
                request_count = int(streams or 1)
        else:
            request_count = int(infer_requests)
        self.request_count = max(1, int(request_count))
        self.infer_queue = ov.AsyncInferQueue(self.compiled_model, int(self.request_count))

    def _prepare_input(self, images: Sequence[np.ndarray]) -> np.ndarray:
        frames = [
            InMemoryYoloVolumeSource._frame_to_model_channels(
                np.asarray(image), int(self.input_channels),
            )
            for image in images
        ]
        if len(frames) != int(self.batch):
            raise RuntimeError(
                f'OpenVINO CPU source produced batch {len(frames)}, but model batch is {self.batch}'
            )
        batch_hwc = np.stack(frames, axis=0)
        if self.input_layout == 'NCHW':
            batch_value = np.moveaxis(batch_hwc, -1, 1)
        else:
            batch_value = batch_hwc
        # Ordinary Ultralytics IRs expose normalized floating-point input. Exports with
        # embedded preprocessing may expose uint8 and must receive the original 0..255 bytes.
        input_type = str(self.input_element_type).lower()
        if input_type in {'u8', 'uint8'}:
            return np.ascontiguousarray(batch_value, dtype=np.uint8)
        if input_type in {'f16', 'float16'}:
            return (
                np.ascontiguousarray(batch_value, dtype=np.float16)
                / np.float16(255.0)
            )
        # BF16 has no native NumPy storage type. start_async safely converts this contiguous
        # FP32 input into a BF16 input tensor when the IR itself exposes BF16.
        return np.ascontiguousarray(batch_value, dtype=np.float32) / np.float32(255.0)

    def infer_source_to_union(
        self,
        source: object,
        *,
        num_frames: int,
        out_size: int,
        conf_threshold: float,
        view_union_mm: np.ndarray,
        view_confmap_mm: Optional[np.ndarray],
        M_out_to_native: np.ndarray,
        native_h: int,
        native_w: int,
        min_conf: float,
        min_radius: float,
    ) -> Dict[str, object]:
        """Run a bounded asynchronous request queue and consume results in frame order."""
        output_queue: 'queue.Queue[object]' = queue.Queue(maxsize=max(2, int(self.request_count)))
        sentinel = object()
        consumer_errors: List[BaseException] = []
        stats = {'prediction_count': 0, 'frames_with_predictions': 0}
        submitted_real = 0

        def _callback(request: object, userdata: object) -> None:
            try:
                copied = [
                    np.array(request.get_output_tensor(index).data, copy=True)
                    for index in range(len(self.output_ports))
                ]
                output_queue.put(('result', userdata, copied))
            except BaseException as exc:
                output_queue.put(('error', userdata, exc))

        self.infer_queue.set_callback(_callback)

        def _consume() -> None:
            nonlocal stats
            while True:
                item = output_queue.get()
                if item is sentinel:
                    return
                kind, userdata, payload = item  # type: ignore[misc]
                if kind == 'error':
                    consumer_errors.append(payload)
                    continue
                if consumer_errors:
                    continue
                start_index, real_count, submitted_batch = (
                    int(value) for value in userdata
                )
                try:
                    payloads = _openvino_cpu_payloads_from_outputs(
                        payload,
                        batch_size=int(submitted_batch),
                        # CUDA's direct proto-union path applies positive --min_conf at
                        # instance selection time. Match that contract before CPU mask
                        # reconstruction so a low-confidence mask cannot survive merely by
                        # touching a high-confidence component in a hybrid view.
                        conf_threshold=max(float(conf_threshold), float(min_conf)),
                        out_size=int(out_size),
                        expected_class_count=self.expected_class_count,
                    )
                    for local_index in range(int(real_count)):
                        frame_index = int(start_index) + int(local_index)
                        instance_count, frame_count = _process_cpu_retina_prediction_frame(
                            frame_index,
                            payloads[int(local_index)],
                            int(out_size),
                            view_union_mm,
                            view_confmap_mm,
                            np.asarray(M_out_to_native, dtype=np.float32),
                            int(native_h),
                            int(native_w),
                            slice_lock=None,
                        )
                        has_foreground = _cleanup_prediction_slice_inplace(
                            view_union_mm,
                            view_confmap_mm,
                            int(frame_index),
                            min_conf=float(min_conf),
                            min_radius=float(min_radius),
                        )
                        stats['prediction_count'] = int(stats['prediction_count']) + int(instance_count)
                        if bool(frame_count) and bool(has_foreground):
                            stats['frames_with_predictions'] = int(stats['frames_with_predictions']) + 1
                except BaseException as exc:
                    consumer_errors.append(exc)

        consumer = threading.Thread(
            target=_consume,
            name='openvino-output-consumer',
            daemon=True,
        )
        consumer.start()
        try:
            for _paths, images, _info in source:  # type: ignore[operator]
                if submitted_real >= int(num_frames):
                    break
                real_count = min(len(images), int(num_frames) - int(submitted_real))
                input_value = self._prepare_input(images)
                userdata = (int(submitted_real), int(real_count), int(input_value.shape[0]))
                self.infer_queue.start_async({self.input_name: input_value}, userdata=userdata)
                submitted_real += int(real_count)
            self.infer_queue.wait_all()
        finally:
            output_queue.put(sentinel)
            consumer.join()
        if consumer_errors:
            raise RuntimeError(
                f'OpenVINO CPU inference/postprocess failed: {consumer_errors[0]}'
            ) from consumer_errors[0]
        if int(submitted_real) != int(num_frames):
            raise RuntimeError(
                f'OpenVINO CPU source produced {submitted_real}/{int(num_frames)} real frames'
            )
        stats['slice_meta'] = _binary_slice_metadata_from_array(view_union_mm)
        stats['device_hole_filled_frames'] = 0
        stats['proto_hole_treated_frames'] = 0
        stats['openvino_request_count'] = int(self.request_count)
        stats['openvino_precision'] = str(self.resolved_precision)
        stats['openvino_input_element_type'] = str(self.input_element_type)
        stats['openvino_model_int8_quantized'] = int(bool(self.model_int8_quantized))
        stats['openvino_class_count'] = (
            int(self.expected_class_count) if self.expected_class_count is not None else -1
        )
        return stats

def run_prediction_volume_in_openvino_worker(
    runner: _OpenVinoCpuSegmenter,
    cfg: PredictConfig,
    task: Dict[str, object],
) -> Dict[str, object]:
    """Run one CPU-eligible Cartesian/Tilted task into the shared result contract."""
    view: ViewInfo = task['view']  # type: ignore[assignment]
    job = task['job']
    kind = str(task['kind'])
    if not cpu_inference_supports_view(view):
        raise ValueError(
            f'OpenVINO CPU workers support Cartesian and Tilted Cartesian only; got {view.name}'
        )
    if kind not in {'fullframe', 'tile'}:
        raise ValueError(f'Unsupported OpenVINO task kind {kind!r}')
    result_mode = str(task.get('result_mode', 'file'))
    if result_mode == HYBRID_DEFERRED_RESULT_MODE:
        raise ValueError('OpenVINO received an unresolved hybrid full-frame task')
    if result_mode == 'd1_owner':
        raise ValueError('OpenVINO CPU workers cannot consume GPU D1 owner tasks')

    slice_offset = int(task.get('slice_start', 0))
    slice_count = int(task.get('slice_count', int(view.num_slices)))
    out_size = int(task['out_size'])
    channel_format = resolve_channel_format(
        task.get('channel_format', DEFAULT_CHANNEL_FORMAT)  # type: ignore[arg-type]
    )
    if int(channel_format.channel_count) != int(cfg.input_channels):
        raise ValueError(
            f'OpenVINO task C={channel_format.channel_count}, worker C={cfg.input_channels}'
        )
    full_processing_shape = view_processing_volume_shape(view, int(out_size))
    declared_processing_shape = tuple(
        int(value) for value in task.get('processing_shape', full_processing_shape)
    )
    processing_h = int(declared_processing_shape[1])
    processing_w = int(declared_processing_shape[2])
    result_shape = (int(slice_count), int(processing_h), int(processing_w))
    native_resize = task.get('native_resize') if isinstance(task.get('native_resize'), dict) else None

    source_mm: Optional[np.memmap] = None
    result_mask: Optional[np.ndarray] = None
    result_conf: Optional[np.ndarray] = None
    result_mask_full: Optional[np.memmap] = None
    result_conf_full: Optional[np.memmap] = None
    source: Optional[object] = None
    try:
        result_mode = str(task.get('result_mode', 'file'))
        if result_mode == 'direct_union':
            union_shape = (
                int(task.get('union_num_slices', slice_count)),
                int(processing_h), int(processing_w),
            )
            result_mask_full = np.memmap(
                Path(str(task['result_mask_path'])), dtype=np.uint8, mode='r+', shape=union_shape,
            )
            result_mask = result_mask_full[slice_offset:slice_offset + slice_count]
            if task.get('result_conf_path'):
                result_conf_full = np.memmap(
                    Path(str(task['result_conf_path'])), dtype=np.uint8, mode='r+', shape=union_shape,
                )
                result_conf = result_conf_full[slice_offset:slice_offset + slice_count]
        else:
            open_mode = 'r+' if bool(task.get('result_workspace_preallocated', False)) else 'w+'
            result_mask = np.memmap(
                Path(str(task['result_mask_path'])), dtype=np.uint8, mode=open_mode, shape=result_shape,
            )
            if task.get('result_conf_path'):
                result_conf = np.memmap(
                    Path(str(task['result_conf_path'])), dtype=np.uint8, mode=open_mode, shape=result_shape,
                )
        if native_resize is not None:
            _wait_for_cube_ready_sentinel(
                str(native_resize['sentinel']),
                request_path=(str(native_resize['request']) if native_resize.get('request') else None),
                failed_path=(str(native_resize['failed']) if native_resize.get('failed') else None),
            )
        source_mm = open_existing_gray_memmap(
            task['source_volume_path'], task['source_shape'], task.get('source_dtype', 'uint8'), mode='r',
        )
        render_callable = _worker_render_callable(
            source_mm,
            view,
            job,
            kind,
            slice_offset=int(slice_offset),
            channel_format=channel_format,
        )
        source = StreamingYoloVolumeSource(
            render_callable,
            num_frames=int(slice_count),
            name=f'openvino-{kind}-{view.name}-{task["job_id"]}',
            batch_size=max(1, int(cfg.batch)),
            out_size=int(out_size),
            render_workers=max(1, int(task.get('render_workers', 1))),
            prefetch_frames=max(1, int(task.get('prefetch_frames', cfg.batch))),
            autostart=True,
            shared_executor=None,
            channel_format=channel_format,
        )
        task_affine = np.asarray(
            task.get('M_out_to_processing')
            if task.get('M_out_to_processing') is not None
            else output_to_view_processing_affine(
                view, np.asarray(task['M_out_to_src'], dtype=np.float32), int(out_size),
            ),
            dtype=np.float32,
        )
        return runner.infer_source_to_union(
            source,
            num_frames=int(slice_count),
            out_size=int(out_size),
            conf_threshold=float(cfg.conf),
            view_union_mm=result_mask,
            view_confmap_mm=result_conf,
            M_out_to_native=task_affine,
            native_h=int(processing_h),
            native_w=int(processing_w),
            min_conf=float(task.get('streaming_cleanup_min_conf', 0.0)),
            min_radius=float(task.get('streaming_cleanup_min_radius', 0.0)),
        )
    finally:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        for mm in (result_mask, result_conf, result_mask_full, result_conf_full, source_mm):
            if mm is not None:
                try:
                    close_memmap_array(mm)
                except Exception:
                    pass

def _cpu_inference_worker_main(
    instance_id: int,
    model_path: str,
    init_dict: Dict[str, object],
    task_queue: object,
    result_queue: object,
) -> None:
    """Persistent socket-local OpenVINO worker process."""
    initialize_runtime_observability()
    persistent_source_memfds: Dict[str, int] = {}
    cpus = [int(value) for value in init_dict.get('numa_affinity_cpus', ())]
    try:
        if cpus and not _sched_setaffinity_all_threads(cpus):
            raise RuntimeError(
                f'OpenVINO CPU worker {instance_id} could not apply affinity {cpus}'
            )
        try:
            cv2.setNumThreads(max(1, int(init_dict.get('cv2_threads', 1))))
        except Exception:
            pass
        set_retina_mask_processor('cpu')
        cfg = PredictConfig(
            imgsz=int(init_dict['imgsz']),
            conf=float(init_dict['conf']),
            device='cpu',
            quantize=None,
            batch=max(1, int(init_dict.get('batch', 1))),
            input_channels=max(1, int(init_dict.get('input_channels', 1))),
            channel_token=str(init_dict.get('channel_token', 'gray')),
        )
        runner = _OpenVinoCpuSegmenter(
            str(model_path),
            imgsz=int(cfg.imgsz),
            batch=int(cfg.batch),
            input_channels=int(cfg.input_channels),
            requested_precision=str(init_dict.get('precision', 'auto')),
            inference_threads=max(1, int(init_dict.get('inference_threads', len(cpus) or 1))),
            physical_cores=max(1, int(init_dict.get('physical_cores', len(cpus) or 1))),
            streams=(
                int(init_dict['streams']) if init_dict.get('streams') is not None else None
            ),
            infer_requests=(
                int(init_dict['infer_requests'])
                if init_dict.get('infer_requests') is not None else None
            ),
        )
        result_queue.put({
            'type': 'ready', 'worker_kind': 'cpu', 'cpu_index': int(instance_id),
            'pid': int(os.getpid()), 'precision': str(runner.resolved_precision),
            'requests': int(runner.request_count), 'model_xml': str(runner.model_xml),
            'threads': int(runner.inference_threads),
            'input_element_type': str(runner.input_element_type),
            'model_int8_quantized': bool(runner.model_int8_quantized),
            'amx_tile': bool(runner.amx_tile_available),
            'amx_bf16': bool('amx_bf16' in runner.cpu_flags),
            'amx_int8': bool('amx_int8' in runner.cpu_flags),
            'openvino_capabilities': sorted(runner.capabilities),
            'class_count': (
                int(runner.expected_class_count)
                if runner.expected_class_count is not None else None
            ),
        })
    except Exception as exc:
        import traceback
        try:
            result_queue.put({
                'type': 'fatal', 'worker_kind': 'cpu', 'cpu_index': int(instance_id),
                'error': repr(exc), 'traceback': traceback.format_exc(),
            })
        except Exception:
            pass
        return

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id = int(task['task_id'])
            transferred_task_fds: List[int] = []
            try:
                task_local = dict(task)
                transferred_task_fds = _materialize_worker_task_memfd_paths(
                    task_local, persistent_source_memfds,
                )
                render_workers = max(
                    1,
                    min(
                        int(init_dict.get('render_workers', 1)),
                        int(task_local.get('slice_count', 1)),
                    ),
                )
                task_local['render_workers'] = int(render_workers)
                task_local['postprocess_workers'] = 1
                started = time.perf_counter()
                stats = run_prediction_volume_in_openvino_worker(runner, cfg, task_local)
                stats = dict(stats)
                stats['worker_compute_seconds'] = max(0.0, time.perf_counter() - started)
                stats['backend'] = 'cpu'
                stats['cpu_instance'] = int(instance_id)
                result_queue.put({
                    'type': 'result', 'worker_kind': 'cpu',
                    'cpu_index': int(instance_id), 'task_id': int(task_id),
                    'ok': True, 'stats': stats,
                })
            except Exception as exc:
                import traceback
                result_queue.put({
                    'type': 'result', 'worker_kind': 'cpu',
                    'cpu_index': int(instance_id), 'task_id': int(task_id),
                    'ok': False, 'error': repr(exc), 'traceback': traceback.format_exc(),
                })
            finally:
                _close_fd_list(transferred_task_fds)
    finally:
        _close_fd_list(persistent_source_memfds.values())
        persistent_source_memfds.clear()

@dataclass
class _DeferredGpuWorkerTaskResult:
    """Task stats whose device-union flush is retiring on a persistent copy lane."""

    stats: Dict[str, object]
    flush_future: Future
    memmaps: Tuple[object, ...]

    def finish(self) -> Dict[str, object]:
        try:
            retired = self.flush_future.result()
            if isinstance(retired, dict):
                self.stats.update(retired)
            return self.stats
        finally:
            for mm in self.memmaps:
                if mm is not None:
                    try:
                        close_memmap_array(mm)
                    except Exception:
                        pass

def run_prediction_volume_in_worker(
    model: object,
    cfg: 'PredictConfig',
    task: Dict[str, object],
) -> Dict[str, object] | _DeferredGpuWorkerTaskResult:
    """Run one independent full-frame or tile task and write its result window.

    v16.4.0 has no grouped-tile/configuration-canvas worker path. Each tile keeps one
    immutable affine and one result volume from resident rendering through inference,
    postprocessing, and the later two-stage parent/bridge gate.
    """
    view: ViewInfo = task['view']  # type: ignore[assignment]
    job = task['job']
    kind = str(task['kind'])
    if str(task.get('result_mode', 'file')) == HYBRID_DEFERRED_RESULT_MODE:
        raise ValueError('CUDA worker received an unresolved hybrid full-frame task')
    if kind not in {'fullframe', 'tile'}:
        raise ValueError(f'Unsupported v16.4.0 worker task kind: {kind!r}')
    if kind == 'fullframe' and not isinstance(job, AugJob):
        raise TypeError(f'Full-frame worker task requires AugJob, got {type(job)!r}')
    if kind == 'tile' and not isinstance(job, DenseTileJob):
        raise TypeError(f'Tile worker task requires DenseTileJob, got {type(job)!r}')

    slice_offset = int(task.get('slice_start', 0))
    slice_count = int(task.get('slice_count', int(view.num_slices)))
    num_frames = int(slice_count)
    out_size = int(task['out_size'])
    channel_format = resolve_channel_format(
        task.get('channel_format', DEFAULT_CHANNEL_FORMAT)  # type: ignore[arg-type]
    )
    if int(channel_format.channel_count) != int(cfg.input_channels):
        raise ValueError(
            f'Worker task {task.get("task_id", "?")} channel format '
            f'{channel_format.token} has C={int(channel_format.channel_count)}, but '
            f'worker PredictConfig requires C={int(cfg.input_channels)}'
        )

    full_processing_shape = view_processing_volume_shape(view, int(out_size))
    declared_processing_shape = tuple(int(v) for v in task.get('processing_shape', full_processing_shape))
    if len(declared_processing_shape) != 3 or int(declared_processing_shape[0]) != int(view.num_slices):
        raise ValueError(
            f'Worker task {task.get("task_id", "?")} has invalid processing shape '
            f'{declared_processing_shape} for {view.name} ({int(view.num_slices)} slices)'
        )
    processing_h = int(declared_processing_shape[1])
    processing_w = int(declared_processing_shape[2])
    result_shape = (int(num_frames), int(processing_h), int(processing_w))
    native_resize = task.get('native_resize') if isinstance(task.get('native_resize'), dict) else None

    source_mm: Optional[np.memmap] = None
    result_mask: Optional[np.ndarray] = None
    result_conf: Optional[np.ndarray] = None
    result_mask_full: Optional[np.memmap] = None
    result_conf_full: Optional[np.memmap] = None
    source: Optional[object] = None
    deferred_result: Optional[_DeferredGpuWorkerTaskResult] = None

    def _open_cpu_render_source() -> Tuple[np.memmap, object]:
        if native_resize is not None:
            _wait_for_cube_ready_sentinel(
                str(native_resize['sentinel']),
                request_path=(str(native_resize['request']) if native_resize.get('request') else None),
                failed_path=(str(native_resize['failed']) if native_resize.get('failed') else None),
            )
        mm = open_existing_gray_memmap(
            task['source_volume_path'], task['source_shape'], task.get('source_dtype', 'uint8'), mode='r',
        )
        try:
            render_callable = _worker_render_callable(
                mm,
                view,
                job,
                kind,
                slice_offset=slice_offset,
                channel_format=channel_format,
            )
            cpu_source = StreamingYoloVolumeSource(
                render_callable,
                num_frames=num_frames,
                name=f"worker-{kind}-{view.name}-{task['job_id']}",
                batch_size=max(1, int(cfg.batch)),
                out_size=out_size,
                render_workers=max(1, int(task.get('render_workers', 1))),
                prefetch_frames=max(1, int(task.get('prefetch_frames', 1))),
                autostart=True,
                shared_executor=None,
                channel_format=channel_format,
            )
        except BaseException:
            close_memmap_array(mm)
            raise
        return mm, cpu_source

    try:
        if str(task.get('result_mode', 'file')) == 'd1_owner':
            # D1 never creates a host-dense task/view union. This sentinel exists only to
            # satisfy the generic prediction API; any host fallback is rejected explicitly.
            result_mask = np.zeros((1, 1, 1), dtype=np.uint8)
            result_conf = None
        elif str(task.get('result_mode', 'file')) == 'direct_union':
            # Every angle variant owns a disjoint per-view accumulator. Full-frame slice
            # leases write non-overlapping z windows directly into that variant's shared union.
            union_shape = (
                int(task.get('union_num_slices', slice_count)), int(processing_h), int(processing_w),
            )
            result_mask_full = np.memmap(
                Path(str(task['result_mask_path'])), dtype=np.uint8, mode='r+', shape=union_shape,
            )
            result_mask = result_mask_full[slice_offset:slice_offset + slice_count]
            if task.get('result_conf_path'):
                result_conf_full = np.memmap(
                    Path(str(task['result_conf_path'])), dtype=np.uint8, mode='r+', shape=union_shape,
                )
                result_conf = result_conf_full[slice_offset:slice_offset + slice_count]
        else:
            result_open_mode = (
                'r+' if bool(task.get('result_workspace_preallocated', False)) else 'w+'
            )
            result_mask = np.memmap(
                Path(str(task['result_mask_path'])),
                dtype=np.uint8,
                mode=result_open_mode,
                shape=result_shape,
            )
            if task.get('result_conf_path'):
                result_conf = np.memmap(
                    Path(str(task['result_conf_path'])),
                    dtype=np.uint8,
                    mode=result_open_mode,
                    shape=result_shape,
                )

        # Keep the source/native view plane resident whenever the worker has VRAM headroom.
        # Full-frame and tile tasks use separate source classes but the same cached native
        # planes; a tile is cropped/warped/resized directly on device before inference.
        gpu_engine = _worker_gpu_render_engine()
        if gpu_engine is not None:
            try:
                resident_view_supported = bool(
                    not is_radial_view(view) or radial_resident_gpu_render_supported(view)
                )
                if native_resize is not None:
                    render_mode = gpu_engine.ensure_volume(
                        str(native_resize['path']),
                        tuple(int(x) for x in native_resize['shape']),
                        str(native_resize.get('dtype', 'uint8')),
                        resize_to_t=int(task['source_shape'][0]),
                        require_radial_texture=bool(task.get('radial_texture_required', is_radial_view(view))),
                    )
                    if render_mode != 'resident':
                        _wait_for_cube_ready_sentinel(
                            str(native_resize['sentinel']),
                            request_path=(str(native_resize['request']) if native_resize.get('request') else None),
                            failed_path=(str(native_resize['failed']) if native_resize.get('failed') else None),
                        )
                        render_mode = gpu_engine.ensure_volume(
                            str(task['source_volume_path']),
                            tuple(int(x) for x in task['source_shape']),
                            str(task.get('source_dtype', 'uint8')),
                            require_radial_texture=bool(task.get('radial_texture_required', is_radial_view(view))),
                        )
                else:
                    render_mode = gpu_engine.ensure_volume(
                        str(task['source_volume_path']),
                        tuple(int(x) for x in task['source_shape']),
                        str(task.get('source_dtype', 'uint8')),
                        require_radial_texture=bool(task.get('radial_texture_required', is_radial_view(view))),
                    )

                if render_mode == 'resident':
                    gpu_engine.run_startup_fused_preflight(
                        gpu_worker_fused_preflight_specs(),
                        out_size=int(out_size),
                        fp16=quantize_uses_fp16(cfg.quantize),
                    )

                if render_mode == 'resident' and resident_view_supported and kind == 'tile':
                    request_affine_grid_cache_entries(10)
                    if is_tilted_view(view):
                        gpu_engine.request_tilted_plan_cache_entries(5)
                    source = GpuTileRenderedYoloSource(
                        gpu_engine,
                        view,
                        job,  # type: ignore[arg-type]
                        slice_offset=slice_offset,
                        num_frames=num_frames,
                        batch_size=max(1, int(cfg.batch)),
                        out_size=out_size,
                        fp16=quantize_uses_fp16(cfg.quantize),
                        name=f"gpu-render-tile-{view.name}-{task['job_id']}",
                        channel_format=channel_format,
                    )
                elif render_mode == 'resident' and resident_view_supported:
                    source = GpuRenderedYoloSource(
                        gpu_engine,
                        view,
                        job,  # type: ignore[arg-type]
                        slice_offset=slice_offset,
                        num_frames=num_frames,
                        batch_size=max(1, int(cfg.batch)),
                        out_size=out_size,
                        fp16=quantize_uses_fp16(cfg.quantize),
                        name=f"gpu-render-fullframe-{view.name}-{task['job_id']}",
                        channel_format=channel_format,
                    )
                elif (
                    kind == 'fullframe'
                    and is_radial_view(view)
                    and radial_streaming_gpu_render_supported(view)
                ):
                    slab_indices = _radial_slab_context_indices(
                        view, slice_offset, num_frames, channel_format,
                    )
                    slab = gpu_engine.prerender_radial_slab(view, slab_indices)
                    source = StreamingYoloVolumeSource(
                        _radial_slab_channel_renderer(
                            slab,
                            slab_indices,
                            view,
                            job,  # type: ignore[arg-type]
                            center_start=slice_offset,
                            channel_format=channel_format,
                        ),
                        num_frames=num_frames,
                        name=f"worker-radialslab-{view.name}-{task['job_id']}",
                        batch_size=max(1, int(cfg.batch)),
                        out_size=out_size,
                        render_workers=2,
                        prefetch_frames=max(1, int(task.get('prefetch_frames', 1))),
                        autostart=True,
                        shared_executor=None,
                        channel_format=channel_format,
                    )
            except _ResidentTensorRTRingFatalError:
                raise
            except Exception as exc:
                print(
                    f"Warning: GPU render path failed for {view.name}/{task['job_id']} ({exc}); "
                    'using the CPU render path for this independent task.'
                )
                source = None

        if source is None:
            if (
                is_tilted_radial_view(view)
                and str(view.name) not in _WORKER_TILTED_RADIAL_CPU_WARNED
            ):
                _WORKER_TILTED_RADIAL_CPU_WARNED.add(str(view.name))
                print(
                    'PERFORMANCE WARNING: tilted-Radial view '
                    f'{view.name!r} is entering the completed-cube CPU renderer. '
                    'This CPU fallback can dominate wall time. '
                    'Look earlier for a resident-upload or CUDA-renderer failure.'
                )
            source_mm, source = _open_cpu_render_source()

        if task.get('M_out_to_processing') is not None:
            task_affine = np.asarray(task['M_out_to_processing'], dtype=np.float32)
        else:
            task_affine = np.asarray(
                output_to_view_processing_affine(
                    view, np.asarray(task['M_out_to_src'], dtype=np.float32), int(out_size),
                ),
                dtype=np.float32,
            )

        def _predict(active_source: object) -> Dict[str, object]:
            return predict_source_and_accumulate(
                model,
                active_source,
                source_label=f"{view.name}-{task['job_id']}",
                num_frames=num_frames,
                out_size=out_size,
                cfg=cfg,
                view_union_mm=result_mask,
                view_confmap_mm=result_conf,
                M_out_to_native=task_affine,
                native_h=int(processing_h),
                native_w=int(processing_w),
                postprocess_workers=int(task.get('postprocess_workers', 1)),
                streaming_cleanup_enabled=bool(task.get('streaming_cleanup_enabled', False)),
                streaming_cleanup_min_conf=float(task.get('streaming_cleanup_min_conf', 0.0)),
                streaming_cleanup_min_radius=float(task.get('streaming_cleanup_min_radius', 0.0)),
                slice_locks=None,
                device_hole_fill=bool(task.get('device_hole_fill', False)),
                defer_device_union_flush=bool(gpu_union_flush_overlap_enabled()),
                device_union_consumer=(
                    (lambda accumulator: _d1_consume_device_union(task, accumulator))
                    if str(task.get('result_mode', 'file')) == 'd1_owner' else None
                ),
                require_device_union=bool(str(task.get('result_mode', 'file')) == 'd1_owner'),
                require_proto_hole_treatment=bool(
                    str(task.get('result_mode', 'file')) == 'd1_owner'
                ),
            )

        retry_with_cpu_render = False
        try:
            stats = _predict(source)
        except _ResidentTensorRTRingFatalError:
            raise
        except Exception as exc:
            if str(task.get('result_mode', 'file')) == 'd1_owner':
                raise
            if not isinstance(source, (GpuRenderedYoloSource, GpuTileRenderedYoloSource)):
                raise
            print(
                f"Warning: lazy GPU render failed for {view.name}/{task['job_id']} ({exc}); "
                'restarting this task with the completed-cube CPU renderer.'
            )
            try:
                source.close()
            except Exception:
                pass
            source = None
            if gpu_engine is not None:
                gpu_engine.disable_resident_after_runtime_failure()
            if result_mask is not None:
                result_mask[...] = np.uint8(0)
            if result_conf is not None:
                result_conf[...] = np.uint8(0)
            retry_with_cpu_render = True

        if retry_with_cpu_render:
            gc.collect()
            if gpu_engine is not None:
                try:
                    gpu_engine.torch.cuda.empty_cache()
                except Exception:
                    pass
            source_mm, source = _open_cpu_render_source()
            stats = _predict(source)

        public_stats: Dict[str, object] = {
            'prediction_count': int(stats.get('prediction_count', 0)),
            'frames_with_predictions': int(stats.get('frames_with_predictions', 0)),
            'device_hole_filled_frames': int(stats.get('device_hole_filled_frames', 0)),
            'proto_hole_treated_frames': int(
                stats.get('proto_hole_treated_frames', stats.get('device_hole_filled_frames', 0))
            ),
            'slice_meta': stats.get('slice_meta'),
        }
        for d1_key in (
            'd1_view_complete', 'd1_covered_slices', 'd1_total_slices',
            'd1_backprojected_task_slices', 'd1_bitset_words',
            'd1_view_compute_seconds', 'd1_layer_ref', 'd1_cvol_stats',
            'd1_publication_seconds', 'd1_nonempty_task_slices',
            'd1_scanned_bbox_pixels', 'd1_view_shadow_path',
            'd1_view_shadow_format', 'd1_view_shadow_stats',
        ):
            if d1_key in stats:
                public_stats[d1_key] = stats[d1_key]
        flush_future = stats.get('_device_union_flush_future')
        if isinstance(flush_future, Future):
            deferred_result = _DeferredGpuWorkerTaskResult(
                stats=public_stats,
                flush_future=flush_future,
                memmaps=(result_mask, result_conf, result_mask_full, result_conf_full),
            )
            result_mask = None
            result_conf = None
            result_mask_full = None
            result_conf_full = None
            return deferred_result
        return public_stats
    finally:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        for mm in (result_mask, result_conf, result_mask_full, result_conf_full, source_mm):
            if mm is not None:
                try:
                    close_memmap_array(mm)
                except Exception:
                    pass

def _pin_cuda_visible_device_token(logical_index: int) -> str:
    """Restrict a worker process to one inherited ``CUDA_VISIBLE_DEVICES`` token before importing CUDA libraries."""
    idx = int(logical_index)
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    if raw is None:
        return str(idx)
    tokens = [tok.strip() for tok in str(raw).split(',') if tok.strip()]
    if idx < 0 or idx >= len(tokens):
        raise RuntimeError(
            f'--device cuda:{idx} is out of range for inherited CUDA_VISIBLE_DEVICES={raw!r} '
            f'({len(tokens)} visible device(s))'
        )
    return tokens[idx]

def _gpu_inference_worker_main(
    gpu_index: int,
    model_path: str,
    init_dict: Dict[str, object],
    task_queue: object,
    result_queue: object,
) -> None:
    """Persistent GPU worker process: pin to one physical GPU, load the model once, serve tasks."""
    global _GPU_WORKER_NUMA_PIN, _GPU_WORKER_NUMA_FULL
    try:
        # Pin the process to its physical GPU before any CUDA context is created, so the model and
        # all tensors live on that device and never contend with the other workers' GPUs. The
        # logical --device index is remapped through the inherited CUDA_VISIBLE_DEVICES list
        # .
        os.environ['CUDA_VISIBLE_DEVICES'] = _pin_cuda_visible_device_token(int(gpu_index))
        initialize_runtime_observability()
        # pin every thread of this worker to its GPU's NUMA node BEFORE any
        # CUDA/model work, so allocator arenas, pinned staging and TRT host scratch land
        # node-local. The parent computed the plan (None = stay unpinned).
        _numa_pin_cpus = init_dict.get('numa_affinity_cpus', None)
        if _numa_pin_cpus and hasattr(os, 'sched_getaffinity'):
            try:
                _GPU_WORKER_NUMA_FULL = {int(c) for c in os.sched_getaffinity(0)}
                if _sched_setaffinity_all_threads([int(c) for c in _numa_pin_cpus]):  # type: ignore[union-attr]
                    _GPU_WORKER_NUMA_PIN = {int(c) for c in _numa_pin_cpus}  # type: ignore[union-attr]
                    print(f'[numa] gpu-worker {int(gpu_index)}: pinned to {len(_GPU_WORKER_NUMA_PIN)} cpu(s)')
            except Exception as _numa_exc:
                _GPU_WORKER_NUMA_PIN = None
                print(f'Warning: [numa] gpu-worker {int(gpu_index)}: pinning failed ({_numa_exc}); running unpinned.')
        try:
            cv2.setNumThreads(max(1, int(init_dict.get('cv2_threads', 1))))
        except Exception:
            pass
        set_retina_mask_processor(str(init_dict.get('retina_processor', 'cpu')))
        set_gpu_worker_fused_preflight_specs(
            init_dict.get('fused_preflight_specs')  # type: ignore[arg-type]
        )
        # propagate the angle-variant GPU fast-path (min_conf, min_radius) into this
        # worker process (globals do not cross the spawn boundary). None disables the fast path.
        _fastpath_min_conf = init_dict.get('angle_variant_gpu_fastpath_min_conf', None)
        _fastpath_min_radius = init_dict.get('angle_variant_gpu_fastpath_min_radius', 0.0)
        set_angle_variant_gpu_fastpath(
            None if _fastpath_min_conf is None else float(_fastpath_min_conf),
            float(_fastpath_min_radius or 0.0),
        )
        cfg = PredictConfig(
            imgsz=int(init_dict['imgsz']),
            conf=float(init_dict['conf']),
            device='cuda:0',
            quantize=resolve_quantize(init_dict.get('quantize')),
            batch=max(1, int(init_dict['batch'])),
            input_channels=max(1, int(init_dict.get('input_channels', 1))),
            channel_token=str(init_dict.get('channel_token', 'gray')),
        )
        if d1_owner_pipeline_enabled():
            _d1_backproject_kernels()
            print(
                f'v16.1.7 D1 backprojection NVRTC preflight passed on cuda:{int(gpu_index)}: '
                'header-free source-geometry atomic-OR kernel compiled before TensorRT load.'
            )
        model = load_ultralytics_model(str(model_path), task='segment')
        ensure_yolo_ready_for_predict(model, cfg)
        validate_yolo_model_input_channels(
            model,
            int(cfg.input_channels),
            channel_token=str(cfg.channel_token),
            context=f'GPU worker cuda:{int(gpu_index)} model load',
        )
        require_channel_aware_yolo_preprocess_patch(str(cfg.channel_token))
        if cpu_retina_masks_enabled():
            try:
                ensure_cpu_retina_mask_predictor_patch()
            except Exception:
                pass
        else:
            # GPU retina mode — reduce per-frame unions at proto resolution.
            try:
                ensure_gpu_retina_proto_union_predictor_patch()
            except Exception:
                pass
        # per-worker GPU render engine (volume residency resolves lazily on
        # the first task, once the shared source volume exists and VRAM headroom is known).
        try:
            _init_worker_gpu_render_engine('cuda:0')
        except Exception as _render_engine_exc:
            print(f'Warning: GPU render engine init failed ({_render_engine_exc}); worker uses CPU rendering.')
        try:
            _init_gpu_union_retirement_manager('cuda:0', int(cfg.imgsz))
        except Exception as _retirement_exc:
            print(
                f'Warning: persistent GPU retirement lanes unavailable ({_retirement_exc}); '
                'using the event-compatible fallback retirement executor.'
            )
        result_queue.put({'type': 'ready', 'gpu_index': int(gpu_index), 'pid': int(os.getpid())})
    except Exception as exc:  # pragma: no cover - worker init failure surfaced to main
        import traceback
        try:
            result_queue.put({'type': 'fatal', 'gpu_index': int(gpu_index), 'error': repr(exc), 'traceback': traceback.format_exc()})
        except Exception:
            pass
        return

    pending_publications: set[Future] = set()
    publication_condition = threading.Condition()
    overlap_announced = False
    persistent_source_memfds: Dict[str, int] = {}

    def _publish_deferred(finished_task_id: int, deferred: _DeferredGpuWorkerTaskResult) -> None:
        try:
            finished_stats = deferred.finish()
            result_queue.put({
                'type': 'result', 'task_id': int(finished_task_id),
                'gpu_index': int(gpu_index), 'ok': True, 'stats': finished_stats,
            })
        except Exception as exc:  # pragma: no cover - surfaced to scheduler
            import traceback
            result_queue.put({
                'type': 'result', 'task_id': int(finished_task_id),
                'gpu_index': int(gpu_index), 'ok': False,
                'error': repr(exc), 'traceback': traceback.format_exc(),
            })

    def _schedule_deferred_publication(
        finished_task_id: int,
        deferred: _DeferredGpuWorkerTaskResult,
    ) -> None:
        future = deferred.flush_future
        with publication_condition:
            pending_publications.add(future)

        def _done(_future: Future) -> None:
            try:
                _publish_deferred(int(finished_task_id), deferred)
            except BaseException as exc:
                # Do not strand worker teardown if the result transport itself is broken.
                try:
                    print(f'Warning: GPU retirement publication transport failed ({exc}).')
                except Exception:
                    pass
            finally:
                with publication_condition:
                    pending_publications.discard(_future)
                    publication_condition.notify_all()

        future.add_done_callback(_done)

    def _wait_for_deferred_publications() -> None:
        with publication_condition:
            while pending_publications:
                publication_condition.wait()

    try:
      while True:
        task = task_queue.get()
        if task is None:
            break
        task_id = int(task['task_id'])
        if str(task.get('task_type', 'inference')) == 'interpolation_pass':
            # The scheduler targets auxiliary work only to a worker with no assigned inference
            # lease.  Finish any asynchronous task retirement before reusing its interpreter.
            _wait_for_deferred_publications()
            allow_full_cpu_affinity = bool(task.get('allow_full_cpu_affinity', False))
            aux_kwargs = dict(task['aux_kwargs'])
            local_cpu_workers = max(1, int(init_dict.get('cpu_workers', 1)))
            if not allow_full_cpu_affinity:
                aux_kwargs['workers'] = max(
                    1, min(int(aux_kwargs.get('workers', local_cpu_workers)), local_cpu_workers),
                )
            numa_widened = False
            if (
                allow_full_cpu_affinity
                and _GPU_WORKER_NUMA_PIN is not None
                and _GPU_WORKER_NUMA_FULL
            ):
                numa_widened = _sched_setaffinity_all_threads(sorted(_GPU_WORKER_NUMA_FULL))
            try:
                aux_stats = _interpolation_process_entry(**aux_kwargs)  # type: ignore[arg-type]
                result_queue.put({
                    'type': 'aux_result', 'task_id': task_id, 'gpu_index': int(gpu_index),
                    'ok': True, 'stats': dict(aux_stats),
                })
            except Exception as exc:  # pragma: no cover - surfaced to the waiting submitter
                import traceback
                result_queue.put({
                    'type': 'aux_result', 'task_id': task_id, 'gpu_index': int(gpu_index), 'ok': False,
                    'error': repr(exc), 'traceback': traceback.format_exc(),
                })
            finally:
                if numa_widened and _GPU_WORKER_NUMA_PIN is not None:
                    _sched_setaffinity_all_threads(sorted(_GPU_WORKER_NUMA_PIN))
            continue
        transferred_task_fds: List[int] = []
        try:
            # The parent affinity plan can be uneven across NUMA nodes. Size this worker's
            # pools from its own whole-core allocation instead of a logical-CPU global average.
            task_local = dict(task)
            transferred_task_fds = _materialize_worker_task_memfd_paths(
                task_local, persistent_source_memfds,
            )
            local_cpu_workers = max(1, int(init_dict.get('cpu_workers', task.get('postprocess_workers', 1))))
            task_local['render_workers'] = max(
                1, min(local_cpu_workers, int(task.get('slice_count', 1))),
            )
            task_local['postprocess_workers'] = int(local_cpu_workers)
            compute_started = time.perf_counter()
            completed = run_prediction_volume_in_worker(model, cfg, task_local)
            compute_seconds = max(0.0, time.perf_counter() - compute_started)
            if isinstance(completed, _DeferredGpuWorkerTaskResult):
                completed.stats['worker_compute_seconds'] = float(compute_seconds)
                # Each future publishes itself when its persistent retirement lane settles.
                # Lane acquisition in predict_source_and_accumulate provides the hard bound;
                # there is no single publication queue or semaphore serializing completions.
                _schedule_deferred_publication(int(task_id), completed)
                result_queue.put({
                    'type': 'compute_released', 'task_id': int(task_id),
                    'gpu_index': int(gpu_index), 'ok': True,
                    'stats': {
                        'worker_compute_seconds': float(compute_seconds),
                        'slice_count': int(task_local.get('slice_count', 0)),
                        'kind': str(task_local.get('kind', '')),
                        'd1_view_complete': bool(completed.stats.get('d1_view_complete', False)),
                        'd1_covered_slices': int(completed.stats.get('d1_covered_slices', 0)),
                        'd1_total_slices': int(completed.stats.get('d1_total_slices', 0)),
                    },
                })
                if not overlap_announced:
                    overlap_announced = True
                    manager = _gpu_union_retirement_manager()
                    lane_count = int(manager.capacity) if manager is not None else 2
                    print(
                        f'v16.1.3 compute/retirement credits active: {lane_count} independent '
                        'event-fenced D2H/publication record(s) per worker; compute completion '
                        'releases the next dispatch credit before result publication. '
                        'YOLO_TTA_GPU_UNION_FLUSH_OVERLAP=0 restores synchronous retirement.'
                    )
            else:
                completed = dict(completed)
                completed['worker_compute_seconds'] = float(compute_seconds)
                result_queue.put({
                    'type': 'result', 'task_id': task_id, 'gpu_index': int(gpu_index),
                    'ok': True, 'stats': completed,
                })
        except _ResidentTensorRTRingFatalError as exc:  # pragma: no cover - unsafe TRT state
            # A failed post/infer-stream drain or binding-address restore means this process's
            # TensorRT contexts can no longer be reused safely. Surface a worker-fatal result
            # and exit instead of dequeuing another view on the compromised backend.
            import traceback
            result_queue.put({
                'type': 'fatal', 'task_id': task_id, 'gpu_index': int(gpu_index),
                'error': repr(exc), 'traceback': traceback.format_exc(),
            })
            return
        except Exception as exc:  # pragma: no cover - per-task failure surfaced to main
            import traceback
            result_queue.put({
                'type': 'result', 'task_id': task_id, 'gpu_index': int(gpu_index), 'ok': False,
                'error': repr(exc), 'traceback': traceback.format_exc(),
            })
        finally:
            _close_fd_list(transferred_task_fds)
    finally:
        _wait_for_deferred_publications()
        _shutdown_d1_worker_pipeline()
        _shutdown_resident_trt_pipeline_cache()
        _shutdown_gpu_union_retirement_manager()
        _close_fd_list(persistent_source_memfds.values())
        persistent_source_memfds.clear()


# Late imports keep callable-only dependency cycles import-safe.
from ._latebind import bind_late_symbols as _bind_late_symbols

_bind_late_symbols(
    __name__,
    globals(),
    {
        "backprojection": (
            "_ResidentTensorRTRingFatalError",
            "_shutdown_resident_trt_pipeline_cache",
        ),
        "config": (
            "DEFAULT_CHANNEL_FORMAT",
            "_resolve_cpu_precision",
            "quantize_uses_fp16",
            "resolve_channel_format",
            "resolve_quantize",
        ),
        "cuda_backend": (
            "GpuRenderedYoloSource",
            "GpuTileRenderedYoloSource",
            "_WORKER_TILTED_RADIAL_CPU_WARNED",
            "_init_worker_gpu_render_engine",
            "_radial_slab_channel_renderer",
            "_radial_slab_context_indices",
            "_wait_for_cube_ready_sentinel",
            "_worker_gpu_render_engine",
            "_worker_render_callable",
            "open_existing_gray_memmap",
            "set_gpu_worker_fused_preflight_specs",
        ),
        "cuda_d1": (
            "_d1_backproject_kernels",
            "_d1_consume_device_union",
            "_shutdown_d1_worker_pipeline",
            "d1_owner_pipeline_enabled",
        ),
        "geometry": (
            "AugJob",
            "DenseTileJob",
            "InMemoryYoloVolumeSource",
            "StreamingYoloVolumeSource",
            "ViewInfo",
            "is_radial_view",
            "is_tilted_radial_view",
            "is_tilted_view",
            "output_to_view_processing_affine",
            "radial_resident_gpu_render_supported",
            "radial_streaming_gpu_render_supported",
            "view_processing_volume_shape",
        ),
        "inference": (
            "CpuRetinaMaskPayload",
            "ModelInputChannelMismatchError",
            "PredictConfig",
            "_cleanup_prediction_slice_inplace",
            "_clip_boxes_np",
            "_gpu_union_retirement_manager",
            "_init_gpu_union_retirement_manager",
            "_process_cpu_retina_prediction_frame",
            "_shutdown_gpu_union_retirement_manager",
            "cpu_retina_masks_enabled",
            "ensure_cpu_retina_mask_predictor_patch",
            "ensure_gpu_retina_proto_union_predictor_patch",
            "ensure_yolo_ready_for_predict",
            "gpu_union_flush_overlap_enabled",
            "load_ultralytics_model",
            "predict_source_and_accumulate",
            "request_affine_grid_cache_entries",
            "require_channel_aware_yolo_preprocess_patch",
            "set_angle_variant_gpu_fastpath",
            "set_retina_mask_processor",
            "validate_yolo_model_input_channels",
        ),
        "runtime": (
            "HYBRID_DEFERRED_RESULT_MODE",
            "_close_fd_list",
            "_interpolation_process_entry",
            "_materialize_worker_task_memfd_paths",
            "_sched_setaffinity_all_threads",
            "close_memmap_array",
            "cpu_inference_supports_view",
            "initialize_runtime_observability",
        ),
    },
)
