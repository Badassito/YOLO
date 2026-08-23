"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import queue
import threading
from collections import OrderedDict
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
)
import numpy as np
from ._deps import cv2, ndi

# Explicit lower-layer dependencies keep imports one-way.
from .config import (
    GIB,
    quantize_uses_fp16,
    resolve_quantize,
)
from .workspace import (
    _env_flag,
    _env_int,
)
from .runtime import (
    _acquire_parallel_pool,
    _release_parallel_pool,
    _settle_parallel_futures,
    choose_parallel_chunk_size,
    choose_slice_parallel_workers,
    flush_array,
    parallel_for_indices_chunked,
    prediction_hot_path_flush_enabled,
)
from .geometry import (
    BatchResultFrameSpec,
    GpuPrefetchingYoloSource,
    InMemoryYoloVolumeSource,
    PredictionVolumeRef,
    StreamingYoloVolumeSource,
    ViewInfo,
    _cupy_external_stream,
    _source_prediction_channel_count,
    make_prediction_ref_yolo_source,
    maybe_wrap_source_with_gpu_input_staging,
    mirror_radial_u_output_to_native_affine,
    prediction_result_frame_spec,
    view_processing_min_radius,
)


if TYPE_CHECKING:
    from .cuda_backend import (
        GpuRenderedYoloSource,
        GpuTileRenderedYoloSource,
    )
    from .backprojection import (
        _trt_binding_layout_for_backend,
        _trt_engine_from_autobackend,
        _try_resident_trt_ring_accumulate,
    )

def load_ultralytics_model(path: str, task: str = 'segment'):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics\n"
            f"Import error: {e}"
        ) from e
    return YOLO(path, task=task)

def canonical_single_device(device: str) -> str:
    raw = str(device or '').strip()
    if not raw:
        return 'cpu'

    token = raw.split(',')[0].strip()
    low = token.lower()
    if low in ('cpu', 'mps'):
        return low
    if low.startswith('cuda'):
        return low
    if token.isdigit():
        return f'cuda:{token}'
    return token

_RETINA_MASK_PROCESSOR_IS_CPU: Optional[bool] = None

def set_retina_mask_processor(processor: str) -> None:
    """Set the resolved retina-mask processor ('cpu' or 'gpu') for the run."""
    global _RETINA_MASK_PROCESSOR_IS_CPU
    _RETINA_MASK_PROCESSOR_IS_CPU = bool(str(processor).strip().lower() == 'cpu')

_ANGLE_VARIANT_GPU_FASTPATH: Optional[Tuple[float, float]] = None

def gpu_retina_flatten_enabled() -> bool:
    """Flatten GPU retina masks (n,H,W) -> union + max-conf planes before PCIe copy.

 Active only when retina masks are resolved on the GPU (not cpu_retina_masks_enabled). The env
 flag YOLO_TTA_GPU_RETINA_FLATTEN (default on) allows forcing the legacy whole-stack copy for
 regression comparison."""
    return _env_flag('YOLO_TTA_GPU_RETINA_FLATTEN', True)

def gpu_retina_warp_enabled() -> bool:
    """Perform the flattened-plane affine warp to view-native space on the GPU.

 When enabled, the union and max-confidence planes are warped to view-native space on the GPU
 (torch grid_sample) before the host copy, so neither affine warp runs on the CPU and only the
 view-native planes cross PCIe. YOLO_TTA_GPU_RETINA_WARP=0 keeps the warps on the CPU (the
 flattened out-size planes are copied down and cv2.warpAffine'd) for regression comparison."""
    return _env_flag('YOLO_TTA_GPU_RETINA_WARP', True)

def gpu_retina_eager_flatten_enabled() -> bool:
    """Run the cheap GPU union/max-conf reduction on the model-stream thread.

 When on the GPU-flatten path, reducing each (n,Hr,Wr) stack to 2 small planes as results stream
 (rather than deferring the reduction to a postprocess worker) lets the full native-resolution
 retina-mask stack be released immediately, so at most ~the bounded pending count of small planes
 (not full stacks) stay resident on the GPU. YOLO_TTA_GPU_RETINA_EAGER_FLATTEN=0 defers it."""
    return _env_flag('YOLO_TTA_GPU_RETINA_EAGER_FLATTEN', True)

def gpu_retina_flatten_pending_limit(worker_count: int) -> int:
    """Bound on queued GPU-resident flattened frames, to cap GPU memory.

 The CPU-RAM-oriented cpu_mask_postprocess_pending_limit defaults to ~num_frames, which is unsafe
 for GPU-resident intermediates: queuing a whole view's worth of flattened planes (or, worse,
 un-reduced stacks) can OOM the device. This caps the in-flight count so GPU residency stays
 bounded while leaving enough work to keep the postprocess workers busy.
 YOLO_TTA_GPU_RETINA_PENDING_FRAMES overrides it."""
    wc = max(1, int(worker_count))
    return max(8, _env_int('YOLO_TTA_GPU_RETINA_PENDING_FRAMES', min(4 * wc, 256)))

def gpu_retina_cleanup_enabled() -> bool:
    """Return whether positive ``--min_radius`` cleanup may run through CuPy.
    
    Hole filling remains a completed-view or task-end operation so cleanup order is preserved."""
    return _env_flag('YOLO_TTA_GPU_RETINA_CLEANUP', True)

def gpu_retina_proto_union_enabled() -> bool:
    """Compute the GPU retina union at PROTO resolution inside construct_result.

 Active only in GPU retina mask mode. Instead of Ultralytics materializing an
 (n, imgsz, imgsz) float retina stack per image (a batch-scaled VRAM transient of
 batch x n x 16 MB that this pipeline immediately reduces to one plane), the patched
 construct_result box-crops the per-instance mask logits at proto scale, reduces them to a
 single max-logit plane, and bilinearly upsamples ONE plane to the network raster. Box-edge
 differences are sub-voxel scale. YOLO_TTA_GPU_PROTO_UNION=0 restores the native retina
 stack + flatten path."""
    return _env_flag('YOLO_TTA_GPU_PROTO_UNION', True)

def gpu_postprocess_side_stream_enabled() -> bool:
    """Run the GPU postprocess tail on a per-thread side CUDA stream."""
    return _env_flag('YOLO_TTA_GPU_POSTPROCESS_STREAM', True)

def gpu_postprocess_pinned_d2h_enabled() -> bool:
    """Stage device->host postprocess copies through per-thread pinned buffers."""
    return _env_flag('YOLO_TTA_GPU_POSTPROCESS_PINNED', True)

def set_angle_variant_gpu_fastpath(min_conf: Optional[float], min_radius: float = 0.0) -> None:
    """Set or clear the process-wide angle-variant GPU cleanup configuration."""
    global _ANGLE_VARIANT_GPU_FASTPATH
    if min_conf is None:
        _ANGLE_VARIANT_GPU_FASTPATH = None
    else:
        _ANGLE_VARIANT_GPU_FASTPATH = (float(min_conf), float(min_radius))

def angle_variant_gpu_fastpath() -> Optional[Tuple[float, float]]:
    """Return the active (min_conf, min_radius) fast-path config, or None when the fast path is off."""
    return _ANGLE_VARIANT_GPU_FASTPATH

def ensure_yolo_ready_for_predict(model: object, cfg: 'PredictConfig') -> None:
    """Keep the active YOLO backend resident on the requested device.

 The scheduling contract wants the GPU to stay hot until work is exhausted. Offloading the model
 between videos can leave Ultralytics with CUDA inputs but CPU weights on the next predict call.
 This helper lazily restores the requested device/dtype when needed and is a cheap no-op while the
 model already matches the active predict configuration."""
    try:
        import torch  # type: ignore
    except Exception:
        return

    target = canonical_single_device(str(cfg.device))
    quantize = resolve_quantize(cfg.quantize)
    wants_fp16 = quantize_uses_fp16(quantize) and str(target).startswith('cuda')
    state = (str(target), quantize)
    if getattr(model, '_tta_predict_state', None) == state:
        return

    candidates: List[object] = [model]
    direct_model = getattr(model, 'model', None)
    if direct_model is not None:
        candidates.append(direct_model)
    predictor = getattr(model, 'predictor', None)
    predictor_model = getattr(predictor, 'model', None) if predictor is not None else None
    if predictor_model is not None and predictor_model is not direct_model:
        candidates.append(predictor_model)

    seen_ids: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        cid = id(candidate)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        to_fn = getattr(candidate, 'to', None)
        if callable(to_fn):
            try:
                to_fn(target)
            except Exception:
                pass

        # Plain FP16/FP32 models may be placed eagerly. INT8 and mixed exported backends
        # own their binding precision and must not be coerced through Module.float/half.
        precision_fn = None
        if wants_fp16:
            precision_fn = getattr(candidate, 'half', None)
        elif quantize in (None, 32):
            precision_fn = getattr(candidate, 'float', None)
        if callable(precision_fn):
            try:
                precision_fn()
            except Exception:
                pass

    if predictor is not None:
        try:
            setattr(predictor, 'device', torch.device(target))
        except Exception:
            pass
        args_obj = getattr(predictor, 'args', None)
        if args_obj is not None:
            try:
                setattr(args_obj, 'device', target)
                setattr(args_obj, 'quantize', quantize)
            except Exception:
                pass

    try:
        setattr(model, '_tta_predict_state', state)
    except Exception:
        pass

class ModelInputChannelMismatchError(RuntimeError):
    """Selected --channel_format cannot satisfy the loaded model input binding."""

def _channel_count_from_bchw_shape(value: object) -> Optional[int]:
    try:
        shape = tuple(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if len(shape) != 4:
        return None
    try:
        # Batch and spatial dimensions are commonly symbolic in ONNX and
        # dynamic TensorRT exports; only C must be statically discoverable.
        channels = int(shape[1])
    except Exception:
        return None
    return channels if channels > 0 else None

def infer_yolo_model_input_channels(model: object) -> Tuple[Optional[int], str]:
    """Best-effort first-convolution or backend-binding channel discovery."""
    # Local import keeps the package dependency graph acyclic.
    from .backprojection import (
        _trt_binding_layout_for_backend,
        _trt_engine_from_autobackend,
    )

    candidates: List[Tuple[str, object]] = [('YOLO', model)]
    direct_model = getattr(model, 'model', None)
    if direct_model is not None:
        candidates.append(('YOLO.model', direct_model))
    predictor = getattr(model, 'predictor', None)
    backend = getattr(predictor, 'model', None) if predictor is not None else None
    if backend is not None:
        candidates.append(('predictor.model', backend))

    seen: set[int] = set()
    for label, candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        # PyTorch and TorchScript modules expose their first image convolution.
        modules_fn = getattr(candidate, 'modules', None)
        if callable(modules_fn):
            try:
                for module in modules_fn():
                    in_channels = getattr(module, 'in_channels', None)
                    weight = getattr(module, 'weight', None)
                    if in_channels is None or weight is None:
                        continue
                    weight_shape = tuple(int(v) for v in getattr(weight, 'shape', ()))
                    if len(weight_shape) == 4 and int(in_channels) > 0:
                        return int(in_channels), f'{label} first convolution'
            except Exception:
                pass

        for attr in ('input_shape', 'shape'):
            count = _channel_count_from_bchw_shape(getattr(candidate, attr, None))
            if count is not None:
                return int(count), f'{label}.{attr}'

        bindings = getattr(candidate, 'bindings', None)
        if isinstance(bindings, dict) and bindings:
            input_name = str(getattr(candidate, 'input_name', '') or '')
            if input_name in bindings:
                binding_items = [(input_name, bindings.get(input_name))]
            elif 'images' in bindings:
                binding_items = [('images', bindings.get('images'))]
            elif len(bindings) == 1:
                binding_items = list(bindings.items())
            else:
                # A multi-binding backend without an identified input may list
                # an output first (for example C=84/116 detections). Do not
                # mistake that output for the model input; the engine/session
                # probes below can identify the real input safely.
                binding_items = []
            for binding_name, binding in binding_items:
                count = _channel_count_from_bchw_shape(getattr(binding, 'shape', None))
                if count is not None:
                    return int(count), f'{label} binding {binding_name}'

        engine = _trt_engine_from_autobackend(candidate)
        if engine is not None:
            try:
                _names, input_name, _outputs, indices = _trt_binding_layout_for_backend(
                    candidate, engine
                )
                if callable(getattr(engine, 'get_tensor_shape', None)):
                    binding_shape = engine.get_tensor_shape(str(input_name))
                else:
                    binding_shape = engine.get_binding_shape(int(indices[input_name]))
                count = _channel_count_from_bchw_shape(binding_shape)
                if count is not None:
                    return int(count), f'{label} TensorRT binding {input_name}'
            except Exception:
                pass

        session = getattr(candidate, 'session', None)
        get_inputs = getattr(session, 'get_inputs', None)
        if callable(get_inputs):
            try:
                inputs = list(get_inputs())
                if inputs:
                    count = _channel_count_from_bchw_shape(getattr(inputs[0], 'shape', None))
                    if count is not None:
                        return int(count), f'{label} ONNX input {getattr(inputs[0], "name", "images")}'
            except Exception:
                pass
    return None, 'unresolved backend'

def validate_yolo_model_input_channels(
    model: object,
    required_channels: int,
    *,
    channel_token: str,
    context: str,
) -> Optional[int]:
    """Fail clearly when a discoverable model binding disagrees with input C."""
    required = max(1, int(required_channels))
    actual, source = infer_yolo_model_input_channels(model)
    if actual is None:
        warning_key = (int(required), str(channel_token))
        if getattr(model, '_tta_unresolved_input_channels_warning', None) != warning_key:
            print(
                f'Warning: {context}: could not statically resolve the loaded model input '
                f'channel count for --channel_format {channel_token} (C={int(required)}). '
                'The backend will enforce the binding shape on its first inference call.'
            )
            try:
                setattr(model, '_tta_unresolved_input_channels_warning', warning_key)
            except Exception:
                pass
        return None
    if int(actual) != int(required):
        raise ModelInputChannelMismatchError(
            f'{context}: --channel_format {channel_token} produces {int(required)} '
            f'input channel(s), but the loaded model expects {int(actual)} '
            f'({source}). Use a model trained/exported with channels={int(required)} '
            'or select a matching --channel_format.'
        )
    validation_key = (int(required), str(source))
    if getattr(model, '_tta_validated_input_channels', None) != validation_key:
        print(
            f'Model input-channel validation passed: {channel_token} -> C={int(required)}; '
            f'loaded model expects C={int(actual)} ({source}).'
        )
        try:
            setattr(model, '_tta_validated_input_channels', validation_key)
        except Exception:
            pass
    return int(actual)

def offload_between_jobs_enabled() -> bool:
    return _env_flag('YOLO_TTA_OFFLOAD_BETWEEN_JOBS', False)

def trim_cuda_memory() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return

    try:
        if not bool(torch.cuda.is_available()):
            return
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, 'ipc_collect', None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        pass

def offload_yolo_from_gpu(model: object) -> None:
    modules: List[object] = []
    direct_model = getattr(model, 'model', None)
    if direct_model is not None:
        modules.append(direct_model)

    predictor = getattr(model, 'predictor', None)
    predictor_model = getattr(predictor, 'model', None) if predictor is not None else None
    if predictor_model is not None and predictor_model is not direct_model:
        modules.append(predictor_model)

    for module in modules:
        try:
            to_fn = getattr(module, 'to', None)
            if callable(to_fn):
                to_fn('cpu')
        except Exception:
            pass

    try:
        setattr(model, '_tta_predict_state', None)
    except Exception:
        pass

    trim_cuda_memory()

def unload_yolo_model(model: object) -> None:
    offload_yolo_from_gpu(model)
    predictor = getattr(model, 'predictor', None)
    if predictor is not None:
        try:
            setattr(model, 'predictor', None)
        except Exception:
            pass
    trim_cuda_memory()

@dataclass
class PredictConfig:
    imgsz: int
    conf: float
    device: str
    quantize: int | str | None
    batch: int = 1
    input_channels: int = 1
    channel_token: str = 'gray'

def async_predict_postprocess_enabled() -> bool:
    """Return True when angle-variant prediction CPU tails may run behind the GPU."""
    return _env_flag('YOLO_TTA_ASYNC_PREDICT_POSTPROCESS', True)

def async_predict_join_workers(default_value: int) -> int:
    return max(1, _env_int('YOLO_TTA_ASYNC_PREDICT_JOIN_WORKERS', max(1, int(default_value))))

def async_predict_pending_frame_limit(num_frames: int) -> int:
    """Optional cap for queued async result-worker futures per source.

 The default 0 means source-sized buffering: the GPU-facing iterator will not
 intentionally wait for CPU result workers while a prediction volume streams.
 Set YOLO_TTA_ASYNC_PREDICT_PENDING_FRAMES to a positive value to reduce
 transient CPU/GPU result memory at the cost of some continuity."""
    requested = _env_int('YOLO_TTA_ASYNC_PREDICT_PENDING_FRAMES', 0)
    if int(requested) <= 0:
        return 0
    return max(1, min(max(1, int(num_frames)), int(requested)))

@dataclass
class PredictionAccumulationHandle:
    """Background CPU accumulation tail for one streamed prediction source."""
    source_label: str
    futures: List[Future]
    view_union_mm: np.ndarray
    view_confmap_mm: Optional[np.ndarray]
    submitted_frames: int = 0
    synthetic_discarded: int = 0
    precompleted_prediction_count: int = 0
    precompleted_frames_with_predictions: int = 0
    pending_limit: int = 0
    radial_padding_processed: int = 0
    radial_padding_union_mm: Optional[np.ndarray] = None
    radial_padding_confmap_mm: Optional[np.ndarray] = None

    def wait(self) -> Dict[str, int]:
        prediction_count = int(self.precompleted_prediction_count)
        frames_with_predictions = int(self.precompleted_frames_with_predictions)
        try:
            for fut in as_completed(list(self.futures)):
                pred_inc, frame_inc = fut.result()
                prediction_count += int(pred_inc)
                frames_with_predictions += int(frame_inc)
        finally:
            if prediction_hot_path_flush_enabled():
                if self.view_confmap_mm is not None:
                    flush_array(self.view_confmap_mm)
                flush_array(self.view_union_mm)
                if self.radial_padding_confmap_mm is not None:
                    flush_array(self.radial_padding_confmap_mm)
                if self.radial_padding_union_mm is not None:
                    flush_array(self.radial_padding_union_mm)
        return {
            'prediction_count': int(prediction_count),
            'frames_with_predictions': int(frames_with_predictions),
            'submitted_frames': int(self.submitted_frames),
            'synthetic_discarded': int(self.synthetic_discarded),
            'radial_padding_processed': int(self.radial_padding_processed),
            'async_accumulation': 1,
        }

DEFAULT_GAUSSIAN_SMOOTHING_SIGMA = 3.0

DEFAULT_GAUSSIAN_SMOOTHING_PASSES = 1

def resolve_gaussian_smoothing_settings(
    gaussian_smoothing_arg: Optional[float],
    gaussian_smoothing_passes_arg: Optional[int],
) -> Tuple[bool, float, int]:
    """Resolve smoothing enablement, sigma, and pass count from the two Gaussian flags."""
    sigma_explicit = gaussian_smoothing_arg is not None
    passes_explicit = gaussian_smoothing_passes_arg is not None
    sigma_f = (
        float(gaussian_smoothing_arg)
        if sigma_explicit
        else float(DEFAULT_GAUSSIAN_SMOOTHING_SIGMA)
    )
    passes_i = (
        int(gaussian_smoothing_passes_arg)
        if passes_explicit
        else int(DEFAULT_GAUSSIAN_SMOOTHING_PASSES)
    )
    enabled = bool((sigma_explicit or passes_explicit) and sigma_f > 0.0 and passes_i > 0)
    if not enabled:
        return False, 0.0, max(0, int(passes_i))
    return True, float(sigma_f), int(passes_i)

CONF_U8_MAX = 255

def quantize_conf_to_u8(conf: float) -> np.uint8:
    conf_clamped = min(1.0, max(0.0, float(conf)))
    return np.uint8(int(round(conf_clamped * float(CONF_U8_MAX))))

def min_conf_to_u8_threshold(min_conf: float) -> int:
    conf_clamped = min(1.0, max(0.0, float(min_conf)))
    return int(math.ceil(conf_clamped * float(CONF_U8_MAX) - 1e-9))

@dataclass(frozen=True)
class CpuRetinaMaskPayload:
    """CPU-owned YOLO segmentation tensors needed to reconstruct retina/native masks off-GPU."""

    proto: np.ndarray             # (C, mask_h, mask_w), float32 CPU
    coeffs: np.ndarray            # (N, C), float32 CPU
    boxes_xyxy: np.ndarray        # (N, 4), scaled to prediction-video pixel coordinates
    confs: np.ndarray             # (N,), float32 CPU
    orig_shape: Tuple[int, int]   # (height, width) of the prediction-video frame
    img_shape: Tuple[int, int]    # (height, width) of the network input tensor
    frame_path: str = ''

@dataclass
class GpuFlattenedRetinaPayload:
    """GPU-resident per-frame union and optional confidence payload.
    
    The payload records device-side counts, cleanup state, readiness events, and references needed to keep asynchronous tensors alive."""

    union_gpu: object
    conf_gpu: Optional[object]
    instance_count: int = 0
    # generic direct-backend compaction retains its scalar count on device.
    # The device-union path records it into a per-frame task tensor and reads totals once at
    # task end; host-output fallbacks stage it alongside their unavoidable mask D2H.
    instance_count_device: Optional[object] = None
    min_conf_applied: bool = False
    run_gpu_cleanup: bool = False
    gpu_min_radius: float = 0.0
    cleanup_done_on_gpu: bool = False
    # CUDA event recorded on the producing stream after union/conf were computed,
    # so the postprocess side stream can order against the producer without a host sync.
    ready_event: Optional[object] = None
    # set by the frame processor when this frame's native plane was written into
    # the per-task device union accumulator (no host write happened; the task-end flush merges).
    accumulated_on_device: bool = False
    # Keep raw head/proto and compaction workspaces alive until ready_event has ordered every
    # consumer. CuPy launches are invisible to torch's allocator stream tracking by themselves.
    device_refs: Optional[Tuple[object, ...]] = field(default=None, repr=False)

@dataclass(frozen=True)
class DeferredCpuRetinaMaskPayload:
    """Compact GPU tensors captured from Ultralytics and copied in result-worker threads."""

    pred: object
    proto: object
    orig_shape: Tuple[int, int]
    img_shape: Tuple[int, int]
    frame_path: str = ''

_ULTRALYTICS_CPU_RETINA_PATCHED = False

_DEFERRED_CLONE_FALLBACK_WARNED = False

def _detach_clone_tensor_if_torch(value: object) -> object:
    """Detach+clone a torch tensor so a deferred payload owns its storage.

 Buffer-reusing inference backends (e.g. TensorRT engines through AutoBackend) hand
 construct_result views of output bindings that the NEXT batch's execute overwrites in
 place. A payload whose realization is deferred to a worker thread must never alias
 those bindings, or it reads a later frame's protos. The clone is a
 device-side copy queued on the current stream — it does not synchronize the host, so
 the deferral still keeps the GPU->CPU copy off the model-stream thread.

 Non-tensor values pass through unchanged. Clone failures (e.g. a transient CUDA OOM)
 PROPAGATE: silently returning the aliased binding would reinstate exactly the
 corruption this exists to prevent — the caller falls back to synchronous
 realization instead."""
    detach = getattr(value, 'detach', None)
    if not callable(detach):
        return value
    detached = detach()
    clone = getattr(detached, 'clone', None)
    if not callable(clone):
        return detached
    return clone()

def cpu_retina_masks_enabled() -> bool:
    """Return the backend-resolved retina-mask placement.

    Parent, CUDA-worker, and OpenVINO-worker initialization publish the backend-local setting.
    CPU remains the safe default only for isolated helper calls made before initialization.
    """
    if _RETINA_MASK_PROCESSOR_IS_CPU is not None:
        return bool(_RETINA_MASK_PROCESSOR_IS_CPU)
    return True

def cpu_retina_roi_only_enabled() -> bool:
    """Use bbox-ROI-only CPU upsampling instead of reconstructing every full-size instance mask."""
    return _env_flag('YOLO_TTA_CPU_RETINA_ROI_ONLY', True)

def cpu_retina_block_detections() -> int:
    """Number of mask logits to reconstruct per CPU matrix-multiply block."""
    return max(1, _env_int('YOLO_TTA_CPU_RETINA_BLOCK_DETECTIONS', 8))

def cpu_retina_deferred_payload_enabled() -> bool:
    """Defer compact pred/proto CPU copies from Ultralytics construct_result to result workers."""
    return _env_flag('YOLO_TTA_CPU_RETINA_DEFER_GPU_COPY', True)

def cpu_mask_postprocess_pending_limit(worker_count: int, num_frames: int) -> int:
    """Bound queued CPU mask-reconstruction frames while allowing RAM-backed buffering.

 Setting YOLO_TTA_CPU_MASK_PENDING_FRAMES=0 removes the frame-count cap for sites that
 intentionally want to absorb the entire CPU backlog in RAM. The default is deliberately much
 larger than the worker count so GPU inference is not throttled by CPU retina reconstruction under
 normal SLURM allocations."""
    workers = max(1, int(worker_count))
    frames = max(1, int(num_frames))
    requested = _env_int('YOLO_TTA_CPU_MASK_PENDING_FRAMES', 0)
    if int(requested) <= 0:
        return max(workers, frames)
    return max(workers, min(frames, max(int(requested), workers * 2)))

_ULTRALYTICS_CHANNEL_AWARE_PREPROCESS_PATCHED = False

def ensure_channel_aware_yolo_preprocess_patch() -> bool:
    """Preserve H×W×C in-memory input order when constructing BCHW tensors.

 Stock Ultralytics' ordinary NumPy-image path assumes BGR-like three-channel
 images and reverses the trailing axis. That would reverse a C...S... neighbor
 stack. Marked batches therefore bypass the color conversion and perform
 an explicit BHWC->BCHW move without changing channel order. Unrelated stock
 image/video sources remain on the original preprocessing path."""
    global _ULTRALYTICS_CHANNEL_AWARE_PREPROCESS_PATCHED

    if _ULTRALYTICS_CHANNEL_AWARE_PREPROCESS_PATCHED:
        return True

    try:
        import torch  # type: ignore
        from ultralytics.engine.predictor import BasePredictor  # type: ignore
    except Exception as exc:  # pragma: no cover - ultralytics is imported lazily on SLURM
        print(f'Warning: channel-aware YOLO preprocess patch could not be installed ({exc})')
        return False

    original_preprocess = BasePredictor.preprocess

    def _tta_channel_aware_preprocess(self, im):  # type: ignore[no-untyped-def]
        gpu_tensor = getattr(im, '_tta_gpu_tensor', None)
        if gpu_tensor is not None:
            try:
                wait_fn = getattr(im, 'wait_ready', None)
                if callable(wait_fn):
                    tensor = wait_fn()
                else:
                    tensor = gpu_tensor
                model_obj = getattr(self, 'model', None)
                use_half = bool(getattr(model_obj, 'fp16', False))
                if use_half and getattr(tensor, 'dtype', None) != torch.float16:
                    tensor = tensor.half()
                elif (not use_half) and getattr(tensor, 'dtype', None) != torch.float32:
                    tensor = tensor.float()
                # GPU-staged batches are already normalized to [0,1].
                return tensor
            except Exception as exc:
                raise RuntimeError(f'Failed to consume GPU-staged YOLO input batch: {exc}') from exc

        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            marked_channels = getattr(im, '_tta_channel_count', None)
            if isinstance(im, np.ndarray):
                seq = [im]
            elif isinstance(im, (list, tuple)):
                seq = list(im)
            else:
                seq = []

            # Preserve legacy one-channel compatibility even for unmarked callers.
            if seq and marked_channels is None:
                first = np.asarray(seq[0])
                if first.ndim == 2 or (first.ndim == 3 and int(first.shape[2]) == 1):
                    marked_channels = 1

            if seq and marked_channels is not None:
                channel_count = max(1, int(marked_channels))
                channel_frames: List[np.ndarray] = []
                valid_batch = True
                for item in seq:
                    arr = np.asarray(item)
                    if arr.ndim == 2:
                        if channel_count != 1:
                            valid_batch = False
                            break
                        formatted = arr[:, :, None]
                    elif arr.ndim == 3 and int(arr.shape[2]) == channel_count:
                        formatted = arr
                    else:
                        valid_batch = False
                        break
                    channel_frames.append(np.ascontiguousarray(formatted, dtype=np.uint8))

                if valid_batch:
                    try:
                        shapes = {tuple(int(v) for v in frame.shape) for frame in channel_frames}
                        if len(shapes) != 1:
                            raise ValueError(
                                f'channel-formatted batch contains mixed shapes: {sorted(shapes)}'
                            )
                        batch = np.moveaxis(np.stack(channel_frames, axis=0), -1, 1)
                        tensor = torch.from_numpy(np.ascontiguousarray(batch))
                        device = getattr(self, 'device', None)
                        if device is not None:
                            tensor = tensor.to(device)
                        model_obj = getattr(self, 'model', None)
                        use_half = bool(getattr(model_obj, 'fp16', False))
                        tensor = tensor.half() if use_half else tensor.float()
                        tensor /= 255.0
                        return tensor
                    except Exception as exc:
                        raise RuntimeError(
                            f'Failed to preprocess {int(channel_count)}-channel in-memory '
                            f'YOLO batch without reordering: {exc}'
                        ) from exc
                raise ValueError(
                    f'Channel-formatted YOLO batch does not consistently contain '
                    f'HxWx{int(channel_count)} uint8 images'
                )

        return original_preprocess(self, im)

    BasePredictor.preprocess = _tta_channel_aware_preprocess
    _ULTRALYTICS_CHANNEL_AWARE_PREPROCESS_PATCHED = True
    print(
        'Channel-aware YOLO preprocess enabled: v15 in-memory H×W×C inputs are '
        'passed as BCHW tensors without BGR/RGB channel reversal.'
    )
    return True

def require_channel_aware_yolo_preprocess_patch(channel_token: str) -> None:
    """Fail before inference rather than silently reordering an H×W×C stack."""
    try:
        installed = bool(ensure_channel_aware_yolo_preprocess_patch())
    except Exception as exc:
        raise RuntimeError(
            f'--channel_format {channel_token} requires the v15 channel-aware '
            f'Ultralytics preprocessing patch, but installation failed: {exc}'
        ) from exc
    if not installed:
        raise RuntimeError(
            f'--channel_format {channel_token} requires the v15 channel-aware '
            'Ultralytics preprocessing patch. Verify the installed Ultralytics '
            'version exposes ultralytics.engine.predictor.BasePredictor.preprocess.'
        )

def _as_numpy_float32_cpu(x: object) -> np.ndarray:
    """Detach torch/array-like tensors into owned, contiguous CPU float32 NumPy arrays."""
    try:
        detach = getattr(x, 'detach', None)
        if callable(detach):
            x = detach()
        to_fn = getattr(x, 'to', None)
        if callable(to_fn):
            try:
                import torch  # type: ignore
                x = to_fn(device='cpu', dtype=torch.float32)
            except Exception:
                cpu_fn = getattr(x, 'cpu', None)
                if callable(cpu_fn):
                    x = cpu_fn()
        cpu_fn = getattr(x, 'cpu', None)
        if callable(cpu_fn):
            x = cpu_fn()
        numpy_fn = getattr(x, 'numpy', None)
        if callable(numpy_fn):
            arr = numpy_fn()
        else:
            arr = np.asarray(x)
    except Exception:
        arr = np.asarray(x)
    return np.ascontiguousarray(arr, dtype=np.float32)

def _clip_boxes_np(boxes: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    h = int(shape[0])
    w = int(shape[1])
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(w))
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(h))
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(w))
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(h))
    return boxes

def _scale_boxes_np(
    img1_shape: Sequence[int],
    boxes: np.ndarray,
    img0_shape: Sequence[int],
    *,
    padding: bool = True,
) -> np.ndarray:
    """NumPy equivalent of Ultralytics ops.scale_boxes for xyxy boxes."""
    if boxes.size <= 0:
        return boxes.astype(np.float32, copy=False)

    img1_h, img1_w = int(img1_shape[0]), int(img1_shape[1])
    img0_h, img0_w = int(img0_shape[0]), int(img0_shape[1])
    gain = min(float(img1_h) / max(1.0, float(img0_h)), float(img1_w) / max(1.0, float(img0_w)))
    if gain <= 0.0:
        return _clip_boxes_np(boxes.astype(np.float32, copy=False), img0_shape)

    pad_x = round((float(img1_w) - round(float(img0_w) * gain)) / 2.0 - 0.1)
    pad_y = round((float(img1_h) - round(float(img0_h) * gain)) / 2.0 - 0.1)
    out = boxes.astype(np.float32, copy=True)
    if bool(padding):
        out[:, [0, 2]] -= float(pad_x)
        out[:, [1, 3]] -= float(pad_y)
    out[:, :4] /= float(gain)
    return _clip_boxes_np(out, img0_shape)

def _scale_masks_crop_slices_np(mask_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> Tuple[slice, slice]:
    """Return the low-resolution crop used by Ultralytics scale_masks before interpolation."""
    im1_h, im1_w = int(mask_shape[0]), int(mask_shape[1])
    im0_h, im0_w = int(target_shape[0]), int(target_shape[1])
    if im1_h == im0_h and im1_w == im0_w:
        return slice(0, im1_h), slice(0, im1_w)

    gain = min(float(im1_h) / max(1.0, float(im0_h)), float(im1_w) / max(1.0, float(im0_w)))
    pad_w = (float(im1_w) - round(float(im0_w) * gain)) / 2.0
    pad_h = (float(im1_h) - round(float(im0_h) * gain)) / 2.0
    top = int(round(pad_h - 0.1))
    left = int(round(pad_w - 0.1))
    bottom = int(im1_h - round(pad_h + 0.1))
    right = int(im1_w - round(pad_w + 0.1))
    top = int(np.clip(top, 0, im1_h))
    left = int(np.clip(left, 0, im1_w))
    bottom = int(np.clip(bottom, top + 1, im1_h)) if im1_h > 0 else 0
    right = int(np.clip(right, left + 1, im1_w)) if im1_w > 0 else 0
    return slice(top, bottom), slice(left, right)

def _bbox_to_integer_roi(box_xyxy: np.ndarray, target_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    h, w = int(target_shape[0]), int(target_shape[1])
    if h <= 0 or w <= 0:
        return None
    x1 = int(math.ceil(max(0.0, float(box_xyxy[0]))))
    y1 = int(math.ceil(max(0.0, float(box_xyxy[1]))))
    x2 = int(math.ceil(min(float(w), float(box_xyxy[2]))))
    y2 = int(math.ceil(min(float(h), float(box_xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return y1, y2, x1, x2

def _resize_lowres_logits_roi(
    low_logits: np.ndarray,
    target_shape: Tuple[int, int],
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """Upsample only the requested high-resolution ROI using align-corners=False geometry.

 This avoids allocating an N x H x W tensor for hundreds of detections. It is equivalent in
 coordinate mapping to resizing the low-resolution retina logits to the prediction-video frame and
 then cropping the bbox ROI, while doing the expensive interpolation only inside the bbox."""
    y1, y2, x1, x2 = (int(v) for v in roi)
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    low = np.asarray(low_logits, dtype=np.float32)
    low_h, low_w = int(low.shape[0]), int(low.shape[1])
    roi_h = int(y2 - y1)
    roi_w = int(x2 - x1)

    if roi_h <= 0 or roi_w <= 0 or low_h <= 0 or low_w <= 0:
        return np.zeros((max(0, roi_h), max(0, roi_w)), dtype=np.float32)

    if low_h == target_h and low_w == target_w:
        return np.ascontiguousarray(low[y1:y2, x1:x2], dtype=np.float32)

    if not cpu_retina_roi_only_enabled():
        full = cv2.resize(low, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(full[y1:y2, x1:x2], dtype=np.float32)

    # Destination pixel-center to source-coordinate mapping for bilinear resize with
    # align_corners=False. Coordinates outside the low-resolution raster are edge-clamped,
    # matching torch.nn.functional.interpolate and cv2.resize behavior.
    xs = ((np.arange(x1, x2, dtype=np.float32) + 0.5) * (float(low_w) / float(target_w))) - 0.5
    ys = ((np.arange(y1, y2, dtype=np.float32) + 0.5) * (float(low_h) / float(target_h))) - 0.5
    xs = np.clip(xs, 0.0, float(low_w - 1))
    ys = np.clip(ys, 0.0, float(low_h - 1))

    x0 = np.floor(xs).astype(np.int32, copy=False)
    y0 = np.floor(ys).astype(np.int32, copy=False)
    x1_idx = np.minimum(x0 + 1, low_w - 1).astype(np.int32, copy=False)
    y1_idx = np.minimum(y0 + 1, low_h - 1).astype(np.int32, copy=False)
    wx = (xs - x0.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    wy = (ys - y0.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    top = (low[y0[:, None], x0[None, :]] * (1.0 - wx[None, :])) + (low[y0[:, None], x1_idx[None, :]] * wx[None, :])
    bottom = (low[y1_idx[:, None], x0[None, :]] * (1.0 - wx[None, :])) + (low[y1_idx[:, None], x1_idx[None, :]] * wx[None, :])
    return np.ascontiguousarray((top * (1.0 - wy[:, None])) + (bottom * wy[:, None]), dtype=np.float32)

def _iter_cpu_retina_payload_rois(
    payload: CpuRetinaMaskPayload,
    target_shape: Tuple[int, int],
) -> Iterator[Tuple[int, float, int, int, int, int, np.ndarray]]:
    """Yield per-instance high-resolution ROI masks reconstructed from CPU protos/coefficients."""
    proto = np.asarray(payload.proto, dtype=np.float32)
    if proto.ndim == 4 and int(proto.shape[0]) == 1:
        proto = proto[0]
    if proto.ndim != 3:
        return

    coeffs = np.asarray(payload.coeffs, dtype=np.float32)
    boxes = np.asarray(payload.boxes_xyxy, dtype=np.float32)
    confs = np.asarray(payload.confs, dtype=np.float32)
    if coeffs.ndim != 2 or coeffs.shape[0] <= 0 or coeffs.shape[1] != int(proto.shape[0]):
        return

    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        return

    mask_crop_y, mask_crop_x = _scale_masks_crop_slices_np(
        (int(proto.shape[1]), int(proto.shape[2])),
        (target_h, target_w),
    )
    c, mh, mw = int(proto.shape[0]), int(proto.shape[1]), int(proto.shape[2])
    proto_flat = np.ascontiguousarray(proto.reshape(c, mh * mw), dtype=np.float32)
    block = cpu_retina_block_detections()
    n = int(coeffs.shape[0])

    for start in range(0, n, block):
        stop = min(n, start + block)
        logits_block = np.matmul(
            np.ascontiguousarray(coeffs[start:stop], dtype=np.float32),
            proto_flat,
        ).reshape((stop - start, mh, mw))

        for local_idx in range(stop - start):
            inst_idx = int(start + local_idx)
            if inst_idx >= int(boxes.shape[0]):
                continue
            roi = _bbox_to_integer_roi(boxes[inst_idx], (target_h, target_w))
            if roi is None:
                continue
            y1, y2, x1, x2 = roi
            low_logits = np.ascontiguousarray(logits_block[local_idx, mask_crop_y, mask_crop_x], dtype=np.float32)
            roi_logits = _resize_lowres_logits_roi(low_logits, (target_h, target_w), roi)
            roi_mask = np.asarray(roi_logits > 0.0, dtype=bool)
            if not np.any(roi_mask):
                continue
            conf_val = float(confs[inst_idx]) if inst_idx < int(confs.shape[0]) else 0.0
            yield inst_idx, conf_val, y1, y2, x1, x2, roi_mask

def _accumulate_cpu_retina_payload_to_prediction_frame(
    payload: CpuRetinaMaskPayload,
    out_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build a prediction-space union/confidence frame from CPU-side retina payload tensors."""
    target_shape = (int(out_size), int(out_size))
    frame_union = np.zeros(target_shape, dtype=np.uint8)
    frame_confmap = np.zeros(target_shape, dtype=np.uint8)
    kept_instances = 0

    # The generated inference videos are square --imgsz frames. If a future source ever reports a
    # different original shape, keep the current --imgsz target because the affine matrices in this
    # pipeline are defined in that prediction-video coordinate system.
    for _inst_idx, conf_val, y1, y2, x1, x2, roi_mask in _iter_cpu_retina_payload_rois(payload, target_shape):
        roi_u8 = roi_mask.astype(np.uint8, copy=False)
        frame_union[y1:y2, x1:x2] |= roi_u8
        conf_u8 = quantize_conf_to_u8(float(conf_val))
        conf_patch = frame_confmap[y1:y2, x1:x2]
        conf_patch[roi_mask] = np.maximum(conf_patch[roi_mask], conf_u8)
        kept_instances += 1

    return frame_union, frame_confmap, int(kept_instances)

def _realize_deferred_cpu_retina_payload(payload: DeferredCpuRetinaMaskPayload) -> CpuRetinaMaskPayload:
    """Copy compact segmentation tensors to CPU and build a CPU-retina payload."""
    pred_cpu = _as_numpy_float32_cpu(payload.pred)
    if pred_cpu.ndim != 2 or int(pred_cpu.shape[0]) <= 0:
        return CpuRetinaMaskPayload(
            proto=np.zeros((0, 0, 0), dtype=np.float32),
            coeffs=np.zeros((0, 0), dtype=np.float32),
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            confs=np.zeros((0,), dtype=np.float32),
            orig_shape=(int(payload.orig_shape[0]), int(payload.orig_shape[1])),
            img_shape=(int(payload.img_shape[0]), int(payload.img_shape[1])),
            frame_path=str(payload.frame_path),
        )

    boxes_scaled = _scale_boxes_np(payload.img_shape, pred_cpu[:, :4], payload.orig_shape)
    pred_cpu[:, :4] = boxes_scaled
    proto_cpu = _as_numpy_float32_cpu(payload.proto)
    if proto_cpu.ndim == 4 and int(proto_cpu.shape[0]) == 1:
        proto_cpu = proto_cpu[0]
    if proto_cpu.ndim != 3:
        proto_cpu = np.zeros((0, 0, 0), dtype=np.float32)
    coeffs = pred_cpu[:, 6:] if int(pred_cpu.shape[1]) > 6 else np.zeros((int(pred_cpu.shape[0]), 0), dtype=np.float32)
    confs = pred_cpu[:, 4] if int(pred_cpu.shape[1]) > 4 else np.zeros((int(pred_cpu.shape[0]),), dtype=np.float32)
    return CpuRetinaMaskPayload(
        proto=np.ascontiguousarray(proto_cpu, dtype=np.float32),
        coeffs=np.ascontiguousarray(coeffs, dtype=np.float32),
        boxes_xyxy=np.ascontiguousarray(pred_cpu[:, :4], dtype=np.float32),
        confs=np.ascontiguousarray(confs, dtype=np.float32),
        orig_shape=(int(payload.orig_shape[0]), int(payload.orig_shape[1])),
        img_shape=(int(payload.img_shape[0]), int(payload.img_shape[1])),
        frame_path=str(payload.frame_path),
    )

def ensure_cpu_retina_mask_predictor_patch() -> bool:
    """Patch Ultralytics segmentation postprocess to return CPU-retina payloads, not GPU masks."""
    global _ULTRALYTICS_CPU_RETINA_PATCHED

    if not cpu_retina_masks_enabled():
        return False
    if _ULTRALYTICS_CPU_RETINA_PATCHED:
        return True

    try:
        from ultralytics.engine.results import Results  # type: ignore
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor  # type: ignore
    except Exception as exc:
        print(f'Warning: CPU retina-mask predictor patch could not be installed; falling back to Ultralytics masks ({exc})')
        return False

    original_construct_result = SegmentationPredictor.construct_result

    def _tta_cpu_retina_construct_result(self, pred, img, orig_img, img_path, proto):  # type: ignore[no-untyped-def]
        if not cpu_retina_masks_enabled():
            return original_construct_result(self, pred, img, orig_img, img_path, proto)

        try:
            img_shape = tuple(int(x) for x in img.shape[2:])
        except Exception:
            img_shape = (int(getattr(orig_img, 'shape', (0, 0))[0]), int(getattr(orig_img, 'shape', (0, 0))[1]))
        try:
            orig_shape = (int(orig_img.shape[0]), int(orig_img.shape[1]))
        except Exception:
            orig_shape = (int(img_shape[0]), int(img_shape[1]))

        if bool(cpu_retina_deferred_payload_enabled()):
            # Keep the Results object lightweight; compact pred/proto tensors are realized in
            # the prediction-result worker, not inside Ultralytics construct_result.
            try:
                deferred_payload: Optional[DeferredCpuRetinaMaskPayload] = DeferredCpuRetinaMaskPayload(
                    pred=_detach_clone_tensor_if_torch(pred),
                    proto=_detach_clone_tensor_if_torch(proto),
                    orig_shape=(int(orig_shape[0]), int(orig_shape[1])),
                    img_shape=(int(img_shape[0]), int(img_shape[1])),
                    frame_path=str(img_path),
                )
            except Exception as clone_exc:
                # a failed clone (e.g. transient CUDA OOM) must NOT fall
                # back to aliasing the backend's reusable output bindings — that silently
                # reinstates the corruption. Realize this frame synchronously below (the
                # CPU copy completes before the next batch executes).
                global _DEFERRED_CLONE_FALLBACK_WARNED
                if not _DEFERRED_CLONE_FALLBACK_WARNED:
                    _DEFERRED_CLONE_FALLBACK_WARNED = True
                    print(
                        f'Warning: deferred CPU retina payload clone failed ({clone_exc}); '
                        'falling back to synchronous realization for affected frames.'
                    )
                deferred_payload = None
            if deferred_payload is not None:
                result = Results(orig_img, path=img_path, names=self.model.names, boxes=None, masks=None)
                setattr(result, '_tta_deferred_cpu_retina_payload', deferred_payload)
                return result

        payload = _realize_deferred_cpu_retina_payload(DeferredCpuRetinaMaskPayload(
            pred=pred,
            proto=proto,
            orig_shape=(int(orig_shape[0]), int(orig_shape[1])),
            img_shape=(int(img_shape[0]), int(img_shape[1])),
            frame_path=str(img_path),
        ))
        boxes_for_result = np.zeros((0, 6), dtype=np.float32)
        if payload.confs.size > 0 and payload.boxes_xyxy.shape[0] == payload.confs.shape[0]:
            boxes_for_result = np.concatenate(
                [payload.boxes_xyxy, payload.confs[:, None], np.zeros((payload.confs.shape[0], 1), dtype=np.float32)],
                axis=1,
            ).astype(np.float32, copy=False)
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=boxes_for_result, masks=None)
        setattr(result, '_tta_cpu_retina_payload', payload)
        return result

    SegmentationPredictor.construct_result = _tta_cpu_retina_construct_result
    _ULTRALYTICS_CPU_RETINA_PATCHED = True
    print(
        'Deferred CPU retina masks enabled: Ultralytics GPU mask upsampling is bypassed; '
        'compact mask protos/coefficients are copied and reconstructed in prediction-result workers.'
    )
    return True

_ULTRALYTICS_GPU_PROTO_UNION_PATCHED = False

def _build_gpu_flattened_payload_from_proto(pred: object, img: object, proto: object) -> Optional['GpuFlattenedRetinaPayload']:
    """Build the flattened union payload at PROTO resolution.

 Instead of letting Ultralytics materialize the (n, imgsz, imgsz) float retina stack that the
 pipeline immediately reduces to one plane, the per-instance mask logits are combined from
 the protos, box-cropped at proto scale, reduced with a single max-logit plane, and ONE plane
 is bilinearly upsampled to the network raster and thresholded at 0 (union(bilinear(l_i)>0)
 becomes bilinear(max_i l_i)>0 — identical away from instance box edges, sub-voxel there).
 Instance-level --min_conf (angle-variant fast path) and the optional per-pixel max-confidence
 plane are applied at proto resolution. Returns None on any unexpected condition so the
 caller falls back to the unpatched Ultralytics path."""
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception:
        return None
    if not isinstance(pred, torch.Tensor) or not isinstance(proto, torch.Tensor):
        return None
    try:
        proto_t = proto
        if proto_t.ndim == 4 and int(proto_t.shape[0]) == 1:
            proto_t = proto_t[0]
        if proto_t.ndim != 3 or pred.ndim != 2:
            return None
        img_h = int(img.shape[2])
        img_w = int(img.shape[3])
        if img_h <= 0 or img_w <= 0:
            return None
        c, mh, mw = (int(proto_t.shape[0]), int(proto_t.shape[1]), int(proto_t.shape[2]))
        if c <= 0 or mh <= 0 or mw <= 0 or int(pred.shape[1]) < 6 + c:
            return None

        fastpath = angle_variant_gpu_fastpath()
        fastpath_min_conf = None if fastpath is None else float(fastpath[0])
        fastpath_min_radius = 0.0 if fastpath is None else float(fastpath[1])
        run_gpu_cleanup = bool(
            fastpath is not None and gpu_retina_cleanup_enabled() and float(fastpath_min_radius) > 0.0
        )

        pred_t = pred
        n = int(pred_t.shape[0])
        confs_t = pred_t[:, 4].to(torch.float32).reshape(-1) if n > 0 else None

        # Angle-variant fast path: drop low-confidence instances before the union (exact
        # instance-level --min_conf, mirroring _try_flatten_gpu_retina_result).
        min_conf_applied = False
        if (
            fastpath_min_conf is not None
            and float(fastpath_min_conf) > 0.0
            and confs_t is not None
            and n > 0
        ):
            keep = confs_t >= float(fastpath_min_conf)
            pred_t = pred_t[keep]
            confs_t = confs_t[keep]
            n = int(pred_t.shape[0])
            min_conf_applied = True

        if n <= 0:
            return GpuFlattenedRetinaPayload(
                union_gpu=None, conf_gpu=None, instance_count=0,
                min_conf_applied=min_conf_applied, run_gpu_cleanup=run_gpu_cleanup,
                gpu_min_radius=float(fastpath_min_radius),
            )

        coeffs = pred_t[:, 6:6 + c].to(torch.float32)
        logits = (coeffs @ proto_t.to(torch.float32).view(c, -1)).view(n, mh, mw)

        # Box crop at proto scale (Ultralytics crop_mask semantics: r >= x1 and r < x2).
        boxes = pred_t[:, :4].to(torch.float32)
        sx = float(mw) / float(img_w)
        sy = float(mh) / float(img_h)
        x1 = (boxes[:, 0] * sx).view(-1, 1, 1)
        y1 = (boxes[:, 1] * sy).view(-1, 1, 1)
        x2 = (boxes[:, 2] * sx).view(-1, 1, 1)
        y2 = (boxes[:, 3] * sy).view(-1, 1, 1)
        cols = torch.arange(mw, device=logits.device, dtype=torch.float32).view(1, 1, -1)
        rows = torch.arange(mh, device=logits.device, dtype=torch.float32).view(1, -1, 1)
        inside = (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)

        # Crop fill: a bounded "confident background" logit rather than -inf. An extreme fill
        # bleeds through the bilinear upsample and erodes up to a full proto cell of box
        # interior; -6 (sigmoid ~0.0025) keeps real logit zero-crossings exact while limiting
        # crop-edge bleed to ~1 px at the network raster. Ultralytics' own non-retina path
        # (ops.process_mask) also crops at proto resolution, with a multiplicative zero fill.
        neg_fill = logits.new_full((), -6.0)
        max_logit = torch.where(inside, logits, neg_fill).amax(dim=0)  # (mh, mw)
        union_gpu = (
            F.interpolate(
                max_logit.reshape(1, 1, mh, mw), size=(img_h, img_w),
                mode='bilinear', align_corners=False,
            ).reshape(img_h, img_w) > 0.0
        ).to(torch.float32)

        conf_gpu = None
        if confs_t is not None and gpu_flatten_conf_tracking_enabled():
            inst_proto = (logits > 0.0) & inside
            conf_plane = (confs_t.clamp(0.0, 1.0).view(-1, 1, 1) * inst_proto).amax(dim=0)
            conf_gpu = F.interpolate(
                conf_plane.reshape(1, 1, mh, mw), size=(img_h, img_w), mode='nearest',
            ).reshape(img_h, img_w)

        ready_event = None
        try:
            if union_gpu.is_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(union_gpu.device))
        except Exception:
            ready_event = None
        return GpuFlattenedRetinaPayload(
            union_gpu=union_gpu, conf_gpu=conf_gpu, instance_count=int(n),
            min_conf_applied=min_conf_applied, run_gpu_cleanup=run_gpu_cleanup,
            gpu_min_radius=float(fastpath_min_radius),
            ready_event=ready_event,
        )
    except Exception:
        return None

def ensure_gpu_retina_proto_union_predictor_patch() -> bool:
    """Patch Ultralytics segmentation postprocess for GPU retina mode.

 Mirrors ensure_cpu_retina_mask_predictor_patch: construct_result returns a lightweight
 Results carrying a prebuilt GpuFlattenedRetinaPayload (union computed at proto resolution)
 instead of an (n, imgsz, imgsz) retina-mask stack. Any failure falls back per-frame to the
 original Ultralytics construct_result."""
    global _ULTRALYTICS_GPU_PROTO_UNION_PATCHED

    if cpu_retina_masks_enabled() or not gpu_retina_proto_union_enabled():
        return False
    if _ULTRALYTICS_GPU_PROTO_UNION_PATCHED:
        return True

    try:
        from ultralytics.engine.results import Results  # type: ignore
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor  # type: ignore
    except Exception as exc:
        print(f'Warning: GPU proto-union predictor patch could not be installed; keeping Ultralytics retina masks ({exc})')
        return False

    original_construct_result = SegmentationPredictor.construct_result

    def _tta_gpu_proto_union_construct_result(self, pred, img, orig_img, img_path, proto):  # type: ignore[no-untyped-def]
        if cpu_retina_masks_enabled() or not gpu_retina_proto_union_enabled():
            return original_construct_result(self, pred, img, orig_img, img_path, proto)
        payload = _build_gpu_flattened_payload_from_proto(pred, img, proto)
        if payload is None:
            return original_construct_result(self, pred, img, orig_img, img_path, proto)
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=None, masks=None)
        setattr(result, '_tta_gpu_flattened_payload', payload)
        return result

    SegmentationPredictor.construct_result = _tta_gpu_proto_union_construct_result
    _ULTRALYTICS_GPU_PROTO_UNION_PATCHED = True
    print(
        'GPU proto-resolution retina union enabled (v13.3.0 R9): per-frame unions are reduced at '
        'proto scale and one plane is upsampled, bypassing the (n, imgsz, imgsz) retina stack.'
    )
    return True

def _affine_theta_for_grid_sample(
    M_out_to_native: np.ndarray, out_size: int, native_h: int, native_w: int,
) -> np.ndarray:
    """Convert a pixel-space output-to-native affine into an ``affine_grid`` theta matrix using ``align_corners=False`` conventions."""
    M = np.asarray(M_out_to_native, dtype=np.float64).reshape(2, 3)
    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    Minv = np.linalg.inv(M3)  # native pixel -> out-size source pixel (3x3)
    nh = float(native_h)
    nw = float(native_w)
    os_ = float(out_size)
    # normalized native output (onx,ony in [-1,1], align_corners=False) -> native output pixel (cx,ry)
    P_out = np.array([
        [nw / 2.0, 0.0, nw / 2.0 - 0.5],
        [0.0, nh / 2.0, nh / 2.0 - 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # source pixel (sx,sy) -> normalized source coords (align_corners=False)
    N_in = np.array([
        [2.0 / os_, 0.0, 1.0 / os_ - 1.0],
        [0.0, 2.0 / os_, 1.0 / os_ - 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    theta3 = N_in @ Minv @ P_out
    return theta3[:2].astype(np.float32, copy=False)

def _affine_theta_from_dst_to_src(
    M_dst_to_src: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int,
) -> np.ndarray:
    """Theta for grid_sample given the dst-pixel -> src-pixel affine directly.

 Generalizes _affine_theta_for_grid_sample to non-square sources (used by the GPU render
 engine, whose AffineSpec already stores M_out_to_src — no inversion needed). Matches
 cv2.warpAffine(src, inv(M_dst_to_src), dsize=(dst_w, dst_h)) pixel-center conventions with
 align_corners=False."""
    M3 = np.vstack([np.asarray(M_dst_to_src, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    dh = float(dst_h)
    dw = float(dst_w)
    sh = float(src_h)
    sw = float(src_w)
    P_dst = np.array([
        [dw / 2.0, 0.0, dw / 2.0 - 0.5],
        [0.0, dh / 2.0, dh / 2.0 - 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    N_src = np.array([
        [2.0 / sw, 0.0, 1.0 / sw - 1.0],
        [0.0, 2.0 / sh, 1.0 / sh - 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    theta3 = N_src @ M3 @ P_dst
    return theta3[:2].astype(np.float32, copy=False)

def _warp_matrix_is_identity(M: np.ndarray, atol: float = 1e-5) -> bool:
    """True when a cv2-style 2x3 affine is the identity within tolerance."""
    try:
        m = np.asarray(M, dtype=np.float64).reshape(2, 3)
    except Exception:
        return False
    return bool(
        abs(m[0, 0] - 1.0) <= atol and abs(m[1, 1] - 1.0) <= atol
        and abs(m[0, 1]) <= atol and abs(m[1, 0]) <= atol
        and abs(m[0, 2]) <= atol and abs(m[1, 2]) <= atol
    )

_AFFINE_GRID_CACHE: 'OrderedDict[Tuple[object, ...], object]' = OrderedDict()

_AFFINE_GRID_CACHE_LOCK = threading.Lock()

_AFFINE_GRID_CACHE_MIN_ENTRIES = 0

def request_affine_grid_cache_entries(entries: int) -> None:
    global _AFFINE_GRID_CACHE_MIN_ENTRIES
    # Capped: each entry is a full dst_h x dst_w x 2 float32 grid, so an unusually dense tile
    # configuration must not be allowed to trade the whole VRAM budget for grid reuse.
    capped = min(int(entries), max(16, _env_int('YOLO_TTA_GPU_WARP_GRID_CACHE_MAX_ENTRIES', 320)))
    _AFFINE_GRID_CACHE_MIN_ENTRIES = max(int(_AFFINE_GRID_CACHE_MIN_ENTRIES), int(capped))

def _affine_grid_cache_entries() -> int:
    return max(
        1,
        _env_int('YOLO_TTA_GPU_WARP_GRID_CACHE_ENTRIES', 12),
        int(_AFFINE_GRID_CACHE_MIN_ENTRIES),
    )

def _get_cached_affine_grid(theta_np: np.ndarray, dst_h: int, dst_w: int, device: object) -> object:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    key = (
        str(device), int(dst_h), int(dst_w),
        tuple(np.round(np.asarray(theta_np, dtype=np.float64).reshape(-1), 9).tolist()),
    )
    with _AFFINE_GRID_CACHE_LOCK:
        cached = _AFFINE_GRID_CACHE.get(key)
        if cached is not None:
            _AFFINE_GRID_CACHE.move_to_end(key)
            return cached
    theta = torch.from_numpy(np.asarray(theta_np, dtype=np.float32)).to(device=device).reshape(1, 2, 3)
    grid = F.affine_grid(theta, [1, 1, int(dst_h), int(dst_w)], align_corners=False)
    # The grid is built once on the creating thread's stream but read from arbitrary streams
    # afterwards; synchronize the producing stream once so later cross-stream reads are ordered.
    try:
        torch.cuda.current_stream(grid.device).synchronize()
    except Exception:
        pass
    with _AFFINE_GRID_CACHE_LOCK:
        _AFFINE_GRID_CACHE[key] = grid
        _AFFINE_GRID_CACHE.move_to_end(key)
        while len(_AFFINE_GRID_CACHE) > _affine_grid_cache_entries():
            _AFFINE_GRID_CACHE.popitem(last=False)
    return grid

_GPU_POSTPROCESS_TLS = threading.local()

def _gpu_postprocess_side_stream(torch_mod: object, device: object) -> Optional[object]:
    """Return this thread's postprocess CUDA stream for ``device`` (None when disabled)."""
    if not gpu_postprocess_side_stream_enabled():
        return None
    try:
        if device is None or getattr(device, 'type', '') != 'cuda':
            return None
        streams = getattr(_GPU_POSTPROCESS_TLS, 'streams', None)
        if streams is None:
            streams = {}
            _GPU_POSTPROCESS_TLS.streams = streams
        dev_key = int(getattr(device, 'index', 0) or 0)
        stream = streams.get(dev_key)
        if stream is None:
            stream = torch_mod.cuda.Stream(device=device)
            streams[dev_key] = stream
        return stream
    except Exception:
        return None

def _tensor_to_host_numpy(torch_mod: object, tensor: object, stream: Optional[object]) -> np.ndarray:
    """Copy a small device tensor to host, via this thread's pinned staging buffer when enabled.

 The returned array aliases a per-thread reusable pinned buffer (one per dtype); callers must
 consume it before requesting another transfer of the same dtype on the same thread."""
    t = tensor.contiguous()
    if not gpu_postprocess_pinned_d2h_enabled():
        return t.cpu().numpy()
    try:
        pools = getattr(_GPU_POSTPROCESS_TLS, 'pinned', None)
        if pools is None:
            pools = {}
            _GPU_POSTPROCESS_TLS.pinned = pools
        dtype_key = str(t.dtype)
        numel = int(t.numel())
        buf = pools.get(dtype_key)
        if buf is None or int(buf.numel()) < numel:
            buf = torch_mod.empty((numel,), dtype=t.dtype, pin_memory=True)
            pools[dtype_key] = buf
        view = buf[:numel].view(t.shape)
        view.copy_(t, non_blocking=True)
        if stream is not None:
            stream.synchronize()
        else:
            torch_mod.cuda.current_stream(t.device).synchronize()
        return view.numpy()
    except Exception:
        return t.cpu().numpy()

def _torch_warp_planes_to_native(
    planes: 'Sequence[object]', M_out_to_native: np.ndarray, out_size: int, native_h: int, native_w: int,
) -> 'List[object]':
    """Warp one or more co-located CUDA planes into view-native space.
    
    Identity transforms return the prepared planes directly; other transforms share a cached nearest-sampling grid."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    prepared = []
    for p in planes:
        t = p if p.dtype == torch.float32 else p.to(torch.float32)
        if int(t.shape[0]) != int(out_size) or int(t.shape[1]) != int(out_size):
            t = F.interpolate(
                t.reshape(1, 1, int(t.shape[0]), int(t.shape[1])),
                size=(int(out_size), int(out_size)), mode='nearest',
            ).reshape(int(out_size), int(out_size))
        prepared.append(t)
    if (
        int(native_h) == int(out_size) and int(native_w) == int(out_size)
        and _warp_matrix_is_identity(M_out_to_native)
    ):
        return prepared
    inp = torch.stack(prepared, dim=0).unsqueeze(0)  # (1, K, out_size, out_size)
    theta_np = _affine_theta_for_grid_sample(M_out_to_native, int(out_size), int(native_h), int(native_w))
    grid = _get_cached_affine_grid(theta_np, int(native_h), int(native_w), inp.device)
    out = F.grid_sample(inp, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    return [out[0, k] for k in range(int(out.shape[1]))]

def _min_radius_filter_ndimage(xp, ndi, mask_bool, min_radius: float):
    """Backend-agnostic --min_radius filter mirroring _filter_connected_components_by_min_radius_scipy.

 Works with (numpy, scipy.ndimage) on the CPU or (cupy, cupyx.scipy.ndimage) on the GPU. Keeps
 connected components whose maximum distance-transform value (radius) is >= min_radius."""
    if float(min_radius) <= 0.0:
        return mask_bool
    structure = xp.ones((3, 3), dtype=bool)
    labels2d, num = ndi.label(mask_bool, structure=structure)
    num_i = int(num)
    if num_i <= 0:
        return xp.zeros(mask_bool.shape, dtype=bool)
    label_ids = xp.arange(1, num_i + 1)
    dist = ndi.distance_transform_edt(mask_bool)
    radii = xp.asarray(ndi.maximum(dist, labels=labels2d, index=label_ids))
    keep_lookup = xp.zeros((num_i + 1,), dtype=bool)
    keep_lookup[1:] = radii >= float(min_radius)
    return keep_lookup[labels2d]

def _try_import_cupy_ndimage():
    """Return (cupy, cupyx.scipy.ndimage) when both import and a CUDA device is available, else None."""
    try:
        import cupy as cp  # type: ignore
        import cupyx.scipy.ndimage as cpx_ndi  # type: ignore
    except Exception:
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return None
    except Exception:
        return None
    return cp, cpx_ndi

def _try_flatten_gpu_retina_result(r, masks_data: object) -> Optional[GpuFlattenedRetinaPayload]:
    """Reduce a GPU retina-mask stack to a device-resident union and optional confidence plane.
    
    Returns ``None`` on unsupported layouts so the caller can use the ordinary result path."""
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if not isinstance(masks_data, torch.Tensor):
        return None
    try:
        masks = masks_data
        if masks.ndim != 3:
            return None
        h = int(masks.shape[1])
        w = int(masks.shape[2])
        n = int(masks.shape[0])

        fastpath = angle_variant_gpu_fastpath()
        fastpath_min_conf = None if fastpath is None else float(fastpath[0])
        fastpath_min_radius = 0.0 if fastpath is None else float(fastpath[1])
        # the per-frame retina GPU hole fill is gone; a completed-view
        # pass or eligible task-end device-union pass fills once in spec order.
        # This frame cleanup therefore has work only for positive --min_radius.
        run_gpu_cleanup = bool(
            fastpath is not None and gpu_retina_cleanup_enabled() and float(fastpath_min_radius) > 0.0
        )

        # Pull confidences onto the mask's device as a (n,) float tensor.
        confs_t = None
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and getattr(boxes, 'conf', None) is not None:
            confs_t = boxes.conf
            try:
                confs_t = confs_t.detach()
            except Exception:
                pass
            confs_t = confs_t.to(device=masks.device, dtype=torch.float32).reshape(-1)
            if int(confs_t.shape[0]) != n:
                # Length mismatch: fall back rather than guessing an alignment.
                return None

        masks_bool = masks > 0

        # angle-variant fast path: drop low-confidence instances on the GPU (exact --min_conf at
        # instance granularity) before the union/flatten.
        min_conf_applied = False
        if (
            fastpath_min_conf is not None
            and float(fastpath_min_conf) > 0.0
            and confs_t is not None
            and n > 0
        ):
            keep = confs_t >= float(fastpath_min_conf)
            masks_bool = masks_bool[keep]
            confs_t = confs_t[keep]
            n = int(masks_bool.shape[0])
            min_conf_applied = True

        if n <= 0:
            # an empty frame needs no zero planes, no warp, no GPU cleanup,
            # and no D2H — union_gpu=None short-circuits the frame processor immediately.
            return GpuFlattenedRetinaPayload(
                union_gpu=None, conf_gpu=None, instance_count=0,
                min_conf_applied=min_conf_applied, run_gpu_cleanup=run_gpu_cleanup,
                gpu_min_radius=float(fastpath_min_radius),
            )

        union_gpu = masks_bool.any(dim=0).to(torch.float32)  # (h, w) 0/1
        # with --min_conf 0 no confidence volume exists anywhere downstream,
        # so skip the per-frame (n,H,W) float transient + amax reduction entirely.
        if confs_t is not None and gpu_flatten_conf_tracking_enabled():
            # Out-of-place clamp: confs_t may be a view onto r.boxes.conf (detach/to/reshape
            # can share storage), so never mutate it in place.
            cf = confs_t.clamp(0.0, 1.0)
            # Per-pixel max confidence among covering instances (pre-quantization, in [0,1]). quantize
            # is monotonic so a later quantize matches the CPU per-instance accumulation. The
            # float<-bool promotion in the multiply keeps a single (n,H,W) transient (no separate cast).
            conf_gpu = (cf.view(-1, 1, 1) * masks_bool).amax(dim=0)
        else:
            conf_gpu = None
        # record readiness on the producing stream so the postprocess side
        # stream can wait on the event instead of the whole default stream.
        ready_event = None
        try:
            if union_gpu.is_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(union_gpu.device))
        except Exception:
            ready_event = None
        return GpuFlattenedRetinaPayload(
            union_gpu=union_gpu, conf_gpu=conf_gpu, instance_count=int(n),
            min_conf_applied=min_conf_applied, run_gpu_cleanup=run_gpu_cleanup,
            gpu_min_radius=float(fastpath_min_radius),
            ready_event=ready_event,
        )
    except Exception:
        return None

def _extract_result_masks_and_confs(r) -> Tuple[Optional[object], Optional[np.ndarray]]:
    """Detach one streamed YOLO result into CPU-owned data for asynchronous postprocess."""
    # the GPU proto-union construct_result patch attaches a prebuilt flattened
    # payload (union already reduced at proto resolution); nothing remains to extract.
    prebuilt_payload = getattr(r, '_tta_gpu_flattened_payload', None)
    if isinstance(prebuilt_payload, GpuFlattenedRetinaPayload):
        return prebuilt_payload, None

    deferred_payload = getattr(r, '_tta_deferred_cpu_retina_payload', None)
    if isinstance(deferred_payload, DeferredCpuRetinaMaskPayload):
        payload = _realize_deferred_cpu_retina_payload(deferred_payload)
        return payload, np.ascontiguousarray(payload.confs, dtype=np.float32)

    cpu_payload = getattr(r, '_tta_cpu_retina_payload', None)
    if isinstance(cpu_payload, CpuRetinaMaskPayload):
        return cpu_payload, np.ascontiguousarray(cpu_payload.confs, dtype=np.float32)

    if getattr(r, 'masks', None) is None or r.masks is None or r.masks.data is None:
        return None, None

    masks_data = r.masks.data  # (n,h,w)

    # in GPU retina mode, flatten (n,H,W) -> union + max-conf on the GPU and copy
    # only those 2 planes. The flattened payload carries its own confidence plane, so the second
    # return value is None. Falls back to the legacy whole-stack copy below on any failure.
    if gpu_retina_flatten_enabled() and not cpu_retina_masks_enabled():
        flattened = _try_flatten_gpu_retina_result(r, masks_data)
        if flattened is not None:
            return flattened, None

    try:
        masks_np = np.asarray(masks_data.detach().cpu().numpy(), dtype=np.uint8)
    except Exception:
        try:
            masks_np = np.asarray(masks_data.cpu().numpy(), dtype=np.uint8)
        except Exception:
            masks_np = np.asarray(masks_data, dtype=np.uint8)

    if masks_np.ndim != 3 or int(masks_np.shape[0]) <= 0:
        return None, None

    num_inst = int(masks_np.shape[0])
    if getattr(r, 'boxes', None) is not None and r.boxes is not None and getattr(r.boxes, 'conf', None) is not None:
        try:
            confs_np = np.asarray(r.boxes.conf.detach().cpu().numpy(), dtype=np.float32)
        except Exception:
            try:
                confs_np = np.asarray(r.boxes.conf.cpu().numpy(), dtype=np.float32)
            except Exception:
                confs_np = np.asarray(r.boxes.conf, dtype=np.float32)
    else:
        confs_np = np.zeros((num_inst,), dtype=np.float32)

    if confs_np.ndim == 0:
        confs_np = np.full((num_inst,), float(confs_np), dtype=np.float32)
    elif int(confs_np.shape[0]) != num_inst:
        confs_np = np.resize(confs_np, (num_inst,)).astype(np.float32, copy=False)

    return np.ascontiguousarray(masks_np), np.ascontiguousarray(confs_np)

def _process_cpu_retina_prediction_frame(
    idx: int,
    payload: CpuRetinaMaskPayload,
    out_size: int,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    slice_lock: Optional[threading.Lock] = None,
) -> Tuple[int, int]:
    """CPU equivalent of retina_masks=True accumulation without allocating GPU HxW masks."""
    frame_union, frame_confmap, kept_instances = _accumulate_cpu_retina_payload_to_prediction_frame(
        payload,
        int(out_size),
    )
    if int(kept_instances) <= 0 or not np.any(frame_union):
        return int(kept_instances), 0

    native_union = cv2.warpAffine(
        frame_union,
        M_out_to_native,
        dsize=(native_w, native_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    native_conf: Optional[np.ndarray] = None
    if frame_confmap is not None and np.any(frame_confmap):
        native_conf = cv2.warpAffine(
            frame_confmap,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8, copy=False)

    def _write_native_outputs() -> None:
        if np.any(native_union):
            view_union_mm[int(idx), :, :] |= native_union.astype(np.uint8, copy=False)

        if view_confmap_mm is not None and native_conf is not None and np.any(native_conf):
            conf_slice = view_confmap_mm[int(idx)]
            np.maximum(conf_slice, native_conf, out=conf_slice)

    if slice_lock is None:
        _write_native_outputs()
    else:
        with slice_lock:
            _write_native_outputs()

    return int(kept_instances), 1

def gpu_device_union_enabled() -> bool:
    """Accumulate per-task native unions on device; one chunked D2H per task."""
    return _env_flag('YOLO_TTA_GPU_DEVICE_UNION', True)

def gpu_device_hole_fill_enabled() -> bool:
    """2D-hole-fill eligible device unions before they are committed."""
    return _env_flag('YOLO_TTA_GPU_HOLE_FILL', True)

def gpu_worker_chunk_hole_fill_enabled() -> bool:
    """Allow split full-frame leases to run a whole-chunk GPU hole fill before handoff.

    Disabled by default because the CuPy connected-component pass and allocator trim are
    task-boundary barriers. Split views instead receive one parallel CPU hole-fill pass after
    their last inference lease, preserving the same per-slice result while keeping workers hot.
    Single-lease views and independent tile tasks retain their existing device-fill behavior.
    """
    return _env_flag('YOLO_TTA_GPU_WORKER_CHUNK_HOLE_FILL', False)

def gpu_union_flush_overlap_enabled() -> bool:
    """Retire CUDA-worker task unions on persistent event-driven D2H lanes.

 The lane manager bounds concurrent accumulators while allowing the worker main thread to
 start later inference tasks and allowing independent retirements to publish out of order."""
    return _env_flag('YOLO_TTA_GPU_UNION_FLUSH_OVERLAP', True)

_CUDA_GRAPH_CAPTURE_EPOCH_LOCK = threading.RLock()

@contextlib.contextmanager
def _cuda_graph_capture_context(torch_mod: object, graph: object, stream: object) -> Iterator[None]:
    """Capture one graph without imposing a process-wide steady-state CUDA lock.

    Newer PyTorch releases support thread-local capture safety directly.  Older releases
    fall back to a short compatibility epoch lock that covers only graph construction, never
    event waits, pinned copies, or host publication.
    """
    try:
        ctx = torch_mod.cuda.graph(
            graph, stream=stream, capture_error_mode='thread_local',
        )
    except TypeError:
        with _CUDA_GRAPH_CAPTURE_EPOCH_LOCK:
            with torch_mod.cuda.graph(graph, stream=stream):
                yield
        return
    with ctx:
        yield

def gpu_union_retirement_lane_count() -> int:
    """Persistent event/copy lanes per CUDA worker."""
    return max(1, min(3, _env_int('YOLO_TTA_GPU_UNION_RETIREMENT_LANES', 2)))

def gpu_union_retirement_chunk_slices() -> int:
    """Slices staged by each half of a lane's persistent double buffer."""
    return max(1, _env_int('YOLO_TTA_GPU_UNION_RETIREMENT_CHUNK_SLICES', 16))

def gpu_union_retirement_event_capacity() -> int:
    """Maximum distinct producer streams fenced without a compatibility device sync."""
    return max(8, _env_int('YOLO_TTA_GPU_UNION_RETIREMENT_EVENT_CAPACITY', 256))

def gpu_union_retirement_force_device_sync() -> bool:
    """Debug/reference mode for comparing event fencing with the retired global barrier."""
    return _env_flag('YOLO_TTA_GPU_UNION_RETIREMENT_FORCE_SYNC', False)

class _GpuUnionRetirementLane:
    """One persistent producer-event fence and double-buffered D2H lane."""

    def __init__(self, torch_mod: object, device: object, lane_id: int, max_plane_pixels: int) -> None:
        self.torch = torch_mod
        self.device = device
        self.lane_id = int(lane_id)
        self.chunk_slices = int(gpu_union_retirement_chunk_slices())
        self.max_plane_pixels = max(1, int(max_plane_pixels))
        self.copy_stream = torch_mod.cuda.Stream(
            device=device, priority=_cuda_stream_priority(torch_mod, high=False),
        )
        self.copy_events = (torch_mod.cuda.Event(), torch_mod.cuda.Event())
        self.stats_event = torch_mod.cuda.Event()
        self.producer_events = [
            torch_mod.cuda.Event()
            for _ in range(int(gpu_union_retirement_event_capacity()))
        ]
        self.pinned_u8 = torch_mod.empty(
            (2 * int(self.chunk_slices) * int(self.max_plane_pixels),),
            dtype=torch_mod.uint8,
            pin_memory=True,
        )
        self.pinned_stats = torch_mod.empty((2,), dtype=torch_mod.int64, pin_memory=True)
        self._active = False

    @property
    def producer_capacity(self) -> int:
        return int(len(self.producer_events))

    def begin(self) -> None:
        if self._active:
            raise RuntimeError(f'GPU retirement lane {self.lane_id} was acquired twice')
        self._active = True

    def end(self) -> None:
        if not self._active:
            raise RuntimeError(f'GPU retirement lane {self.lane_id} was released twice')
        self._active = False

    def pinned_views(self, h: int, w: int, chunk: int) -> Tuple[object, object]:
        plane = int(h) * int(w)
        need = 2 * int(chunk) * int(plane)
        if int(plane) > int(self.max_plane_pixels) or int(need) > int(self.pinned_u8.numel()):
            raise RuntimeError(
                f'GPU retirement lane {self.lane_id} capacity '
                f'{self.max_plane_pixels} pixels is smaller than {int(h)}x{int(w)}'
            )
        return (
            self.pinned_u8[: int(chunk) * int(plane)].view(int(chunk), int(h), int(w)),
            self.pinned_u8[
                int(self.chunk_slices) * int(self.max_plane_pixels):
                int(self.chunk_slices) * int(self.max_plane_pixels) + int(chunk) * int(plane)
            ].view(int(chunk), int(h), int(w)),
        )

class _GpuUnionRetirementManager:
    """Bounded persistent multi-lane executor for event-driven task retirement."""

    def __init__(self, torch_mod: object, device: object, max_plane_pixels: int) -> None:
        self.torch = torch_mod
        self.device = device
        self.lanes = [
            _GpuUnionRetirementLane(torch_mod, device, lane_id, int(max_plane_pixels))
            for lane_id in range(int(gpu_union_retirement_lane_count()))
        ]
        self.available: 'queue.Queue[_GpuUnionRetirementLane]' = queue.Queue()
        for lane in self.lanes:
            self.available.put(lane)
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.lanes), thread_name_prefix='gpu-union-retirement',
        )
        self.closed = False

    @property
    def capacity(self) -> int:
        return int(len(self.lanes))

    def acquire(self) -> _GpuUnionRetirementLane:
        if self.closed:
            raise RuntimeError('GPU retirement manager is closed')
        lane = self.available.get()
        lane.begin()
        return lane

    def release(self, lane: _GpuUnionRetirementLane) -> None:
        lane.end()
        self.available.put(lane)

    def submit(self, func: Callable[[], object]) -> Future:
        if self.closed:
            raise RuntimeError('GPU retirement manager is closed')
        return self.executor.submit(func)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.executor.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            self.executor.shutdown(wait=True)
        for lane in self.lanes:
            try:
                lane.copy_stream.synchronize()
            except Exception:
                pass
        self.lanes.clear()

_GPU_UNION_RETIREMENT_MANAGER: Optional[_GpuUnionRetirementManager] = None

_GPU_UNION_RETIREMENT_MANAGER_LOCK = threading.Lock()

_GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR: Optional[ThreadPoolExecutor] = None

def _init_gpu_union_retirement_manager(device_str: str, max_plane_side: int) -> Optional[_GpuUnionRetirementManager]:
    global _GPU_UNION_RETIREMENT_MANAGER
    if not gpu_union_flush_overlap_enabled():
        return None
    with _GPU_UNION_RETIREMENT_MANAGER_LOCK:
        if _GPU_UNION_RETIREMENT_MANAGER is not None:
            return _GPU_UNION_RETIREMENT_MANAGER
        import torch  # type: ignore
        device = torch.device(str(device_str))
        manager = _GpuUnionRetirementManager(
            torch, device, max(1, int(max_plane_side)) * max(1, int(max_plane_side)),
        )
        _GPU_UNION_RETIREMENT_MANAGER = manager
        print(
            f'v16.1.3 event-driven GPU union retirement initialized: '
            f'{manager.capacity} persistent lane(s), '
            f'{gpu_union_retirement_chunk_slices()} slice(s)/D2H chunk, '
            f'{gpu_union_retirement_event_capacity()} producer event(s)/lane.'
        )
        return manager

def _gpu_union_retirement_manager() -> Optional[_GpuUnionRetirementManager]:
    return _GPU_UNION_RETIREMENT_MANAGER

def _gpu_union_retirement_fallback_executor() -> ThreadPoolExecutor:
    global _GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR
    with _GPU_UNION_RETIREMENT_MANAGER_LOCK:
        if _GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR is None:
            _GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix='gpu-union-retirement-fallback',
            )
        return _GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR

def _shutdown_gpu_union_retirement_manager() -> None:
    global _GPU_UNION_RETIREMENT_MANAGER, _GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR
    with _GPU_UNION_RETIREMENT_MANAGER_LOCK:
        manager = _GPU_UNION_RETIREMENT_MANAGER
        fallback = _GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR
        _GPU_UNION_RETIREMENT_MANAGER = None
        _GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR = None
    if manager is not None:
        manager.close()
    if fallback is not None:
        try:
            fallback.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            fallback.shutdown(wait=True)

class _DeviceUnionAccumulator:
    """Accumulate one task's native mask and confidence slices on the GPU.
    
    Written slices flush through bounded pinned chunks; any host fallback disables device-only hole-fill assumptions."""

    def __init__(self, torch_mod: object, device: object, num_frames: int, native_h: int, native_w: int, want_conf: bool) -> None:
        self.torch = torch_mod
        self.device = device
        self.union_dev = torch_mod.zeros(
            (int(num_frames), int(native_h), int(native_w)), dtype=torch_mod.uint8, device=device,
        )
        self.conf_dev = (
            torch_mod.zeros((int(num_frames), int(native_h), int(native_w)), dtype=torch_mod.uint8, device=device)
            if bool(want_conf) else None
        )
        # device-compacted generic payloads write one scalar per slice. Static/legacy
        # payloads leave zero here and continue returning their host-known stats normally.
        self.prediction_counts_dev = torch_mod.zeros(
            (int(num_frames),), dtype=torch_mod.int32, device=device,
        )
        # Per-slice device-write tracking (GIL-atomic independent cells; indices are disjoint
        # across postprocess threads). host_written records any frame that bypassed the device
        # union (CPU fallback / cupy-cleanup frames) and wrote the host window directly.
        self.written = np.zeros((int(num_frames),), dtype=bool)
        self.host_written = False
        self._producer_streams: Dict[int, object] = {}
        self._producer_streams_lock = threading.Lock()
        self._retirement_sealed = False
        try:
            self.register_producer_stream(torch_mod.cuda.current_stream(device))
        except Exception:
            pass

    def mark_host_write(self) -> None:
        self.host_written = True

    def register_producer_stream(self, stream: Optional[object]) -> None:
        """Register every CUDA stream that can write this task union."""
        if stream is None:
            try:
                stream = self.torch.cuda.current_stream(self.device)
            except Exception:
                return
        if self._retirement_sealed:
            raise RuntimeError('device union received a producer after retirement was sealed')
        with self._producer_streams_lock:
            self._producer_streams[id(stream)] = stream

    def write_frame(
        self,
        idx: int,
        union_bool_t: object,
        conf_u8_t: Optional[object] = None,
        prediction_count_dev: Optional[object] = None,
        producer_stream: Optional[object] = None,
    ) -> None:
        self.register_producer_stream(producer_stream)
        # Runs on the calling thread's side stream (inside the frame processor's stream ctx);
        # slice indices are disjoint across threads, so no lock is needed.
        self.union_dev[int(idx)] = union_bool_t.to(self.torch.uint8)
        if self.conf_dev is not None and conf_u8_t is not None:
            self.conf_dev[int(idx)] = conf_u8_t
        if prediction_count_dev is not None:
            self.prediction_counts_dev[int(idx)].copy_(
                prediction_count_dev.reshape(-1)[0], non_blocking=True,
            )
        self.written[int(idx)] = True

    def take_device_prediction_stats(
        self,
        *,
        retirement_lane: Optional[_GpuUnionRetirementLane] = None,
        synchronize_device: bool = True,
    ) -> Tuple[int, int]:
        """Read task prediction/frame totals on the retirement copy stream."""
        if self.prediction_counts_dev is None:
            return 0, 0
        try:
            if bool(synchronize_device):
                self.synchronize_for_retirement(retirement_lane)
            counts = self.prediction_counts_dev
            if retirement_lane is None:
                stats = self.torch.stack([
                    counts.to(self.torch.int64).sum(),
                    (counts > 0).to(self.torch.int64).sum(),
                ]).cpu().numpy()
                return int(stats[0]), int(stats[1])
            stream = retirement_lane.copy_stream
            with self.torch.cuda.stream(stream):
                stats_dev = self.torch.stack([
                    counts.to(self.torch.int64).sum(),
                    (counts > 0).to(self.torch.int64).sum(),
                ])
                retirement_lane.pinned_stats.copy_(stats_dev, non_blocking=True)
                retirement_lane.stats_event.record(stream)
            retirement_lane.stats_event.synchronize()
            stats_np = retirement_lane.pinned_stats.numpy()
            return int(stats_np[0]), int(stats_np[1])
        except Exception:
            return 0, 0

    def fill_holes_2d(self) -> int:
        """Fill enclosed per-slice 2D background on the device.

        Each block runs in a nested scope so its CuPy labels and Boolean temporaries lose
        their final references before the CuPy pool is trimmed. Returns the filled frame
        count, or zero so the caller performs the CPU pass when the device path is unsafe.
        """
        torch = self.torch
        if self.union_dev is None or bool(self.host_written):
            return 0
        cp_mod = _try_import_cupy_ndimage()
        if cp_mod is None:
            return 0
        cp, cpx_ndi = cp_mod
        n, h, w = (int(x) for x in self.union_dev.shape)
        if n <= 0 or h < 3 or w < 3:
            return 0
        dev_idx = int(getattr(self.device, 'index', 0) or 0)
        structure = None
        filled = 0
        try:
            torch.cuda.synchronize(self.device)
            block = max(1, _env_int('YOLO_TTA_GPU_HOLE_FILL_BLOCK', 64))
            with cp.cuda.Device(dev_idx):
                while block > 1:
                    need = int(block) * h * w * 6 + 512 * 1024 * 1024
                    free_bytes, _total = torch.cuda.mem_get_info(self.device)
                    if int(free_bytes) >= int(need):
                        break
                    block //= 2
                structure = cp.zeros((3, 3, 3), dtype=cp.bool_)
                structure[1, 1, 1] = True
                structure[1, 0, 1] = True
                structure[1, 2, 1] = True
                structure[1, 1, 0] = True
                structure[1, 1, 2] = True

                def _fill_block(z0: int, z1: int) -> None:
                    blk = cp.asarray(self.union_dev[int(z0):int(z1)])
                    bg = blk == 0
                    labels, num = cpx_ndi.label(bg, structure=structure)
                    if int(num) <= 0:
                        return
                    touches = cp.zeros((int(num) + 1,), dtype=cp.bool_)
                    touches[cp.unique(labels[:, 0, :])] = True
                    touches[cp.unique(labels[:, -1, :])] = True
                    touches[cp.unique(labels[:, :, 0])] = True
                    touches[cp.unique(labels[:, :, -1])] = True
                    enclosed = bg & ~touches[labels]
                    if bool(enclosed.any()):
                        blk[enclosed] = cp.uint8(1)

                for z0 in range(0, n, int(block)):
                    _fill_block(int(z0), min(n, int(z0) + int(block)))
                cp.cuda.get_current_stream().synchronize()
                filled = int(n)
        except Exception:
            filled = 0
        finally:
            structure = None
            gc.collect()
            try:
                with cp.cuda.Device(dev_idx):
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        return int(filled)

    def compute_slice_metadata(self) -> Optional[Dict[str, np.ndarray]]:
        """Return compact per-slice foreground metadata from the device union.

        Host-written fallback frames return ``None`` because their device rows are
        incomplete. Call after device hole filling and before the task flush.
        """
        torch = self.torch
        if self.union_dev is None or bool(self.host_written):
            return None
        try:
            torch.cuda.synchronize(self.device)
            u = self.union_dev
            n, h, w = (int(x) for x in u.shape)
            rows = u.amax(dim=2) > 0   # (n, h)
            cols = u.amax(dim=1) > 0   # (n, w)
            any_t = rows.any(dim=1)    # (n,)
            rows_u8 = rows.to(torch.uint8)
            cols_u8 = cols.to(torch.uint8)
            y0 = rows_u8.argmax(dim=1)
            y1 = int(h) - rows_u8.flip(1).argmax(dim=1)
            x0 = cols_u8.argmax(dim=1)
            x1 = int(w) - cols_u8.flip(1).argmax(dim=1)
            bbox_t = torch.stack([y0, y1, x0, x1], dim=1).to(torch.int64)
            any_np = any_t.cpu().numpy()
            bbox_np = bbox_t.cpu().numpy()
            rows_np = rows.cpu().numpy()
            bbox_np[~any_np] = 0  # argmax on all-zero rows would report a full-extent bbox
            return {
                'slice_any': np.ascontiguousarray(any_np),
                'slice_bboxes': np.ascontiguousarray(bbox_np),
                # Bit-packed (n, ceil(h/8)) row-occupancy — small enough for the mp result queue.
                'slice_row_any': np.packbits(np.ascontiguousarray(rows_np), axis=1),
                'slice_row_count': np.asarray([int(h)], dtype=np.int64),
            }
        except Exception:
            return None

    def synchronize_for_retirement(
        self,
        retirement_lane: Optional[_GpuUnionRetirementLane] = None,
    ) -> None:
        """Seal producers with events; use a whole-device barrier only in fallback/debug mode."""
        if self._retirement_sealed:
            return
        self._retirement_sealed = True
        if retirement_lane is None or gpu_union_retirement_force_device_sync():
            self.torch.cuda.synchronize(self.device)
            return
        with self._producer_streams_lock:
            producer_streams = list(self._producer_streams.values())
        if len(producer_streams) > int(retirement_lane.producer_capacity):
            # This is a defensive compatibility path, not the steady state.  Increasing
            # YOLO_TTA_GPU_UNION_RETIREMENT_EVENT_CAPACITY removes it.
            self.torch.cuda.synchronize(self.device)
            return
        for event, stream in zip(retirement_lane.producer_events, producer_streams):
            event.record(stream)
            retirement_lane.copy_stream.wait_event(event)

    def flush_into(
        self,
        view_union_mm: np.ndarray,
        view_confmap_mm: Optional[np.ndarray],
        chunk_slices: int = 64,
        *,
        retirement_lane: Optional[_GpuUnionRetirementLane] = None,
        synchronize_device: bool = True,
        collect_slice_metadata: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Commit the task union through a persistent event-driven D2H lane."""
        torch = self.torch
        if bool(synchronize_device):
            self.synchronize_for_retirement(retirement_lane)
        n, h, w = (int(x) for x in self.union_dev.shape)
        chunk = max(1, min(
            int(retirement_lane.chunk_slices) if retirement_lane is not None else int(chunk_slices),
            n,
        ))
        plane = int(h) * int(w)
        written = self.written
        metadata_enabled = bool(collect_slice_metadata and not self.host_written)
        slice_any = np.zeros((n,), dtype=bool) if metadata_enabled else None
        slice_bboxes = np.zeros((n, 4), dtype=np.int64) if metadata_enabled else None
        slice_row_any = (
            np.zeros((n, int((h + 7) // 8)), dtype=np.uint8)
            if metadata_enabled else None
        )

        def _collect_metadata(z0: int, z1: int, host: np.ndarray) -> None:
            if not metadata_enabled:
                return
            rows = np.any(host, axis=2)
            cols = np.any(host, axis=1)
            wr = np.asarray(written[int(z0):int(z1)], dtype=bool)
            if not bool(wr.all()):
                rows[~wr] = False
                cols[~wr] = False
            any_local = np.any(rows, axis=1)
            slice_any[int(z0):int(z1)] = any_local
            slice_row_any[int(z0):int(z1)] = np.packbits(rows, axis=1)
            if not bool(any_local.any()):
                return
            y0 = np.argmax(rows, axis=1).astype(np.int64, copy=False)
            y1 = int(h) - np.argmax(rows[:, ::-1], axis=1).astype(np.int64, copy=False)
            x0 = np.argmax(cols, axis=1).astype(np.int64, copy=False)
            x1 = int(w) - np.argmax(cols[:, ::-1], axis=1).astype(np.int64, copy=False)
            local_bbox = np.stack([y0, y1, x0, x1], axis=1)
            local_bbox[~any_local] = 0
            slice_bboxes[int(z0):int(z1)] = local_bbox

        owned_pin: Optional[object] = None
        if retirement_lane is not None:
            pin_views = retirement_lane.pinned_views(h, w, chunk)
            copy_stream = retirement_lane.copy_stream
            events = retirement_lane.copy_events
        else:
            owned_pin = torch.empty((2 * chunk * plane,), dtype=torch.uint8, pin_memory=True)
            pin_views = (
                owned_pin[: chunk * plane].view(chunk, h, w),
                owned_pin[chunk * plane: 2 * chunk * plane].view(chunk, h, w),
            )
            copy_stream = torch.cuda.Stream(device=self.device)
            events = (torch.cuda.Event(), torch.cuda.Event())
        pin_np = (pin_views[0].numpy(), pin_views[1].numpy())

        def _drain(
            src_dev: object,
            dst_mm: np.ndarray,
            *,
            collect_metadata: bool = False,
            confidence: bool = False,
        ) -> None:
            chunks = [
                (z0, min(n, z0 + chunk))
                for z0 in range(0, n, chunk)
                if bool(written[z0:min(n, z0 + chunk)].any())
            ]
            if not chunks:
                return

            def _issue(ci: int) -> None:
                z0_i, z1_i = chunks[ci]
                buf = ci % 2
                with torch.cuda.stream(copy_stream):
                    pin_views[buf][: z1_i - z0_i].copy_(
                        src_dev[z0_i:z1_i], non_blocking=True,
                    )
                    events[buf].record(copy_stream)

            _issue(0)
            for ci, (z0, z1) in enumerate(chunks):
                if ci + 1 < len(chunks):
                    _issue(ci + 1)
                events[ci % 2].synchronize()
                host = pin_np[ci % 2][: z1 - z0]
                if bool(collect_metadata):
                    _collect_metadata(int(z0), int(z1), host)
                wr = written[z0:z1]
                if bool(wr.all()):
                    dst = np.asarray(dst_mm[z0:z1])
                    if bool(confidence):
                        np.maximum(dst, host, out=dst)
                    else:
                        np.bitwise_or(dst, host, out=dst)
                else:
                    for zi in range(z1 - z0):
                        if wr[zi]:
                            dst = np.asarray(dst_mm[z0 + zi])
                            if bool(confidence):
                                np.maximum(dst, host[zi], out=dst)
                            else:
                                np.bitwise_or(dst, host[zi], out=dst)

        _drain(self.union_dev, view_union_mm, collect_metadata=bool(metadata_enabled))
        if self.conf_dev is not None and view_confmap_mm is not None:
            _drain(self.conf_dev, view_confmap_mm, collect_metadata=False, confidence=True)
        owned_pin = None
        self.union_dev = None
        self.conf_dev = None
        self.prediction_counts_dev = None
        if not metadata_enabled:
            return None
        return {
            'slice_any': np.ascontiguousarray(slice_any),
            'slice_bboxes': np.ascontiguousarray(slice_bboxes),
            'slice_row_any': np.ascontiguousarray(slice_row_any),
            'slice_row_count': np.asarray([int(h)], dtype=np.int64),
        }

def _try_create_device_union_accumulator(
    device_str: str, num_frames: int, native_h: int, native_w: int, *, want_conf: bool,
) -> Optional[_DeviceUnionAccumulator]:
    """Build the per-task device union when VRAM allows; None -> per-frame D2H."""
    if not gpu_device_union_enabled():
        return None
    try:
        import torch  # type: ignore
        if not str(device_str).startswith('cuda') or not bool(torch.cuda.is_available()):
            return None
        device = torch.device(str(device_str))
        need = int(num_frames) * int(native_h) * int(native_w) * (2 if bool(want_conf) else 1)
        free_bytes, _total = torch.cuda.mem_get_info(device)
        if int(free_bytes) < int(need) + 2 * GIB:
            return None
        return _DeviceUnionAccumulator(torch, device, int(num_frames), int(native_h), int(native_w), bool(want_conf))
    except Exception:
        return None

def _process_gpu_flattened_prediction_frame(
    idx: int,
    payload: GpuFlattenedRetinaPayload,
    out_size: int,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    slice_lock: Optional[threading.Lock] = None,
    device_union: Optional[_DeviceUnionAccumulator] = None,
) -> Tuple[int, int]:
    """Accumulate one GPU-flattened retina result into native view space.
    
    Union and confidence planes stay on a side CUDA stream through warp and optional radius cleanup. Eligible tasks write directly into the device union; host fallback preserves output on any GPU-path failure. Hole filling remains a completed-view or task-end operation."""
    instance_count = max(0, int(payload.instance_count))
    union_gpu = payload.union_gpu
    if union_gpu is None:
        return int(instance_count), 0

    track_conf = view_confmap_mm is not None
    native_union_np: Optional[np.ndarray] = None
    native_conf_np: Optional[np.ndarray] = None
    cleaned_on_gpu = False

    try:
        import torch  # type: ignore
        if not gpu_retina_warp_enabled():
            raise RuntimeError('gpu retina warp disabled')

        # the whole postprocess tail runs on this thread's side CUDA stream,
        # ordered against the producer via the payload's recorded event, so warp/quantize/D2H
        # no longer serialize with TensorRT kernel issue on the default stream.
        device = getattr(union_gpu, 'device', None)
        side_stream = _gpu_postprocess_side_stream(torch, device)
        stream_ctx = torch.cuda.stream(side_stream) if side_stream is not None else contextlib.nullcontext()
        with stream_ctx:
            if side_stream is not None:
                ready_evt = getattr(payload, 'ready_event', None)
                if ready_evt is not None:
                    side_stream.wait_event(ready_evt)
                else:
                    side_stream.wait_stream(torch.cuda.default_stream(device))
                # Payload tensors are produced on the inference/default stream but can be
                # released as soon as this worker returns, while the side-stream write is still
                # in flight. Tell the caching allocator about every cross-stream use.
                _record_candidates: List[object] = [
                    union_gpu, payload.conf_gpu, payload.instance_count_device,
                ]
                if payload.device_refs is not None:
                    _record_candidates.extend(payload.device_refs)
                for _candidate in _record_candidates:
                    if isinstance(_candidate, torch.Tensor):
                        try:
                            _candidate.record_stream(side_stream)
                        except Exception:
                            pass

            # Warp union (and conf) to view-native space on the GPU (nearest, zero-padded). Both
            # planes share one grid_sample; identity warps skip it entirely and non-identity
            # warps reuse a cached grid.
            warp_conf = bool(track_conf and payload.conf_gpu is not None)
            warped = _torch_warp_planes_to_native(
                [union_gpu, payload.conf_gpu] if warp_conf else [union_gpu],
                M_out_to_native, int(out_size), int(native_h), int(native_w),
            )
            native_union_t = warped[0]
            native_union_bool_t = native_union_t > 0.5

            native_conf_t = None
            if warp_conf:
                native_conf_f = warped[1]
                # Quantize to u8 (round-half-even) only where the union is foreground.
                native_conf_t = torch.where(
                    native_union_bool_t,
                    (native_conf_f.clamp(0.0, 1.0) * float(CONF_U8_MAX)).round(),
                    torch.zeros((), dtype=native_conf_f.dtype, device=native_conf_f.device),
                ).clamp(0.0, 255.0).to(torch.uint8)

            # with a per-task device union, the warped plane stays ON DEVICE —
            # no per-frame D2H, no host |=; the task-end flush does one chunked transfer.
            # Frames that still need the cupy --min_radius cleanup (which lands on host) fall
            # through to the legacy path; the flush's OR/max merge keeps both consistent.
            if device_union is not None and not (
                bool(payload.run_gpu_cleanup) and float(payload.gpu_min_radius) > 0.0
            ):
                device_union.write_frame(
                    int(idx), native_union_bool_t, native_conf_t,
                    prediction_count_dev=payload.instance_count_device,
                    producer_stream=side_stream,
                )
                payload.accumulated_on_device = True
                payload.cleanup_done_on_gpu = False
                # frames_with_predictions: reported from the pre-warp instance count (checking
                # post-warp emptiness would force a device sync per frame; the stat is
                # informational only).
                if payload.instance_count_device is not None:
                    return 0, 0  # task-end device reduction supplies both stats
                return int(instance_count), (1 if int(instance_count) > 0 else 0)

            # connected-component --min_radius on the GPU (cupy), in native
            # space, only when a positive radius is set. The old per-frame retina 2D
            # hole fill (cupyx binary_fill_holes, an iterative-dilation kernel/sync
            # storm) is removed: the completed-view or eligible task-end device-union
            # pass fills once, preserving --min_conf -> --min_radius -> hole fill.
            if bool(payload.run_gpu_cleanup) and float(payload.gpu_min_radius) > 0.0:
                cp_mod = _try_import_cupy_ndimage()
                if cp_mod is not None:
                    cp, cpx_ndi = cp_mod
                    try:
                        # Pin the cupy device to the torch tensor's CUDA index so cp.asarray does
                        # not default to device 0 when inference runs on a non-default GPU.
                        _dev_idx = getattr(native_union_bool_t.device, 'index', None)
                        _cp_dev = cp.cuda.Device(int(_dev_idx)) if _dev_idx is not None else cp.cuda.Device()
                        _cp_stream = (
                            _cupy_external_stream(cp, side_stream)
                            if side_stream is not None else contextlib.nullcontext()
                        )
                        with _cp_dev, _cp_stream:
                            # Move to cupy via uint8 (torch bool tensors do not reliably expose
                            # __cuda_array_interface__), then back to a boolean mask for the CC ops.
                            union_cp = cp.asarray(native_union_bool_t.to(torch.uint8).contiguous()) > 0
                            union_cp = _min_radius_filter_ndimage(cp, cpx_ndi, union_cp, float(payload.gpu_min_radius))
                            native_union_np = np.ascontiguousarray(cp.asnumpy(union_cp).astype(np.uint8, copy=False) > 0)
                        cleaned_on_gpu = True
                    except Exception:
                        native_union_np = None  # fall back to torch->cpu union, CPU cleanup downstream

            if native_union_np is None:
                # D2H through this thread's pinned staging buffer (non-blocking
                # copy + stream sync); copy out immediately since the buffer is reused per frame.
                native_union_np = np.ascontiguousarray(
                    _tensor_to_host_numpy(torch, native_union_bool_t, side_stream)
                )
                native_union_np = native_union_np > 0
            if native_conf_t is not None:
                native_conf_np = np.ascontiguousarray(
                    _tensor_to_host_numpy(torch, native_conf_t, side_stream).astype(np.uint8, copy=True)
                )
                if bool(cleaned_on_gpu):
                    # Keep confidence only where the cleaned union remains foreground.
                    native_conf_np = np.where(native_union_np, native_conf_np, np.uint8(0)).astype(np.uint8, copy=False)
    except Exception:
        # Robust CPU fallback: copy the flattened GPU planes down and warp on the CPU (cv2).
        native_union_np = None
        native_conf_np = None
        cleaned_on_gpu = False
        try:
            import torch  # type: ignore
            frame_union = np.ascontiguousarray((union_gpu.detach().cpu().numpy() > 0).astype(np.uint8))
        except Exception:
            return int(instance_count), 0
        if int(frame_union.shape[0]) != int(out_size) or int(frame_union.shape[1]) != int(out_size):
            frame_union = cv2.resize(frame_union, (int(out_size), int(out_size)), interpolation=cv2.INTER_NEAREST)
        native_union_np = cv2.warpAffine(
            frame_union, M_out_to_native, dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ) > 0
        if track_conf and payload.conf_gpu is not None:
            conf_plane = payload.conf_gpu.detach().cpu().numpy()
            conf_u8 = np.clip(np.rint(np.clip(conf_plane, 0.0, 1.0) * float(CONF_U8_MAX)), 0, 255).astype(np.uint8)
            if int(conf_u8.shape[0]) != int(out_size) or int(conf_u8.shape[1]) != int(out_size):
                conf_u8 = cv2.resize(conf_u8, (int(out_size), int(out_size)), interpolation=cv2.INTER_NEAREST)
            native_conf_np = cv2.warpAffine(
                conf_u8, M_out_to_native, dsize=(native_w, native_h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(np.uint8, copy=False)
            native_conf_np = np.where(native_union_np, native_conf_np, np.uint8(0)).astype(np.uint8, copy=False)

    if payload.instance_count_device is not None and not bool(payload.accumulated_on_device):
        # This host/cupy fallback already performs a mask D2H; only the normal device-union
        # path is synchronization-free per frame. Preserve exact informational counts here.
        try:
            instance_count = int(payload.instance_count_device.detach().cpu().reshape(-1)[0])
        except Exception:
            instance_count = 0

    if native_union_np is None or not np.any(native_union_np):
        # Empty after warp/cleanup: nothing written, so report no contributing frame (matches the
        # CPU retina path). Still record whether the GPU cleanup ran so the CPU cleanup stays skipped.
        payload.cleanup_done_on_gpu = bool(cleaned_on_gpu)
        return int(instance_count), 0

    def _write_native_outputs() -> None:
        view_union_mm[int(idx), :, :] |= native_union_np.astype(np.uint8, copy=False)
        if view_confmap_mm is not None and native_conf_np is not None and np.any(native_conf_np):
            conf_slice = view_confmap_mm[int(idx)]
            np.maximum(conf_slice, native_conf_np, out=conf_slice)

    # this frame bypassed the device union (cupy cleanup or CPU fallback) and
    # writes the host window directly — the device-side hole fill must stand down for this task.
    if device_union is not None:
        device_union.mark_host_write()
    if slice_lock is None:
        _write_native_outputs()
    else:
        with slice_lock:
            _write_native_outputs()

    payload.cleanup_done_on_gpu = bool(cleaned_on_gpu)
    return int(instance_count), 1

def _process_prediction_frame(
    idx: int,
    masks_np: Optional[object],
    confs_np: Optional[np.ndarray],
    out_size: int,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    slice_lock: Optional[threading.Lock] = None,
    device_union: Optional[_DeviceUnionAccumulator] = None,
) -> Tuple[int, int]:
    """Collapse one streamed result directly into unpacked native-view union + confidence volumes."""
    if isinstance(masks_np, GpuFlattenedRetinaPayload):
        return _process_gpu_flattened_prediction_frame(
            idx=idx,
            payload=masks_np,
            out_size=out_size,
            view_union_mm=view_union_mm,
            view_confmap_mm=view_confmap_mm,
            M_out_to_native=M_out_to_native,
            native_h=native_h,
            native_w=native_w,
            slice_lock=slice_lock,
            device_union=device_union,
        )

    if isinstance(masks_np, CpuRetinaMaskPayload):
        # host-side accumulation path — device hole fill stands down for the task.
        if device_union is not None:
            device_union.mark_host_write()
        return _process_cpu_retina_prediction_frame(
            idx=idx,
            payload=masks_np,
            out_size=out_size,
            view_union_mm=view_union_mm,
            view_confmap_mm=view_confmap_mm,
            M_out_to_native=M_out_to_native,
            native_h=native_h,
            native_w=native_w,
            slice_lock=slice_lock,
        )

    if masks_np is None:
        return 0, 0
    masks_arr = np.asarray(masks_np)
    if masks_arr.ndim != 3 or int(masks_arr.shape[0]) <= 0:
        return 0, 0
    # raw retina-stack frames accumulate host-side below.
    if device_union is not None:
        device_union.mark_host_write()

    track_conf = view_confmap_mm is not None
    frame_union = np.zeros((out_size, out_size), dtype=np.uint8)
    frame_confmap = np.zeros((out_size, out_size), dtype=np.uint8) if track_conf else None
    num_inst = int(masks_arr.shape[0])

    for inst_idx in range(num_inst):
        inst = np.asarray(masks_arr[inst_idx], dtype=np.uint8)
        if inst.shape[0] != out_size or inst.shape[1] != out_size:
            inst = cv2.resize(inst, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        inst = (inst > 0).astype(np.uint8, copy=False)
        if not np.any(inst):
            continue

        frame_union |= inst
        if track_conf and frame_confmap is not None:
            conf_val = float(confs_np[inst_idx]) if (confs_np is not None and inst_idx < int(confs_np.shape[0])) else 0.0
            conf_u8 = quantize_conf_to_u8(conf_val)
            inst_bool = inst > 0
            frame_confmap[inst_bool] = np.maximum(frame_confmap[inst_bool], conf_u8)

    if not np.any(frame_union):
        return int(num_inst), 1

    native_union = cv2.warpAffine(
        frame_union,
        M_out_to_native,
        dsize=(native_w, native_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    native_conf: Optional[np.ndarray] = None
    if frame_confmap is not None and np.any(frame_confmap):
        native_conf = cv2.warpAffine(
            frame_confmap,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8, copy=False)

    def _write_native_outputs() -> None:
        if np.any(native_union):
            view_union_mm[int(idx), :, :] |= native_union.astype(np.uint8, copy=False)

        if view_confmap_mm is not None and native_conf is not None and np.any(native_conf):
            conf_slice = view_confmap_mm[int(idx)]
            np.maximum(conf_slice, native_conf, out=conf_slice)

    if slice_lock is None:
        _write_native_outputs()
    else:
        with slice_lock:
            _write_native_outputs()

    return int(num_inst), 1


def _prediction_accumulation_target(
    spec: BatchResultFrameSpec,
    *,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    radial_padding_union_mm: Optional[np.ndarray],
    radial_padding_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_w: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, np.ndarray, bool]:
    """Resolve storage, destination index, affine, and stat policy for one result."""

    if not bool(spec.is_radial_padding):
        target_index = int(spec.task_index)
        if target_index < 0 or target_index >= int(view_union_mm.shape[0]):
            raise IndexError(
                f'prediction result {int(spec.result_index)} maps outside task union '
                f'{tuple(int(v) for v in view_union_mm.shape)}'
            )
        return (
            view_union_mm,
            view_confmap_mm,
            target_index,
            np.asarray(M_out_to_native, dtype=np.float32),
            True,
        )

    if radial_padding_union_mm is not None:
        target_union = radial_padding_union_mm
        target_conf = radial_padding_confmap_mm
        target_index = int(spec.radial_padding_ordinal or 0)
    else:
        target_union = view_union_mm
        target_conf = view_confmap_mm
        target_index = int(spec.global_destination_index)
    if target_index < 0 or target_index >= int(target_union.shape[0]):
        raise IndexError(
            f'Radial padding result {int(spec.result_index)} maps to {target_index}, but '
            f'target shape is {tuple(int(v) for v in target_union.shape)}; provide a '
            'task-local radial padding sink for split leases'
        )
    return (
        target_union,
        target_conf,
        target_index,
        (
            mirror_radial_u_output_to_native_affine(M_out_to_native, int(native_w))
            if bool(spec.mirror_radial_u)
            else np.asarray(M_out_to_native, dtype=np.float32)
        ),
        False,
    )

_DIRECT_PREDICT_ANNOUNCED = False

def direct_predict_enabled() -> bool:
    return _env_flag('YOLO_TTA_DIRECT_PREDICT', True)

def resident_trt_ring_enabled() -> bool:
    """Enable the fixed batch-1 resident-render TensorRT pipeline."""
    return _env_flag('YOLO_TTA_RESIDENT_TRT_RING', True)

def resident_trt_cuda_graphs_enabled() -> bool:
    """Capture each ring slot's static TensorRT and mask-decode launches when supported."""
    return _env_flag('YOLO_TTA_RESIDENT_TRT_CUDA_GRAPHS', True)

def resident_trt_native_warp_enabled() -> bool:
    """Allow the resident ring to fuse a non-identity network->native affine."""
    return _env_flag('YOLO_TTA_RESIDENT_TRT_NATIVE_WARP', True)

def _cuda_stream_priority(torch_mod: object, *, high: bool) -> int:
    """Return a supported CUDA stream priority, using zero when the runtime does not expose a priority range."""
    try:
        least, greatest = torch_mod.cuda.get_stream_priority_range()
        return int(greatest if bool(high) else least)
    except Exception:
        return -1 if bool(high) else 0

@dataclass(frozen=True)
class ResidentRingUnitDescriptor:
    """Per-unit post-warp geometry and task-union destination for a ring slot."""

    unit_index: int
    destination_index: int
    native_h: int
    native_w: int
    M_out_to_native: np.ndarray

class _ResidentGpuPipelineSlot:
    """One static-address input/output/event slot in the resident two-entry ring."""

    def __init__(
        self,
        torch_mod: object,
        device: object,
        out_size: int,
        input_channels: int,
        input_dtype: object,
        slot_id: int,
    ) -> None:
        self.slot_id = int(slot_id)
        self.input_channels = int(input_channels)
        if self.input_channels <= 0:
            raise RuntimeError(
                f'resident ring requires a positive input channel count, got {self.input_channels}'
            )
        if input_dtype not in (torch_mod.float16, torch_mod.float32):
            raise RuntimeError(
                f'resident ring does not support TensorRT input dtype {input_dtype}; '
                'only float16/float32 normalized image bindings are supported'
            )
        self.input = torch_mod.empty(
            (1, self.input_channels, int(out_size), int(out_size)),
            dtype=input_dtype,
            device=device,
        )
        # Stable metadata address for renderer launches. A one-thread setter kernel
        # updates its value on the render stream without a reusable-host-buffer race.
        self.render_meta = torch_mod.empty((2,), dtype=torch_mod.int32, device=device)
        # Distinct high-priority streams allow the two TensorRT execution contexts to
        # overlap. Mask decode/quantize/union uses a third, ordinary-priority stream.
        high_priority = _cuda_stream_priority(torch_mod, high=True)
        low_priority = _cuda_stream_priority(torch_mod, high=False)
        self.infer_stream = torch_mod.cuda.Stream(device=device, priority=high_priority)
        self.post_stream = torch_mod.cuda.Stream(device=device, priority=low_priority)
        # Events are allocated once and re-recorded. render_done protects input reads,
        # infer_done protects input reuse, and post_done protects output-binding reuse.
        self.render_done = torch_mod.cuda.Event(enable_timing=False, blocking=False)
        self.infer_done = torch_mod.cuda.Event(enable_timing=False, blocking=False)
        self.post_done = torch_mod.cuda.Event(enable_timing=False, blocking=False)
        self.infer_valid = False
        self.post_valid = False
        self.frame_index = -1
        self.absolute_index = -1
        self.synthetic = False
        self.context = None
        self.binding_addresses: Optional[List[int]] = None
        self.head = None
        self.proto = None
        self.compact_indices = None
        self.compact_count = None
        self.max_logit = None
        self.proto_tmp = None
        self.conf_proto = None
        self.native_union = None
        self.native_conf = None
        self.unit_descriptor: Optional[ResidentRingUnitDescriptor] = None
        self.identity_native_warp = False
        self.native_to_out: Tuple[np.float32, ...] = tuple()
        self.infer_graph = None
        self.post_graph = None
        self.render_graph = None
        self.render_graph_key = None
        # Prepared once per resident source/task. Graph replay compares this cached key
        # instead of rebuilding large Radial azimuth-byte geometry keys for every frame.
        self.render_expected_key = None
        # Keep zero-copy cupy array views alive for the lifetime of captured graphs.
        self._cupy_refs: Dict[str, object] = {}
        # renderer views are separate from postprocess graph views: the latter mapping
        # is rebuilt by _ResidentTensorRTRingExecutor after the slots are allocated.
        self._render_cupy_refs: Dict[str, object] = {}

_RESIDENT_MASK_KERNELS: Optional[object] = None

_RESIDENT_MASK_KERNELS_FAILED = False

def _resident_mask_kernels() -> Optional[object]:
    """Compile the small device compaction/proto-union kernels once with NVRTC.

 The compaction count remains in device memory. The following proto kernel reads
 that count directly, so there is no per-frame ``nonzero``, ``tolist`` or count
 readback. CuPy is already an optional dependency of the GPU cleanup path; failure
 to import/compile it simply disables this stricter fast path."""
    global _RESIDENT_MASK_KERNELS, _RESIDENT_MASK_KERNELS_FAILED
    if _RESIDENT_MASK_KERNELS is not None:
        return _RESIDENT_MASK_KERNELS
    if _RESIDENT_MASK_KERNELS_FAILED:
        return None
    try:
        import cupy as cp  # type: ignore
        src = r'''
        #include <cuda_fp16.h>
        extern "C" __global__ void compact_f32(
            const float* head, int anchors, float threshold, int* indices, int* count) {
          int a = blockDim.x * blockIdx.x + threadIdx.x;
          if (a < anchors && head[4 * anchors + a] >= threshold) {
            int dst = atomicAdd(count, 1);
            indices[dst] = a;
          }
        }
        extern "C" __global__ void compact_f16(
            const half* head, int anchors, float threshold, int* indices, int* count) {
          int a = blockDim.x * blockIdx.x + threadIdx.x;
          if (a < anchors && __half2float(head[4 * anchors + a]) >= threshold) {
            int dst = atomicAdd(count, 1);
            indices[dst] = a;
          }
        }

        __device__ __forceinline__ float rd(const float* p, int i) { return p[i]; }
        __device__ __forceinline__ float rd(const half* p, int i) { return __half2float(p[i]); }

        template <typename H, typename P>
        __device__ void proto_union_body(
            const H* head, const P* proto, const int* indices, const int* count,
            int anchors, int masks, int ph, int pw, int ih, int iw,
            float* max_logit, float* conf_proto) {
          int p = blockDim.x * blockIdx.x + threadIdx.x;
          int pixels = ph * pw;
          if (p >= pixels) return;
          int y = p / pw;
          int x = p - y * pw;
          float best = -6.0f;
          float best_conf = 0.0f;
          int n = *count;
          for (int j = 0; j < n; ++j) {
            int a = indices[j];
            float cx = rd(head, 0 * anchors + a);
            float cy = rd(head, 1 * anchors + a);
            float bw = rd(head, 2 * anchors + a);
            float bh = rd(head, 3 * anchors + a);
            float x1 = (cx - 0.5f * bw) * ((float)pw / (float)iw);
            float y1 = (cy - 0.5f * bh) * ((float)ph / (float)ih);
            float x2 = (cx + 0.5f * bw) * ((float)pw / (float)iw);
            float y2 = (cy + 0.5f * bh) * ((float)ph / (float)ih);
            if ((float)x < x1 || (float)x >= x2 || (float)y < y1 || (float)y >= y2) continue;
            float logit = 0.0f;
            #pragma unroll 4
            for (int c = 0; c < masks; ++c) {
              logit += rd(head, (5 + c) * anchors + a) * rd(proto, c * pixels + p);
            }
            best = fmaxf(best, logit);
            if (conf_proto != nullptr && logit > 0.0f) {
              best_conf = fmaxf(best_conf, rd(head, 4 * anchors + a));
            }
          }
          max_logit[p] = best;
          if (conf_proto != nullptr) conf_proto[p] = best_conf;
        }

        #define DECL_UNION(NAME, HT, PT) \
        extern "C" __global__ void NAME( \
            const HT* head, const PT* proto, const int* indices, const int* count, \
            int anchors, int masks, int ph, int pw, int ih, int iw, \
            float* max_logit, float* conf_proto) { \
          proto_union_body(head, proto, indices, count, anchors, masks, ph, pw, ih, iw, \
                           max_logit, conf_proto); \
        }
        DECL_UNION(union_f32_f32, float, float)
        DECL_UNION(union_f32_f16, float, half)
        DECL_UNION(union_f16_f32, half, float)
        DECL_UNION(union_f16_f16, half, half)

        extern "C" __global__ void proto_threshold_signed(
            const float* src, int ph, int pw, float* dst) {
          int p = blockDim.x * blockIdx.x + threadIdx.x;
          int pixels = ph * pw;
          if (p >= pixels) return;
          dst[p] = src[p] > 0.0f ? 1.0f : -1.0f;
        }

        extern "C" __global__ void proto_dilate_signed(
            const float* src, int ph, int pw, int radius, float* dst) {
          int p = blockDim.x * blockIdx.x + threadIdx.x;
          int pixels = ph * pw;
          if (p >= pixels) return;
          int y = p / pw;
          int x = p - y * pw;
          bool fg = false;
          for (int dy = -radius; dy <= radius && !fg; ++dy) {
            int yy = max(0, min(ph - 1, y + dy));
            for (int dx = -radius; dx <= radius; ++dx) {
              int xx = max(0, min(pw - 1, x + dx));
              if (src[yy * pw + xx] > 0.0f) { fg = true; break; }
            }
          }
          dst[p] = fg ? 1.0f : -1.0f;
        }

        extern "C" __global__ void proto_erode_signed(
            const float* src, int ph, int pw, int radius, float* dst) {
          int p = blockDim.x * blockIdx.x + threadIdx.x;
          int pixels = ph * pw;
          if (p >= pixels) return;
          int y = p / pw;
          int x = p - y * pw;
          bool fg = true;
          for (int dy = -radius; dy <= radius && fg; ++dy) {
            int yy = max(0, min(ph - 1, y + dy));
            for (int dx = -radius; dx <= radius; ++dx) {
              int xx = max(0, min(pw - 1, x + dx));
              if (src[yy * pw + xx] <= 0.0f) { fg = false; break; }
            }
          }
          dst[p] = fg ? 1.0f : -1.0f;
        }

        extern "C" __global__ void upsample_quantize(
            const float* max_logit, const float* conf_proto,
            int ph, int pw, int oh, int ow, unsigned char* out_union,
            unsigned char* out_conf) {
          int q = blockDim.x * blockIdx.x + threadIdx.x;
          int out_pixels = oh * ow;
          if (q >= out_pixels) return;
          int oy = q / ow;
          int ox = q - oy * ow;
          float sy = ((float)oy + 0.5f) * ((float)ph / (float)oh) - 0.5f;
          float sx = ((float)ox + 0.5f) * ((float)pw / (float)ow) - 0.5f;
          int y0r = (int)floorf(sy), x0r = (int)floorf(sx);
          float fy = sy - (float)y0r, fx = sx - (float)x0r;
          int y0 = max(0, min(ph - 1, y0r));
          int x0 = max(0, min(pw - 1, x0r));
          int y1 = max(0, min(ph - 1, y0r + 1));
          int x1 = max(0, min(pw - 1, x0r + 1));
          float v00 = max_logit[y0 * pw + x0];
          float v01 = max_logit[y0 * pw + x1];
          float v10 = max_logit[y1 * pw + x0];
          float v11 = max_logit[y1 * pw + x1];
          float v0 = v00 + fx * (v01 - v00);
          float v1 = v10 + fx * (v11 - v10);
          unsigned char fg = (v0 + fy * (v1 - v0)) > 0.0f ? 1 : 0;
          out_union[q] = fg;
          if (out_conf != nullptr) {
            int ny = min(ph - 1, (int)((long long)oy * ph / oh));
            int nx = min(pw - 1, (int)((long long)ox * pw / ow));
            float cf = fminf(1.0f, fmaxf(0.0f, conf_proto[ny * pw + nx]));
            out_conf[q] = fg ? (unsigned char)__float2uint_rn(cf * 255.0f) : 0;
          }
        }

        extern "C" __global__ void upsample_quantize_affine(
            const float* max_logit, const float* conf_proto,
            int ph, int pw, int ih, int iw, int oh, int ow,
            float m00, float m01, float m02,
            float m10, float m11, float m12,
            unsigned char* out_union, unsigned char* out_conf) {
          int q = blockDim.x * blockIdx.x + threadIdx.x;
          int out_pixels = oh * ow;
          if (q >= out_pixels) return;
          int oy = q / ow;
          int ox = q - oy * ow;

          // Pixel-center convention: integer (ox,oy) is a native pixel center.  The
          // precomputed inverse affine maps it to the network raster, matching
          // _affine_theta_for_grid_sample with align_corners=False.  Outside the
          // network pixel-center extent is constant zero padding.  Inside, compose
          // that coordinate directly with the align_corners=False proto upsample;
          // threshold only after this one bilinear sample (there is no intermediate
          // out_size binary plane and no full-resolution grid tensor).
          float ix = m00 * (float)ox + m01 * (float)oy + m02;
          float iy = m10 * (float)ox + m11 * (float)oy + m12;
          if (ix < -0.5f || ix >= (float)iw - 0.5f ||
              iy < -0.5f || iy >= (float)ih - 0.5f) {
            out_union[q] = 0;
            if (out_conf != nullptr) out_conf[q] = 0;
            return;
          }

          float sx = (ix + 0.5f) * ((float)pw / (float)iw) - 0.5f;
          float sy = (iy + 0.5f) * ((float)ph / (float)ih) - 0.5f;
          int y0r = (int)floorf(sy), x0r = (int)floorf(sx);
          float fy = sy - (float)y0r, fx = sx - (float)x0r;
          int y0 = max(0, min(ph - 1, y0r));
          int x0 = max(0, min(pw - 1, x0r));
          int y1 = max(0, min(ph - 1, y0r + 1));
          int x1 = max(0, min(pw - 1, x0r + 1));
          float v00 = max_logit[y0 * pw + x0];
          float v01 = max_logit[y0 * pw + x1];
          float v10 = max_logit[y1 * pw + x0];
          float v11 = max_logit[y1 * pw + x1];
          float v0 = v00 + fx * (v01 - v00);
          float v1 = v10 + fx * (v11 - v10);
          unsigned char fg = (v0 + fy * (v1 - v0)) > 0.0f ? 1 : 0;
          out_union[q] = fg;
          if (out_conf != nullptr) {
            // Confidence remains nearest-proto sampled, as in the generic path.
            int ny = max(0, min(ph - 1, (int)floorf(iy * ((float)ph / (float)ih))));
            int nx = max(0, min(pw - 1, (int)floorf(ix * ((float)pw / (float)iw))));
            float cf = fminf(1.0f, fmaxf(0.0f, conf_proto[ny * pw + nx]));
            out_conf[q] = fg ? (unsigned char)__float2uint_rn(cf * 255.0f) : 0;
          }
        }
        '''
        names = (
            'compact_f32', 'compact_f16',
            'union_f32_f32', 'union_f32_f16', 'union_f16_f32', 'union_f16_f16',
            'proto_threshold_signed', 'proto_dilate_signed', 'proto_erode_signed',
            'upsample_quantize', 'upsample_quantize_affine',
        )
        module = cp.RawModule(code=src, options=('--std=c++11',), name_expressions=names)
        _RESIDENT_MASK_KERNELS = argparse.Namespace(
            cp=cp,
            **{name: module.get_function(name) for name in names},
        )
        return _RESIDENT_MASK_KERNELS
    except Exception:
        _RESIDENT_MASK_KERNELS_FAILED = True
        return None

class _DirectPredictResult:
    """Minimal per-frame result for the direct loop.

 Duck-typed for _extract_result_masks_and_confs: it only carries the prebuilt
 GpuFlattenedRetinaPayload; no Ultralytics Results object, boxes, paths, or orig
 images are constructed."""

    __slots__ = ('_tta_gpu_flattened_payload',)

    def __init__(self, payload: GpuFlattenedRetinaPayload) -> None:
        self._tta_gpu_flattened_payload = payload

def _direct_predict_applicable(cfg: 'PredictConfig') -> bool:
    """Direct loop preconditions: proto-union consume path + a CUDA device."""
    if not direct_predict_enabled():
        return False
    if cpu_retina_masks_enabled() or not gpu_retina_proto_union_enabled():
        return False
    try:
        import torch  # type: ignore
    except Exception:
        return False
    try:
        if not str(canonical_single_device(str(cfg.device))).startswith('cuda'):
            return False
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def _ensure_predictor_for_direct_predict(model: object, cfg: 'PredictConfig') -> Optional[object]:
    """Create/fetch the Ultralytics predictor + backend without running stream_inference.

 Model.predict(..., stream=True) constructs the predictor and calls setup_model
 (AutoBackend load and device/precision placement) eagerly but returns an UNITERATED generator,
 so no source setup, warmup, or inference runs. The predictor's (patched) preprocess and
 its AutoBackend are then driven directly by _direct_predict_stream."""
    try:
        predictor = getattr(model, 'predictor', None)
        if predictor is None or getattr(predictor, 'model', None) is None:
            _ = model.predict(
                source=np.zeros((32, 32, max(1, int(cfg.input_channels))), dtype=np.uint8),
                task='segment',
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=1.0,
                save=False,
                stream=True,
                retina_masks=True,
                batch=max(1, int(cfg.batch)),
                device=cfg.device,
                quantize=cfg.quantize,
                verbose=False,
            )
            predictor = getattr(model, 'predictor', None)
        if predictor is None or getattr(predictor, 'model', None) is None:
            return None
        ensure_yolo_ready_for_predict(model, cfg)
        validate_yolo_model_input_channels(
            model,
            int(cfg.input_channels),
            channel_token=str(cfg.channel_token),
            context=f'direct predictor setup on {cfg.device}',
        )
        return predictor
    except ModelInputChannelMismatchError:
        raise
    except Exception as exc:
        print(f'Warning: direct predict setup failed ({exc}); using model.predict for this source.')
        return None

def _split_segmentation_backend_outputs(preds: object) -> Optional[Tuple[object, object]]:
    """Return (head, protos) from a raw segmentation forward.

 Mirrors SegmentationPredictor.postprocess's output handling: preds[0] is the
 (B, 4+nc+nm, A) detection head, preds[1] the (B, nm, mh, mw) prototype tensor (torch
.pt backends nest it as the last element of a tuple). None when the structure does not
 match, so the caller can fail fast instead of mis-decoding."""
    try:
        if not isinstance(preds, (list, tuple)) or len(preds) < 2:
            return None
        head = preds[0]
        proto = preds[1]
        if isinstance(proto, (list, tuple)):
            proto = proto[-1]
        if getattr(head, 'ndim', 0) != 3 or getattr(proto, 'ndim', 0) != 4:
            return None
        return head, proto
    except Exception:
        return None

_DIRECT_DEVICE_COMPACTION_ANNOUNCED = False

_DIRECT_DEVICE_COMPACTION_FALLBACK_WARNED = False

def direct_device_compaction_enabled() -> bool:
    """Fuse the generic direct loop's confidence gate + proto union without host counts."""
    return _env_flag('YOLO_TTA_DIRECT_DEVICE_COMPACTION', True)

def _build_direct_device_compacted_payload(
    head_frame: object,
    proto_frame: object,
    im: object,
    confidence_threshold: float,
    min_conf_applied: bool = False,
) -> Optional[GpuFlattenedRetinaPayload]:
    """Build one flattened mask using the resident kernels on the current CUDA stream.

 Unlike the legacy direct path this never evaluates a data-dependent tensor shape on the
 host. ``compact_count`` is consumed by the proto kernel in device memory and travels with
 the payload for task-end accounting."""
    if not direct_device_compaction_enabled():
        return None
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        kernels = _resident_mask_kernels()
        if kernels is None:
            return None
        # AutoBackend/TensorRT may reuse static output bindings on the next enqueue while this
        # payload is queued to a postprocess worker. Snapshot the compact head/proto tensors;
        # this is still orders of magnitude smaller than materializing an instance-mask stack.
        head = head_frame.contiguous().clone()
        proto = proto_frame.contiguous().clone()
        if int(head.ndim) != 2 or int(proto.ndim) != 3:
            return None
        if head.dtype not in (torch.float16, torch.float32):
            return None
        if proto.dtype not in (torch.float16, torch.float32):
            return None
        masks, ph, pw = (int(v) for v in proto.shape)
        if int(head.shape[0]) != 5 + int(masks):
            return None  # this pipeline supports one segmentation class
        anchors = int(head.shape[1])
        if anchors <= 0 or masks <= 0 or ph <= 0 or pw <= 0:
            return None
        img_h, img_w = int(im.shape[-2]), int(im.shape[-1])
        device = head.device
        indices = torch.empty((anchors,), dtype=torch.int32, device=device)
        count = torch.zeros((1,), dtype=torch.int32, device=device)
        max_logit = torch.empty((ph, pw), dtype=torch.float32, device=device)
        conf_proto = (
            torch.empty((ph, pw), dtype=torch.float32, device=device)
            if gpu_flatten_conf_tracking_enabled() else None
        )

        cp = kernels.cp
        stream = torch.cuda.current_stream(device)
        external = _cupy_external_stream(cp, stream)
        cp_head = cp.asarray(head)
        cp_proto = cp.asarray(proto)
        cp_indices = cp.asarray(indices)
        cp_count = cp.asarray(count)
        cp_max_logit = cp.asarray(max_logit)
        cp_conf_proto = cp.asarray(conf_proto) if conf_proto is not None else None
        compact = kernels.compact_f16 if head.dtype == torch.float16 else kernels.compact_f32
        compact(
            ((anchors + 255) // 256,), (256,),
            (
                cp_head, np.int32(anchors), np.float32(confidence_threshold),
                cp_indices, cp_count,
            ),
            stream=external,
        )
        htag = 'f16' if head.dtype == torch.float16 else 'f32'
        ptag = 'f16' if proto.dtype == torch.float16 else 'f32'
        union_kernel = getattr(kernels, f'union_{htag}_{ptag}')
        union_kernel(
            (((ph * pw) + 255) // 256,), (256,),
            (
                cp_head, cp_proto, cp_indices, cp_count,
                np.int32(anchors), np.int32(masks), np.int32(ph), np.int32(pw),
                np.int32(img_h), np.int32(img_w), cp_max_logit,
                cp_conf_proto if cp_conf_proto is not None else np.uintp(0),
            ),
            stream=external,
        )

        # One plane, not one plane per instance. The terminal view warp remains in the existing
        # post path; the resident-ring specialization fuses this upsample with that warp.
        union_gpu = (
            F.interpolate(
                max_logit.reshape(1, 1, ph, pw),
                size=(img_h, img_w), mode='bilinear', align_corners=False,
            ).reshape(img_h, img_w) > 0.0
        ).to(torch.float32)
        conf_gpu = None
        if conf_proto is not None:
            conf_gpu = F.interpolate(
                conf_proto.reshape(1, 1, ph, pw),
                size=(img_h, img_w), mode='nearest',
            ).reshape(img_h, img_w)

        ready_event = torch.cuda.Event()
        ready_event.record(stream)
        fastpath = angle_variant_gpu_fastpath()
        fastpath_radius = 0.0 if fastpath is None else float(fastpath[1])
        payload = GpuFlattenedRetinaPayload(
            union_gpu=union_gpu,
            conf_gpu=conf_gpu,
            instance_count=-1,
            instance_count_device=count,
            min_conf_applied=bool(min_conf_applied),
            run_gpu_cleanup=bool(
                fastpath is not None and gpu_retina_cleanup_enabled() and fastpath_radius > 0.0
            ),
            gpu_min_radius=float(fastpath_radius),
            ready_event=ready_event,
            device_refs=(
                head, proto, indices, count, max_logit, conf_proto,
                cp_head, cp_proto, cp_indices, cp_count, cp_max_logit, cp_conf_proto,
            ),
        )
        return payload
    except Exception:
        return None

def _direct_predict_stream(
    predictor: object,
    source: object,
    cfg: 'PredictConfig',
    source_label: str,
) -> Iterator[_DirectPredictResult]:
    """Run the direct preprocess, backend-forward, confidence-gate, and proto-union loop.
    
    The loop preserves padded-frame accounting while avoiding per-frame Results and avoidable host count reads."""
    import torch  # type: ignore

    backend = getattr(predictor, 'model', None)
    if backend is None:
        raise RuntimeError('direct predict: predictor has no backend model')
    names = getattr(backend, 'names', None) or {}
    nc = max(1, len(names))
    conf_thres = float(cfg.conf)
    device_compaction_active = bool(direct_device_compaction_enabled())

    global _DIRECT_PREDICT_ANNOUNCED, _DIRECT_DEVICE_COMPACTION_ANNOUNCED, _DIRECT_DEVICE_COMPACTION_FALLBACK_WARNED
    if not _DIRECT_PREDICT_ANNOUNCED:
        _DIRECT_PREDICT_ANNOUNCED = True
        print(
            'Direct backend predict loop active (v13.3.6 C2): stream_inference, NMS and '
            'per-frame Results are bypassed; confidence-gated instances feed the '
            'proto-resolution union directly (YOLO_TTA_DIRECT_PREDICT=0 restores model.predict).'
        )

    start_fn = getattr(source, 'start', None)
    if callable(start_fn):
        try:
            start_fn()
        except Exception:
            pass

    with torch.inference_mode():
        for _paths, im0s, _info in iter(source):
            im = predictor.preprocess(im0s)
            preds = backend(im)
            split = _split_segmentation_backend_outputs(preds)
            if split is None:
                raise RuntimeError(
                    f'direct predict: unexpected backend output structure for {source_label}; '
                    'set YOLO_TTA_DIRECT_PREDICT=0 to use model.predict'
                )
            head, protos = split
            bsz = int(im.shape[0])
            no = int(head.shape[1])
            nm = int(protos.shape[1])
            if no != 4 + nc + nm or int(head.shape[0]) != bsz or int(protos.shape[0]) != bsz:
                raise RuntimeError(
                    f'direct predict: head/proto layout mismatch for {source_label} '
                    f'(head channels {no}, nc {nc}, protos {nm}); '
                    'set YOLO_TTA_DIRECT_PREDICT=0 to use model.predict'
                )
            if nc == 1 and device_compaction_active:
                _direct_fastpath = angle_variant_gpu_fastpath()
                _direct_effective_threshold = max(
                    float(conf_thres),
                    float(_direct_fastpath[0]) if _direct_fastpath is not None else float(conf_thres),
                )
                compacted_payloads = [
                    _build_direct_device_compacted_payload(
                        head[int(i)], protos[int(i)], im, float(_direct_effective_threshold),
                        min_conf_applied=bool(
                            _direct_fastpath is not None and float(_direct_fastpath[0]) > 0.0
                        ),
                    )
                    for i in range(int(bsz))
                ]
                if all(payload is not None for payload in compacted_payloads):
                    if not _DIRECT_DEVICE_COMPACTION_ANNOUNCED:
                        _DIRECT_DEVICE_COMPACTION_ANNOUNCED = True
                        print(
                            'Direct device compaction active (v13.3.12 C3): confidence filtering, '
                            'proto dot/crop union, and instance counts remain on device; '
                            'YOLO_TTA_DIRECT_DEVICE_COMPACTION=0 restores the synchronized fallback.'
                        )
                    for payload in compacted_payloads:
                        yield _DirectPredictResult(payload)  # type: ignore[arg-type]
                    continue
                if not _DIRECT_DEVICE_COMPACTION_FALLBACK_WARNED:
                    _DIRECT_DEVICE_COMPACTION_FALLBACK_WARNED = True
                    print(
                        'Warning: direct device compaction unavailable; this source uses the '
                        'legacy nonzero/count fallback. The resident TensorRT ring remains independent.'
                    )
                # A deterministic dtype/CuPy/NVRTC failure or OOM should not repeat the same
                # clone/allocation/probe on every remaining batch in this source.
                device_compaction_active = False
            scores = head[:, 4:4 + nc, :]
            if nc == 1:
                conf_all = scores[:, 0, :]
                cls_all = None
            else:
                conf_all, cls_all = scores.max(dim=1)
            keep_mask = conf_all >= conf_thres  # (B, A)
            nz = keep_mask.nonzero()  # host sync; row-major => grouped by image
            if int(nz.shape[0]) > 0:
                counts = torch.bincount(nz[:, 0], minlength=bsz).tolist()  # host sync (tiny)
            else:
                counts = [0] * bsz
            offset = 0
            for i in range(bsz):
                n_i = int(counts[i])
                if n_i <= 0:
                    pred_i = head.new_zeros((0, 6 + nm))
                else:
                    rows = nz[offset:offset + n_i, 1]
                    det = head[i].index_select(1, rows)  # (4+nc+nm, n) — copies out of binding memory
                    xy = det[0:2, :].transpose(0, 1).to(torch.float32)
                    half_wh = det[2:4, :].transpose(0, 1).to(torch.float32) * 0.5
                    conf_i = conf_all[i].index_select(0, rows).to(torch.float32).reshape(-1, 1)
                    cls_i = (
                        torch.zeros_like(conf_i)
                        if cls_all is None
                        else cls_all[i].index_select(0, rows).to(torch.float32).reshape(-1, 1)
                    )
                    coeffs = det[4 + nc:, :].transpose(0, 1).to(torch.float32)
                    pred_i = torch.cat([xy - half_wh, xy + half_wh, conf_i, cls_i, coeffs], dim=1)
                offset += n_i
                payload = _build_gpu_flattened_payload_from_proto(pred_i, im, protos[i])
                if payload is None:
                    raise RuntimeError(
                        f'direct predict: payload construction failed for {source_label}; '
                        'set YOLO_TTA_DIRECT_PREDICT=0 to use model.predict'
                    )
                yield _DirectPredictResult(payload)

def predict_source_and_accumulate(
    model,
    source: object,
    *,
    source_label: str,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_workers: int = 1,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
    device_hole_fill: bool = False,
    defer_device_union_flush: bool = False,
    device_union_consumer: Optional[Callable[['_DeviceUnionAccumulator'], Dict[str, object]]] = None,
    require_device_union: bool = False,
    require_proto_hole_treatment: bool = False,
    radial_padding_union_mm: Optional[np.ndarray] = None,
    radial_padding_confmap_mm: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run YOLO predict(stream=True) on an in-memory source and accumulate native masks.

 The predictor consumes ``InMemoryYoloVolumeSource`` instances in the path,
 so the GPU no longer reads augmented FFV1 videos from scratch. CPU-side result
 handling remains bounded by ``cpu_mask_postprocess_pending_limit`` and runs
 behind the streamed GPU inference. When the angle-variant cleanup
 path is enabled, slice-local filtering is appended to the same streamed
 postprocess unit so a completed prediction slice is already cleaned before
 the full view volume has finished inferencing."""
    # only compute the GPU flatten's max-conf plane when a confidence
    # volume actually exists downstream.
    # Local import keeps the package dependency graph acyclic.
    from .cuda_backend import (
        GpuRenderedYoloSource,
        GpuTileRenderedYoloSource,
    )
    from .backprojection import _try_resident_trt_ring_accumulate

    set_gpu_flatten_conf_tracking(view_confmap_mm is not None)
    ensure_yolo_ready_for_predict(model, cfg)
    source_channels = _source_prediction_channel_count(source, cfg)
    if int(source_channels) != int(cfg.input_channels):
        raise ValueError(
            f'{source_label}: source yields C={int(source_channels)}, but '
            f'--channel_format {cfg.channel_token} requires C={int(cfg.input_channels)}'
        )
    validate_yolo_model_input_channels(
        model,
        int(cfg.input_channels),
        channel_token=str(cfg.channel_token),
        context=f'prediction source {source_label}',
    )
    unwrapped_source = source
    source = maybe_wrap_source_with_gpu_input_staging(source, cfg, source_label)
    # the call above may create a GpuPrefetchingYoloSource that nothing
    # else owns. If iteration is abandoned by an exception (model.predict setup, a
    # mid-stream CUDA error, a postprocess future failure), its producer thread and
    # VRAM-staged queue leaked for the life of the process. Close the wrapper created
    # here on the way out; on success this is an idempotent no-op (the sentinel path in
    # GpuPrefetchingYoloSource.__next__ already closed it).
    owned_staging_wrapper = (
        source
        if (source is not unwrapped_source and isinstance(source, GpuPrefetchingYoloSource))
        else None
    )
    try:
        if isinstance(source, (
            InMemoryYoloVolumeSource, StreamingYoloVolumeSource, GpuPrefetchingYoloSource,
            GpuRenderedYoloSource, GpuTileRenderedYoloSource,
        )):
            require_channel_aware_yolo_preprocess_patch(str(cfg.channel_token))
        use_custom_cpu_retina = False
        if cpu_retina_masks_enabled():
            use_custom_cpu_retina = bool(ensure_cpu_retina_mask_predictor_patch())
            if not use_custom_cpu_retina:
                print('Warning: CPU retina predictor patch unavailable; using Ultralytics native retina_masks=True for this source.')
        else:
            # GPU retina mode — reduce unions at proto resolution inside
            # construct_result instead of materializing (n, imgsz, imgsz) retina stacks.
            ensure_gpu_retina_proto_union_predictor_patch()

        prediction_count = 0
        frames_with_predictions = 0

        # drive the backend directly on the proto-union fast path; fall back
        # to Ultralytics stream_inference when the direct loop's preconditions do not hold.
        results = None
        predictor_direct = None
        if not use_custom_cpu_retina and _direct_predict_applicable(cfg):
            predictor_direct = _ensure_predictor_for_direct_predict(model, cfg)
            if predictor_direct is not None:
                results = _direct_predict_stream(predictor_direct, source, cfg, source_label)
        if results is None:
            results = model.predict(
                source=source,
                task='segment',
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=1.0,
                save=False,
                stream=True,
                retina_masks=not bool(use_custom_cpu_retina),
                batch=max(1, int(cfg.batch)),
                device=cfg.device,
                quantize=cfg.quantize,
                verbose=False,
            )
            validate_yolo_model_input_channels(
                model,
                int(cfg.input_channels),
                channel_token=str(cfg.channel_token),
                context=f'initialized prediction backend for {source_label}',
            )

        worker_count = max(1, min(int(postprocess_workers), int(num_frames)))
        pending_limit = cpu_mask_postprocess_pending_limit(worker_count, int(num_frames))
        stream_cleanup = bool(streaming_cleanup_enabled)
        stream_backend = cleanup_backend() if stream_cleanup else ''
        stream_structure2 = np.ones((3, 3), dtype=bool) if stream_cleanup else None
        stream_min_conf = float(streaming_cleanup_min_conf)
        stream_min_radius = float(streaming_cleanup_min_radius)
        stream_min_conf_u8 = int(min_conf_to_u8_threshold(stream_min_conf)) if stream_min_conf > 0.0 else 0

        # Admit raw device-union accumulation whenever no host-only cleanup must run before
        # union. Every angle-variant task therefore retains its masks on device; only
        # positive per-slice radius cleanup, unsupported confidence cleanup, CPU retina masks,
        # or insufficient VRAM force the per-frame host path.
        device_union: Optional[_DeviceUnionAccumulator] = None
        fastpath = angle_variant_gpu_fastpath()
        preunion_min_conf = (
            float(stream_min_conf)
            if stream_cleanup and fastpath is not None and stream_min_conf > 0.0
            else None
        )
        host_cleanup_required = bool(
            stream_cleanup
            and (
                (stream_min_conf > 0.0 and preunion_min_conf is None)
                or stream_min_radius > 0.0
            )
        )
        if not cpu_retina_masks_enabled() and not host_cleanup_required:
            device_union = _try_create_device_union_accumulator(
                canonical_single_device(str(cfg.device)),
                int(num_frames), int(native_h), int(native_w),
                want_conf=view_confmap_mm is not None,
            )
            if device_union is not None:
                print(
                    f'Device union accumulation active for {source_label}: '
                    f'{int(num_frames)}x{int(native_h)}x{int(native_w)} u8 on device.'
                )
        if bool(require_device_union) and device_union is None:
            raise RuntimeError(
                f'{source_label}: the D1 fast path requires a task-local device union; '
                'disable YOLO_TTA_V1613_FAST_BUNDLE to use host-dense compatibility'
            )

        # A capable resident batch-1 TensorRT source runs render, inference, confidence
        # compaction, proto union, and destination warping as one device pipeline. Every
        # full-frame or tile task owns one immutable output-to-destination affine.
        specialized_stats: Optional[Dict[str, int]] = None
        if predictor_direct is not None:
            specialized_stats = _try_resident_trt_ring_accumulate(
                predictor_direct, source, cfg,
                num_frames=int(num_frames),
                out_size=int(out_size),
                M_out_to_native=M_out_to_native,
                native_h=int(native_h),
                native_w=int(native_w),
                device_union=device_union,
                preunion_min_conf=preunion_min_conf,
            )
            if specialized_stats is not None:
                prediction_count = int(specialized_stats['prediction_count'])
                frames_with_predictions = int(specialized_stats['frames_with_predictions'])

        source_padding_count = max(0, int(getattr(source, 'radial_padding_count', 0) or 0))
        effective_slice_locks = slice_locks
        if effective_slice_locks is None and source_padding_count > 0:
            effective_slice_locks = [
                threading.Lock() for _ in range(max(1, int(view_union_mm.shape[0])))
            ]

        def _process_prediction_unit(
            spec: BatchResultFrameSpec,
            masks_obj: Optional[object],
            confs_arr: Optional[np.ndarray],
        ) -> Tuple[int, int]:
            target_union, target_conf, target_index, target_affine, count_stats = (
                _prediction_accumulation_target(
                    spec,
                    view_union_mm=view_union_mm,
                    view_confmap_mm=view_confmap_mm,
                    radial_padding_union_mm=radial_padding_union_mm,
                    radial_padding_confmap_mm=radial_padding_confmap_mm,
                    M_out_to_native=M_out_to_native,
                    native_w=int(native_w),
                )
            )
            slice_lock = None
            if effective_slice_locks is not None and len(effective_slice_locks) > 0:
                lock_index = (
                    int(spec.global_destination_index)
                    if bool(spec.is_radial_padding) else int(spec.task_index)
                )
                slice_lock = effective_slice_locks[lock_index % len(effective_slice_locks)]
            if isinstance(masks_obj, GpuFlattenedRetinaPayload):
                # The payload builder sees only the process-global/native radius. Override it
                # with this source's scaled processing-grid radius before GPU cleanup.
                masks_obj.gpu_min_radius = float(stream_min_radius)
                masks_obj.run_gpu_cleanup = bool(
                    angle_variant_gpu_fastpath() is not None
                    and gpu_retina_cleanup_enabled()
                    and float(stream_min_radius) > 0.0
                )
            # Hold the destination shard across both union and cleanup. Ordinary slice 0,
            # wrapped slice 0, and repeated multi-wrap destinations can otherwise clean the
            # same host plane concurrently after _process_prediction_frame releases its lock.
            with (slice_lock if slice_lock is not None else contextlib.nullcontext()):
                pred_inc, frame_inc = _process_prediction_frame(
                    idx=int(target_index),
                    masks_np=masks_obj,
                    confs_np=confs_arr,
                    out_size=out_size,
                    view_union_mm=target_union,
                    view_confmap_mm=target_conf,
                    M_out_to_native=target_affine,
                    native_h=native_h,
                    native_w=native_w,
                    slice_lock=None,
                    # The task-local device union has only logical frames. Radial extension
                    # slots are few and retire through their explicit host/auxiliary sink.
                    device_union=(device_union if count_stats else None),
                )
                # skip the per-slice CPU streaming cleanup when the GPU fast path already ran
                # min_conf + --min_radius + hole fill on this slice (cleanup_done_on_gpu is set by the frame
                # processor only when the on-GPU connected-component cleanup actually completed).
                # also skip it for device-accumulated frames — their host slice is
                # written by the task-end flush, and this cleanup was gated to scan-only work anyway.
                if (
                    stream_cleanup
                    and not bool(getattr(masks_obj, 'cleanup_done_on_gpu', False))
                    and not bool(getattr(masks_obj, 'accumulated_on_device', False))
                ):
                    cleaned_has_foreground = _cleanup_prediction_slice_inplace(
                        target_union,
                        target_conf,
                        int(target_index),
                        min_conf=stream_min_conf,
                        min_radius=stream_min_radius,
                        backend=stream_backend,
                        structure2=stream_structure2,
                        min_conf_u8=stream_min_conf_u8,
                    )
                    frame_inc = 1 if bool(cleaned_has_foreground) else 0
            if not count_stats:
                # Wrapped slots improve the logical slice union but are not additional
                # source frames; keep task/frame telemetry bounded by num_frames.
                return 0, 0
            return int(pred_inc), int(frame_inc)

        def _extract_and_process_result(spec: BatchResultFrameSpec, result_obj: object) -> Tuple[int, int]:
            masks_np, confs_np = _extract_result_masks_and_confs(result_obj)
            try:
                del result_obj
            except Exception:
                pass
            return _process_prediction_unit(spec, masks_np, confs_np)

        # on the GPU-flatten path, eagerly reduce each (n,Hr,Wr) GPU stack to 2 small
        # planes on this (stream) thread so the full stack is released immediately, and bound the queue so
        # only a capped number of GPU-resident flattened frames stay alive (avoids device OOM).
        gpu_flatten_eager = bool(
            gpu_retina_flatten_enabled() and not cpu_retina_masks_enabled() and gpu_retina_eager_flatten_enabled()
        )
        effective_pending_limit = int(pending_limit)
        if gpu_flatten_eager:
            effective_pending_limit = max(1, min(int(pending_limit), gpu_retina_flatten_pending_limit(worker_count)))

        radial_padding_processed = 0
        if specialized_stats is not None:
            # The resident ring wrote the device union and task metadata directly.
            pass
        elif worker_count <= 1:
            for idx, r in enumerate(results):
                spec = prediction_result_frame_spec(source, int(idx), num_frames=int(num_frames))
                if spec is None:
                    continue
                radial_padding_processed += int(bool(spec.is_radial_padding))
                masks_np, confs_np = _extract_result_masks_and_confs(r)
                pred_inc, frame_inc = _process_prediction_unit(spec, masks_np, confs_np)
                prediction_count += int(pred_inc)
                frames_with_predictions += int(frame_inc)
        else:
            pending: List[Future] = []
            # checkout-cached pool — one build per worker process, not per task.
            executor = _acquire_parallel_pool(worker_count)
            try:
                for idx, r in enumerate(results):
                    spec = prediction_result_frame_spec(source, int(idx), num_frames=int(num_frames))
                    if spec is None:
                        continue
                    radial_padding_processed += int(bool(spec.is_radial_padding))

                    if gpu_flatten_eager:
                        masks_np, confs_np = _extract_result_masks_and_confs(r)
                        try:
                            del r
                        except Exception:
                            pass
                        pending.append(executor.submit(_process_prediction_unit, spec, masks_np, confs_np))
                    else:
                        pending.append(executor.submit(_extract_and_process_result, spec, r))
                    if len(pending) >= effective_pending_limit:
                        fut = pending.pop(0)
                        pred_inc, frame_inc = fut.result()
                        prediction_count += int(pred_inc)
                        frames_with_predictions += int(frame_inc)

                while pending:
                    fut = pending.pop(0)
                    pred_inc, frame_inc = fut.result()
                    prediction_count += int(pred_inc)
                    frames_with_predictions += int(frame_inc)
            finally:
                _settle_parallel_futures(pending)
                _release_parallel_pool(worker_count, executor)

        # Keep the inference handoff to one producer-stream seal. Device counts and slice
        # metadata are consumed on the retirement lane from the same D2H chunks that commit
        # the union, instead of two whole-device synchronizations/reductions before return.
        device_hole_filled_frames = int(
            specialized_stats.get('proto_hole_treated_frames', 0)
            if specialized_stats is not None else 0
        )
        if (
            device_union is not None
            and device_hole_filled_frames <= 0
            and bool(device_hole_fill)
            and gpu_device_hole_fill_enabled()
        ):
            device_hole_filled_frames = int(device_union.fill_holes_2d())

        # D1 consumes the task union in-place after the resident proto treatment and
        # before any D2H/full-view publication. The callback backprojects into a persistent
        # owner-GPU source-space bitset and may return one asynchronous cvol publication future
        # when this lease completes the view.
        if device_union_consumer is not None:
            if device_union is None:
                raise RuntimeError(f'{source_label}: D1 callback received no device union')
            if bool(device_union.host_written):
                raise RuntimeError(
                    f'{source_label}: D1 callback cannot accept host-written fallback slices'
                )
            if bool(require_proto_hole_treatment):
                if specialized_stats is None:
                    raise RuntimeError(
                        f'{source_label}: D1 requires the resident TensorRT ring so D3 proto '
                        'topology treatment occurs before backprojection'
                    )
                if int(device_hole_filled_frames) != int(num_frames):
                    raise RuntimeError(
                        f'{source_label}: D3 treated {int(device_hole_filled_frames)}/'
                        f'{int(num_frames)} frames before D1 backprojection'
                    )
            device_union.synchronize_for_retirement(None)
            if specialized_stats is None:
                compacted_predictions, compacted_frames = device_union.take_device_prediction_stats(
                    retirement_lane=None, synchronize_device=False,
                )
                prediction_count += int(compacted_predictions)
                frames_with_predictions += int(compacted_frames)
            consumed = dict(device_union_consumer(device_union) or {})
            publication_future = consumed.pop('_publication_future', None)
            return {
                'prediction_count': int(prediction_count),
                'frames_with_predictions': int(frames_with_predictions),
                'device_hole_filled_frames': int(device_hole_filled_frames),
                'proto_hole_treated_frames': int(device_hole_filled_frames),
                'slice_meta': None,
                'radial_padding_processed': int(radial_padding_processed),
                **consumed,
                # Reuse the established worker-private future channel; the deferred result
                # wrapper is agnostic to whether the future retires D2H or publishes cvol.
                '_device_union_flush_future': publication_future,
            }

        device_union_flush_future: Optional[Future] = None
        slice_meta: Optional[Dict[str, np.ndarray]] = None
        if device_union is not None:
            retirement_manager = _gpu_union_retirement_manager()
            retirement_lane: Optional[_GpuUnionRetirementLane] = (
                retirement_manager.acquire() if retirement_manager is not None else None
            )
            try:
                device_union.synchronize_for_retirement(retirement_lane)
            except BaseException:
                if retirement_manager is not None and retirement_lane is not None:
                    retirement_manager.release(retirement_lane)
                raise
            if bool(defer_device_union_flush) and gpu_union_flush_overlap_enabled():
                base_prediction_count = int(prediction_count)
                base_frames_with_predictions = int(frames_with_predictions)
                count_device_predictions = bool(specialized_stats is None)
                filled_frames_for_result = int(device_hole_filled_frames)

                def _retire_device_union() -> Dict[str, object]:
                    try:
                        compacted_predictions = 0
                        compacted_frames = 0
                        if count_device_predictions:
                            compacted_predictions, compacted_frames = (
                                device_union.take_device_prediction_stats(
                                    retirement_lane=retirement_lane,
                                    synchronize_device=False,
                                )
                            )
                        retired_meta = device_union.flush_into(
                            view_union_mm,
                            view_confmap_mm,
                            retirement_lane=retirement_lane,
                            synchronize_device=False,
                            collect_slice_metadata=True,
                        )
                        if prediction_hot_path_flush_enabled():
                            if view_confmap_mm is not None:
                                flush_array(view_confmap_mm)
                            flush_array(view_union_mm)
                            if radial_padding_confmap_mm is not None:
                                flush_array(radial_padding_confmap_mm)
                            if radial_padding_union_mm is not None:
                                flush_array(radial_padding_union_mm)
                        return {
                            'prediction_count': int(base_prediction_count + compacted_predictions),
                            'frames_with_predictions': int(base_frames_with_predictions + compacted_frames),
                            'device_hole_filled_frames': int(filled_frames_for_result),
                            'slice_meta': (
                                None if int(radial_padding_processed) > 0 else retired_meta
                            ),
                            'radial_padding_processed': int(radial_padding_processed),
                        }
                    finally:
                        if retirement_manager is not None and retirement_lane is not None:
                            retirement_manager.release(retirement_lane)

                try:
                    device_union_flush_future = (
                        retirement_manager.submit(_retire_device_union)
                        if retirement_manager is not None
                        else _gpu_union_retirement_fallback_executor().submit(_retire_device_union)
                    )
                except BaseException:
                    if retirement_manager is not None and retirement_lane is not None:
                        retirement_manager.release(retirement_lane)
                    raise
            else:
                try:
                    if specialized_stats is None:
                        compacted_predictions, compacted_frames = (
                            device_union.take_device_prediction_stats(
                                retirement_lane=retirement_lane,
                                synchronize_device=False,
                            )
                        )
                        prediction_count += int(compacted_predictions)
                        frames_with_predictions += int(compacted_frames)
                    slice_meta = device_union.flush_into(
                        view_union_mm,
                        view_confmap_mm,
                        retirement_lane=retirement_lane,
                        synchronize_device=False,
                        collect_slice_metadata=True,
                    )
                finally:
                    if retirement_manager is not None and retirement_lane is not None:
                        retirement_manager.release(retirement_lane)

        if prediction_hot_path_flush_enabled() and device_union_flush_future is None:
            if view_confmap_mm is not None:
                flush_array(view_confmap_mm)
            flush_array(view_union_mm)
            if radial_padding_confmap_mm is not None:
                flush_array(radial_padding_confmap_mm)
            if radial_padding_union_mm is not None:
                flush_array(radial_padding_union_mm)

        return {
            'prediction_count': int(prediction_count),
            'frames_with_predictions': int(frames_with_predictions),
            'device_hole_filled_frames': int(device_hole_filled_frames),
            'slice_meta': None if int(radial_padding_processed) > 0 else slice_meta,
            'radial_padding_processed': int(radial_padding_processed),
            # Private worker protocol; removed before the stats cross the process queue.
            '_device_union_flush_future': device_union_flush_future,
        }
    finally:
        if owned_staging_wrapper is not None:
            try:
                owned_staging_wrapper.close()
            except Exception:
                pass

def predict_source_and_submit_accumulation(
    model,
    source: object,
    *,
    source_label: str,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_executor: ThreadPoolExecutor,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
    radial_padding_union_mm: Optional[np.ndarray] = None,
    radial_padding_confmap_mm: Optional[np.ndarray] = None,
) -> PredictionAccumulationHandle:
    """Run YOLO streaming inference and enqueue result accumulation without draining it."""
    # only compute the GPU flatten's max-conf plane when a confidence
    # volume actually exists downstream.
    # Local import keeps the package dependency graph acyclic.
    from .cuda_backend import (
        GpuRenderedYoloSource,
        GpuTileRenderedYoloSource,
    )

    set_gpu_flatten_conf_tracking(view_confmap_mm is not None)
    ensure_yolo_ready_for_predict(model, cfg)
    source_channels = _source_prediction_channel_count(source, cfg)
    if int(source_channels) != int(cfg.input_channels):
        raise ValueError(
            f'{source_label}: source yields C={int(source_channels)}, but '
            f'--channel_format {cfg.channel_token} requires C={int(cfg.input_channels)}'
        )
    validate_yolo_model_input_channels(
        model,
        int(cfg.input_channels),
        channel_token=str(cfg.channel_token),
        context=f'prediction source {source_label}',
    )
    unwrapped_source = source
    source = maybe_wrap_source_with_gpu_input_staging(source, cfg, source_label)
    # the call above may create a GpuPrefetchingYoloSource that nothing
    # else owns. If iteration is abandoned by an exception (model.predict setup, a
    # mid-stream CUDA error, a postprocess future failure), its producer thread and
    # VRAM-staged queue leaked for the life of the process. Close the wrapper created
    # here on the way out; on success this is an idempotent no-op (the sentinel path in
    # GpuPrefetchingYoloSource.__next__ already closed it).
    owned_staging_wrapper = (
        source
        if (source is not unwrapped_source and isinstance(source, GpuPrefetchingYoloSource))
        else None
    )
    try:
        if isinstance(source, (
            InMemoryYoloVolumeSource, StreamingYoloVolumeSource, GpuPrefetchingYoloSource,
            GpuRenderedYoloSource, GpuTileRenderedYoloSource,
        )):
            require_channel_aware_yolo_preprocess_patch(str(cfg.channel_token))
        use_custom_cpu_retina = False
        if cpu_retina_masks_enabled():
            use_custom_cpu_retina = bool(ensure_cpu_retina_mask_predictor_patch())
            if not use_custom_cpu_retina:
                print('Warning: CPU retina predictor patch unavailable; using Ultralytics native retina_masks=True for this source.')
        else:
            # GPU retina mode — reduce unions at proto resolution inside
            # construct_result instead of materializing (n, imgsz, imgsz) retina stacks.
            ensure_gpu_retina_proto_union_predictor_patch()

        # same direct-loop preference as predict_source_and_accumulate.
        results = None
        if not use_custom_cpu_retina and _direct_predict_applicable(cfg):
            predictor_direct = _ensure_predictor_for_direct_predict(model, cfg)
            if predictor_direct is not None:
                results = _direct_predict_stream(predictor_direct, source, cfg, source_label)
        if results is None:
            results = model.predict(
                source=source,
                task='segment',
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=1.0,
                save=False,
                stream=True,
                retina_masks=not bool(use_custom_cpu_retina),
                batch=max(1, int(cfg.batch)),
                device=cfg.device,
                quantize=cfg.quantize,
                verbose=False,
            )
            validate_yolo_model_input_channels(
                model,
                int(cfg.input_channels),
                channel_token=str(cfg.channel_token),
                context=f'initialized prediction backend for {source_label}',
            )

        stream_cleanup = bool(streaming_cleanup_enabled)
        stream_backend = cleanup_backend() if stream_cleanup else ''
        stream_structure2 = np.ones((3, 3), dtype=bool) if stream_cleanup else None
        stream_min_conf = float(streaming_cleanup_min_conf)
        stream_min_radius = float(streaming_cleanup_min_radius)
        stream_min_conf_u8 = int(min_conf_to_u8_threshold(stream_min_conf)) if stream_min_conf > 0.0 else 0

        source_padding_count = max(0, int(getattr(source, 'radial_padding_count', 0) or 0))
        effective_slice_locks = slice_locks
        if effective_slice_locks is None and source_padding_count > 0:
            effective_slice_locks = [
                threading.Lock() for _ in range(max(1, int(view_union_mm.shape[0])))
            ]

        def _process_prediction_unit(
            spec: BatchResultFrameSpec,
            masks_obj: Optional[object],
            confs_arr: Optional[np.ndarray],
        ) -> Tuple[int, int]:
            target_union, target_conf, target_index, target_affine, count_stats = (
                _prediction_accumulation_target(
                    spec,
                    view_union_mm=view_union_mm,
                    view_confmap_mm=view_confmap_mm,
                    radial_padding_union_mm=radial_padding_union_mm,
                    radial_padding_confmap_mm=radial_padding_confmap_mm,
                    M_out_to_native=M_out_to_native,
                    native_w=int(native_w),
                )
            )
            slice_lock = None
            if effective_slice_locks is not None and len(effective_slice_locks) > 0:
                lock_index = (
                    int(spec.global_destination_index)
                    if bool(spec.is_radial_padding) else int(spec.task_index)
                )
                slice_lock = effective_slice_locks[lock_index % len(effective_slice_locks)]
            if isinstance(masks_obj, GpuFlattenedRetinaPayload):
                masks_obj.gpu_min_radius = float(stream_min_radius)
                masks_obj.run_gpu_cleanup = bool(
                    angle_variant_gpu_fastpath() is not None
                    and gpu_retina_cleanup_enabled()
                    and float(stream_min_radius) > 0.0
                )
            with (slice_lock if slice_lock is not None else contextlib.nullcontext()):
                pred_inc, frame_inc = _process_prediction_frame(
                    idx=int(target_index),
                    masks_np=masks_obj,
                    confs_np=confs_arr,
                    out_size=out_size,
                    view_union_mm=target_union,
                    view_confmap_mm=target_conf,
                    M_out_to_native=target_affine,
                    native_h=native_h,
                    native_w=native_w,
                    slice_lock=None,
                )
                # Keep union and streamed cleanup atomic for repeated wrapped destinations.
                if stream_cleanup and not bool(getattr(masks_obj, 'cleanup_done_on_gpu', False)):
                    cleaned_has_foreground = _cleanup_prediction_slice_inplace(
                        target_union,
                        target_conf,
                        int(target_index),
                        min_conf=stream_min_conf,
                        min_radius=stream_min_radius,
                        backend=stream_backend,
                        structure2=stream_structure2,
                        min_conf_u8=stream_min_conf_u8,
                    )
                    frame_inc = 1 if bool(cleaned_has_foreground) else 0
            if not count_stats:
                return 0, 0
            return int(pred_inc), int(frame_inc)

        def _extract_and_process_result(spec: BatchResultFrameSpec, result_obj: object) -> Tuple[int, int]:
            masks_np, confs_np = _extract_result_masks_and_confs(result_obj)
            try:
                del result_obj
            except Exception:
                pass
            return _process_prediction_unit(spec, masks_np, confs_np)

        futures: List[Future] = []
        submitted_frames = 0
        synthetic_discarded = 0
        radial_padding_processed = 0
        precompleted_prediction_count = 0
        precompleted_frames_with_predictions = 0
        pending_limit = async_predict_pending_frame_limit(int(num_frames))

        # eagerly reduce GPU stacks to 2 small planes on this thread and bound the queue
        # so GPU-resident flattened frames stay capped (see predict_source_and_accumulate).
        gpu_flatten_eager = bool(
            gpu_retina_flatten_enabled() and not cpu_retina_masks_enabled() and gpu_retina_eager_flatten_enabled()
        )
        if gpu_flatten_eager:
            gpu_cap = gpu_retina_flatten_pending_limit(max(1, int(getattr(postprocess_executor, '_max_workers', 1) or 1)))
            pending_limit = max(1, min(int(pending_limit) if int(pending_limit) > 0 else gpu_cap, gpu_cap))

        def _join_one_pending() -> None:
            nonlocal futures, precompleted_prediction_count, precompleted_frames_with_predictions
            if not futures:
                return
            done, remaining = wait(set(futures), return_when=FIRST_COMPLETED)
            futures = list(remaining)
            for fut_done in done:
                pred_inc, frame_inc = fut_done.result()
                precompleted_prediction_count += int(pred_inc)
                precompleted_frames_with_predictions += int(frame_inc)

        for idx, r in enumerate(results):
            spec = prediction_result_frame_spec(source, int(idx), num_frames=int(num_frames))
            if spec is None:
                synthetic_discarded += 1
                continue
            if spec.is_radial_padding:
                radial_padding_processed += 1
            else:
                submitted_frames += 1
            if gpu_flatten_eager:
                masks_np, confs_np = _extract_result_masks_and_confs(r)
                try:
                    del r
                except Exception:
                    pass
                futures.append(postprocess_executor.submit(_process_prediction_unit, spec, masks_np, confs_np))
            else:
                futures.append(postprocess_executor.submit(_extract_and_process_result, spec, r))
            while int(pending_limit) > 0 and len(futures) >= int(pending_limit):
                _join_one_pending()

        return PredictionAccumulationHandle(
            source_label=str(source_label),
            futures=futures,
            view_union_mm=view_union_mm,
            view_confmap_mm=view_confmap_mm,
            submitted_frames=int(submitted_frames),
            synthetic_discarded=int(synthetic_discarded),
            precompleted_prediction_count=int(precompleted_prediction_count),
            precompleted_frames_with_predictions=int(precompleted_frames_with_predictions),
            pending_limit=int(pending_limit),
            radial_padding_processed=int(radial_padding_processed),
            radial_padding_union_mm=radial_padding_union_mm,
            radial_padding_confmap_mm=radial_padding_confmap_mm,
        )
    finally:
        if owned_staging_wrapper is not None:
            try:
                owned_staging_wrapper.close()
            except Exception:
                pass

def predict_in_memory_volume_and_submit_accumulation(
    model,
    prediction_volume: PredictionVolumeRef,
    *,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_executor: ThreadPoolExecutor,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
    radial_padding_union_mm: Optional[np.ndarray] = None,
    radial_padding_confmap_mm: Optional[np.ndarray] = None,
) -> PredictionAccumulationHandle:
    source = make_prediction_ref_yolo_source(
        prediction_volume,
        batch_size=max(1, int(cfg.batch)),
        max_frames=int(num_frames),
    )
    return predict_source_and_submit_accumulation(
        model,
        source,
        source_label=prediction_volume.name,
        num_frames=int(num_frames),
        out_size=int(out_size),
        cfg=cfg,
        view_union_mm=view_union_mm,
        view_confmap_mm=view_confmap_mm,
        M_out_to_native=M_out_to_native,
        native_h=int(native_h),
        native_w=int(native_w),
        postprocess_executor=postprocess_executor,
        streaming_cleanup_enabled=bool(streaming_cleanup_enabled),
        streaming_cleanup_min_conf=float(streaming_cleanup_min_conf),
        streaming_cleanup_min_radius=float(streaming_cleanup_min_radius),
        slice_locks=slice_locks,
        radial_padding_union_mm=radial_padding_union_mm,
        radial_padding_confmap_mm=radial_padding_confmap_mm,
    )

def predict_in_memory_volume_and_accumulate(
    model,
    prediction_volume: PredictionVolumeRef,
    *,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_workers: int = 1,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
    radial_padding_union_mm: Optional[np.ndarray] = None,
    radial_padding_confmap_mm: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    source = make_prediction_ref_yolo_source(
        prediction_volume,
        batch_size=max(1, int(cfg.batch)),
        max_frames=int(num_frames),
    )
    return predict_source_and_accumulate(
        model,
        source,
        source_label=prediction_volume.name,
        num_frames=int(num_frames),
        out_size=int(out_size),
        cfg=cfg,
        view_union_mm=view_union_mm,
        view_confmap_mm=view_confmap_mm,
        M_out_to_native=M_out_to_native,
        native_h=int(native_h),
        native_w=int(native_w),
        postprocess_workers=int(postprocess_workers),
        streaming_cleanup_enabled=bool(streaming_cleanup_enabled),
        streaming_cleanup_min_conf=float(streaming_cleanup_min_conf),
        streaming_cleanup_min_radius=float(streaming_cleanup_min_radius),
        slice_locks=slice_locks,
        radial_padding_union_mm=radial_padding_union_mm,
        radial_padding_confmap_mm=radial_padding_confmap_mm,
    )

def cleanup_backend() -> str:
    """Return the per-slice cleanup backend.

 OpenCV is the default because the hot operations used here release the GIL and scale better
 under Python thread pools. Set YOLO_TTA_CLEANUP_BACKEND=scipy to recover the older scipy.ndimage
 cleanup path for debugging or strict regression comparison."""
    backend = os.environ.get('YOLO_TTA_CLEANUP_BACKEND', 'opencv').strip().lower()
    if backend not in {'opencv', 'scipy'}:
        backend = 'opencv'
    return backend

def _cv2_connected_components(mask_u8: np.ndarray, connectivity: int = 8) -> Tuple[int, np.ndarray]:
    return cv2.connectedComponents(
        np.ascontiguousarray(mask_u8, dtype=np.uint8),
        connectivity=int(connectivity),
        ltype=cv2.CV_32S,
    )

def _fill_holes_2d_scipy(mask_bool: np.ndarray) -> np.ndarray:
    return np.asarray(ndi.binary_fill_holes(np.asarray(mask_bool, dtype=bool)), dtype=bool)

def _fill_holes_2d_opencv(mask_bool: np.ndarray) -> np.ndarray:
    """Fill 2D holes using background connected components.

 This matches scipy.ndimage.binary_fill_holes' default 2D background connectivity (4-connected)
 while avoiding the slower Python-visible scipy path for thousands of large slices."""
    mask_u8 = np.ascontiguousarray(np.asarray(mask_bool, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)
    if bool(np.all(mask_u8)):
        return np.ones(mask_u8.shape, dtype=bool)

    bg_u8 = (mask_u8 == 0).astype(np.uint8, copy=False)
    num_labels, labels2d = _cv2_connected_components(bg_u8, connectivity=4)
    if int(num_labels) <= 1:
        return mask_u8.astype(bool, copy=False)

    touches_boundary = np.zeros((int(num_labels),), dtype=bool)
    touches_boundary[np.unique(labels2d[0, :])] = True
    touches_boundary[np.unique(labels2d[-1, :])] = True
    touches_boundary[np.unique(labels2d[:, 0])] = True
    touches_boundary[np.unique(labels2d[:, -1])] = True

    enclosed_bg = (labels2d > 0) & (~touches_boundary[labels2d])
    if np.any(enclosed_bg):
        mask_u8 = mask_u8.copy()
        mask_u8[enclosed_bg] = np.uint8(1)
    return mask_u8.astype(bool, copy=False)

def _fill_holes_2d_opencv_u8_inplace(
    arr_u8: np.ndarray,
    known_bbox: Optional[Sequence[int]] = None,
) -> None:
    """Fill enclosed 4-connected background inside the foreground bbox plus halo.

    A one-pixel exterior background halo gives the cropped connected-component problem
    exactly the same outside/background semantics as the full plane.  When foreground
    touches a global image edge the crop also touches that edge, so boundary-connected
    background remains outside.  Empty and full-foreground slices return without labels.
    """
    arr = np.asarray(arr_u8, dtype=np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        return
    full_h, full_w = (int(arr.shape[0]), int(arr.shape[1]))
    if known_bbox is not None:
        try:
            y0_known, y1_known, x0_known, x1_known = (int(v) for v in known_bbox)
            y0_known = max(0, min(full_h, y0_known))
            y1_known = max(y0_known, min(full_h, y1_known))
            x0_known = max(0, min(full_w, x0_known))
            x1_known = max(x0_known, min(full_w, x1_known))
            x0, y0 = int(x0_known), int(y0_known)
            width, height = int(x1_known - x0_known), int(y1_known - y0_known)
        except Exception:
            x0, y0, width, height = (int(v) for v in cv2.boundingRect(arr))
    else:
        x0, y0, width, height = (int(v) for v in cv2.boundingRect(arr))
    if width <= 0 or height <= 0:
        return
    y0h = max(0, int(y0) - 1)
    x0h = max(0, int(x0) - 1)
    y1h = min(full_h, int(y0) + int(height) + 1)
    x1h = min(full_w, int(x0) + int(width) + 1)
    sub = arr[y0h:y1h, x0h:x1h]
    bg = np.ascontiguousarray(sub == 0, dtype=np.uint8)
    if not bool(np.any(bg)):
        return
    num_labels, labels2d = _cv2_connected_components(bg, connectivity=4)
    if int(num_labels) <= 1:
        return
    touches_boundary = np.zeros((int(num_labels),), dtype=bool)
    touches_boundary[np.unique(labels2d[0, :])] = True
    touches_boundary[np.unique(labels2d[-1, :])] = True
    touches_boundary[np.unique(labels2d[:, 0])] = True
    touches_boundary[np.unique(labels2d[:, -1])] = True
    enclosed_bg = (labels2d > 0) & (~touches_boundary[labels2d])
    if bool(np.any(enclosed_bg)):
        sub[enclosed_bg] = np.uint8(1)

def _filter_connected_components_by_min_radius_scipy(
    mask_bool: np.ndarray,
    structure2: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    labels2d, num = ndi.label(mask_bool, structure=structure2)
    if int(num) <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)

    label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
    dist = np.asarray(ndi.distance_transform_edt(mask_bool), dtype=np.float32)
    radii = np.asarray(ndi.maximum(dist, labels=labels2d, index=label_ids), dtype=np.float32)
    keep_ids = label_ids[radii >= float(min_radius)]
    if keep_ids.size <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    return np.isin(labels2d, keep_ids)

def _filter_connected_components_by_min_radius_opencv(
    mask_bool: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    mask_u8 = np.ascontiguousarray(np.asarray(mask_bool, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)

    num_labels, labels2d = _cv2_connected_components(mask_u8, connectivity=8)
    if int(num_labels) <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radii = np.zeros((int(num_labels),), dtype=np.float32)
    np.maximum.at(radii, labels2d.ravel(), np.asarray(dist, dtype=np.float32).ravel())
    keep_lookup = radii >= float(min_radius)
    keep_lookup[0] = False
    return keep_lookup[labels2d]

def _filter_connected_components_by_min_conf_opencv(
    mask_bool: np.ndarray,
    conf_slice: np.ndarray,
    min_conf_u8: int,
) -> np.ndarray:
    mask_u8 = np.ascontiguousarray(np.asarray(mask_bool, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)

    num_labels, labels2d = _cv2_connected_components(mask_u8, connectivity=8)
    if int(num_labels) <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)

    maxima = np.zeros((int(num_labels),), dtype=np.uint8)
    np.maximum.at(maxima, labels2d.ravel(), np.asarray(conf_slice, dtype=np.uint8).ravel())
    keep_lookup = maxima >= int(min_conf_u8)
    keep_lookup[0] = False
    return keep_lookup[labels2d]

def _filter_connected_components_by_min_conf_scipy(
    mask_bool: np.ndarray,
    conf_slice: np.ndarray,
    min_conf_u8: int,
    structure2: np.ndarray,
) -> np.ndarray:
    labels2d, num = ndi.label(mask_bool, structure=structure2)
    if int(num) <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
    maxima = np.asarray(ndi.maximum(conf_slice, labels=labels2d, index=label_ids), dtype=np.uint8)
    keep_ids = label_ids[maxima >= int(min_conf_u8)]
    if keep_ids.size <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    return np.isin(labels2d, keep_ids)

def _cleanup_prediction_slice_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    idx: int,
    *,
    min_conf: float = 0.0,
    min_radius: float = 0.0,
    backend: Optional[str] = None,
    structure2: Optional[np.ndarray] = None,
    min_conf_u8: Optional[int] = None,
) -> bool:
    """Apply slice-local cleanup in specification order: confidence gating, radius filtering, then hole filling."""
    idx_i = int(idx)
    if confmap_mm is None and float(min_conf) <= 0.0 and float(min_radius) <= 0.0:
        # nothing to filter and no confidence map to zero — skip the full
        # slice bool-cast + unconditional write-back (a pure read+rewrite no-op at
        # min_conf 0 --min_radius 0) and report foreground from a read-only scan.
        return bool(np.any(np.asarray(mask_mm[idx_i])))
    backend_norm = cleanup_backend() if backend is None else str(backend)
    structure = np.ones((3, 3), dtype=bool) if structure2 is None else structure2
    min_conf_u8_i = (
        int(min_conf_to_u8_threshold(float(min_conf)))
        if min_conf_u8 is None and float(min_conf) > 0.0
        else int(min_conf_u8 or 0)
    )

    mask_slice = np.asarray(mask_mm[idx_i], dtype=bool)
    conf_slice = None if confmap_mm is None else np.asarray(confmap_mm[idx_i], dtype=np.uint8)

    if np.any(mask_slice) and conf_slice is not None and float(min_conf) > 0.0:
        if backend_norm == 'opencv':
            mask_slice = _filter_connected_components_by_min_conf_opencv(
                mask_slice,
                conf_slice,
                int(min_conf_u8_i),
            )
        else:
            mask_slice = _filter_connected_components_by_min_conf_scipy(
                mask_slice,
                conf_slice,
                int(min_conf_u8_i),
                structure,
            )

    # apply --min_radius BEFORE hole filling. Spec items 5-6 require the order
    # min_conf -> --min_radius -> hole fill, with hole filling applied to the FINAL per-view
    # volume. Hole filling is therefore performed as a separate volume-level 2D pass in
    # cleanup_view_volume_after_prediction_inplace after the (: native per-view) radius
    # filter completes, rather than here in the per-slice unit (which previously hole-filled before radius).
    if np.any(mask_slice) and float(min_radius) > 0.0:
        if backend_norm == 'opencv':
            mask_slice = _filter_connected_components_by_min_radius_opencv(
                mask_slice,
                float(min_radius),
            )
        else:
            mask_slice = _filter_connected_components_by_min_radius_scipy(
                mask_slice,
                structure,
                float(min_radius),
            )

    has_foreground = bool(np.any(mask_slice))
    mask_mm[idx_i, :, :] = mask_slice.astype(np.uint8, copy=False)
    if conf_slice is not None:
        if has_foreground:
            conf_slice[~mask_slice] = np.uint8(0)
        else:
            conf_slice.fill(np.uint8(0))
        confmap_mm[idx_i, :, :] = conf_slice.astype(np.uint8, copy=False)
    return bool(has_foreground)

_INFERENCE_BATCH_SIZE = 1

_GPU_FLATTEN_TRACK_CONF = True

def set_gpu_flatten_conf_tracking(enabled: bool) -> None:
    global _GPU_FLATTEN_TRACK_CONF
    _GPU_FLATTEN_TRACK_CONF = bool(enabled)

def gpu_flatten_conf_tracking_enabled() -> bool:
    return bool(_GPU_FLATTEN_TRACK_CONF)

def set_inference_batch_size(batch_size: int) -> None:
    global _INFERENCE_BATCH_SIZE
    _INFERENCE_BATCH_SIZE = max(1, int(batch_size))

def inference_batch_size() -> int:
    return max(1, int(_INFERENCE_BATCH_SIZE))

def fused_slice_cleanup_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray] = None,
    *,
    min_conf: float = 0.0,
    min_radius: float = 0.0,
    workers: int = 1,
    desc: str = 'Fused slice cleanup',
) -> None:
    """Slice-parallel cleanup with chunked worker fan-out.

 The old path delegated one future per slice and, when ``--min_radius`` was active, performed a
 Python loop plus one EDT per connected component. On large sparse slices that makes the stage
 look effectively single-threaded even though it nominally uses a thread pool. The updated path
 keeps the same semantics but:
 - submits chunked slice ranges so worker threads stay busy with lower dispatch overhead
 - computes connected-component radii with one EDT + one reduce per slice instead of one EDT
 per component"""
    num_slices = int(mask_mm.shape[0])
    structure2 = np.ones((3, 3), dtype=bool)
    min_conf_u8 = int(min_conf_to_u8_threshold(float(min_conf))) if float(min_conf) > 0.0 else 0
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)
    chunk_size = choose_parallel_chunk_size(num_slices, worker_count, target_chunks_per_worker=2, min_chunk_size=1)
    backend = cleanup_backend()

    def _process(i: int) -> None:
        _cleanup_prediction_slice_inplace(
            mask_mm,
            confmap_mm,
            int(i),
            min_conf=float(min_conf),
            min_radius=float(min_radius),
            backend=backend,
            structure2=structure2,
            min_conf_u8=int(min_conf_u8),
        )

    parallel_for_indices_chunked(
        num_slices,
        _process,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
    )
    flush_array(mask_mm)
    if confmap_mm is not None:
        flush_array(confmap_mm)

def fill_view_volume_holes_2d_inplace(
    mask_mm: np.ndarray,
    *,
    workers: int = 1,
    desc: str = '2D hole fill (per-view volume)',
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
) -> None:
    """Fill enclosed 2D background components in completed view slices.

    When an inference worker supplied valid per-slice foreground metadata, only
    known-nonempty slices are submitted and their half-open ``(y0,y1,x0,x1)`` bbox is
    passed directly to the topology primitive. Missing or malformed metadata falls back
    to the authoritative full-slice scan.
    """
    num_slices = int(mask_mm.shape[0])
    if num_slices <= 0:
        return
    backend = cleanup_backend()

    metadata_ok = False
    any_arr: Optional[np.ndarray] = None
    bbox_arr: Optional[np.ndarray] = None
    if known_slice_any is not None:
        try:
            any_arr = np.asarray(known_slice_any, dtype=bool).reshape(-1)
            if int(any_arr.shape[0]) != int(num_slices):
                raise ValueError('slice-any length mismatch')
            if known_slice_bboxes is not None:
                bbox_arr = np.asarray(known_slice_bboxes, dtype=np.int64)
                if tuple(int(v) for v in bbox_arr.shape) != (int(num_slices), 4):
                    raise ValueError('slice-bbox shape mismatch')
            metadata_ok = True
        except Exception:
            metadata_ok = False
            any_arr = None
            bbox_arr = None

    active_indices = (
        np.flatnonzero(any_arr).astype(np.int64, copy=False)
        if metadata_ok and any_arr is not None
        else np.arange(int(num_slices), dtype=np.int64)
    )
    active_count = int(active_indices.size)
    if metadata_ok:
        print(
            f'{desc}: inference-worker foreground metadata scheduled {active_count}/{num_slices} '
            'slice(s); empty full-plane scans skipped.'
        )
    if active_count <= 0:
        return

    worker_count = choose_slice_parallel_workers(int(workers), active_count)
    chunk_size = choose_parallel_chunk_size(active_count, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _process(position: int) -> None:
        idx_i = int(active_indices[int(position)])
        arr = np.asarray(mask_mm[idx_i])
        known_bbox: Optional[np.ndarray] = None
        if bbox_arr is not None:
            candidate = np.asarray(bbox_arr[idx_i], dtype=np.int64).reshape(4)
            y0c, y1c, x0c, x1c = (int(v) for v in candidate)
            if (
                0 <= y0c < y1c <= int(arr.shape[0])
                and 0 <= x0c < x1c <= int(arr.shape[1])
            ):
                known_bbox = candidate
        # Without trusted metadata, keep the authoritative emptiness test. With metadata,
        # the committed GPU union has already proved this slice nonempty. An invalid bbox
        # falls back to the backend's normal full-slice bbox discovery for this slice only.
        if not metadata_ok and not bool(arr.any()):
            return
        if backend == 'opencv':
            _fill_holes_2d_opencv_u8_inplace(arr, known_bbox=known_bbox)
        else:
            if known_bbox is None:
                filled = _fill_holes_2d_scipy(arr > 0)
                holes = filled & (arr == 0)
                if bool(np.any(holes)):
                    arr[holes] = np.uint8(1)
                return
            y0, y1, x0, x1 = (int(v) for v in known_bbox)
            y0 = max(0, y0 - 1); x0 = max(0, x0 - 1)
            y1 = min(int(arr.shape[0]), y1 + 1); x1 = min(int(arr.shape[1]), x1 + 1)
            if y1 <= y0 or x1 <= x0:
                return
            sub = arr[y0:y1, x0:x1]
            filled = _fill_holes_2d_scipy(sub > 0)
            holes = filled & (sub == 0)
            if bool(np.any(holes)):
                sub[holes] = np.uint8(1)

    parallel_for_indices_chunked(
        active_count,
        _process,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
    )
    flush_array(mask_mm)

def cleanup_view_volume_after_prediction_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    view: ViewInfo,
    min_conf: float,
    min_radius: float,
    *,
    workers: int = 1,
    precleaned_slice_cleanup: bool = False,
    skip_hole_fill: bool = False,
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    threshold_plane_shape: Optional[Tuple[int, int]] = None,
) -> None:
    # non-radial masks can still be on the canonical inference grid here.
    # Convert the native-view radius once and perform every component operation on that smaller
    # raster. Radial already owns a deliberately folded native raster and keeps its historical
    # threshold unchanged.
    threshold_shape = (
        tuple(int(v) for v in threshold_plane_shape)
        if threshold_plane_shape is not None
        else tuple(int(v) for v in np.asarray(mask_mm).shape[-2:])
    )
    native_min_radius = view_processing_min_radius(
        view, float(min_radius), threshold_shape,
    )

    if bool(precleaned_slice_cleanup):
        # angle-variant inference has already applied per-slice --min_conf and the full
        # view-native --min_radius as results streamed in (streaming_cleanup_min_radius now equals
        # the full radius for every view), so nothing radius-related remains before the final
        # hole-fill pass.
        pass
    else:
        effective_confmap_mm = confmap_mm if float(min_conf) > 0.0 else None

        fused_slice_cleanup_inplace(
            mask_mm,
            effective_confmap_mm,
            min_conf=float(min_conf),
            min_radius=float(native_min_radius),
            workers=int(workers),
            desc=f'Fused cleanup ({view.name})',
        )

    # 2D hole filling is the FINAL per-view step, applied after
    # min_conf and the view-native --min_radius filter. The previous order hole-filled before
    # min_radius inside the per-slice unit.
    # skipped when every slice of this view was already hole-filled on device by
    # the GPU workers (identical per-slice semantics; the CPU pass would be pure recompute).
    if bool(skip_hole_fill):
        print(f'2D hole fill ({view.name}): done on device during accumulation (v13.3.3 S2); CPU pass skipped.')
    else:
        fill_view_volume_holes_2d_inplace(
            mask_mm,
            workers=int(workers),
            desc=f'2D hole fill ({view.name})',
            known_slice_any=known_slice_any,
            known_slice_bboxes=known_slice_bboxes,
        )

    flush_array(mask_mm)
    if confmap_mm is not None:
        flush_array(confmap_mm)
