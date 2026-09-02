"""Dependency-light SAM boundary for the v19 LTA prototype.

The module deliberately imports neither PyTorch nor ``sam3`` at import time.
Configuration/help, local-bundle preflight, and session planning therefore work
on hosts that do not have the model runtime installed.  The official builder is
loaded only by :func:`build_local_sam_predictor` after an explicit local
checkpoint has already been resolved.
"""

from __future__ import annotations

import functools
import gc
import gzip
import hashlib
import inspect
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Callable, Optional, Tuple


LTA_SESSION_FRAMES = 30
LTA_MAX_NUM_OBJECTS = 128
LTA_MULTIPLEX_COUNT = 16
LTA_CHANNEL_POLICY = "implicit_rgb_v1"

_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth"})
_PREFERRED_CHECKPOINT_NAMES = (
    "sam3.1_multiplex.pt",
    "sam3.pt",
)
_PINNED_SAM_BPE_SHA256 = (
    "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
)
_PINNED_SAM_BPE_PREFIX = b'"bpe_simple_vocab_16e6.txt#version: 0.2\n'
_SAM_REMOVED_OBJECT_SCORE = -1e4
_SAM31_BUILD_PATCH_LOCK = threading.RLock()


def cuda_capability_supports_fa3(major: int, minor: int = 0) -> bool:
    """Return the pinned v19 FA3 policy (Hopper+, not Ada/4090)."""

    if isinstance(major, bool) or isinstance(minor, bool):
        raise TypeError("CUDA capability fields must be integers")
    return int(major) >= 9


@dataclass(frozen=True)
class LocalSamBundle:
    """Resolved local-only SAM model assets.

    ``root`` is the user-selected file or directory after strict resolution.
    ``checkpoint_path`` is always one existing local file.  ``bpe_path`` is
    optional because the pinned SAM package carries its own local vocabulary.
    """

    root: Path
    checkpoint_path: Path
    model_version: str
    checkpoint_identity_sha256: str
    bpe_path: Optional[Path] = None

    def __post_init__(self) -> None:
        root = Path(self.root)
        checkpoint = Path(self.checkpoint_path)
        bpe = None if self.bpe_path is None else Path(self.bpe_path)
        model_version = str(self.model_version).strip().lower()
        identity = str(self.checkpoint_identity_sha256).strip().lower()
        if not root.is_absolute() or not root.exists():
            raise ValueError("SAM bundle root must be an existing absolute path")
        if not checkpoint.is_absolute() or not checkpoint.is_file():
            raise ValueError("SAM checkpoint must be an existing absolute file")
        if checkpoint.suffix.lower() not in _CHECKPOINT_SUFFIXES:
            raise ValueError("SAM checkpoint must use .pt or .pth")
        if bpe is not None and (not bpe.is_absolute() or not bpe.is_file()):
            raise ValueError("SAM BPE path must be an existing absolute file")
        if model_version not in {"sam3", "sam3.1"}:
            raise ValueError("SAM bundle model_version must be 'sam3' or 'sam3.1'")
        if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
            raise ValueError("SAM checkpoint identity must be a SHA-256 hex digest")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "checkpoint_path", checkpoint)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "checkpoint_identity_sha256", identity)
        object.__setattr__(self, "bpe_path", bpe)


@dataclass(frozen=True)
class SamSessionPlan:
    """One fixed, non-overlapping ordered SAM session."""

    sequence_id: str
    session_index: int
    frame_start: int
    frame_stop: int

    def __post_init__(self) -> None:
        sequence_id = str(self.sequence_id).strip()
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")
        if isinstance(self.session_index, bool) or int(self.session_index) < 0:
            raise ValueError("session_index must be >= 0")
        if isinstance(self.frame_start, bool) or int(self.frame_start) < 0:
            raise ValueError("frame_start must be >= 0")
        if isinstance(self.frame_stop, bool) or int(self.frame_stop) <= int(self.frame_start):
            raise ValueError("frame_stop must be greater than frame_start")
        if int(self.frame_stop) - int(self.frame_start) > LTA_SESSION_FRAMES:
            raise ValueError(
                f"one LTA session may contain at most {LTA_SESSION_FRAMES} frames"
            )
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "session_index", int(self.session_index))
        object.__setattr__(self, "frame_start", int(self.frame_start))
        object.__setattr__(self, "frame_stop", int(self.frame_stop))

    @property
    def frame_count(self) -> int:
        return int(self.frame_stop) - int(self.frame_start)

    @property
    def frame_indices(self) -> range:
        return range(int(self.frame_start), int(self.frame_stop))


@dataclass(frozen=True)
class SamPromptBox:
    """One normalized positive exemplar box in ``xywh`` form."""

    exemplar_id: str
    frame_index: int
    xywh: Tuple[float, float, float, float]
    positive: bool = True

    def __post_init__(self) -> None:
        exemplar_id = str(self.exemplar_id).strip()
        if not exemplar_id:
            raise ValueError("exemplar_id must not be empty")
        if isinstance(self.frame_index, bool) or int(self.frame_index) < 0:
            raise ValueError("frame_index must be >= 0")
        if len(tuple(self.xywh)) != 4:
            raise ValueError("xywh must contain four normalized values")
        x, y, width, height = (float(value) for value in self.xywh)
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError("xywh values must be finite")
        if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
            raise ValueError("xywh origin must be non-negative and extent must be positive")
        if x + width > 1.0 + 1e-12 or y + height > 1.0 + 1e-12:
            raise ValueError("xywh must remain inside the normalized image")
        if not bool(self.positive):
            raise ValueError("the v19 LTA prototype accepts positive exemplar boxes only")
        object.__setattr__(self, "exemplar_id", exemplar_id)
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "xywh", (x, y, width, height))
        object.__setattr__(self, "positive", True)


@dataclass(frozen=True)
class SamFramePrediction:
    """Runtime-neutral SAM result passed to LTA geometry/output code."""

    sequence_id: str
    session_index: int
    frame_index: int
    object_id: int
    initial_detection_score: float
    binary_mask: object
    frame_tracker_score: Optional[float] = None

    def __post_init__(self) -> None:
        sequence_id = str(self.sequence_id).strip()
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")
        for field_name, value in (
            ("session_index", self.session_index),
            ("frame_index", self.frame_index),
            ("object_id", self.object_id),
        ):
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        detection = resolve_confidence(self.initial_detection_score)
        tracker = (
            None
            if self.frame_tracker_score is None
            else resolve_confidence(self.frame_tracker_score)
        )
        if self.binary_mask is None:
            raise ValueError("binary_mask must not be None")
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "session_index", int(self.session_index))
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "object_id", int(self.object_id))
        object.__setattr__(self, "initial_detection_score", detection)
        object.__setattr__(self, "frame_tracker_score", tracker)


def resolve_confidence(value: object) -> float:
    """Resolve the shared ``--conf`` scalar accepted by LTA."""

    try:
        resolved = float(value)
    except Exception as exc:
        raise ValueError(f"--conf must be a finite value in [0,1]; got {value!r}") from exc
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"--conf must be a finite value in [0,1]; got {value!r}")
    return resolved


def resolve_local_sam_bundle(value: str | Path) -> LocalSamBundle:
    """Resolve an ordinary local file/directory without any remote fallback."""

    try:
        root = Path(value).expanduser().resolve(strict=True)
    except Exception as exc:
        raise ValueError(f"--model is not an existing local filesystem path: {value!r}") from exc

    if root.is_file():
        checkpoint = root
        bundle_root = root
        bpe = None
    elif root.is_dir():
        preferred = [root / name for name in _PREFERRED_CHECKPOINT_NAMES]
        matches = [path for path in preferred if path.is_file()]
        if not matches:
            matches = sorted(
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES
            )
        if len(matches) != 1:
            detail = ", ".join(path.name for path in matches) or "none"
            raise ValueError(
                "--model directory must contain exactly one SAM checkpoint "
                f"(.pt/.pth); found: {detail}"
            )
        checkpoint = matches[0].resolve(strict=True)
        bundle_root = root
        packaged_bpe = root / "bpe_simple_vocab_16e6.txt.gz"
        bpe = packaged_bpe.resolve(strict=True) if packaged_bpe.is_file() else None
    else:  # pragma: no cover - strict resolution leaves only file/dir on normal hosts
        raise ValueError("--model must resolve to a file or directory")

    checkpoint_name = checkpoint.name.lower()
    if "3.1" in checkpoint_name or "multiplex" in checkpoint_name:
        model_version = "sam3.1"
    elif checkpoint_name == "sam3.pt" or checkpoint_name.startswith("sam3_"):
        model_version = "sam3"
    else:
        raise ValueError(
            "--model checkpoint name does not identify SAM 3 or SAM 3.1; use "
            "sam3.pt or sam3.1_multiplex.pt in the pinned local bundle"
        )
    return LocalSamBundle(
        root=bundle_root,
        checkpoint_path=checkpoint,
        model_version=model_version,
        checkpoint_identity_sha256=_local_file_identity_sha256(checkpoint),
        bpe_path=bpe,
    )


def _local_file_identity_sha256(path: Path) -> str:
    stat = path.stat()
    payload = "\0".join(
        (
            str(path.resolve()),
            str(int(stat.st_size)),
            str(int(stat.st_mtime_ns)),
            str(int(stat.st_ctime_ns)),
            str(int(stat.st_dev)),
            str(int(stat.st_ino)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def revalidate_local_sam_bundle(bundle: LocalSamBundle) -> None:
    """Fail if the selected local checkpoint changed after preflight."""

    if not isinstance(bundle, LocalSamBundle):
        raise TypeError("bundle must be a LocalSamBundle")
    try:
        current = _local_file_identity_sha256(bundle.checkpoint_path.resolve(strict=True))
    except Exception as exc:
        raise RuntimeError(
            f"local SAM checkpoint disappeared during the run: {bundle.checkpoint_path}"
        ) from exc
    if current != bundle.checkpoint_identity_sha256:
        raise RuntimeError(
            f"local SAM checkpoint changed during the run: {bundle.checkpoint_path}"
        )


def resolve_installed_sam_bpe(*, package_root: Optional[Path] = None) -> Path:
    """Resolve the pinned SAM package's local BPE asset without network access."""

    if package_root is None:
        try:
            import sam3  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency gated
            raise RuntimeError("the pinned local sam3 package is not importable") from exc
        package_root = Path(sam3.__file__).resolve().parent
    candidate = Path(package_root) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    try:
        resolved = candidate.resolve(strict=True)
    except Exception as exc:
        raise RuntimeError(f"local SAM BPE asset is missing: {candidate}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"local SAM BPE asset is not a file: {resolved}")
    return _validate_pinned_sam_bpe(resolved)


def _validate_pinned_sam_bpe(path: Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != _PINNED_SAM_BPE_SHA256:
        raise RuntimeError(
            "local SAM BPE asset does not match the pinned vocabulary: "
            f"{resolved}"
        )
    try:
        with gzip.open(resolved, "rb") as handle:
            prefix = handle.read(len(_PINNED_SAM_BPE_PREFIX))
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"local SAM BPE asset is not readable gzip: {resolved}") from exc
    if prefix != _PINNED_SAM_BPE_PREFIX:
        raise RuntimeError(f"local SAM BPE vocabulary header is invalid: {resolved}")
    return resolved


def patch_sam_init_state_signature(predictor: object) -> Tuple[str, ...]:
    """Filter wrapper-only kwargs before the pinned model ``init_state`` call.

    Current SAM 3.1's base predictor forwards ``offload_state_to_cpu`` even
    though the multiplex model does not accept it.  Wrapping only ``init_state``
    retains upstream session ownership while making the call signature exact.
    """

    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    if not callable(init_state):
        raise TypeError("SAM predictor model must expose init_state")
    signature = inspect.signature(init_state)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return ()
    allowed = frozenset(signature.parameters)
    incompatible = "offload_state_to_cpu"
    if incompatible in allowed:
        return ()

    @functools.wraps(init_state)
    def filtered_init_state(*args: object, **kwargs: object) -> object:
        filtered = dict(kwargs)
        filtered.pop(incompatible, None)
        return init_state(*args, **filtered)

    setattr(model, "init_state", filtered_init_state)
    return (incompatible,)


def _same_local_path(value: object, expected: Path) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    try:
        return Path(value).resolve(strict=True) == Path(expected).resolve(strict=True)
    except Exception:
        return False


def _approved_egpu_fp32_name(name: str) -> bool:
    parts = str(name).split(".")
    return (
        len(parts) == 7
        and parts[:4] == ["detector", "transformer", "decoder", "layers"]
        and parts[4].isdigit()
        and parts[5] in {"linear1", "linear2"}
        and parts[6] in {"weight", "bias"}
    )


def _audit_sam31_predictor_storage(
    predictor: object,
    *,
    runtime_torch: object,
    weight_storage: str,
) -> None:
    model = getattr(predictor, "model", None)
    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        raise RuntimeError("the assembled SAM 3.1 predictor exposes no tensor inventory")
    tensors = tuple(named_parameters()) + tuple(named_buffers())
    if not tensors:
        raise RuntimeError("the assembled SAM 3.1 predictor tensor inventory is empty")
    current_device = int(runtime_torch.cuda.current_device())
    placement_errors = []
    dtype_errors = []
    for name, tensor in tensors:
        device = getattr(tensor, "device", None)
        device_type = str(getattr(device, "type", ""))
        get_device = getattr(tensor, "get_device", None)
        tensor_device = int(get_device()) if callable(get_device) else -1
        if bool(getattr(tensor, "is_meta", False)) or (
            device_type != "cuda" or tensor_device != current_device
        ):
            placement_errors.append(f"{name}:{device}")
            continue
        is_floating_point = getattr(tensor, "is_floating_point", None)
        if not callable(is_floating_point) or not bool(is_floating_point()):
            continue
        dtype = getattr(tensor, "dtype", None)
        if weight_storage == "float32":
            if dtype != runtime_torch.float32:
                dtype_errors.append(f"{name}:{dtype}")
        elif _approved_egpu_fp32_name(name):
            if dtype != runtime_torch.float32:
                dtype_errors.append(f"{name}:{dtype} (expected float32)")
        elif dtype != runtime_torch.bfloat16:
            dtype_errors.append(f"{name}:{dtype} (expected bfloat16)")
    if placement_errors or dtype_errors:
        raise RuntimeError(
            "assembled SAM 3.1 tensor audit failed: "
            f"placement={placement_errors[:12]}, dtype={dtype_errors[:12]}"
        )
def _build_real_sam31_predictor_single_load_unlocked(
    *,
    builder: Callable[..., object],
    kwargs: Mapping[str, object],
    bundle: LocalSamBundle,
    runtime_torch: object,
    weight_storage: str,
    construction_device: str,
) -> object:
    """Build the merged predictor without the upstream throwaway checkpoint load.

    The official SAM 3.1 builder first loads the merged 3.5 GiB predictor state
    into a tracker-only model, where every ``tracker.*``/``detector.*`` key is
    incompatible, then loads the same file again into the assembled predictor.
    That redundant materialization can terminate a 16 GiB Windows host.  This
    adapter leaves model construction upstream-owned, skips only that known
    incompatible first load, builds the parameter shell on CPU, assigns the
    memory-mapped assembled state without a second copy, audits its exact
    state-dict result, and chunks the final host-to-CUDA tensor copies so a
    Windows eGPU does not need one large BAR1 transfer mapping.
    """

    try:
        import sam3.model_builder as sam_model_builder  # type: ignore
        from sam3.model.sam3_multiplex_tracking import (  # type: ignore
            Sam3MultiplexTrackingWithInteractivity,
        )
    except Exception as exc:  # pragma: no cover - real dependency gated
        raise RuntimeError("the pinned SAM 3.1 builder internals are unavailable") from exc

    official_builder = getattr(sam_model_builder, "build_sam3_predictor", None)
    tracker_builder = getattr(
        sam_model_builder, "build_sam3_multiplex_video_model", None
    )
    torch_load = getattr(runtime_torch, "load", None)
    if builder is not official_builder or not callable(tracker_builder) or not callable(torch_load):
        raise RuntimeError("the pinned SAM 3.1 builder identity has changed")

    checkpoint = bundle.checkpoint_path.resolve(strict=True)
    original_state_loader = Sam3MultiplexTrackingWithInteractivity.load_state_dict
    original_cuda = Sam3MultiplexTrackingWithInteractivity.cuda
    torch_linspace = getattr(runtime_torch, "linspace", None)
    if not callable(torch_linspace):
        raise RuntimeError("the pinned PyTorch runtime exposes no linspace constructor")
    checkpoint_load_count = 0
    meta_drop_path_linspace_count = 0
    state_audits: list[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []

    def tracker_without_merged_checkpoint(*args: object, **inner: object) -> object:
        requested = inner.get("checkpoint_path")
        if not _same_local_path(requested, checkpoint):
            raise RuntimeError(
                "SAM 3.1 tracker construction did not receive the selected local checkpoint"
            )
        inner["checkpoint_path"] = None
        inner["load_from_HF"] = False
        inner["device"] = construction_device
        return tracker_builder(*args, **inner)

    def cpu_scalar_linspace(*args: object, **inner: object) -> object:
        nonlocal meta_drop_path_linspace_count
        # The pinned ViT constructor materializes its stochastic-depth schedule
        # with ``[x.item() for x in torch.linspace(0, 0.1, 32)]``.  A meta
        # tensor cannot service ``item()``, so permit exactly that scalar-only
        # constructor call on CPU.  Do not turn this into a general escape hatch
        # for tensors that should remain meta until the checkpoint is assigned.
        if (
            len(args) != 3
            or inner
            or float(args[0]) != 0.0
            or not math.isclose(float(args[1]), 0.1, rel_tol=0.0, abs_tol=1e-12)
            or isinstance(args[2], bool)
            or int(args[2]) != 32
        ):
            raise RuntimeError(
                "unexpected torch.linspace call during pinned SAM 3.1 meta construction"
            )
        meta_drop_path_linspace_count += 1
        inner["device"] = "cpu"
        return torch_linspace(*args, **inner)

    def mmap_local_checkpoint(file: object, *args: object, **inner: object) -> object:
        nonlocal checkpoint_load_count
        if _same_local_path(file, checkpoint):
            checkpoint_load_count += 1
            inner["mmap"] = True
            inner["map_location"] = "cpu"
            inner["weights_only"] = True
        return torch_load(file, *args, **inner)

    def audited_state_load(self: object, *args: object, **inner: object) -> object:
        if construction_device == "meta":
            state_dict = args[0] if args else inner.get("state_dict")
            if not isinstance(state_dict, Mapping):
                raise RuntimeError(
                    "SAM meta construction received no mapping state dictionary"
                )
            named_buffers = getattr(self, "named_buffers", None)
            if not callable(named_buffers):
                raise RuntimeError("SAM meta construction exposes no buffer inventory")
            state_keys = frozenset(str(key) for key in state_dict)
            constructor_only_buffers = [
                (
                    str(name),
                    tuple(int(value) for value in tensor.shape),
                    str(getattr(tensor, "device", "")),
                )
                for name, tensor in named_buffers()
                if int(tensor.numel()) > 0 and str(name) not in state_keys
            ]
            expected_constructor_only = [
                (
                    "detector.backbone.language_backbone.encoder.attn_mask",
                    (32, 32),
                    "meta",
                )
            ]
            if constructor_only_buffers not in ([], expected_constructor_only):
                raise RuntimeError(
                    "SAM meta construction created non-checkpoint buffers outside the "
                    f"pinned causal mask: {constructor_only_buffers[:20]}"
                )
        inner["assign"] = True
        result = original_state_loader(self, *args, **inner)
        missing = tuple(str(value) for value in getattr(result, "missing_keys", ()))
        unexpected = tuple(str(value) for value in getattr(result, "unexpected_keys", ()))
        state_audits.append((missing, unexpected))
        if weight_storage == "bfloat16_egpu":
            cast = getattr(self, "bfloat16", None)
            if not callable(cast):
                raise RuntimeError("the assembled SAM 3.1 model cannot cast to bfloat16")
            cast()
            detector = getattr(self, "detector", None)
            transformer = getattr(detector, "transformer", None)
            decoder = getattr(transformer, "decoder", None)
            layers = tuple(getattr(decoder, "layers", ()))
            if not layers:
                raise RuntimeError("the pinned SAM decoder exposes no FFN layers")
            fp32_modules = 0
            for layer in layers:
                for name in ("linear1", "linear2"):
                    module = getattr(layer, name, None)
                    to_float = getattr(module, "float", None)
                    if not callable(to_float):
                        raise RuntimeError(
                            f"the pinned SAM decoder layer exposes no {name} FP32 boundary"
                        )
                    to_float()
                    fp32_modules += 1
            if fp32_modules != len(layers) * 2:
                raise RuntimeError("the pinned SAM decoder FFN FP32 audit failed")
        return result

    def chunked_cuda(self: object, device: object = None) -> object:
        cuda = getattr(runtime_torch, "cuda")
        torch_device = getattr(runtime_torch, "device")
        target = torch_device(
            "cuda",
            int(cuda.current_device()) if device is None else int(device),
        )
        chunk_bytes = 4 * 1024 * 1024
        if construction_device == "meta":
            named_buffers = getattr(self, "named_buffers", None)
            residual = []
            if callable(named_buffers):
                residual = [
                    (str(name), tuple(int(value) for value in tensor.shape), str(tensor.dtype))
                    for name, tensor in named_buffers()
                    if bool(getattr(tensor, "is_meta", False)) and int(tensor.numel()) > 0
                ]
            expected_causal = (
                "detector.backbone.language_backbone.encoder.attn_mask",
                (32, 32),
            )
            residual_identity = [(name, shape) for name, shape, _dtype in residual]
            if residual_identity == [expected_causal]:
                encoder = self.detector.backbone.language_backbone.encoder
                build_causal_mask = getattr(encoder, "build_causal_mask", None)
                if not callable(build_causal_mask):
                    raise RuntimeError("SAM text encoder exposes no causal-mask builder")
                with runtime_torch.device("cpu"):
                    causal_mask = build_causal_mask()
                if weight_storage == "bfloat16_egpu":
                    causal_mask = causal_mask.to(dtype=runtime_torch.bfloat16)
                encoder._buffers["attn_mask"] = causal_mask
                residual = []
            if residual:
                raise RuntimeError(
                    "SAM meta construction left initialized buffers absent from the checkpoint: "
                    f"{residual[:20]}"
                )

        def move_tensor(tensor: object) -> object:
            if bool(getattr(tensor, "is_cuda", False)):
                return tensor.to(device=target)
            if bool(getattr(tensor, "is_meta", False)) and int(tensor.numel()) > 0:
                raise RuntimeError(
                    "SAM meta construction left an initialized tensor absent from the checkpoint"
                )
            destination = runtime_torch.empty_like(tensor, device=target)
            numel = int(tensor.numel())
            if numel == 0:
                return destination
            if int(tensor.element_size()) * numel <= chunk_bytes:
                destination.copy_(tensor, non_blocking=False)
                return destination
            if not bool(tensor.is_contiguous()) or not bool(destination.is_contiguous()):
                destination.copy_(tensor, non_blocking=False)
                return destination
            chunk_elements = max(1, chunk_bytes // int(tensor.element_size()))
            source_flat = tensor.view(-1)
            destination_flat = destination.view(-1)
            for start in range(0, numel, chunk_elements):
                stop = min(numel, start + chunk_elements)
                destination_flat[start:stop].copy_(
                    source_flat[start:stop],
                    non_blocking=False,
                )
            return destination

        apply = getattr(self, "_apply", None)
        if not callable(apply):
            raise RuntimeError("the assembled SAM 3.1 model has no tensor apply boundary")
        return apply(move_tensor)

    try:
        setattr(
            sam_model_builder,
            "build_sam3_multiplex_video_model",
            tracker_without_merged_checkpoint,
        )
        setattr(runtime_torch, "load", mmap_local_checkpoint)
        if construction_device == "meta":
            setattr(runtime_torch, "linspace", cpu_scalar_linspace)
        setattr(
            Sam3MultiplexTrackingWithInteractivity,
            "load_state_dict",
            audited_state_load,
        )
        setattr(Sam3MultiplexTrackingWithInteractivity, "cuda", chunked_cuda)
        if construction_device == "meta":
            with runtime_torch.device("meta"):
                predictor = builder(**dict(kwargs))
        else:
            predictor = builder(**dict(kwargs))
    finally:
        setattr(sam_model_builder, "build_sam3_multiplex_video_model", tracker_builder)
        setattr(runtime_torch, "load", torch_load)
        setattr(runtime_torch, "linspace", torch_linspace)
        setattr(
            Sam3MultiplexTrackingWithInteractivity,
            "load_state_dict",
            original_state_loader,
        )
        setattr(Sam3MultiplexTrackingWithInteractivity, "cuda", original_cuda)

    try:
        if construction_device == "meta" and meta_drop_path_linspace_count != 2:
            raise RuntimeError(
                "the pinned SAM 3.1 meta constructor did not materialize exactly two "
                "CPU scalar drop-path schedules; observed "
                f"{meta_drop_path_linspace_count}"
            )
        if checkpoint_load_count != 1:
            raise RuntimeError(
                "the pinned SAM 3.1 builder did not perform exactly one local merged "
                f"checkpoint load; observed {checkpoint_load_count}"
            )
        if len(state_audits) != 1:
            raise RuntimeError(
                "the pinned SAM 3.1 builder did not expose exactly one assembled "
                f"state-dict audit; observed {len(state_audits)}"
            )
        missing, unexpected = state_audits[0]
        if missing or unexpected:
            raise RuntimeError(
                "local SAM 3.1 checkpoint does not exactly match the assembled predictor: "
                f"missing={len(missing)} {list(missing[:12])}, "
                f"unexpected={len(unexpected)} {list(unexpected[:12])}"
            )
        _audit_sam31_predictor_storage(
            predictor,
            runtime_torch=runtime_torch,
            weight_storage=weight_storage,
        )
    except BaseException:
        failed_predictor = predictor
        predictor = None
        shutdown = getattr(failed_predictor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        del failed_predictor
        gc.collect()
        cuda = getattr(runtime_torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        raise
    return predictor


def _build_real_sam31_predictor_single_load(
    *,
    builder: Callable[..., object],
    kwargs: Mapping[str, object],
    bundle: LocalSamBundle,
    runtime_torch: object,
    weight_storage: str,
    construction_device: str,
) -> object:
    """Serialize the process-global pinned SAM builder patch transaction."""

    with _SAM31_BUILD_PATCH_LOCK:
        try:
            return _build_real_sam31_predictor_single_load_unlocked(
                builder=builder,
                kwargs=kwargs,
                bundle=bundle,
                runtime_torch=runtime_torch,
                weight_storage=weight_storage,
                construction_device=construction_device,
            )
        except BaseException:
            gc.collect()
            cuda = getattr(runtime_torch, "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
            raise


def plan_sam_sessions(sequence_id: str, frame_count: int) -> Tuple[SamSessionPlan, ...]:
    """Partition one ordered runtime view into fixed 30-frame sessions."""

    if isinstance(frame_count, bool) or int(frame_count) < 1:
        raise ValueError("frame_count must be >= 1")
    total = int(frame_count)
    plans = []
    for session_index, frame_start in enumerate(range(0, total, LTA_SESSION_FRAMES)):
        plans.append(
            SamSessionPlan(
                sequence_id=str(sequence_id),
                session_index=int(session_index),
                frame_start=int(frame_start),
                frame_stop=min(total, int(frame_start) + LTA_SESSION_FRAMES),
            )
        )
    return tuple(plans)


def build_local_sam_predictor(
    bundle: LocalSamBundle,
    *,
    version: Optional[str] = None,
    builder: Optional[Callable[..., object]] = None,
    compile: bool = False,
    warm_up: bool = False,
    async_loading_frames: bool = True,
    conf: float = 0.15,
    device_id: Optional[int] = None,
    use_fa3: Optional[bool] = None,
    use_rope_real: Optional[bool] = None,
    weight_storage: str = "float32",
    max_num_objects: int = LTA_MAX_NUM_OBJECTS,
    construction_device: str = "cpu",
) -> object:
    """Construct the pinned SAM predictor from an explicit local checkpoint.

    The default official builders download only when ``checkpoint_path`` is
    omitted.  This function never omits it.  ``builder`` is injectable so the
    preflight contract can be tested without importing SAM or PyTorch.
    """

    if not isinstance(bundle, LocalSamBundle):
        raise TypeError("bundle must be a LocalSamBundle")
    resolved_version = str(version or bundle.model_version).strip().lower()
    if resolved_version not in {"sam3", "sam3.1"}:
        raise ValueError("SAM version must be 'sam3' or 'sam3.1'")
    if resolved_version != bundle.model_version:
        raise ValueError(
            f"SAM builder version {resolved_version!r} does not match local bundle "
            f"{bundle.model_version!r}"
        )
    resolved_weight_storage = str(weight_storage).strip().lower()
    if resolved_weight_storage not in {"float32", "bfloat16_egpu"}:
        raise ValueError(
            "SAM weight_storage must be 'float32' or 'bfloat16_egpu'"
        )
    if resolved_weight_storage != "float32" and resolved_version != "sam3.1":
        raise ValueError("bfloat16_egpu weight storage is supported only for SAM 3.1")
    if isinstance(max_num_objects, bool) or not 1 <= int(max_num_objects) <= LTA_MAX_NUM_OBJECTS:
        raise ValueError(
            f"SAM max_num_objects must be in [1,{LTA_MAX_NUM_OBJECTS}]"
        )
    resolved_construction_device = str(construction_device).strip().lower()
    if resolved_construction_device not in {"cpu", "meta"}:
        raise ValueError("SAM construction_device must be 'cpu' or 'meta'")
    if resolved_construction_device == "meta" and resolved_version != "sam3.1":
        raise ValueError("meta construction is supported only for SAM 3.1")
    if resolved_construction_device == "meta" and (bool(compile) or bool(warm_up)):
        raise ValueError(
            "SAM meta construction requires compile=False and warm_up=False; "
            "compile/warm-up must be applied only after CUDA materialization"
        )
    runtime_torch = None
    if builder is None:
        try:
            import torch as runtime_torch  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency/hardware gated
            raise RuntimeError(
                "LTA SAM runtime is unavailable; install the pinned local PyTorch runtime"
            ) from exc
        if device_id is None:
            raise ValueError("device_id is required for the real SAM runtime")
        resolved_device_id = int(device_id)
        if resolved_device_id < 0 or resolved_device_id >= int(runtime_torch.cuda.device_count()):
            raise ValueError(
                f"SAM device_id {resolved_device_id} is outside CUDA device_count="
                f"{int(runtime_torch.cuda.device_count())}"
            )
        runtime_torch.cuda.set_device(resolved_device_id)
        if use_fa3 is None:
            capability = runtime_torch.cuda.get_device_capability(resolved_device_id)
            use_fa3 = cuda_capability_supports_fa3(
                int(capability[0]), int(capability[1])
            )
        # Bind CUDA ownership before importing SAM: model_builder performs
        # import-time CUDA capability setup and must observe the worker device.
        try:
            from sam3.model_builder import build_sam3_predictor  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency gated
            raise RuntimeError(
                "LTA SAM runtime is unavailable; install the pinned local sam3 runtime"
            ) from exc
        builder = build_sam3_predictor

    confidence = resolve_confidence(conf)

    kwargs = {
        "checkpoint_path": str(bundle.checkpoint_path),
        "version": resolved_version,
        "compile": bool(compile),
        "warm_up": bool(warm_up),
        "async_loading_frames": bool(async_loading_frames),
    }
    if bundle.bpe_path is not None:
        bpe_path = (
            _validate_pinned_sam_bpe(bundle.bpe_path)
            if runtime_torch is not None
            else bundle.bpe_path
        )
        kwargs["bpe_path"] = str(bpe_path)
    elif runtime_torch is not None:
        kwargs["bpe_path"] = str(resolve_installed_sam_bpe())
    if resolved_version == "sam3.1":
        kwargs["max_num_objects"] = int(max_num_objects)
        kwargs["multiplex_count"] = int(LTA_MULTIPLEX_COUNT)
        kwargs["use_fa3"] = bool(use_fa3) if use_fa3 is not None else False
        kwargs["use_rope_real"] = (
            bool(use_rope_real) if use_rope_real is not None else bool(use_fa3)
        )
    if runtime_torch is not None and resolved_version == "sam3.1":
        predictor = _build_real_sam31_predictor_single_load(
            builder=builder,
            kwargs=kwargs,
            bundle=bundle,
            runtime_torch=runtime_torch,
            weight_storage=resolved_weight_storage,
            construction_device=resolved_construction_device,
        )
    else:
        predictor = builder(**kwargs)
    if runtime_torch is not None:
        patch_sam_init_state_signature(predictor)
    updated_controls = configure_predictor_confidence(predictor, confidence)
    expected_controls = {
        "model.score_threshold_detection",
        "model.image_only_det_thresh",
    }
    if resolved_version == "sam3.1":
        expected_controls.add("predictor.default_output_prob_thresh")
    missing_controls = sorted(expected_controls - set(updated_controls))
    if missing_controls:
        raise RuntimeError(
            "the pinned SAM predictor is missing required instance-admission "
            f"controls {missing_controls}; refusing to ignore --conf"
        )
    if runtime_torch is not None and int(runtime_torch.cuda.current_device()) != int(device_id):
        raise RuntimeError(
            f"SAM builder changed CUDA ownership from cuda:{int(device_id)} to "
            f"cuda:{int(runtime_torch.cuda.current_device())}"
        )
    return predictor


def configure_predictor_confidence(predictor: object, conf: object) -> Tuple[str, ...]:
    """Map the shared LTA ``--conf`` threshold onto exposed SAM controls.

    SAM 3 and 3.1 expose slightly different wrapper/model attributes.  The
    returned names are manifested/tested so a pinned runtime cannot silently
    ignore every threshold mapping after an upstream API change.
    """

    confidence = resolve_confidence(conf)
    updated = []
    if hasattr(predictor, "default_output_prob_thresh"):
        setattr(predictor, "default_output_prob_thresh", confidence)
        updated.append("predictor.default_output_prob_thresh")
    model = getattr(predictor, "model", None)
    for name in ("score_threshold_detection", "image_only_det_thresh"):
        if model is not None and hasattr(model, name):
            setattr(model, name, confidence)
            updated.append(f"model.{name}")
    return tuple(updated)


def _sequence_values(value: object, *, name: str) -> Tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"SAM output {name} must be an ordered array/sequence")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"SAM output {name} must be iterable") from exc


def _normalize_binary_mask(mask: object, *, name: str) -> object:
    """Normalize one official SAM mask to a rank-2 boolean array/tensor."""

    shape_raw = getattr(mask, "shape", None)
    if shape_raw is None:
        raise ValueError(f"SAM output {name} mask exposes no shape")
    shape = tuple(int(value) for value in tuple(shape_raw))
    normalized = mask
    if len(shape) == 3 and shape[0] == 1:
        normalized = mask[0]  # type: ignore[index]
        shape = tuple(int(value) for value in tuple(getattr(normalized, "shape")))
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"SAM output {name} mask must resolve to nonempty HxW; got {shape}")
    dtype = str(getattr(normalized, "dtype", "")).lower()
    if "bool" not in dtype:
        raise ValueError(f"SAM output {name} mask must be boolean; got dtype={dtype or 'unknown'}")
    return normalized


def _raise_on_dropped_objects(outputs: Mapping[str, object]) -> None:
    stats = outputs.get("frame_stats")
    if not isinstance(stats, Mapping):
        raise ValueError("SAM output frame_stats must be a mapping")
    if "num_obj_dropped" not in stats:
        raise ValueError("SAM output frame_stats has no num_obj_dropped field")
    dropped = stats["num_obj_dropped"]
    try:
        count = int(dropped or 0)
    except Exception as exc:
        raise ValueError(f"SAM frame_stats has invalid dropped-object count {dropped!r}") from exc
    if count > 0:
        raise RuntimeError(
            f"SAM Object Multiplex dropped {count} object(s); LTA refuses silent overflow"
        )


def normalize_video_frame_output(
    outputs: Mapping[str, object],
    *,
    sequence_id: str,
    session_index: int,
    global_frame_index: int,
    frame_tracker_scores: Optional[object] = None,
    require_drop_stats: bool = True,
) -> Tuple[SamFramePrediction, ...]:
    """Normalize one official video output without importing NumPy/Torch."""

    object_ids = _sequence_values(outputs.get("out_obj_ids"), name="out_obj_ids")
    scores = _sequence_values(outputs.get("out_probs"), name="out_probs")
    masks = _sequence_values(outputs.get("out_binary_masks"), name="out_binary_masks")
    # The official public response does not expose its internal framewise
    # tracker score. A future pinned adapter may supply it explicitly through
    # this argument; never guess unofficial output keys here.
    tracker_scores = _sequence_values(frame_tracker_scores, name="tracker_scores")
    if len(object_ids) != len(scores) or len(object_ids) != len(masks):
        raise ValueError(
            "SAM video output lengths differ for object ids, scores, and masks: "
            f"{len(object_ids)}/{len(scores)}/{len(masks)}"
        )
    if tracker_scores and len(tracker_scores) != len(object_ids):
        raise ValueError(
            "SAM video tracker-score length differs from object outputs: "
            f"{len(tracker_scores)}/{len(object_ids)}"
        )
    normalized_ids = tuple(int(value) for value in object_ids)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("SAM video output contains duplicate object ids in one frame")
    if bool(require_drop_stats):
        _raise_on_dropped_objects(outputs)
    else:
        stats = outputs.get("frame_stats")
        if stats is not None:
            _raise_on_dropped_objects(outputs)
    predictions = []
    for index, (object_id, score) in enumerate(zip(object_ids, scores)):
        resolved_score = float(score)
        if resolved_score == _SAM_REMOVED_OBJECT_SCORE:
            continue
        normalized_mask = _normalize_binary_mask(
            masks[index], name=f"frame {global_frame_index} object {object_id}"
        )
        predictions.append(
            SamFramePrediction(
                sequence_id=sequence_id,
                session_index=int(session_index),
                frame_index=int(global_frame_index),
                object_id=int(object_id),
                initial_detection_score=resolved_score,
                frame_tracker_score=(
                    None if not tracker_scores else float(tracker_scores[index])
                ),
                binary_mask=normalized_mask,
            )
        )
    return tuple(predictions)


def run_video_session(
    predictor: object,
    *,
    resource: object,
    session: SamSessionPlan,
    prompt: SamPromptBox,
    conf: object,
    offload_video_to_cpu: bool = False,
) -> Tuple[SamFramePrediction, ...]:
    """Run one fixed official SAM video session through its request API."""

    if not isinstance(session, SamSessionPlan):
        raise TypeError("session must be a SamSessionPlan")
    if not isinstance(prompt, SamPromptBox):
        raise TypeError("prompt must be a SamPromptBox")
    if not isinstance(resource, list):
        raise TypeError(
            "the v19 LTA prototype requires an explicit ordered list of decoded PIL frames"
        )
    if len(resource) != int(session.frame_count):
        raise ValueError(
            "decoded SAM resource length does not match the fixed session: "
            f"{len(resource)} != {session.frame_count}"
        )
    if not int(session.frame_start) <= int(prompt.frame_index) < int(session.frame_stop):
        raise ValueError("the session's positive prompt must address a frame inside the session")
    confidence = resolve_confidence(conf)
    handle_request = getattr(predictor, "handle_request", None)
    handle_stream_request = getattr(predictor, "handle_stream_request", None)
    if not callable(handle_request) or not callable(handle_stream_request):
        raise TypeError("predictor must expose handle_request and handle_stream_request")

    started = handle_request(
        {
            "type": "start_session",
            "resource_path": resource,
            "offload_video_to_cpu": bool(offload_video_to_cpu),
        }
    )
    if not isinstance(started, Mapping) or not str(started.get("session_id", "")).strip():
        raise RuntimeError("SAM start_session returned no session_id")
    session_id = str(started["session_id"])
    local_prompt_frame = int(prompt.frame_index) - int(session.frame_start)
    collected = []
    seen_local_frames: set[int] = set()
    active_error: Optional[BaseException] = None
    stream: Optional[Iterable[object]] = None
    try:
        prompted = handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": local_prompt_frame,
                "bounding_boxes": [list(prompt.xywh)],
                "bounding_box_labels": [1],
                "output_prob_thresh": confidence,
            }
        )
        if not isinstance(prompted, Mapping):
            raise RuntimeError("SAM add_prompt returned a non-mapping response")
        prompted_frame = int(prompted.get("frame_index", -1))
        if prompted_frame != local_prompt_frame:
            raise RuntimeError(
                "SAM add_prompt returned the wrong frame: "
                f"{prompted_frame} != {local_prompt_frame}"
            )
        prompted_outputs = prompted.get("outputs")
        if not isinstance(prompted_outputs, Mapping):
            raise RuntimeError("SAM add_prompt response has no mapping outputs")
        _raise_on_dropped_objects(prompted_outputs)
        stream = handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "both",
                "max_frame_num_to_track": int(session.frame_count),
                "output_prob_thresh": confidence,
            }
        )
        for response in stream:
            if not isinstance(response, Mapping):
                raise RuntimeError("SAM propagation yielded a non-mapping response")
            local_frame = int(response.get("frame_index", -1))
            if not 0 <= local_frame < int(session.frame_count):
                raise RuntimeError(
                    f"SAM propagation yielded out-of-range frame {local_frame} "
                    f"for {session.frame_count}-frame session"
                )
            if local_frame in seen_local_frames:
                raise RuntimeError(f"SAM propagation yielded duplicate frame {local_frame}")
            seen_local_frames.add(local_frame)
            outputs = response.get("outputs")
            if not isinstance(outputs, Mapping):
                raise RuntimeError("SAM propagation response has no mapping outputs")
            collected.extend(
                normalize_video_frame_output(
                    outputs,
                    sequence_id=session.sequence_id,
                    session_index=session.session_index,
                    global_frame_index=int(session.frame_start) + local_frame,
                )
            )
        expected_frames = set(range(int(session.frame_count)))
        if seen_local_frames != expected_frames:
            missing = sorted(expected_frames - seen_local_frames)
            extra = sorted(seen_local_frames - expected_frames)
            raise RuntimeError(
                "SAM propagation did not settle every session frame exactly once; "
                f"missing={missing[:12]}, extra={extra[:12]}"
            )
    except BaseException as exc:
        active_error = exc
        collected.clear()
        raise
    finally:
        cleanup_errors = []
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            try:
                close_stream()
            except Exception as stream_close_exc:
                cleanup_errors.append(("propagation iterator", stream_close_exc))
        try:
            handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                }
            )
        except Exception as close_exc:
            cleanup_errors.append(("close_session", close_exc))
        if active_error is not None:
            add_note = getattr(active_error, "add_note", None)
            if callable(add_note):
                for boundary, cleanup_error in cleanup_errors:
                    add_note(
                        f"SAM {boundary} also failed while handling the primary error: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        elif cleanup_errors:
            boundary, cleanup_error = cleanup_errors[0]
            add_note = getattr(cleanup_error, "add_note", None)
            if callable(add_note):
                for secondary_boundary, secondary_error in cleanup_errors[1:]:
                    add_note(
                        f"SAM {secondary_boundary} also failed during cleanup: "
                        f"{type(secondary_error).__name__}: {secondary_error}"
                    )
            raise cleanup_error
    return tuple(sorted(collected, key=lambda item: (item.frame_index, item.object_id)))


def run_image_frame(
    processor: object,
    *,
    image: object,
    sequence_id: str,
    frame_index: int,
    prompt: SamPromptBox,
    conf: object,
) -> Tuple[SamFramePrediction, ...]:
    """Run one official ``Sam3Processor``-compatible image call."""

    if not isinstance(prompt, SamPromptBox):
        raise TypeError("prompt must be a SamPromptBox")
    confidence = resolve_confidence(conf)
    set_image = getattr(processor, "set_image", None)
    add_prompt = getattr(processor, "add_geometric_prompt", None)
    set_threshold = getattr(processor, "set_confidence_threshold", None)
    if not callable(set_image) or not callable(add_prompt):
        raise TypeError("processor must expose set_image and add_geometric_prompt")
    if callable(set_threshold):
        set_threshold(confidence)
    state = set_image(image)
    x, y, width, height = prompt.xywh
    cxcywh = [x + width * 0.5, y + height * 0.5, width, height]
    state = add_prompt(box=cxcywh, label=True, state=state)
    if not isinstance(state, Mapping):
        raise RuntimeError("SAM image processor returned a non-mapping state")
    scores = _sequence_values(state.get("scores"), name="scores")
    masks = _sequence_values(state.get("masks"), name="masks")
    if len(scores) != len(masks):
        raise ValueError(
            f"SAM image output lengths differ for scores and masks: {len(scores)}/{len(masks)}"
        )
    return tuple(
        SamFramePrediction(
            sequence_id=str(sequence_id),
            session_index=int(frame_index),
            frame_index=int(frame_index),
            object_id=int(index),
            initial_detection_score=float(score),
            frame_tracker_score=None,
            binary_mask=_normalize_binary_mask(
                masks[index], name=f"image frame {frame_index} object {index}"
            ),
        )
        for index, score in enumerate(scores)
    )


__all__ = (
    "LTA_CHANNEL_POLICY",
    "LTA_MAX_NUM_OBJECTS",
    "LTA_MULTIPLEX_COUNT",
    "LTA_SESSION_FRAMES",
    "LocalSamBundle",
    "SamFramePrediction",
    "SamPromptBox",
    "SamSessionPlan",
    "build_local_sam_predictor",
    "configure_predictor_confidence",
    "cuda_capability_supports_fa3",
    "normalize_video_frame_output",
    "plan_sam_sessions",
    "resolve_confidence",
    "resolve_local_sam_bundle",
    "resolve_installed_sam_bpe",
    "revalidate_local_sam_bundle",
    "run_image_frame",
    "run_video_session",
    "patch_sam_init_state_signature",
)
