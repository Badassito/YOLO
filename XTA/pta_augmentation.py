"""External augmentation policy discovery and paired execution for PTA.

This module is a leaf in the XTA import graph.  It owns policy-file
inspection, CPU/GPU policy loading, deterministic per-thread CPU pipeline
construction, and validation of paired image/mask results without importing
the PTA dataset engine.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import inspect
import math
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple, Union

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc


@dataclass(frozen=True)
class AugmentationDefinition:
    """Import-free description of an external augmentation policy.

    Deferred mode deliberately parses only Python syntax and export names.  It
    therefore does not import Albumentations/Torch, allocate transform
    objects, or execute arbitrary policy-file code during dataset generation.
    """

    path: Path
    content_sha256: str
    export_name: str


def inspect_augmentation_definition(path_arg: str) -> AugmentationDefinition:
    path = Path(path_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"--augmentation file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"--augmentation must point to a Python file, not a directory: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"--augmentation must point to a .py file: {path}")
    content = path.read_bytes()
    content_sha256 = hashlib.sha256(content).hexdigest()
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Could not parse augmentation file {path}: {exc}") from exc

    supported = {
        "custom_transforms",
        "augmentation",
        "build_augmentation",
        "build_gpu_augmentation",
    }
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(str(node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    defined.add(str(target.id))
    recognized = sorted(supported & defined)
    if not recognized:
        raise ValueError(
            f"Augmentation file {path} must define exactly one of: "
            "custom_transforms, augmentation, build_augmentation, build_gpu_augmentation"
        )
    if len(recognized) != 1:
        raise ValueError(
            f"Augmentation file {path} defines multiple supported exports {recognized}; define exactly one"
        )
    return AugmentationDefinition(path, content_sha256, recognized[0])


def assert_augmentation_definition_unchanged(
    definition: Optional[AugmentationDefinition],
) -> None:
    """Reject successful publication if the external policy changed mid-run."""

    if definition is None:
        return
    try:
        current = inspect_augmentation_definition(str(definition.path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "PTA augmentation policy changed during execution; refusing a complete "
            f"manifest: path={definition.path}, unavailable_or_invalid={exc}"
        ) from exc
    changed: List[str] = []
    if current.path != definition.path:
        changed.append("path")
    if current.content_sha256 != definition.content_sha256:
        changed.append("sha256")
    if current.export_name != definition.export_name:
        changed.append("export")
    if changed:
        raise RuntimeError(
            "PTA augmentation policy changed during execution; refusing a complete "
            f"manifest: path={definition.path}, fields={changed}"
        )


@dataclass
class LoadedAugmentation:
    path: Path
    content_sha256: str
    export_name: str
    albumentations_version: str
    pipeline_builder: Callable[[], object]
    thread_state: threading.local = field(default_factory=threading.local)

    def pipeline_for_current_thread(self) -> object:
        pipeline = getattr(self.thread_state, "pipeline", None)
        if pipeline is None:
            try:
                pipeline = self.pipeline_builder()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to construct augmentation pipeline from {self.path}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            validate_seedable_augmentation_pipeline(pipeline, path=self.path)
            mask_interpolation_setter = getattr(pipeline, "set_mask_interpolation", None)
            if callable(mask_interpolation_setter):
                mask_interpolation_setter(cv2.INTER_NEAREST)
            self.thread_state.pipeline = pipeline
        return pipeline


@dataclass
class LoadedGpuAugmentation:
    """Fork-inherited factory for the single-file CUDA policy API.

    The parent imports the definition before the persistent pool is forked but
    never constructs CUDA state.  Each child invokes the factory only after it
    has been assigned exactly one CUDA-visible device.
    """

    path: Path
    content_sha256: str
    export_name: str
    runtime_name: str
    policy_builder: Callable[..., object]

    @property
    def albumentations_version(self) -> str:
        return ""

    def build_for_device(self, *, device: str, batch_size: int) -> object:
        try:
            policy = self.policy_builder(device=str(device), batch_size=int(batch_size))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to construct GPU augmentation policy from {self.path} "
                f"for {device}: {type(exc).__name__}: {exc}"
            ) from exc
        apply_batch = getattr(policy, "apply_batch", None)
        if not callable(apply_batch):
            raise TypeError(
                f"{self.path}: build_gpu_augmentation() must return an object with "
                "apply_batch(image, mask, seeds, output_size)"
            )
        return policy


OfflineAugmentation = Union[LoadedAugmentation, LoadedGpuAugmentation]


def validate_seedable_augmentation_pipeline(pipeline: object, *, path: Path) -> None:
    if not callable(pipeline):
        raise TypeError(f"Augmentation pipeline from {path} is not callable: {type(pipeline).__name__}")
    seed_setter = getattr(pipeline, "set_random_seed", None)
    if not callable(seed_setter):
        raise TypeError(
            f"Augmentation pipeline from {path} must provide set_random_seed(seed). "
            "Install a current Albumentations 2.x-compatible release and return an A.Compose-compatible pipeline."
        )


def _load_external_python_module(path: Path, content_sha256: str) -> object:
    module_name = f"_pta_v4_augmentation_{content_sha256[:24]}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for augmentation file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    parent_text = str(path.parent)
    sys.path.insert(0, parent_text)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            f"Failed to import augmentation file {path}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            sys.path.remove(parent_text)
        except ValueError:
            pass
    return module


def load_augmentation_definition(path_arg: str) -> LoadedAugmentation:
    definition = inspect_augmentation_definition(path_arg)
    if definition.export_name == "build_gpu_augmentation":
        raise ValueError(
            f"{definition.path}: build_gpu_augmentation is a CUDA policy; "
            "use the offline GPU loader"
        )
    path = definition.path
    content_sha256 = definition.content_sha256
    try:
        albumentations = importlib.import_module("albumentations")
    except Exception as exc:
        raise RuntimeError(
            "--augmentation requires Albumentations: pip install albumentations"
        ) from exc

    module = _load_external_python_module(path, content_sha256)
    module_vars = vars(module)
    export_name = definition.export_name
    if export_name not in module_vars:
        raise ValueError(f"Augmentation export {export_name!r} was not created when importing {path}")
    exported = module_vars[export_name]
    pipeline_builder: Callable[[], object]
    if export_name == "custom_transforms":
        if not isinstance(exported, (list, tuple)):
            raise TypeError(f"{path}: custom_transforms must be a list or tuple")
        if not exported:
            raise ValueError(f"{path}: custom_transforms must contain at least one transform")
        transforms_template = copy.deepcopy(list(exported))

        def _build_from_list() -> object:
            return albumentations.Compose(copy.deepcopy(transforms_template))

        pipeline_builder = _build_from_list
    elif export_name == "augmentation":
        validate_seedable_augmentation_pipeline(exported, path=path)
        try:
            pipeline_template = copy.deepcopy(exported)
        except Exception as exc:
            raise TypeError(
                f"{path}: augmentation must be deepcopy-compatible so each worker has isolated random state: {exc}"
            ) from exc

        def _build_from_object() -> object:
            return copy.deepcopy(pipeline_template)

        pipeline_builder = _build_from_object
    else:
        if not callable(exported):
            raise TypeError(f"{path}: build_augmentation must be callable")
        try:
            signature = inspect.signature(exported)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{path}: could not inspect build_augmentation(): {exc}") from exc
        required = [
            param for param in signature.parameters.values()
            if param.default is inspect.Parameter.empty
            and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        if required:
            raise TypeError(f"{path}: build_augmentation() must not require arguments")

        def _build_from_factory() -> object:
            pipeline = exported()
            try:
                return copy.deepcopy(pipeline)
            except Exception as exc:
                raise TypeError(
                    f"{path}: build_augmentation() must return a deepcopy-compatible pipeline "
                    f"so workers cannot share random state: {exc}"
                ) from exc

        pipeline_builder = _build_from_factory

    loaded = LoadedAugmentation(
        path=path,
        content_sha256=content_sha256,
        export_name=export_name,
        albumentations_version=str(getattr(albumentations, "__version__", "unknown")),
        pipeline_builder=pipeline_builder,
    )
    # Validate before any output directory is cleaned. Reuse this instance on
    # the main thread; worker threads receive their own independently built one.
    probe = pipeline_builder()
    validate_seedable_augmentation_pipeline(probe, path=path)
    mask_interpolation_setter = getattr(probe, "set_mask_interpolation", None)
    if callable(mask_interpolation_setter):
        mask_interpolation_setter(cv2.INTER_NEAREST)
    loaded.thread_state.pipeline = probe
    return loaded


def load_gpu_augmentation_definition(path_arg: str) -> LoadedGpuAugmentation:
    definition = inspect_augmentation_definition(path_arg)
    if definition.export_name != "build_gpu_augmentation":
        raise ValueError(
            f"{definition.path}: GPU offline execution requires the single export "
            "build_gpu_augmentation"
        )
    module = _load_external_python_module(definition.path, definition.content_sha256)
    builder = getattr(module, definition.export_name, None)
    if not callable(builder):
        raise TypeError(f"{definition.path}: build_gpu_augmentation must be callable")
    try:
        signature = inspect.signature(builder)
        signature.bind(device="cuda:0", batch_size=1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{definition.path}: build_gpu_augmentation must accept keyword arguments "
            f"device and batch_size: {exc}"
        ) from exc
    runtime_name = str(getattr(module, "PTA_GPU_RUNTIME", "torch-cuda"))
    return LoadedGpuAugmentation(
        path=definition.path,
        content_sha256=definition.content_sha256,
        export_name=definition.export_name,
        runtime_name=runtime_name,
        policy_builder=builder,
    )


def load_offline_augmentation_definition(path_arg: str) -> OfflineAugmentation:
    definition = inspect_augmentation_definition(path_arg)
    if definition.export_name == "build_gpu_augmentation":
        return load_gpu_augmentation_definition(path_arg)
    return load_augmentation_definition(path_arg)


def _augmented_image_to_uint8(value: object, *, context: str) -> np.ndarray:
    """Normalize an augmented HxW or HxWxC image without collapsing channels."""

    arr = np.asarray(value)
    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ValueError(f"{context}: augmented image must be a nonempty 2D/3D array, got shape={arr.shape}")
    if arr.ndim == 3 and int(arr.shape[2]) < 1:
        raise ValueError(f"{context}: augmented image must contain at least one channel, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
        raise TypeError(f"{context}: augmented image must be numeric, got dtype={arr.dtype}")
    if arr.dtype == np.bool_:
        return np.ascontiguousarray(arr.astype(np.uint8) * np.uint8(255))
    if np.issubdtype(arr.dtype, np.floating):
        # NaN/inf propagate through min/max, so two scans replace
        # the previous isfinite + min + max triple pass.
        amin = float(np.min(arr)) if arr.size else 0.0
        amax = float(np.max(arr)) if arr.size else 0.0
        if not (math.isfinite(amin) and math.isfinite(amax)):
            raise ValueError(f"{context}: augmented image contains NaN or infinity")
        if amin >= 0.0 and amax <= 1.0:
            arr = np.rint(arr.astype(np.float32, copy=False) * 255.0).astype(np.uint8)
        elif amin >= 0.0 and amax <= 255.0:
            arr = np.rint(arr).astype(np.uint8)
        else:
            raise ValueError(
                f"{context}: floating augmented image range [{amin:g}, {amax:g}] cannot be written as uint8; "
                "omit Normalize/ToTensor transforms or return values in [0,1] or [0,255]"
            )
    elif arr.dtype == np.uint16:
        arr = np.clip(np.rint(arr.astype(np.float32) / 257.0), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        amin = int(np.min(arr))
        amax = int(np.max(arr))
        if amin < 0 or amax > 255:
            raise ValueError(f"{context}: integer augmented image range [{amin}, {amax}] is outside [0,255]")
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _augmented_mask_to_binary(value: object, *, context: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.size == 0 or arr.ndim != 2:
        raise ValueError(f"{context}: augmented mask must be a nonempty 2D array, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
        raise TypeError(f"{context}: augmented mask must be numeric, got dtype={arr.dtype}")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{context}: augmented mask contains NaN or infinity")
    return np.ascontiguousarray((arr >= 0.5).astype(np.uint8))


def apply_augmentation_pair(
    augmentation: LoadedAugmentation,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    context: str,
    copy_inputs: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # Callers replaying many seeded copies of one source item make
    # ONE private copy up front and pass copy_inputs=False, instead of paying
    # two full-frame memcpys on every call.
    pipeline = augmentation.pipeline_for_current_thread()
    seed_setter = getattr(pipeline, "set_random_seed")
    if copy_inputs:
        image_in = np.ascontiguousarray(np.asarray(image).copy())
        mask_in = np.ascontiguousarray(np.asarray(mask).copy())
    else:
        image_in = np.asarray(image)
        mask_in = np.asarray(mask)
    try:
        seed_setter(int(seed))
        result = pipeline(
            image=image_in,
            mask=mask_in,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{context}: augmentation failed using {augmentation.path} with seed={int(seed)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(result, Mapping):
        raise TypeError(f"{context}: augmentation pipeline must return a mapping, got {type(result).__name__}")
    if "image" not in result or "mask" not in result:
        raise ValueError(f"{context}: augmentation result must contain both 'image' and 'mask' keys")
    out_image = _augmented_image_to_uint8(result["image"], context=context)
    out_mask = _augmented_mask_to_binary(result["mask"], context=context)
    input_channel_count = 1 if image_in.ndim == 2 else int(image_in.shape[2])
    output_channel_count = 1 if out_image.ndim == 2 else int(out_image.shape[2])
    if output_channel_count != input_channel_count:
        raise ValueError(
            f"{context}: augmentation changed the image channel count from "
            f"{input_channel_count} to {output_channel_count}"
        )
    if image_in.ndim == 2 and out_image.ndim == 3 and int(out_image.shape[2]) == 1:
        out_image = np.ascontiguousarray(out_image[:, :, 0])
    elif image_in.ndim == 3 and int(image_in.shape[2]) == 1 and out_image.ndim == 2:
        out_image = np.ascontiguousarray(out_image[:, :, None])
    if tuple(out_image.shape[:2]) != tuple(out_mask.shape[:2]):
        raise ValueError(
            f"{context}: augmented image/mask dimensions differ: image={out_image.shape}, mask={out_mask.shape}"
        )
    return out_image, out_mask


def assert_augmentation_did_not_synthesize_mask(
    original_mask: np.ndarray,
    augmented_mask: np.ndarray,
    *,
    context: str,
    original_known_empty: bool = False,
) -> None:
    """Reject only true empty-mask synthesis, not contour degeneracy changes."""

    if not original_known_empty and np.any(np.asarray(original_mask) > 0):
        return
    augmented_count = int(np.count_nonzero(np.asarray(augmented_mask) > 0))
    if augmented_count <= 0:
        return
    raise RuntimeError(
        f"{context}: augmentation introduced {augmented_count} mask pixel(s) into a truly empty mask. "
        "Check fill_mask/fill values and custom mask transforms; PTA augmentations may move, enlarge, "
        "shrink, or remove existing labels but may not synthesize them."
    )


__all__ = (
    "AugmentationDefinition",
    "LoadedAugmentation",
    "LoadedGpuAugmentation",
    "OfflineAugmentation",
    "_augmented_image_to_uint8",
    "_augmented_mask_to_binary",
    "_load_external_python_module",
    "apply_augmentation_pair",
    "assert_augmentation_definition_unchanged",
    "assert_augmentation_did_not_synthesize_mask",
    "inspect_augmentation_definition",
    "load_augmentation_definition",
    "load_gpu_augmentation_definition",
    "load_offline_augmentation_definition",
    "validate_seedable_augmentation_pipeline",
)
