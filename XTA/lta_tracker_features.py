"""Pinned SAM 3.1 visual features for the mask-only LTA tracker bridge.

The normal multiplex preparation runs grounding merely to retain its visual
backbone output. Mask-only LTA does not consume detections. This adapter calls
the same visual trunk and both tracker necks, preserving the BF16 cast before
the tracker projections used by the pinned single-rank grounding path.

Installed SAM source is never patched. Unsupported custom models retain their
original feature bridge; failures inside the supported path are propagated.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import operator
from typing import Any


TRACKER_FEATURE_POLICY = "sam3_1_visual_backbone_only_exact_bf16"
TRACKER_FEATURE_AUDIT_KEY = "lta_tracker_feature_preparation"


def _class_identity(value: object) -> tuple[str, str]:
    return type(value).__module__, type(value).__name__


def _unsupported_reason(model: object) -> str | None:
    if _class_identity(model) != (
        "sam3.model.sam3_multiplex_tracking", "Sam3MultiplexTrackingWithInteractivity"
    ):
        return "custom_model"
    detector = getattr(model, "detector", None)
    if _class_identity(detector) != (
        "sam3.model.sam3_multiplex_detector", "Sam3MultiplexDetector"
    ):
        return "custom_detector"
    if _class_identity(getattr(detector, "backbone", None)) != (
        "sam3.model.vl_combiner", "SAM3VLBackboneTri"
    ):
        return "custom_backbone"
    if (
        getattr(model, "world_size", None) != 1
        or getattr(detector, "world_size", None) != 1
        or getattr(model, "rank", None) != 0
        or getattr(detector, "rank", None) != 0
    ):
        return "distributed_world"
    if (
        not bool(getattr(model, "is_multiplex", False))
        or not bool(getattr(detector, "is_multiplex", False))
        or not bool(getattr(detector, "gather_backbone_out", False))
    ):
        return "custom_feature_layout"
    return None


def _record_preparation(
    state: MutableMapping[str, Any], *, fallback_reason: str | None
) -> dict[str, object]:
    audit = state.setdefault(TRACKER_FEATURE_AUDIT_KEY, {
        "policy": TRACKER_FEATURE_POLICY,
        "feature_only_preparations": 0,
        "fallback_preparations": 0,
        "fallback_reasons": {},
        "fpn_preprojection_dtype": "bfloat16",
        "sam3_detection_neck_requested": False,
        "interactive_and_propagation_necks_requested": True,
    })
    if not isinstance(audit, MutableMapping) or audit.get("policy") != TRACKER_FEATURE_POLICY:
        raise RuntimeError("LTA tracker feature audit state is incompatible")
    if fallback_reason is None:
        audit["feature_only_preparations"] += 1
    else:
        audit["fallback_preparations"] += 1
        reasons = audit["fallback_reasons"]
        reasons[fallback_reason] = int(reasons.get(fallback_reason, 0)) + 1
    return {
        "policy": TRACKER_FEATURE_POLICY if fallback_reason is None else "original_feature_bridge",
        "fallback_reason": fallback_reason,
    }


def prepare_tracker_frame_features(
    model: object,
    state: MutableMapping[str, Any],
    frame_idx: int,
    reverse: bool,
) -> dict[str, object]:
    """Populate the exact one-frame tracker cache without grounding detection.

    The caller retains its existing inference-mode/autocast context. The
    pinned Sam3MultiplexPredictorWrapper enters CUDA BF16 autocast for its
    process lifetime; the original preparation methods add no new context.
    The frame image stored in the cache is the original normalized image object; only
    the backbone input is promoted to float32 on the detector's device, just
    as in ``Sam3Image._get_img_feats``. Cache pruning follows the original
    directional adjacent-frame rule exactly.
    """

    if not isinstance(state, MutableMapping):
        raise TypeError("tracker inference state must be a mutable mapping")
    if isinstance(frame_idx, bool):
        raise TypeError("frame_idx must be an integer")
    frame_index = operator.index(frame_idx)
    if frame_index < 0:
        raise ValueError("frame_idx must be nonnegative")
    if not isinstance(reverse, bool):
        raise TypeError("reverse must be bool")
    reason = _unsupported_reason(model)
    if reason is not None:
        original = getattr(model, "_prepare_backbone_feats", None)
        if not callable(original):
            raise RuntimeError("SAM model exposes no original shared feature bridge")
        original(state, frame_index, reverse=reverse)
        return _record_preparation(state, fallback_reason=reason)

    import torch

    detector = model.detector
    backbone = detector.backbone
    tracker = model.tracker
    if bool(getattr(model, "training", False)) or bool(getattr(backbone, "training", False)):
        raise RuntimeError("LTA tracker feature preparation requires evaluation mode")
    if frame_index >= int(state["num_frames"]):
        raise ValueError("frame_idx lies outside the tracker input sequence")
    cache = state["feature_cache"]
    if not isinstance(cache, MutableMapping):
        raise RuntimeError("SAM feature_cache must be a mutable mapping")
    input_batch = state["input_batch"]
    image_batch = input_batch.img_batch
    if not hasattr(image_batch, "tensors") or getattr(image_batch, "mask", None) is not None:
        raise RuntimeError("pinned SAM frame batch must use unmasked NestedTensor storage")
    # Local LTA sessions have one fixed FindStage per frame. Streaming/circular
    # remapping would need its own qualification and must not silently select
    # another image when this adapter is active.
    find_input = input_batch.find_inputs[frame_index]
    image_ids = find_input.img_ids
    if image_ids.numel() != 1 or int(image_ids.reshape(-1)[0].item()) != frame_index:
        raise RuntimeError("pinned SAM frame identity differs from the requested local frame")
    original_image = image_batch.tensors[frame_index]
    if not isinstance(original_image, torch.Tensor) or original_image.ndim != 3:
        raise RuntimeError("pinned SAM frame image must be a CHW tensor")
    image = original_image.unsqueeze(0).to(dtype=torch.float32, device=detector.device)
    features = backbone.forward_image(
        image,
        need_sam3_out=False,
        need_interactive_out=True,
        need_propagation_out=True,
    )
    if not isinstance(features, Mapping):
        raise RuntimeError("SAM visual backbone returned a non-mapping feature set")
    prepared = {}
    # Preserve pinned cache construction order: interactive projections first,
    # then propagation projections. No grounding output contributes to either.
    for key, decoder in (
        ("interactive", tracker.interactive_sam_mask_decoder),
        ("sam2_backbone_out", tracker.sam_mask_decoder),
    ):
        branch = features.get(key)
        if not isinstance(branch, Mapping) or branch.get("vision_mask") is not None:
            raise RuntimeError(f"SAM visual backbone returned incompatible {key} features")
        pyramid = tuple(branch.get("backbone_fpn", ()))
        positions = branch.get("vision_pos_enc")
        if len(pyramid) != 3 or not isinstance(positions, (tuple, list)) or len(positions) != 3:
            raise RuntimeError(f"SAM {key} features require exactly three pyramid levels")
        for level, (feature, position) in enumerate(zip(pyramid, positions)):
            tensor = getattr(feature, "tensors", None)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 4
                or tensor.shape[0] != 1
                or getattr(feature, "mask", None) is not None
                or not isinstance(position, torch.Tensor)
                or position.ndim != 4
                or position.shape[0] != 1
                or tensor.shape[-2:] != position.shape[-2:]
            ):
                raise RuntimeError(f"SAM {key} pyramid level {level} has incompatible geometry")
        # Grounding's _build_multigpu_buffer_next_chunk applies this BF16 cast
        # even with one rank and FP32 model storage. Keep it before decoder
        # projections; casting after them can change the tracker features.
        projected = [feature.tensors.to(dtype=torch.bfloat16) for feature in pyramid]
        projected[0] = decoder.conv_s0(projected[0])
        projected[1] = decoder.conv_s1(projected[1])
        prepared[key] = {
            "vision_features": projected[-1],
            "vision_mask": None,
            "vision_pos_enc": positions,
            "backbone_fpn": [
                type(feature)(tensor, None) for feature, tensor in zip(pyramid, projected)
            ],
        }
    cache[frame_index] = (original_image, prepared)
    cache.pop(frame_index + 1 if reverse else frame_index - 1, None)
    return _record_preparation(state, fallback_reason=None)


__all__ = (
    "TRACKER_FEATURE_AUDIT_KEY",
    "TRACKER_FEATURE_POLICY",
    "prepare_tracker_frame_features",
)
