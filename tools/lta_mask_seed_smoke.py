"""Exercise authoritative-mask seeding through the pinned LTA tracker adapter.

This uses a pinned private SAM2 tracker boundary after creating one interactive
object. It is diagnostic-only and must not be treated as a public SAM API or an
approved LTA publication path.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for entry in (ROOT, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from lta_gpu_smoke import (
    _MeasuredPredictor,
    configure_constrained_gpu_batches,
    cuda_snapshot,
    find_case_video,
    find_indexed_exemplar,
    install_sdpa_fallback,
    resolve_pinned_sam_runtime_provenance,
    validate_smoke_window,
)
from lta_point_smoke import (
    PointClick,
    _distance_peaks,
    _pixel_center_normalized,
    write_strategy_artifacts,
)
from lta_tile_smoke import (
    decode_rgb_tile_frames,
    probe_video_dimensions,
)
from XTA.lta_experimental import (
    MASK_SEED_ANCHOR_IOU,
    _resolve_empty_frame_limit,
    resolve_mask_seed_capacity,
    run_mask_seed_session,
)
from XTA.lta_inputs import parse_yolo_segmentation_label
from XTA.lta_tiles import (
    plan_object_tile,
    rasterize_polygons,
    select_polygon_row,
    transform_polygon_to_tile,
)
from XTA.lta_sam import (
    LTA_MAX_NUM_OBJECTS,
    LTA_MULTIPLEX_COUNT,
    LTA_SESSION_FRAMES,
    SamSessionPlan,
    build_local_sam_predictor,
    resolve_local_sam_bundle,
)




def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--exemplar-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--prompt-frame", type=int, required=True)
    parser.add_argument("--exemplar-index", type=int, required=True)
    parser.add_argument("--label-row", type=int, required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument(
        "--seed-scope",
        choices=("selected", "all-in-tile"),
        default="selected",
    )
    parser.add_argument(
        "--propagation-mode",
        choices=("merged", "tracker-only"),
        default="tracker-only",
    )
    parser.add_argument(
        "--weight-storage",
        choices=("bfloat16_egpu", "float32"),
        default="bfloat16_egpu",
    )
    return parser


def main() -> None:
    import numpy as np
    import torch

    args = _build_parser().parse_args()
    validate_smoke_window(args.start_frame, args.prompt_frame)
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"--output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = resolve_local_sam_bundle(args.model)
    video = find_case_video(Path(args.input_root).resolve(strict=True), "direct")
    _image, label = find_indexed_exemplar(Path(args.exemplar_root).resolve(strict=True), args.exemplar_index)
    label_digest, polygons = parse_yolo_segmentation_label(label)
    selected_polygon = select_polygon_row(polygons, int(args.label_row))
    width, height = probe_video_dimensions(video)
    tile = plan_object_tile(
        selected_polygon,
        source_width=width,
        source_height=height,
    )
    local_polygons = tuple(transform_polygon_to_tile(item, tile) for item in polygons)
    local_selected_polygon = transform_polygon_to_tile(selected_polygon, tile)
    if args.seed_scope == "selected":
        ground_truth = rasterize_polygons((local_selected_polygon,))
        object_masks = None
        peak = _distance_peaks(ground_truth, 1)[0]
        seed = PointClick(*_pixel_center_normalized(peak[0], peak[1], tile.size), True, "object_creation_seed")
    else:
        masks = tuple(rasterize_polygons((polygon,)) for polygon in local_polygons)
        object_masks = tuple(mask for mask in masks if bool(mask.any()))
        ground_truth = rasterize_polygons(
            tuple(polygon for polygon, mask in zip(local_polygons, masks) if bool(mask.any()))
        )
        seed = None
        if args.propagation_mode != "tracker-only":
            raise ValueError("--seed-scope all-in-tile requires --propagation-mode tracker-only")
    seeded_object_count = 1 if object_masks is None else len(object_masks)
    model_object_capacity = resolve_mask_seed_capacity(seeded_object_count)
    runtime = resolve_pinned_sam_runtime_provenance()
    session = SamSessionPlan(
        sequence_id="lta__private_mask_seed",
        session_index=0,
        frame_start=args.start_frame,
        frame_stop=args.start_frame + LTA_SESSION_FRAMES,
    )

    torch.cuda.set_device(args.device)
    predictor = None
    restore_sdpa = None
    measured = None
    resource = None
    active_error: BaseException | None = None
    memory = [cuda_snapshot(torch, args.device, "before_builder")]
    try:
        predictor = build_local_sam_predictor(
            bundle,
            device_id=args.device,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            conf=args.conf,
            weight_storage=args.weight_storage,
            max_num_objects=model_object_capacity,
            construction_device="meta",
        )
        constrained = configure_constrained_gpu_batches(predictor)
        restore_sdpa = install_sdpa_fallback()
        measured = _MeasuredPredictor(predictor, torch, args.device, memory)
        resource = decode_rgb_tile_frames(video, args.start_frame, tile)
        result = run_mask_seed_session(
            measured,
            predictor,
            resource=resource,
            session=session,
            prompt_frame=args.prompt_frame,
            ground_truth=ground_truth,
            seed=seed,
            object_masks=object_masks,
            conf=args.conf,
            propagation_mode=args.propagation_mode,
        )
        result["peak_allocated_mib"] = int(torch.cuda.max_memory_allocated(args.device) // (1024 * 1024))
        artifacts = write_strategy_artifacts(
            output,
            result=result,
            resource=resource,
            ground_truth=ground_truth,
            session=session,
            prompt_frame=args.prompt_frame,
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors: list[tuple[str, Exception]] = []
        if restore_sdpa is not None:
            try:
                restore_sdpa()
            except Exception as exc:
                cleanup_errors.append(("SDPA restoration", exc))
        if predictor is not None:
            shutdown = getattr(predictor, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    cleanup_errors.append(("predictor shutdown", exc))
        measured = None
        resource = None
        predictor = None
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            cleanup_errors.append(("CUDA garbage collection", exc))
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

    summary = {
        "status": "execution_complete",
        "execution_completed": True,
        "diagnostic_schema": "lta.private-mask-seed/1",
        "private_unstable_api": True,
        "propagation_mode": str(args.propagation_mode),
        "seed_scope": str(args.seed_scope),
        "seeded_object_count": result["seeded_object_count"],
        "model_object_capacity": model_object_capacity,
        "runtime": runtime,
        "model": {
            "bundle_root": str(bundle.root),
            "checkpoint_path": str(bundle.checkpoint_path),
            "model_version": bundle.model_version,
            "checkpoint_identity_sha256": bundle.checkpoint_identity_sha256,
        },
        "device": {
            "index": int(args.device),
            "name": torch.cuda.get_device_name(int(args.device)),
            "capability": list(torch.cuda.get_device_capability(int(args.device))),
        },
        "video": {
            "path": str(video),
            "source_width": int(width),
            "source_height": int(height),
        },
        "label": {
            "path": str(label),
            "sha256": str(label_digest),
            "encoded_index": int(args.exemplar_index),
            "zero_based_row": int(args.label_row),
        },
        "tile_xyxy": list(tile.xyxy),
        "session": {
            "sequence_id": session.sequence_id,
            "session_index": int(session.session_index),
            "frame_start": int(session.frame_start),
            "frame_stop_exclusive": int(session.frame_stop),
            "frame_count": int(session.frame_count),
            "prompt_frame_global": int(args.prompt_frame),
            "prompt_frame_local": int(args.prompt_frame) - int(session.frame_start),
        },
        "settings": {
            "conf": float(args.conf),
            "weight_storage": str(args.weight_storage),
            "construction_device": "meta",
            "use_fa3": False,
            "use_rope_real": False,
            "compile": False,
            "warm_up": False,
            "async_loading_frames": False,
            "offload_video_to_cpu_requested": True,
            "multiplex_count": int(LTA_MULTIPLEX_COUNT),
            "max_num_objects": int(model_object_capacity),
            "sdpa_backends": ["flash_attention", "efficient_attention", "math"],
        },
        "anchor_integrity_gate": {
            "minimum_per_object_iou": float(MASK_SEED_ANCHOR_IOU),
            "requires_exact_object_ids": True,
            "passed": bool(result["anchor_integrity_passed"]),
        },
        "diagnostic_propagation_gate": {
            "definition": (
                "anchor integrity passed and at least one non-anchor frame returned "
                "a nonempty mask"
            ),
            "minimum_non_anchor_active_frames": 1,
            "non_anchor_active_frames": result["non_anchor_active_frames"],
            "passed": bool(result["diagnostic_propagation_gate_passed"]),
            "quality_or_publication_claim": False,
        },
        "drop_stats_applicable": bool(result["drop_stats_applicable"]),
        "anchor_seed_metrics": result["final_metrics"],
        "anchor_seed_object_metrics": result["seed_object_metrics"],
        "propagation_response_count": result["propagation_response_count"],
        "propagation_active_frames": result["propagation_active_frames"],
        "anchor_preview_propagation_iou": result["anchor_preview_propagation_iou"],
        "anchor_propagation_object_metrics": result[
            "anchor_propagation_object_metrics"
        ],
        "anchor_expected_object_ids": result["anchor_expected_object_ids"],
        "anchor_returned_object_ids": result["anchor_returned_object_ids"],
        "peak_allocated_mib": result["peak_allocated_mib"],
        "constrained_batches": constrained,
        "artifacts": artifacts,
        "memory_events": memory,
    }
    path = output / "summary.json"
    summary["summary_path"] = str(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
