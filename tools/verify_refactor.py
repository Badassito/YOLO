"""Verify the package against the checked-in historical refactor inventory."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "volume_tta"
MANIFEST = PACKAGE / "_refactor_manifest.json"

# These definitions received small, intentional seam fixes after physical extraction.
INTENTIONALLY_CHANGED = {
    ("backprojection", "HybridBackprojectionQueue"),
    ("config", "build_argparser"),
    ("config", "resolve_backend_batches"),
    ("config", "resolve_backend_precisions"),
    ("cuda_backend", "_GpuWorkerRenderEngine"),
    ("cuda_backend", "_radial_slab_channel_renderer"),
    ("cuda_d1", "_nrrd_layer_key"),
    ("cuda_d1", "_nrrd_layer_name"),
    ("geometry", "ChannelFormattedFrameRenderer"),
    ("geometry", "channel_view_slice_index"),
    ("geometry", "make_dense_tile_channel_renderer"),
    ("geometry", "make_fullframe_channel_renderer"),
    ("geometry", "render_dense_tile_frame_for_job"),
    ("geometry", "render_fullframe_frame_for_job"),
    ("geometry", "write_aug_job_meta"),
    ("geometry", "write_dense_tile_job_meta"),
    ("inference", "cpu_retina_masks_enabled"),
    ("media", "abort_streaming_producers"),
    ("media", "decode_video_to_memmap_gray8_streaming"),
    ("media", "resize_volume_to_processing_cube_gray8_streaming"),
    ("outputs", "_publish_staged_file_atomically"),
    ("outputs", "_MemberParallelGzipPayloadWriter"),
    ("outputs", "_announce_nrrd_cpu_deflate_backend"),
    ("outputs", "_nrrd_gzip_executor"),
    ("outputs", "_nrrd_member_codec_candidates"),
    ("outputs", "_nrrd_member_codec_self_test"),
    ("outputs", "_nrrd_member_codec_spec"),
    ("outputs", "_open_nrrd_payload_writer"),
    ("outputs", "_select_nrrd_member_codec"),
    ("outputs", "_try_gpu_downbin_volume"),
    ("outputs", "_try_gpu_downbin_volume_on_device"),
    ("outputs", "NrrdLayerSink"),
    ("outputs", "write_binary_tiff_sequence_from_pattern"),
    ("outputs", "write_layer_nrrd_with_low_quality_mirrors"),
    ("outputs", "write_single_layer_nrrd_from_ref"),
    ("outputs", "write_view_images"),
    ("outputs", "write_yolo_labels_from_pattern"),
    ("outputs", "nrrd_gzip_compresslevel"),
    ("outputs", "nrrd_member_codec_requested"),
    ("pipeline", "main"),
    ("runtime", "_GpuWorkerAuxInterpolationPool"),
    ("runtime", "_materialize_worker_task_memfd_paths"),
    ("runtime", "copy_workspace_array"),
    ("runtime", "interpolate_view_volume_pass_maybe_process"),
    ("runtime", "interpolation_process_start_method"),
    ("runtime", "reset_runtime_state_for_new_run"),
    ("workers", "run_prediction_volume_in_worker"),
    ("topology", "_try_label_slices_stage_a_gpu"),
    ("topology", "build_slice_endpoint_seeds_from_label_volume"),
    ("interpolation", "SliceEndpointSeed"),
    ("interpolation", "NrrdLayerRef"),
    ("interpolation", "interpolate_view_volume_pass_inplace"),
    ("interpolation", "interpolation_planning_backend_name"),
    ("finalization", "assemble_view_volumes_and_projected_layers_fused"),
    ("assembly", "finalize_consolidated_tile_volume_for_parent"),
    ("assembly", "materialize_nrrd_view_layer"),
    ("geometry", "is_tilted_view"),
    ("outputs", "write_summary_file"),
    ("outputs", "nrrd_layer_output_suffix"),
}

# The public wrapper now owns the full-run cleanup boundary; the preserved orchestration
# body moved intact (plus reviewed v17.0.7 fixes) behind this private implementation name.
INTENTIONALLY_RENAMED_CHANGED = {
    ("pipeline", "main"): "_main_impl",
}

INTENTIONALLY_VERSIONED = {
    ("config", "451b35336c86c625bd71b77e55c8a09bef571c75405977484e6e0e6debadcd51"):
        "SCRIPT_VERSION",
    ("config", "bbaeec59e08232583950d10ce19229b162f82f5f41ec60afe2dff19fc2e9c6b2"):
        "SCRIPT_VERSION_COMPACT",
}

INTENTIONALLY_RELOCATED = {
    ("runtime", "70e22341666e8e63ad2a0a0239676cd85eb4aba6e256a55379ec7958cbc35799"):
        "workers",
    ("runtime", "a6ecc26570ba0d1a5feda101d96bfa587013159b1bf5370b5875aff9ed3ff212"):
        "workers",
    ("inference", "bf2ffc53f405ceff8a38bc5f846d09608a4ac82575a58d8938ceb0204d8bc99b"):
        "backprojection",
    ("inference", "fcc88a83030aa0dac3b506d345ee748d01c4cdc9e0e3b08aa9549cbfde0d44ca"):
        "backprojection",
}

# v17.0.11 removes definitions and state that had no caller or runtime effect.  Keep the
# immutable source inventory intact and account for each retired statement by its original
# digest, so adding a similarly named definition later cannot silently satisfy this audit.
INTENTIONALLY_REMOVED = {
    ("config", "0b77703bf375bcd802f74a77ca9009db17a87296bb829376ed2f30f368250243"):
        "OUTPUT_NRRD_PREFIX",
    ("config", "e9cbdca394845cae9bdb26ad2d5cdfd5dea831d31b29c8b305e97300328760ee"):
        "RADIAL_TEXTURE_VARIANT_LABEL",
    ("config", "813fa551257393b30cd4587ca2fdfa2de5cf42351be75a57f94c0e03f0b210ca"):
        "resolve_save_options",
    ("config", "071ba93675e9d91466da964da542560b322ea80c4498c011939bd24ebacca524"):
        "_parse_quantize_arg",
    ("finalization", "357c81e4223f2c6b6fd247ce2c44088ad228499def98cce9d3ab8ae430bfa5a0"):
        "assemble_views_concurrency",
    ("finalization", "7e3ee2b874386829aab5aa902979c279a4b412b6e835f3c55d92aba377867431"):
        "assemble_view_volume_from_projected_layers",
    ("geometry", "f294bcd6122f87aa1128cb47877d0a2761738d2b50fc91e304a6491d5afd17f0"):
        "tile_jobs_uniform_crop_shape",
    ("inference", "beac6387c6688fec98b2fc6023e8f034b6489e34bae549240bcbc139e24938eb"):
        "background_model_load_enabled",
    ("inference", "65033f136d739e98455671add94f47d951b82b708e7a02629c68916e5301deec"):
        "_canonical_single_device_token",
    ("inference", "820c3099703d78a02772d1e785c98008df67ecc0e18073787f43b43a8f0df783"):
        "parse_device_list",
    ("inference", "34e6bdedc1de53c986f30028efc59349d36b7d872296ca5fe7ad0c3b655d629a"):
        "is_cpu_device_list",
    ("inference", "7ec809ae495bc5b969690a8cbd4193a63cfb192b26a83dbc113229d7101b427f"):
        "resolve_retina_mask_processor",
    ("interpolation", "0f8e158500f0a81ec31578174c31fde27e072bcc4b4f54718f63f88a7d3b2f62"):
        "_component_records_directly_overlap",
    ("outputs", "bed6cab37b1b3c47b34c2204852d8e6d5d77d787c56193da32a91587b72dfc75"):
        "_NRRD_GZIP_EXECUTOR",
}

# The compiled overlap helper lived inside a larger top-level conditional.  Pin both the
# original inventory digest and the reviewed replacement digest so the verifier still
# authenticates every sibling kernel in that statement after the one dead helper is pruned.
INTENTIONALLY_PRUNED_REPLACEMENTS = {
    ("interpolation", "e2a3ab6f0b2bb8abfd8cb880317a197a446bd80d52fd49f7c4bc72608dbfe529"):
        (
            "1ccb9d3b22d1623694a7d8a0ef561e5981771359c8c7b86dc34e5d2d7611ed14",
            "_numba_blocks_overlap_any_kernel",
        ),
}


def digest(node: ast.AST) -> str:
    normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_path = ROOT / str(manifest["source"])
    # The checked-in manifest is the immutable provenance record. The user intentionally
    # retired the historical monolith after extraction; when a private archival copy is
    # present we still authenticate it, but its absence no longer disables verification.
    if source_path.is_file():
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_sha256 != str(manifest["source_sha256"]):
            raise RuntimeError(
                f"refactor source checksum mismatch: {source_sha256} != {manifest['source_sha256']}"
            )

    available: dict[str, Counter[str]] = {}
    trees: dict[str, ast.Module] = {}
    top_level: dict[str, list[ast.stmt]] = {}
    for module in {str(item["module"]) for item in manifest["statements"]}:
        tree = ast.parse(
            (PACKAGE / f"{module}.py").read_text(encoding="utf-8"),
            filename=str(PACKAGE / f"{module}.py"),
        )
        trees[module] = tree
        top_level[module] = list(tree.body)
        available[module] = Counter(digest(node) for node in tree.body)

    inventory_keys = {
        (str(item["module"]), str(item["sha256"]))
        for item in manifest["statements"]
    }
    untracked_pruning = sorted(
        (set(INTENTIONALLY_REMOVED) | set(INTENTIONALLY_PRUNED_REPLACEMENTS))
        - inventory_keys
    )
    if untracked_pruning:
        raise RuntimeError(
            f"reviewed pruning entries are absent from the immutable inventory: {untracked_pruning!r}"
        )

    retired_names = {
        (module, name)
        for (module, _statement_hash), name in INTENTIONALLY_REMOVED.items()
    }
    retired_names.update(
        (module, name)
        for (module, _statement_hash), (_replacement_hash, name)
        in INTENTIONALLY_PRUNED_REPLACEMENTS.items()
    )
    remaining_retired_names: list[tuple[str, str]] = []
    for module, name in sorted(retired_names):
        for node in ast.walk(trees[module]):
            declares_name = (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ) or (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == name
            )
            if declares_name:
                remaining_retired_names.append((module, name))
                break
    if remaining_retired_names:
        raise RuntimeError(
            f"reviewed dead bindings still exist: {remaining_retired_names!r}"
        )

    missing_pruned_replacements = [
        (module, replacement_hash)
        for (module, _original_hash), (replacement_hash, _name)
        in INTENTIONALLY_PRUNED_REPLACEMENTS.items()
        if available[module][replacement_hash] != 1
    ]
    if missing_pruned_replacements:
        raise RuntimeError(
            "missing or duplicate reviewed pruning replacements: "
            f"{missing_pruned_replacements!r}"
        )

    missing_changed = [
        (module, name)
        for module, name in sorted(INTENTIONALLY_CHANGED)
        if sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == INTENTIONALLY_RENAMED_CHANGED.get((module, name), name)
            for node in top_level[module]
        ) != 1
    ]
    if missing_changed:
        raise RuntimeError(f"missing or duplicate reviewed seam functions: {missing_changed!r}")
    for (module, public_name), implementation_name in INTENTIONALLY_RENAMED_CHANGED.items():
        public_matches = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == public_name
            for node in top_level[module]
        )
        if public_matches != 1:
            raise RuntimeError(
                f"missing or duplicate public wrapper for {module}.{implementation_name}: "
                f"{module}.{public_name}"
            )

    missing_versions: list[tuple[str, str]] = []
    for (module, _original_hash), variable_name in INTENTIONALLY_VERSIONED.items():
        matches = 0
        for node in top_level[module]:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            matches += sum(isinstance(target, ast.Name) and target.id == variable_name for target in targets)
        if matches != 1:
            missing_versions.append((module, variable_name))
    if missing_versions:
        raise RuntimeError(f"missing or duplicate version declarations: {missing_versions!r}")

    missing: list[dict[str, object]] = []
    preserved = 0
    changed = 0
    removed = 0
    for item in manifest["statements"]:
        module = str(item["module"])
        name = item.get("name")
        statement_hash = str(item["sha256"])
        inventory_key = (module, statement_hash)
        if inventory_key in INTENTIONALLY_REMOVED:
            removed += 1
            continue
        if inventory_key in INTENTIONALLY_PRUNED_REPLACEMENTS:
            replacement_hash, _removed_name = INTENTIONALLY_PRUNED_REPLACEMENTS[inventory_key]
            available[module][replacement_hash] -= 1
            changed += 1
            continue
        if (
            (module, name) in INTENTIONALLY_CHANGED
            or (module, statement_hash) in INTENTIONALLY_VERSIONED
        ):
            changed += 1
            continue
        destination = INTENTIONALLY_RELOCATED.get((module, statement_hash), module)
        if available[destination][statement_hash] < 1:
            missing.append(item)
            continue
        available[destination][statement_hash] -= 1
        preserved += 1

    if missing:
        raise RuntimeError(f"missing {len(missing)} preserved statements: {missing!r}")
    expected = int(manifest["statement_count"])
    if preserved + changed + removed != expected:
        raise RuntimeError(
            "statement accounting mismatch: "
            f"{preserved} preserved + {changed} changed + {removed} removed != {expected}"
        )
    print(
        f"accounted for {preserved} unchanged original top-level statements and "
        f"{changed} reviewed original-statement changes plus {removed} reviewed removals "
        f"({expected} original statements total)"
    )


if __name__ == "__main__":
    main()
