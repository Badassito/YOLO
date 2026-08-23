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
    ("geometry", "InMemoryYoloVolumeSource"),
    ("geometry", "PredictionVolumeRef"),
    ("geometry", "StreamingYoloVolumeSource"),
    ("geometry", "channel_view_slice_index"),
    ("geometry", "make_dense_tile_channel_renderer"),
    ("geometry", "make_fullframe_channel_renderer"),
    ("geometry", "make_in_memory_yolo_source"),
    ("geometry", "make_prediction_ref_yolo_source"),
    ("geometry", "materialize_dense_tile_prediction_volume_for_job"),
    ("geometry", "materialize_fullframe_prediction_volume_for_job"),
    ("geometry", "maybe_eager_stage_prediction_ref_on_gpu"),
    ("geometry", "render_dense_tile_frame_for_job"),
    ("geometry", "render_fullframe_frame_for_job"),
    ("geometry", "write_aug_job_meta"),
    ("geometry", "write_dense_tile_job_meta"),
    ("inference", "cpu_retina_masks_enabled"),
    ("inference", "PredictionAccumulationHandle"),
    ("inference", "_DeviceUnionAccumulator"),
    ("inference", "predict_in_memory_volume_and_accumulate"),
    ("inference", "predict_in_memory_volume_and_submit_accumulation"),
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
    ("runtime", "RuntimeTelemetry"),
    ("workers", "run_prediction_volume_in_worker"),
    ("workers", "_OpenVinoCpuSegmenter"),
    ("workers", "run_prediction_volume_in_openvino_worker"),
    ("topology", "_try_label_slices_stage_a_gpu"),
    ("topology", "build_slice_endpoint_seeds_from_label_volume"),
    ("interpolation", "SliceEndpointSeed"),
    ("interpolation", "NrrdLayerRef"),
    ("interpolation", "SliceBridgeRenderPlan"),
    ("interpolation", "_paint_linear_slice_bridge_plan_onto_slice"),
    ("interpolation", "_paste_local_mask_onto_slice"),
    ("interpolation", "_plan_slice_seed_bridges"),
    ("interpolation", "interpolate_view_volume_pass_inplace"),
    ("interpolation", "interpolation_planning_backend_name"),
    ("finalization", "assemble_view_volumes_and_projected_layers_fused"),
    ("finalization", "assemble_current_view_union_volume"),
    ("assembly", "finalize_consolidated_tile_volume_for_parent"),
    ("assembly", "materialize_nrrd_view_layer"),
    ("geometry", "is_tilted_view"),
    ("outputs", "write_summary_file"),
    ("outputs", "nrrd_layer_output_suffix"),
    ("cuda_backend", "GpuRenderedYoloSource"),
    ("cuda_backend", "GpuTileRenderedYoloSource"),
    ("cuda_backend", "_radial_slab_context_indices"),
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
    ("config", "7eeed39e30c270fc4e56bbef52e6bc94b6e61bce6599988450ac920bb180a67f"):
        "SCRIPT_BASENAME",
}

# Functions that need to call back into a higher architectural layer carry this marker
# immediately above an explicit function-local import.  Treat that narrow import seam as
# a reviewed AST change without weakening the historical inventory for the function body.
LOCAL_IMPORT_SEAM_MARKER = "# Local import keeps the package dependency graph acyclic."

# Each entry pins both the complete reviewed top-level definition and the exact marker-to-
# import associations inside it.  The second digest covers the import's relative offset,
# enclosing lexical scopes, and normalized ImportFrom AST.  Comments are absent from Python's
# AST, so pinning only the definition digest would still let a marker move to a different
# already-existing local import without review.
REVIEWED_LOCAL_IMPORT_SEAMS = {
    ("assembly", "prepare_view_volume_after_fullframe"): (
        "97fba54f3926ed644b98139bf880f6d37f711afd4f8b6ad2d15c064e394755fe",
        "ef76a565816d431f18d15e145b7a2c42ffd316df45905e9daee00b8b5fd26477",
    ),
    ("assembly", "finalize_consolidated_tile_volume_for_parent"): (
        "bc3ecd1d7d9f2d9e9f158a290d0e075bae85565b2b299cfe84f08b08ec7491a4",
        "c955712cc1202c0be55b529d57026f25537c1d43e0e9c692ad6cdc85b3b913f5",
    ),
    ("backprojection", "backproject_tilted_volume_to_volume"): (
        "7d3d558f67fad8df3dacb4fe385e7f4d2d600fe63a815c63cf414b9a8d66044a",
        "101a6fb2b4446cf0d71be4e025f1224cfecb863185f21e3482bfcfc9cf3a3231",
    ),
    ("cuda_backend", "_GpuWorkerRenderEngine"): (
        "0fd9ae9c9ad1205a1a3466d79b1c8e030bab0a5907b30da25d2da26e4d0437e2",
        "6af12722154b40f75aca9726a04dcfad36c02c9160737a4a39c5ab3697562e09",
    ),
    ("geometry", "GpuPrefetchingYoloSource"): (
        "d34ae87abc324d9aa32dd8906bad4d1ccb3cbab27bad863c4b1979404bfb5bc8",
        "b230c59c54aa3f8c1c0efe9ffd6c4d6172f49531e530ce0a02acfc252c19a35d",
    ),
    ("geometry", "gpu_input_staging_enabled"): (
        "3ac4bb523c36af4f98daf48f3153a3813cc9459f2aa879846459c4c3e3352e70",
        "27c0b26fbbf4d1ccfa8a5af862a09cce91b81ea32d281a1b5301c84d4eb2876e",
    ),
    ("geometry", "gpu_input_staging_preflight_reserve"): (
        "0fab02a75013aaff85c7b63328c518f3c2022e2de9fe8c5f64471b2a4cdde919",
        "0dab677f37a32eea1b3a0138dcf73c6b876b41093fee8d61af77c89abbeb9699",
    ),
    ("geometry", "maybe_wrap_source_with_gpu_input_staging"): (
        "496bf1060982040fb935f932af7624277ab7f4c8f8abe4f01f7864f1ab1f4213",
        "28dda3f8902c87b048214bf8d3dd28f3117785ea42503b66e20db91056c6c2e2",
    ),
    ("geometry", "ensure_ultralytics_accepts_in_memory_volume_source"): (
        "60badc92d2dc2b6667dd10e4d14364d82fb8496c819c2e6a523682eca0802030",
        "857b70aaccd5a89c0104cd8a7bb39fea7e92d30adc84025789fc7f03cfc81eb4",
    ),
    ("geometry", "_materialize_prediction_volume_from_renderer"): (
        "9f25cfa6cfa5031abf332f6001711bd58a9dfd344a831a40cfe90cb7c6069d49",
        "c307c54502fdd0c1e867c6f0b72e74e7664fa6d875764b18cb182b65fe04ebd1",
    ),
    ("inference", "infer_yolo_model_input_channels"): (
        "7e8d329f12766affd78ee938e593bb989e8535d157c375b143f4fbbeee1bec6c",
        "d0e25f9ae060e0c7d74bece86ae3befb49f781d4a703061bb311fcb2c8d4f410",
    ),
    ("inference", "predict_source_and_accumulate"): (
        "ade96e190b2d8e1dca2bf1e691e39e7cf2a5814008b4b98fcd88a86098485fdf",
        "ccf37e9656817910658d67d9c07dd114ce6eb29a820fda1bfa904ac246d955d8",
    ),
    ("inference", "predict_source_and_submit_accumulation"): (
        "a815666f85c4b0df61997937dea9279b32fe72b98afc1030727db1c3217c4838",
        "1ce8d95596c7f073801ec32a1245bb5751595f1b3d6b5a2884063a126b89efdf",
    ),
    ("interpolation", "SliceComponentTableCache"): (
        "fcb31853f671ae2dc0a8a7e9bada7e481186cd2b2d5c3369f803e850ff9bceab",
        "29f4bad74e6bec994ef8fa59ab290dcb079b0a5ae918342c108b472e66ab66ab",
    ),
    ("interpolation", "_find_slice_projection_candidates_numba"): (
        "cbeebf1855fd025de032082a83f62ed77b1120e6e16bdeba779c4f0badfc8e29",
        "7fbd6e38c3db9d3821d9622ed891b1b7fbfe0f838c27de910d0944c749cfdcda",
    ),
    ("interpolation", "_find_slice_projection_candidates_python"): (
        "2964a4a06f74b43fbdca643ab9663fd3332fa8f854a2ae69f3fae162a7775dc0",
        "94740a539572d09297cb90016159014fc849ececb6d5c529e605783041542bfd",
    ),
    ("interpolation", "_build_slice_endpoint_seeds"): (
        "fc5d6a613a8cd296ee5723e56650361859962addc085e130d3cb70d47946eb7e",
        "126545d0d25722c4df5918428643130e3c6a1eb639a0c61476fd7a259ad57cdf",
    ),
    ("interpolation", "interpolate_view_volume_pass_inplace"): (
        "13aa2ffa922299f34b8db3b23568f10715911298ca2a6bc8fe74abf6079b85a6",
        "0a585dbad86412327820dccb21e479e86057fa65bb0ae01a016b198238a3f661",
    ),
    ("interpolation", "RawBBoxMaskStore"): (
        "5e076700cb529dfbbb7a0e6bb7f7b582249b312252ab7510afe9da701402aa0a",
        "a19c94672d63a393b3a647ec73bba9da4791667f7b0082fa9f739ee56c50d16c",
    ),
    ("media", "resolve_radial_azimuth_angles"): (
        "938f099878fc06c1be11510b28e23a799ab12ff42f496cc4e54f0126fc80541f",
        "8d5afec87722d10ab7ef8bbc7606ce19d0bdae2fd6e68cf91fe3cd23ec5d77ca",
    ),
    ("runtime", "gpu_worker_default_seconds_per_frame"): (
        "8a1c116c3636a5c90ea5f13260fad70cd2360f8dc703fe079e5209f0f8e1948c",
        "efe66c8507863ddaabdad07859530397e9d25215132e7b9c854a6260bfe1281c",
    ),
    ("runtime", "gpu_worker_task_cost_key"): (
        "7a962f913cbd65ea9efcf961dc0c240d9b4ec53048f5422c071259ae1b6928a6",
        "bdd65ef45ec484d03fe3b3521df8ef368fde4d34eb9eeaa88529aedb4dd85082",
    ),
    ("runtime", "cpu_inference_supports_view"): (
        "832bf116476c2bd39211d1d7232ca1b6d18bbe7d084445c3caa5809bb284e0d0",
        "3c719dcf6dbaac2ea8151cbf0f55b27f848103316013a3aa0b26a8b24ba7439a",
    ),
    ("runtime", "cpu_inference_task_priority"): (
        "bad6fb69cdc53cbb9b104e739de56a99392093d5c8b16666d6687a9cdb133090",
        "50a97e164a4f6a098fbf7e773de161a7a1ad4a8ce05c9c37f286606a965df3bb",
    ),
    ("runtime", "_interpolation_process_entry"): (
        "169ad5a655a0fa423b523db1b94d02980586180992f506701a56baa0848eaaf6",
        "babab2eb1d231e00bab18c5e3d624b34d92401f4a0fadd98b03035b6d38e0775",
    ),
    ("runtime", "interpolate_view_volume_pass_maybe_process"): (
        "c33d75c4f4bfa4ed742665fd1b66d68e2ef60026cfcf4432f336fe892dfea740",
        "539803830c71cfc4b66cdf2c096504b9cf2d4e970aa94d8a5121c57ed15acd98",
    ),
    ("topology", "_try_label_slices_stage_a_gpu"): (
        "6dce9807e442982e580688a8466cb4076b7cf5d22db2599d943ac51b915aceff",
        "79f6cfde9d6e366582897499240cf9ea21081d92328d16c658d9113891e0004e",
    ),
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


def reviewed_local_import_seams(
    module: str,
    module_source: str,
    tree: ast.Module,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Validate and fingerprint every explicitly reviewed function-local import seam."""
    source_lines = module_source.splitlines()
    marker_lines: list[int] = []
    malformed_marker_lines: list[int] = []
    for line_number, line in enumerate(source_lines, start=1):
        if LOCAL_IMPORT_SEAM_MARKER not in line:
            continue
        if line.strip() != LOCAL_IMPORT_SEAM_MARKER:
            malformed_marker_lines.append(line_number)
        else:
            marker_lines.append(line_number)
    if malformed_marker_lines:
        raise RuntimeError(
            f"{module}: local-import seam marker must be the complete comment on lines "
            f"{malformed_marker_lines!r}"
        )

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    imports_by_line: dict[int, list[ast.ImportFrom]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports_by_line.setdefault(node.lineno, []).append(node)

    seams_by_top_level: dict[ast.stmt, list[tuple[int, str, str]]] = {}
    for marker_line in marker_lines:
        import_line = marker_line + 1
        import_nodes = imports_by_line.get(import_line, [])
        if len(import_nodes) != 1:
            raise RuntimeError(
                f"{module}:{marker_line}: local-import seam marker must be immediately "
                "followed by exactly one from-import"
            )
        import_node = import_nodes[0]
        if import_node.level < 1:
            raise RuntimeError(
                f"{module}:{import_line}: reviewed local-import seam must use a relative import"
            )

        marker_indent = source_lines[marker_line - 1][
            : len(source_lines[marker_line - 1])
            - len(source_lines[marker_line - 1].lstrip(" \t"))
        ]
        import_indent = source_lines[import_line - 1][
            : len(source_lines[import_line - 1])
            - len(source_lines[import_line - 1].lstrip(" \t"))
        ]
        if marker_indent != import_indent:
            raise RuntimeError(
                f"{module}:{marker_line}: local-import seam marker and import must have "
                "identical indentation"
            )

        ancestors: list[ast.AST] = []
        cursor: ast.AST = import_node
        while cursor in parents:
            cursor = parents[cursor]
            ancestors.append(cursor)
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ancestors):
            raise RuntimeError(
                f"{module}:{import_line}: reviewed import is not function-local"
            )
        top_level_nodes = [node for node in ancestors if parents.get(node) is tree]
        if len(top_level_nodes) != 1 or not isinstance(
            top_level_nodes[0],
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            raise RuntimeError(
                f"{module}:{import_line}: reviewed import must belong to one top-level definition"
            )
        top_level_node = top_level_nodes[0]
        lexical_scope = ".".join(
            node.name
            for node in reversed(ancestors)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        import_ast = ast.dump(import_node, annotate_fields=True, include_attributes=False)
        seams_by_top_level.setdefault(top_level_node, []).append(
            (import_node.lineno - top_level_node.lineno, lexical_scope, import_ast)
        )

    reviewed: dict[tuple[str, str], tuple[str, str]] = {}
    for top_level_node, seams in seams_by_top_level.items():
        assert isinstance(top_level_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        seam_payload = json.dumps(sorted(seams), separators=(",", ":"))
        reviewed[(module, top_level_node.name)] = (
            digest(top_level_node),
            hashlib.sha256(seam_payload.encode("utf-8")).hexdigest(),
        )
    return reviewed


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
    local_import_seams: dict[tuple[str, str], tuple[str, str]] = {}
    for module in {str(item["module"]) for item in manifest["statements"]}:
        module_path = PACKAGE / f"{module}.py"
        module_source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(module_source, filename=str(module_path))
        trees[module] = tree
        top_level[module] = list(tree.body)
        available[module] = Counter(digest(node) for node in tree.body)
        local_import_seams.update(reviewed_local_import_seams(module, module_source, tree))

    unexpected_local_import_seams = sorted(
        set(local_import_seams) - set(REVIEWED_LOCAL_IMPORT_SEAMS)
    )
    missing_local_import_seams = sorted(
        set(REVIEWED_LOCAL_IMPORT_SEAMS) - set(local_import_seams)
    )
    changed_local_import_seams = sorted(
        key
        for key in set(local_import_seams) & set(REVIEWED_LOCAL_IMPORT_SEAMS)
        if local_import_seams[key] != REVIEWED_LOCAL_IMPORT_SEAMS[key]
    )
    if unexpected_local_import_seams or missing_local_import_seams or changed_local_import_seams:
        raise RuntimeError(
            "local-import seam review mismatch: "
            f"unexpected={unexpected_local_import_seams!r}, "
            f"missing={missing_local_import_seams!r}, "
            f"changed={changed_local_import_seams!r}"
        )

    effective_changed = INTENTIONALLY_CHANGED | set(REVIEWED_LOCAL_IMPORT_SEAMS)

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
        for module, name in sorted(effective_changed)
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
            (module, name) in effective_changed
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
