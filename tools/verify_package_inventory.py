"""Verify current modules against the checked-in package statement inventory."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "XTA"
MANIFEST = PACKAGE / "_package_inventory.json"

# Pin the historical statements independently of the appended v20 review data.
# Formatting or review metadata may change; statement names/hashes may not.
IMMUTABLE_INVENTORY_STATEMENTS_SHA256 = '0c32fe9dcf8531e246e996cd276a010659f5564edb87f445b44dc7065147dfc5'

# These definitions have reviewed, intentional implementation changes.
INTENTIONALLY_CHANGED = {
    ("assembly", "materialize_interpolation_component_nrrd_view_layer"),
    ("assembly", "project_view_volume_to_orthogonal_volume"),
    ("backprojection", "_backproject_cartesian_azimuthal_generic"),
    ("backprojection", "_backproject_tilted_azimuthal_volume_to_volume"),
    ("backprojection", "backproject_azimuthal_volume_to_volume"),
    ("cuda_backend", "union_conf_volume_into_volume_inplace"),
    ("cuda_d1", "_d1_finalize_bitset_layer"),
    ("finalization", "_v14_apply_component_removal_plan"),
    ("finalization", "union_volume_into_volume"),
    ("inference", "cleanup_view_volume_after_prediction_inplace"),
    ("inference", "fill_view_volume_holes_2d_inplace"),
    ("inference", "fused_slice_cleanup_inplace"),
    ("interpolation", "IncrementalRawBBoxMaskStoreWriter"),
    ("interpolation", "PreparedViewResult"),
    ("interpolation", "_build_linear_slice_bridge_plan"),
    ("interpolation", "_component_record_mirrored_u"),
    ("interpolation", "_drain_volume_to_mmap"),
    ("interpolation", "_estimate_linear_slice_bridge_min_radius_from_plan"),
    ("interpolation", "_write_raw_bbox_payload_store"),
    ("interpolation", "materialize_raw_bbox_mask_store_workspace"),
    ("interpolation", "write_raw_bbox_mask_store"),
    ("media", "LazyProcessingCube"),
    ("media", "decode_video_to_memmap_gray8"),
    ("media", "resize_categorical_volume_to_processing_cube_uint8"),
    ("media", "resize_volume_t_axis_only_gray8_slab"),
    ("media", "resize_volume_to_processing_cube_gray8"),
    ("media", "restore_mask_volume_to_original_shape"),
    ("media", "should_resize_to_processing_cube"),
    ("outputs", "resize_binary_mask_volume_to_shape"),
    ("outputs", "resize_gray_volume_to_shape"),
    ("pipeline", "_main_impl"),
    ("runtime", "_ensure_process_backed_interpolation_volume"),
    ("runtime", "_mount_fstype_for_path"),
    ("runtime", "close_memmap_array"),
    ("runtime", "open_raw_store_payload_writer"),
    ("topology", "fill_3d_voids_inplace_streaming"),
    ("topology", "_adjacent_gid_pair_codes"),
    ("topology", "label_foreground_volume_streaming"),
    ("backprojection", "_MainProcessGpuStageCoordinator"),
    ("backprojection", "_ResidentTensorRTRingExecutor"),
    ("backprojection", "_azimuthal_resident_backproject_kernel"),
    ("backprojection", "_resident_trt_pipeline_acquire"),
    ("backprojection", "_try_resident_trt_ring_accumulate"),
    ("backprojection", "HybridBackprojectionQueue"),
    ("config", "build_argparser"),
    ("config", "resolve_save_request"),
    ("config", "resolve_backend_batches"),
    ("config", "resolve_backend_precisions"),
    ("cuda_backend", "_GpuWorkerRenderEngine"),
    ("cuda_backend", "_fused_direct_render_kernels"),
    ("cuda_backend", "_azimuthal_slab_channel_renderer"),
    ("cuda_d1", "_d1_backproject_kernels"),
    ("cuda_d1", "_d1_consume_device_union"),
    ("cuda_d1", "_D1WorkerViewState"),
    ("cuda_d1", "_shutdown_d1_worker_pipeline"),
    ("cuda_d1", "_d1_get_or_create_state"),
    ("cuda_d1", "_nrrd_layer_key"),
    ("cuda_d1", "_nrrd_layer_name"),
    ("cuda_d1", "tile_dense_worker_result_limit_bytes"),
    ("cuda_d1", "tile_dense_worker_result_limit_tasks"),
    ("finalization", "apply_keep_largest_objects_inplace"),
    ("geometry", "ChannelFormattedFrameRenderer"),
    ("geometry", "InMemoryYoloVolumeSource"),
    ("geometry", "PredictionVolumeRef"),
    ("geometry", "StreamingYoloVolumeSource"),
    ("geometry", "channel_view_slice_index"),
    ("geometry", "build_view_frame_cache"),
    ("geometry", "dense_tile_positions"),
    ("geometry", "gpu_input_staging_ahead_sources"),
    ("geometry", "get_azimuthal_sampler"),
    ("geometry", "make_dense_tile_channel_renderer"),
    ("geometry", "make_fullframe_channel_renderer"),
    ("geometry", "make_in_memory_yolo_source"),
    ("geometry", "make_prediction_ref_yolo_source"),
    ("geometry", "materialize_dense_tile_prediction_volume_for_job"),
    ("geometry", "materialize_fullframe_prediction_volume_for_job"),
    ("geometry", "maybe_eager_stage_prediction_ref_on_gpu"),
    ("geometry", "queued_streaming_source_cpu_warmup_slots"),
    ("geometry", "render_dense_tile_frame_for_job"),
    ("geometry", "render_fullframe_frame_for_job"),
    ("geometry", "streaming_prediction_source_prefetch_frames"),
    ("geometry", "streaming_prediction_source_workers"),
    ("geometry", "should_cache_view_frames"),
    ("geometry", "tile_crop_border_pixels"),
    ("geometry", "tile_parent_crop_window"),
    ("geometry", "write_aug_job_meta"),
    ("geometry", "write_dense_tile_job_meta"),
    ("geometry", "resolve_tile_configs"),
    ("geometry", "extract_azimuthal_slice_frame"),
    ("inference", "cpu_retina_masks_enabled"),
    ("inference", "PredictionAccumulationHandle"),
    ("inference", "_DeviceUnionAccumulator"),
    ("inference", "_ResidentGpuPipelineSlot"),
    ("inference", "_resident_mask_kernels"),
    ("inference", "_try_create_device_union_accumulator"),
    ("inference", "gpu_union_retirement_lane_count"),
    ("inference", "predict_in_memory_volume_and_accumulate"),
    ("inference", "predict_in_memory_volume_and_submit_accumulation"),
    ("media", "abort_streaming_producers"),
    ("media", "decode_video_to_memmap_gray8_streaming"),
    ("media", "processing_volume_mode"),
    ("media", "resize_volume_to_processing_cube_gray8_streaming"),
    ("media", "_cube_t_axis_resize_backend"),
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
    ("runtime", "_record_runtime_feature_gauges"),
    ("runtime", "_materialize_worker_task_memfd_paths"),
    ("runtime", "copy_workspace_array"),
    ("runtime", "choose_scratch_dir"),
    ("runtime", "interpolate_view_volume_pass_maybe_process"),
    ("runtime", "interpolation_process_start_method"),
    ("runtime", "reset_runtime_state_for_new_run"),
    ("runtime", "RuntimeTelemetry"),
    ("workers", "run_prediction_volume_in_worker"),
    ("workers", "_gpu_inference_worker_main"),
    ("workers", "_OpenVinoCpuSegmenter"),
    ("workers", "run_prediction_volume_in_openvino_worker"),
    ("topology", "_try_label_slices_stage_a_gpu"),
    ("topology", "build_slice_endpoint_seeds_from_label_volume"),
    ("interpolation", "SliceEndpointSeed"),
    ("interpolation", "NrrdLayerRef"),
    ("interpolation", "SliceBridgeRenderPlan"),
    ("interpolation", "SliceSeedBridgePlanResult"),
    ("interpolation", "_paint_linear_slice_bridge_plan_onto_slice"),
    ("interpolation", "_paste_local_mask_onto_slice"),
    ("interpolation", "_plan_slice_seed_bridges"),
    ("interpolation", "interpolate_view_volume_pass_inplace"),
    ("interpolation", "interpolation_planning_backend_name"),
    ("finalization", "assemble_view_volumes_and_projected_layers_fused"),
    ("finalization", "assemble_current_view_union_volume"),
    ("finalization", "_v1401_embedded_plane_ridges"),
    ("finalization", "_v14_plan_components_and_write_sparse_audits"),
    ("finalization", "_union_projected_layer_ref_into_volume"),
    ("assembly", "finalize_consolidated_tile_volume_for_parent"),
    ("assembly", "gate_tile_residual_against_parent_bridge"),
    ("assembly", "gate_tile_result_against_parent_mask"),
    ("assembly", "materialize_nrrd_view_layer"),
    ("assembly", "_try_apply_gaussian_smoothing_gpu_chunked_inplace"),
    ("assembly", "apply_gaussian_smoothing_inplace"),
    ("assembly", "spill_waiting_tile_result_to_raw_store"),
    ("geometry", "is_tilted_view"),
    ("outputs", "write_summary_file"),
    ("outputs", "nrrd_layer_output_suffix"),
    ("cuda_backend", "GpuRenderedYoloSource"),
    ("cuda_backend", "GpuTileRenderedYoloSource"),
    ("cuda_backend", "_azimuthal_slab_context_indices"),
    ("workspace", "v1613_d1_pipeline_active"),
    ("workspace", "v1613_fast_bundle_active"),
    ("workspace", "available_anon_work_bytes"),
}

# The public wrapper owns the full-run cleanup boundary and delegates to this private
# implementation name.
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

# Non-definition bindings whose reviewed contract changed after the immutable baseline.
# Pin the original statement digest and require the named replacement binding to remain
# unique, matching the version-binding treatment without misclassifying it as metadata.
INTENTIONALLY_CHANGED_BINDINGS = {
    ("config", "9a8d538aa3d7fa8f8d2cf55e46f6ac5b31ff4bc6b5823d7bf242954e9055c6df"):
        "SAVE_OPTION_TOKENS",
    ("geometry", "a4f438f50fb19a43076e30f5f4b09acf5b68ca6487f501f2c51f2e2b4bd86623"):
        "_AZIMUTHAL_SAMPLER_CACHE",
}

# Functions that need to call back into a higher architectural layer carry this marker
# immediately above an explicit function-local import.  Treat that narrow import seam as
# a reviewed AST change without weakening statement coverage for the function body.
LOCAL_IMPORT_SEAM_MARKER = "# Local import keeps the package dependency graph acyclic."

# Each entry pins both the complete reviewed top-level definition and the exact marker-to-
# import associations inside it.  The second digest covers the import's relative offset,
# enclosing lexical scopes, and normalized ImportFrom AST.  Comments are absent from Python's
# AST, so pinning only the definition digest would still let a marker move to a different
# already-existing local import without review.
REVIEWED_LOCAL_IMPORT_SEAMS = {
    ('assembly', 'prepare_view_volume_after_fullframe'): (
        '8c00e3949e69d5428f6bc946d11febbbd2763ce4acad285b378e9e960ea633f0',
        '322145486bb7f21b5f9ea590068bd323184072c7c7e9b2968473349950200ebe',
    ),
    ('assembly', 'finalize_consolidated_tile_volume_for_parent'): (
        'bc3ecd1d7d9f2d9e9f158a290d0e075bae85565b2b299cfe84f08b08ec7491a4',
        'c955712cc1202c0be55b529d57026f25537c1d43e0e9c692ad6cdc85b3b913f5',
    ),
    ('backprojection', 'backproject_tilted_volume_to_volume'): (
        'ae645bd53d368d0216171d90afe0bef10b7b1dd1dee80499b0eca6f0c8d49a9c',
        '101a6fb2b4446cf0d71be4e025f1224cfecb863185f21e3482bfcfc9cf3a3231',
    ),
    ('cuda_backend', '_GpuWorkerRenderEngine'): (
        'f3f2605d857dbbabe839f06c9a17bef2d89e6c47173fec85e8eb912f1005ced5',
        'e25d3d3f292164aee026e8e9c8d49caf37a2be6241c7c36f65aca52d31d3bbf5',
    ),
    ('geometry', 'GpuPrefetchingYoloSource'): (
        '1234b0ac1454e2d643e3688a7b94aab0510961d19ff4b0c91d4720f000568c17',
        'b230c59c54aa3f8c1c0efe9ffd6c4d6172f49531e530ce0a02acfc252c19a35d',
    ),
    ('geometry', 'gpu_input_staging_enabled'): (
        '3ac4bb523c36af4f98daf48f3153a3813cc9459f2aa879846459c4c3e3352e70',
        '27c0b26fbbf4d1ccfa8a5af862a09cce91b81ea32d281a1b5301c84d4eb2876e',
    ),
    ('geometry', 'gpu_input_staging_preflight_reserve'): (
        '0fab02a75013aaff85c7b63328c518f3c2022e2de9fe8c5f64471b2a4cdde919',
        '0dab677f37a32eea1b3a0138dcf73c6b876b41093fee8d61af77c89abbeb9699',
    ),
    ('geometry', 'maybe_wrap_source_with_gpu_input_staging'): (
        '496bf1060982040fb935f932af7624277ab7f4c8f8abe4f01f7864f1ab1f4213',
        '28dda3f8902c87b048214bf8d3dd28f3117785ea42503b66e20db91056c6c2e2',
    ),
    ('geometry', 'ensure_ultralytics_accepts_in_memory_volume_source'): (
        '60badc92d2dc2b6667dd10e4d14364d82fb8496c819c2e6a523682eca0802030',
        '857b70aaccd5a89c0104cd8a7bb39fea7e92d30adc84025789fc7f03cfc81eb4',
    ),
    ('geometry', '_materialize_prediction_volume_from_renderer'): (
        '134a732f44c0e24d20a4cd2bd779bb48b291088a4f56d0d71d009dc909de9da9',
        '4a8cb9169fcfbc4bf1fa1658615d8f3ae1377e8ac32cc1aeaa65707f37e22cd3',
    ),
    ('inference', 'infer_yolo_model_input_channels'): (
        '7e8d329f12766affd78ee938e593bb989e8535d157c375b143f4fbbeee1bec6c',
        'd0e25f9ae060e0c7d74bece86ae3befb49f781d4a703061bb311fcb2c8d4f410',
    ),
    ('inference', 'predict_source_and_accumulate'): (
        '6b3048e20bd43692dc08ba6ba9dd1e8db8ac6a6199cd60aff241a11a301f4fb4',
        'ccf37e9656817910658d67d9c07dd114ce6eb29a820fda1bfa904ac246d955d8',
    ),
    ('inference', 'predict_source_and_submit_accumulation'): (
        '025d1c3210be00b360d318306e71fcef8211d7ce116a43a77ca46a42dcead5eb',
        '1ce8d95596c7f073801ec32a1245bb5751595f1b3d6b5a2884063a126b89efdf',
    ),
    ('interpolation', 'SliceComponentTableCache'): (
        'fcb31853f671ae2dc0a8a7e9bada7e481186cd2b2d5c3369f803e850ff9bceab',
        '29f4bad74e6bec994ef8fa59ab290dcb079b0a5ae918342c108b472e66ab66ab',
    ),
    ('interpolation', '_find_slice_projection_candidates_numba'): (
        'cbeebf1855fd025de032082a83f62ed77b1120e6e16bdeba779c4f0badfc8e29',
        '7fbd6e38c3db9d3821d9622ed891b1b7fbfe0f838c27de910d0944c749cfdcda',
    ),
    ('interpolation', '_find_slice_projection_candidates_python'): (
        '2964a4a06f74b43fbdca643ab9663fd3332fa8f854a2ae69f3fae162a7775dc0',
        '94740a539572d09297cb90016159014fc849ececb6d5c529e605783041542bfd',
    ),
    ('interpolation', '_build_slice_endpoint_seeds'): (
        'bf29b06e72824fa78871fe502eb7ea79057bd352138dbf90bda55ca62f0d30ae',
        '126545d0d25722c4df5918428643130e3c6a1eb639a0c61476fd7a259ad57cdf',
    ),
    ('interpolation', 'interpolate_view_volume_pass_inplace'): (
        '0c91e9acd48ec2d9b7470e9b8143b0a77329ac59ecb01428288055595934e2b6',
        '0a585dbad86412327820dccb21e479e86057fa65bb0ae01a016b198238a3f661',
    ),
    ('interpolation', 'RawBBoxMaskStore'): (
        '5e076700cb529dfbbb7a0e6bb7f7b582249b312252ab7510afe9da701402aa0a',
        'a19c94672d63a393b3a647ec73bba9da4791667f7b0082fa9f739ee56c50d16c',
    ),
    ('media', 'resolve_azimuthal_azimuth_angles'): (
        'be46c0979af4d4395c6538c789df7ca43659c4c5f66b01175d0e9fafb7117087',
        '94f8c4ba510aa3756118ce2ae981a3ce25f3d6eebc81c2b096045efe004b6b12',
    ),
    ('runtime', 'gpu_worker_default_seconds_per_frame'): (
        'cbdc743efad682f4c852ac135af8a38a9c3a85103499dfdedd36f92dbe0618d6',
        '8417466843f56b5afeaeef3a2d20fd67a91a90b0355438581bb5b2eadb1f7623',
    ),
    ('runtime', 'gpu_worker_task_cost_key'): (
        'f98176aac67c05de805593dccc266ab9e6f595c2989a180d638cb2948f05f552',
        '96f2a949e6948895eba3a583fa9f3da197a532674e991b8f98b921a196bff5ea',
    ),
    ('runtime', 'cpu_inference_supports_view'): (
        'ef761cc4da5ee4ba113200b9985d925659139b97d4d8c67201f0c8ffb989aa2a',
        '2175b73531245efd35fd7b3ffef54a31b2eba271bb82f19d9a1f710796bd3c2f',
    ),
    ('runtime', 'cpu_inference_task_priority'): (
        'cff4f59a9287337a997965bbfb9a63ebc1c1c1bd252cf11318317736d347c144',
        '50a97e164a4f6a098fbf7e773de161a7a1ad4a8ce05c9c37f286606a965df3bb',
    ),
    ('runtime', '_interpolation_process_entry'): (
        '83c1a7c9b379a384e2d285231ec33dbc595870b09e793b62be7fc8baaf0bff91',
        'babab2eb1d231e00bab18c5e3d624b34d92401f4a0fadd98b03035b6d38e0775',
    ),
    ('runtime', 'interpolate_view_volume_pass_maybe_process'): (
        '00604305e8c8b8fab50881a1bc8a3e41b9265bd7a3d58515619527622e499f92',
        'fa8ba3a3af40efa0605c8b9e41e422281719adee141eeb6376b2e589eb88b213',
    ),
    ('topology', '_try_label_slices_stage_a_gpu'): (
        '6dce9807e442982e580688a8466cb4076b7cf5d22db2599d943ac51b915aceff',
        '79f6cfde9d6e366582897499240cf9ea21081d92328d16c658d9113891e0004e',
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

# Keep the baseline inventory intact and account for each retired statement by its baseline
# digest, so adding a similarly named definition later cannot silently satisfy this audit.
INTENTIONALLY_REMOVED = {
    ("interpolation", "3e5d65dbd592dcec65808a55789c42f16ae2685e2992c890415c1a049c8dd124"):
        "_component_record_to_local_canvas",
    ("interpolation", "512225d0daf2f7d71f236c27fae5effa7c0a410f1175366cabc0a1987718843d"):
        "_local_half_width_for_component_records",
    ("interpolation", "7a2a71d0dec9df268cbf9efe76cf84c6812c1c0d0ad5e5856838ae7a52196314"):
        "_component_to_local_canvas",
    ("interpolation", "bcf33c955845b6f1e36dcc21bb95ade865cd980493a2e2631b7e0245c8ab94ea"):
        "_local_half_width_for_components",
    ("runtime", "6f3557199f1b478a7f72c087dd15e2d28bdf5a49b6b4a8ff29707dc038b2cfb6"):
        "raw_store_memfd_enabled",
    ("runtime", "49fc4622c58ba272d947cb15e4790c1eca1ca8f3f0bed161ba4ef678137ee088"):
        "_create_memfd_backed_payload_path",
    ("runtime", "1bebc854b4c96d9f0f827c2d5df1b735fcd1ce404fe82841a2787ce01af879a5"):
        "flush_array",
    ("runtime", "1fcce8668dc1d71e4ea52d70cd57a9ab19eb0a97fc27b474cdf722be725b6a02"):
        "prediction_volume_build_flush_enabled",
    ("runtime", "dd953b74f4d66f1146464d5faa74a8e3ce2d664e7b7890f9071c9f6e21a9b003"):
        "prediction_hot_path_flush_enabled",
    ("config", "0b77703bf375bcd802f74a77ca9009db17a87296bb829376ed2f30f368250243"):
        "OUTPUT_NRRD_PREFIX",
    ("config", "4b5b1cd71ab26699413794dc1b3b0a1b1a7b91dbb917f0321bd4fda5fe2b95d8"):
        "LEGACY_OUTPUT_NRRD_PREFIX",
    ("config", "479a50756ba923fbe000d52aad5dea92511533a044b7903224de6715fd9301a7"):
        "variant_nrrd_stem",
    ("config", "e9cbdca394845cae9bdb26ad2d5cdfd5dea831d31b29c8b305e97300328760ee"):
        "AZIMUTHAL_TEXTURE_VARIANT_LABEL",
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
    ("runtime", "4ae29b5a9b0c7626a2e0f2e4daefd6fc1f9cb3d37a1d44888dc14e563306a144"):
        "scratch_shm_required_free_bytes",
    ("runtime", "2c5358a6d7546d61f2bb33b5bfe44152266dd35372e50c366636f611f221948d"):
        "_auto_shm_scratch_candidate",
}

# The compiled overlap helper lived inside a larger top-level conditional.  Pin both the
# baseline inventory digest and the reviewed replacement digest so the verifier still
# authenticates every sibling kernel in that statement after the one dead helper is pruned.
INTENTIONALLY_PRUNED_REPLACEMENTS = {
    ("interpolation", "e2a3ab6f0b2bb8abfd8cb880317a197a446bd80d52fd49f7c4bc72608dbfe529"):
        (
            "4e01060dae0bb88826bf2b113bcb0f5e9d10f2ade407861176dd712774bb430e",
            "_numba_blocks_overlap_any_kernel",
        ),
}

# v20 additions to formerly preserved definitions are pinned individually. These
# are semantic changes, never accepted by the mechanical Azimuthal rename map.
# Key: immutable (module, baseline AST digest). Value: current name/digest/reason.
REVIEWED_V20_STATEMENT_REPLACEMENTS = {
    ('geometry', 'c8e5256bd662acf22cc3768a96593fce8084353e0c54c2bb00f80c64d771075a'): (
        'ViewInfo',
        '698023677c765317066783bf66b23a0d618d2cc37b315ffe8148255c0124306a',
        'Distinct shell trajectory fields retain azimuthal metadata and provide explicit radius/patch coordinates.',
    ),
    ('geometry', '7032646b8c76f03c753731ba4ae6b573a3b2c999679b88c411b778834c1c0992'): (
        'get_view_infos',
        'f77b27e9a095317082509cb8bd88f5a98dd561d3818ae74a351340ed9511c116',
        'Compile shell requests after unchanged existing view-family order.',
    ),
    ('geometry', '7971d05259012d2615747c9fb8af8bd5e7454f1a354624341bf0d32fc37dad82'): (
        'get_view_frame_by_index',
        '9459aacb1f573c141b78224b6ca3be8596dabe93050f9e74574e04143cfca6d0',
        'Dispatch new radial trajectories through bounded native shell rendering after source readiness.',
    ),
    ('cuda_backend', 'c8972e4acbdde22a8fe221daae7d343900a8d4862710528e4643374ca19093b7'): (
        '_fused_preflight_family',
        '7543d6727bbe5978c32a0e2f978f9560c83beb092fa334e6b9b98aacd687f6b4',
        'Exclude shell views from fused kernels that only understand Cartesian and angular planes.',
    ),
    ('finalization', 'fd81cf9fc0a5587f79c19bbb789f2ee537872fa0bbbbeeda884375ee660b75c3'): (
        '_v14_sample_one_normal_section',
        '9f05a7abde1d13bd2fd201fd4f7cebd2c830c089178e6deac61901083a8347b4',
        'Use the physical radial_distance variable name; square-root and annulus arithmetic are unchanged.',
    ),
    ('interpolation', '225ccfafc2b01273762daff87a7b9e9d1b1aecaa5d29a0d1e65aabbc617822b8'): (
        '_view_uses_interpolation',
        'bef22390c1d652d4e07a88261acd5ea5d16685f4c11517e7aad08e87e46e2ff5',
        'Allow interpolation within independent radial radius stacks.',
    ),
    ('cuda_d1', 'c476ceeb33c77a59a0833f212f1a773eaf632c11819693121f46eb762742ebbc'): (
        '_d1_view_family_ids',
        'e9c75f246b0b9b6102d6758f0c917ef9ea0de1a1a997bfd7716b047e74ff43df',
        'Reject shell views at the incompatible D1 kernel boundary; native union projection handles them.',
    ),
    ('inference', '58b9a7e4a8df5eba9cd5a011380d1baae25576338d2ecad05669ba8f3882a571'): (
        '_build_direct_device_compacted_payload',
        '01022e9cde7573fbd0883364c2aefde85cce82dffa0020cb4962cdb815f02324',
        'Match the generic compact CUDA kernel eight-argument signature with image height, width and null optional bounding boxes; required by shell inference.',
    ),
    ('geometry', 'e0203b6d1f34c8f3463b5a344c6e7b40a22d6fe2ba67552bf9323b20ea9b59f9'): (
        'view_output_token',
        'd4b65eef0239cccad875fc7906131975aea307b033653b6d982c4a864a00002c',
        'Give upright and tilted shell patches distinct filename-safe Radial output tokens retaining patch indices.',
    ),
    ('runtime', '105ed98db9ab4f6929a3244915a10ed1a3c14104f15068420e44bb9c89fa5b50'): (
        '_sched_setaffinity_all_threads',
        '0f11d5b4d48f7454adcb92d28c0cd2872e50481f0a1556c33f3405783afec7f4',
        'Dispatch Windows to a verified process/thread affinity implementation with rollback; preserve the Linux affinity implementation.',
    ),
    ('inference', '9931df03e8cd8b9142d3ab7cae0b896ed054b3648fd9700fa2fe7149fe2be16b'): (
        '_split_segmentation_backend_outputs',
        '1ca4b7224262330ee4bc332ca2f16571a87fdb0abbde91321edb17ce60a38ce7',
        'Accept the measured current Ultralytics PyTorch ((head, prototype), auxiliary dict) segmentation layout while retaining flat exported and legacy tuple outputs; reject unsupported tensor dimensions.',
    ),
}


def stable_ast_dump(node: ast.AST) -> str:
    """Serialize an AST without Python 3.13's default empty-field elision."""
    dump_options = {
        "annotate_fields": True,
        "include_attributes": False,
    }
    if "show_empty" in inspect.signature(ast.dump).parameters:
        dump_options["show_empty"] = True
    return ast.dump(node, **dump_options)


def digest(node: ast.AST) -> str:
    normalized = stable_ast_dump(node)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def current_baseline_name(name: object) -> object:
    """Map historical declaration names only; never normalize current source ASTs.

    The old Radial family became Azimuthal in v20. New Radial shell definitions
    must not satisfy those historical statements merely by reusing their names.
    """
    if not isinstance(name, str):
        return name
    return name.replace('radial', 'azimuthal').replace('Radial', 'Azimuthal').replace('RADIAL', 'AZIMUTHAL')


def azimuthal_rename_replacements(manifest: dict[str, object]) -> dict[tuple[str, str], str]:
    """Validate exact baseline-to-rename AST hash pairs, retaining all original hashes."""
    statements = manifest['statements']
    baseline = {(str(item['module']), str(item['sha256'])): item for item in statements}
    review = manifest.get('v20_azimuthal_rename', {})
    replacements = {}
    for item in review.get('statements', []):
        key = (str(item['module']), str(item['baseline_sha256']))
        if key not in baseline:
            raise RuntimeError(f'Azimuthal rename references an absent immutable statement: {key!r}')
        if key in replacements:
            raise RuntimeError(f'duplicate Azimuthal rename review: {key!r}')
        original_name = baseline[key].get('name')
        if item.get('baseline_name') != original_name or item.get('current_name') != current_baseline_name(original_name):
            raise RuntimeError(f'Azimuthal rename declaration identity mismatch: {key!r}')
        replacement = str(item['renamed_sha256'])
        if len(replacement) != 64 or any(character not in '0123456789abcdef' for character in replacement):
            raise RuntimeError(f'invalid reviewed Azimuthal AST digest: {key!r}')
        replacements[key] = replacement
    return replacements


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
        import_ast = stable_ast_dump(import_node)
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
    baseline_digest = hashlib.sha256(json.dumps(
        manifest['statements'], sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    if baseline_digest != IMMUTABLE_INVENTORY_STATEMENTS_SHA256:
        raise RuntimeError('immutable inventory digest mismatch; retain historical statement records and add explicit reviews')
    rename_replacements = azimuthal_rename_replacements(manifest)

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
    untracked_changed_bindings = sorted(
        set(INTENTIONALLY_CHANGED_BINDINGS) - inventory_keys
    )
    if untracked_changed_bindings:
        raise RuntimeError(
            "reviewed changed-binding entries are absent from the immutable inventory: "
            f"{untracked_changed_bindings!r}"
        )
    untracked_v20 = sorted(set(REVIEWED_V20_STATEMENT_REPLACEMENTS) - inventory_keys)
    if untracked_v20:
        raise RuntimeError(f'v20 replacement reviews are absent from the immutable inventory: {untracked_v20!r}')
    for (module, _baseline_hash), (name, replacement_hash, reason) in REVIEWED_V20_STATEMENT_REPLACEMENTS.items():
        matches = [node for node in top_level[module] if getattr(node, 'name', None) == name]
        if not reason or len(matches) != 1 or digest(matches[0]) != replacement_hash:
            raise RuntimeError(f'v20 reviewed definition changed or is missing: {module}.{name}')
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
        for (module, _baseline_hash), (replacement_hash, _name)
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
    for (module, _baseline_hash), variable_name in INTENTIONALLY_VERSIONED.items():
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

    missing_changed_bindings: list[tuple[str, str]] = []
    for (module, _baseline_hash), variable_name in INTENTIONALLY_CHANGED_BINDINGS.items():
        matches = 0
        for node in top_level[module]:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            matches += sum(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in targets
            )
        if matches != 1:
            missing_changed_bindings.append((module, variable_name))
    if missing_changed_bindings:
        raise RuntimeError(
            "missing or duplicate reviewed changed bindings: "
            f"{missing_changed_bindings!r}"
        )

    missing: list[dict[str, object]] = []
    preserved = 0
    changed = 0
    removed = 0
    for item in manifest["statements"]:
        module = str(item["module"])
        name = current_baseline_name(item.get("name"))
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
        if inventory_key in REVIEWED_V20_STATEMENT_REPLACEMENTS:
            _name, replacement_hash, _reason = REVIEWED_V20_STATEMENT_REPLACEMENTS[inventory_key]
            available[module][replacement_hash] -= 1
            changed += 1
            continue
        if (
            (module, name) in effective_changed
            or (module, statement_hash) in INTENTIONALLY_VERSIONED
            or inventory_key in INTENTIONALLY_CHANGED_BINDINGS
        ):
            changed += 1
            continue
        destination = INTENTIONALLY_RELOCATED.get((module, statement_hash), module)
        expected_hash = rename_replacements.get(inventory_key, statement_hash)
        if available[destination][expected_hash] < 1:
            missing.append(item)
            continue
        available[destination][expected_hash] -= 1
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
        "package inventory verified: "
        f"preserved={preserved}, reviewed_changes={changed}, reviewed_removals={removed}"
    )


if __name__ == "__main__":
    main()
