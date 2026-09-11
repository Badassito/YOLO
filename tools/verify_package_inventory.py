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

# The v21 appendix records exact reviewed additions and supersessions. Keep the
# immutable statement inventory and all v20 pins intact; only this explicit
# release review may replace their effective current-definition digests.
REVIEWED_V21_SHA256 = '47204e5024f23a6e8db72c215fd0dadf9e46a5bfa49aef50056add8ffbfc15e9'

# The patch review supersedes effective v21/v20 digests without rewriting any
# earlier release records. Every changed definition and top-level binding is
# independently checked against this authenticated appendix.
REVIEWED_V21_0_1_SHA256 = '63b96f6742d72dc7f79b9b7cd130eee93a6e9530249e40bf0c3f697c2389fc6a'
REVIEWED_V21_0_2_SHA256 = 'f33ab49e331d8d175194c8d70919e7e398a399f39e1503c891b969c260d60a35'
REVIEWED_V21_0_3_SHA256 = '9d09663dab895ff2c9da78175b5e5acbf6a6bd96adfcff66a441a2c97ea90eeb'
REVIEWED_V21_0_4_SHA256 = '8c8875883d5568194c7a709023ae50061b2a7b6c8a04f5fd48bc3fc5bd24cec7'
REVIEWED_V21_0_5_SHA256 = '9078a719026147d7af6b990634090f78a0874ba1297f49f89e062700f791a9e3'
REVIEWED_V21_0_6_SHA256 = 'fc3c2ca5ab0af6257ee9b4fd3b4bf9f66a9c669007a87377a9d7d47cb13858c7'
REVIEWED_V21_1_SHA256 = '534ad7dabb3bccbbf95883fa7cad678081e85f4efa1b5e851781b173da017f37'
# This method was previously covered by the immutable full Radial module and
# class pins. Name its exact pre-v21.0.6 AST before reviewing the upload change.
REVIEWED_PRESERVED_RADIAL_UPLOAD_SHA256 = '5f12562dafcb991f93b1702b8976e33a5117e4e01de357c3cb99c98afe52599a'

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
        'd52de78c1ba5464c0fb4b0234b96c9c0cdf430e93100974defce364d55001721',
        'c9595db954da98f5df4678a2bbf2416a70dc432e38c05097ed3396e6d0c823f6',
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
        '369e578af8e515cc079c738f58d91518a06a6e834e1f55d74533dffad1ea4d35',
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
        '28eec86038df26fe016c533232272e809cf340c62a8acc83a262fac84d190308',
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
    ('interpolation', '7f10eb5c476c42205a2aa865d3465c1ebabcad7a5a7d2a7042f4bd0d00c27187'): (
        '_raw_store_chunks_cache_key',
        'b8948f64f81c29198edc73b608230a46a89e633957a2074ae88ddf6d0334fd30',
        'Keep logical layer-path cache identities distinct across equally named memfds and stable through backing-descriptor retirement; retain per-layer shared mmap refcounts.',
    ),

    ('outputs', '9a9119faff78922a2fb2f826ef0ee290c9612194bb6ca984490a77420bede185'): (
        '_write_one_decomposed_nrrd_layer_payload',
        'c5ff52a5c1982a4eed513b1c35a01fa9fd385313515ac03049a394ab39291984',
        'Stream native bbox row bands and cached zero spans only for software member writers with no dense-block observer; retain restored, dense-observer and hardware routes.',
    ),

    ('cuda_d1', 'bb0b1ecf37d4c52647a15defe499c22667663ca95fb7de4ec6cfbf0021143061'): (
        '_d1_submit_publication',
        '5c92d44692a1f8910a8b8e332c06102828089a552de95881345de07d23d7f4d8',
        'Reuse bounded source-bitset publication with explicit native-shell provenance while preserving legacy output semantics.',
    ),
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
    ('backprojection', '5530061ab3afe471577e68c8c7466e0147b9da5ac845fe5f93cb8ddfe16fff69'): (
        '_ResidentTensorRTRingExecutor',
        '3645e9c5a49b9fd9f74c3fa325f4d9d6f2ba872bbbb6d7de10f9d5a848179041',
        'Retain idle TensorRT ring buffers across radial generic tasks; fence context handoffs, restore/rebind tensor addresses, and discard/recapture the changed borrowed-context inference graph before reuse.',
    ),
    ('backprojection', '9b514b1cfc81798ca0b8ac84760457aa09df110e5294af349e7b9743c8871671'): (
        '_resident_trt_pipeline_acquire',
        'f13530361829bad0e0f79b7db8cd47cd838b002bd97837b30a9ef12f6da87a14',
        'Resume retained ring contexts and track source renderer/volume identity before reuse.',
    ),
    ('backprojection', '7a65755d524b14876ed83f1b64ebcfabe777ba3690f855b09c84e267a5924509'): (
        '_try_resident_trt_ring_accumulate',
        'a19f4eb0d7c6dc7395e792abd2934b67d9ecc27f8a35aba35d8a75e0763c9d08',
        'Soft-suspend compatible radial generic tasks and run all-view fused preflight only after single-channel static TensorRT admission, before borrowing context bindings.',
    ),
    ('workers', 'f6c091cc8da972d22cd13d9135cc862ecbcc92b2c9ccca540d48d8a7b6eeb794'): (
        'run_prediction_volume_in_worker',
        '6dd4ac73111d24f25196e3442378ab5a434eb44dee84c6fb4fc88cdfff6fad2b',
        'Avoid irrelevant eager fused preflight for generic paths and emit flushed per-family render/result provenance, including CPU fallback.',
    ),
    ('assembly', 'fbe3c9b679863c1c1e49085f5e3b98ca730c8bc2f6d96979ed390a7a2b190a05'): (
        'project_view_volume_to_orthogonal_volume',
        '75a39a1f829c04ed66d55c9da77ebcfc41df8a5bd12c1690874f16c100e5975e',
        'Forward existing exact per-shell mask bounding boxes into the factored Radial projector without changing other view dispatch.',
    ),
    ('pipeline', 'a2c4a36ea39c43fb451b6667bdb8aa0507d56f849d4d7b3768dd46476af98fc2'): (
        '_main_impl',
        '60496d3e8cc73cceb8d0608c33dfb7c6510671379bfd07fab7cf70ddba47e6a4',
        'Authorize skipped non-interpolated drain copies only when component-ref retirement is active, tiling is absent and temporary artifacts are not retained; existing terminal ownership retires the original.',
    ),
    ('assembly', '43e838c80f5d1abeb35369a50161eb5363f95cfaa560d9b6f7cf437dc1c20567'): (
        'materialize_nrrd_view_layer',
        '487f1f3d7a8af4456500621fb6e18ccf2b791967b93851267b12b23d0eeb4f81',
        'Fatal projection failures abort/discard incomplete CPU sink stores and raw scratch without retry, including fatal failures during a transactional dense retry; preserve the original fatal exception and quarantined GPU owners.',
    ),
    ('interpolation', '7780e8aa5a241f9942c17c57e60f3acb446da4da3fb7396dd8bd01ddcd146b17'): (
        'IncrementalRawBBoxMaskStoreWriter',
        'fc3eeba9edd8e3982f4cc2c7ac90f94a0d5d8f80ddaab98acef7f24d3a3cba4c',
        'Validate complete encoded raw/packed record addressing and padding before atomic reservation; append owned payload synchronously without rescanning pixels, preserve failure invalidation, and provide serialized positional-write fallback on Windows.',
    ),
}


# New v20 helpers are authenticated separately from historical accounting.
# Kernel source is a string constant inside its factory's AST, so its digest
# covers the actual CUDA arithmetic as well as Python compilation/fallback logic.
REVIEWED_V20_ADDED_DEFINITIONS = {
    ('interpolation', '_raw_store_chunks_cache_key'): (
        'b8948f64f81c29198edc73b608230a46a89e633957a2074ae88ddf6d0334fd30',
        'Keep logical layer-path cache identities distinct across equally named memfds and stable through backing-descriptor retirement; retain per-layer shared mmap refcounts.',
    ),

    ('nrrd_spans', 'canonical_zero_member'): (
        'c1c7188458aa910ccf03590a1f551763090a650b3465d63629cdf5b843e664a5',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('nrrd_spans', 'stream_native_crop_spans'): (
        '9eeda37617a222f06413c78a3d15d45c1a0040876ded2da1cec031d12fc84011',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('outputs', '_MemberParallelGzipPayloadWriter'): (
        'f18fc29b7bb1c13a73e8cbc8260ca228561bdc6c611447347cc1ab9df8426d7d',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('outputs', '_write_one_decomposed_nrrd_layer_payload'): (
        'c5ff52a5c1982a4eed513b1c35a01fa9fd385313515ac03049a394ab39291984',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('publication_memory', 'publication_output_reserve'): (
        'de3daa4868896f867ad2a826766d692c70e92ddfe1b5a71c29d29af0a8c408e1',
        'Reserve actual gzip windows, mirror canvases and global compressed spools before retained RAM admission.',
    ),

    ('cuda_d1', '_d1_get_or_create_state'): (
        '37cff0c7bc9c3e636a5fe68f5bc75fd180835667ab091371dce8a132db5ce6cf',
        'Publish exact packed source crops without dense expansion; carry bounded parent RAM grants and preserve NumPy/raw fallbacks.',
    ),
    ('interpolation', 'IncrementalRawBBoxMaskStoreWriter'): (
        'fc3eeba9edd8e3982f4cc2c7ac90f94a0d5d8f80ddaab98acef7f24d3a3cba4c',
        'Support parent-owned RAM payloads with private prepublication disk spill, bounded copying and binary descriptor writes.',
    ),
    ('packed_publication', 'PackedOwnerCrop'): (
        'a23fcab310937aecd6e0bd5a10bbcf42efb49da674ac56c1ff1faf217a56b8ac',
        'Direct source-bitset bbox/count and row-packbits encoding with exact addressing, padding and bounded output blocks.',
    ),
    ('packed_publication', 'encode_owner_packed_block'): (
        '5584fce731c199ad90439018360b8f346a04ac6f528e68a4552973a62b498d0d',
        'Direct source-bitset bbox/count and row-packbits encoding with exact addressing, padding and bounded output blocks.',
    ),
    ('publication_memory', 'plan_native_publication_memory'): (
        '3fc6e04c0fee91391d33b66d6f0f5f500894044d93740899513f7938a0c98420',
        'Pre-dispatch worst-case retained-layer admission, physical/cgroup headroom, future-work reserve, parent descriptor lifetime and disk spill policy.',
    ),
    ('publication_memory', 'publication_ram_headroom'): (
        'c019bc77c877709c2106e25116cb8cc47c3ff255ce4f53613f232dc268fc4ea3',
        'Pre-dispatch worst-case retained-layer admission, physical/cgroup headroom, future-work reserve, parent descriptor lifetime and disk spill policy.',
    ),
    ('publication_memory', 'retained_payload_plan'): (
        '7a06a971b4815a9fbc9777cae369dc78d724206725e677297ece64bd8cd76555',
        'Pre-dispatch worst-case retained-layer admission, physical/cgroup headroom, future-work reserve, parent descriptor lifetime and disk spill policy.',
    ),
    ('topology', '_compiled_adjacent_gid_pair_codes'): (
        '5f6698213b323cc9b2cfa6b4ae4846b328efc7d0ac17cf71267e11b8a32f80df',
        'Use exact bounded row-run intersections before the retained pixel-hash adjacency fallback.',
    ),
    ('topology_runs', '_run_adjacent_pair_codes'): (
        '68093630d4d287bc4d316feb8b30cefdd752275b812588f37b60091546fe3b1c',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),
    ('topology_runs', 'run_adjacent_pair_codes'): (
        '2e3bf434f38c21856f794382ea355c18a76a39580bd070f422401a469a8e8619',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),

    ('cylindrical_owner', '_bucket_shell_pixels_numpy'): (
        'f247ed177d854a52dad612649a85d5ee3f19d81f427e8eb19b2acb9edb586f68',
        'Stable vectorized shell buckets preserve exact pixel order when optional JIT compilation is unavailable.',
    ),
    ('cylindrical_owner', 'shutdown_radial_owners'): (
        '934300c210658758a74c761fa460398ee226e7b4c684e774afddb57c4cec08db',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'preflight_radial_owner'): (
        'ddb926a9b0dbd536691f1bf5ac15f321a1225a03625cfbd8d11f2d368648b5db',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'active_radial_owners'): (
        '24f66e6f282c5e1990c14a45130c63e815ecb418062bc96b3dd698c5f7e436a3',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'consume_radial_device_union'): (
        'd6a6f408781926417ded5fe2eb809e94fb9d2a23fe7dfe41dcbb1f26c38e9b88',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'RadialOwner'): (
        '710eb39c851132e4d7865a03c5a6bebb4f0b2dd10b84d17dd39277620e1a4042',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', '_bucket_shell_pixels'): (
        '234cadc5b2e81fcaacd479e6c8054f9e82243bb4e69e6f0cec30bb83ee19e1e1',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'is_radial_owner_task'): (
        'ef6221af1b8063ddd35778f54a66801ed2fc1c5abd0617554c84d75d8fe34755',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'radial_owner_eligible'): (
        '4b3759b7a8f350f77087dec613f5d21b13cb5bb7d7a633d86cd743bb80b19478',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'radial_runtime_provenance'): (
        'f84efc29da511d5b2cad028396a7fb1aca93973c381393de1c96b14862ae4c16',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'radial_owner_enabled'): (
        '37fe66f517ab6dff9ff2d6a1f60ac12b3ed1865bb406ce7eb58416a3bc5eabb3',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'DeviceOnlyRadialTarget'): (
        'cf86365ca84e4b8aad65f863ae1f87477a637df9cc142e45a657cad8ac799801',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('tta_scheduler', 'TtaScheduler'): (
        '596f4ca180ab0586b60731ff0a6e04d7f3a905ddf6c6d5198d284d57a4f9110d',
        'Keep native radial tasks on one owner even when the multi-owner experiment is requested.',
    ),
    ('cuda_d1', '_d1_submit_publication'): (
        '5c92d44692a1f8910a8b8e332c06102828089a552de95881345de07d23d7f4d8',
        'Reuse bounded source-bitset publication with explicit native-shell provenance while preserving legacy output semantics.',
    ),
    ('cuda_d1', '_d1_finalize_bitset_layer'): (
        '44ef5a4b81c1630ca48eee683bf10622e49b8cade7ebedff33db8daf867c0a43',
        'Reuse bounded source-bitset publication with explicit native-shell provenance while preserving legacy output semantics.',
    ),
    ('inference', '_prediction_accumulation_target'): (
        '9adaefe94493deba1ad8a04d0d7a191f167507576ccb48a307aab18cf7cba858',
        'Support shape-only native device results, reject duplicate/missing radius results and avoid host cleanup on device-only frames.',
    ),
    ('inference', 'predict_source_and_accumulate'): (
        '28eec86038df26fe016c533232272e809cf340c62a8acc83a262fac84d190308',
        'Support shape-only native device results, reject duplicate/missing radius results and avoid host cleanup on device-only frames.',
    ),
    ('workers', '_gpu_inference_worker_main'): (
        'fe54c592cd78e1e67dedca3f736a934900a77bb95531421191c2d32731b0b0bb',
        'Dispatch native shell consumers without old D1 geometry/proto cleanup; retain native-owner retirement barriers and fatal CUDA ownership.',
    ),
    ('workers', '_release_gpu_worker_inference_assets'): (
        'cc82e02acb0e19f80792e32e4b516c0063ea28b706d3e2b1562cb01456816a7d',
        'Dispatch native shell consumers without old D1 geometry/proto cleanup; retain native-owner retirement barriers and fatal CUDA ownership.',
    ),
    ('workers', 'run_prediction_volume_in_worker'): (
        '6dd4ac73111d24f25196e3442378ab5a434eb44dee84c6fb4fc88cdfff6fad2b',
        'Dispatch native shell consumers without old D1 geometry/proto cleanup; retain native-owner retirement barriers and fatal CUDA ownership.',
    ),
    ('pipeline', '_main_impl'): (
        '60496d3e8cc73cceb8d0608c33dfb7c6510671379bfd07fab7cf70ddba47e6a4',
        'Route eligible shell views through the native owner contract, preserve compatibility and empty-layer policy, and log effective execution provenance.',
    ),
    ('cylindrical_cuda_projection', '_pack_radial_source_block'): (
        '8af8b8387c7a92773486ef7eb7decdae0f8df0bb960f65c678490055e2129b4f',
        'Copy exact cropped uint8 source intervals across shells and row boundaries using validated integer addresses and bounded output ownership.',
    ),
    ('cylindrical_projection', '_build_radial_plane_plan_reference'): (
        'c6d7097429247bd42b40ebee82506a068f2cd934f2aeba7e930671f9fa31c557',
        'Retain the original two-pass CSR assembly as an exact numerical reference for qualification.',
    ),
    ('cuda_backend', 'radial_native_kernel_enabled'): (
        '5fb5888625044f859e03b198d29f1f06f1180ecf2c6b7fb3c617f3dc6fbda0ad',
        'Explicit opt-out for the scalar native radial CUDA renderer; default enablement retains a logged Torch fallback.',
    ),
    ('cuda_backend', '_radial_native_kernels'): (
        'd452ff9b1a55bb4ce7d57da2830fb3c6570601e71b7b018bd992e23113e2e5bd',
        'Compile and cache the bounded scalar-geometry radial CUDA sampler; preserve float32 accumulation/uint8 rounding, deferred logical-T values and zero-extended taps.',
    ),
    ('backprojection', '_resident_trt_pipeline_suspend_for_radial'): (
        'fb076a9d5e579a9900ee47e7fbe50e126fb1c28b55ac12e7f2869c1cb196d03f',
        'Suspend only idle matching model/source radial ring contexts; mismatches retain full invalidation and active/failed handoffs fail closed.',
    ),
    ('workers', '_announce_worker_render_path'): (
        'd587d682581704f2d325c323d443c89884b941955efeb2b3c85b2224c90f33a2',
        'Emit one flushed render/result route record per family and task-kind combination in each worker.',
    ),
    ('cylindrical_projection', '_RadialPlanePlanTooLarge'): (
        '0df1909707cf449838365cfd7cd18ecc011e63b0194b2147fb5124397f2612e3',
        'Explicit bounded-plan admission exception selects the unchanged reference before any output is delivered.',
    ),
    ('cylindrical_projection', 'RadialPlanePlan'): (
        'f0ef083a324833515f2888a4cecf76bb6cabf61d225b1287ec007530c7d1ea1a',
        'Readonly 2D shell ownership and CSR periodic-column plan with explicit resident byte accounting.',
    ),
    ('cylindrical_projection', 'clear_radial_plane_plan_cache'): (
        'fcffb209a2ab00bb2137946a1d00fc33f0aa22c76f483659a2a5f6a3222ab54e',
        'Advance the cache generation and detach active build Futures without cancellation; old callers complete without repopulating or removing newer-generation entries.',
    ),
    ('cylindrical_projection', '_plane_geometry'): (
        'a795e43d7e1dea439d1e80c025602c89901cdf8d40a816c10593988f75936e43',
        'Resolve source/processing base-plane axes independently of the varying stack axis.',
    ),
    ('cylindrical_projection', '_plane_occurrence_strips'): (
        '98efa5553f4e6440cdfbc50e4b95ff54b17b0e982b576dfd0bcab950942aeb63',
        'Preserve exact NumPy nearest-shell and repeated-addition wrap equations while allowing bounded single-pass construction strips.',
    ),
    ('cylindrical_projection', '_build_radial_plane_plan'): (
        'a54394857ccfe6a62e40206623e76317615a627814483152dd7a4a5f58abd63e',
        'Assemble exact readonly CSR tables in one geometry pass; bound strip occurrence retention by the plan budget and preserve per-pixel wrap order.',
    ),
    ('cylindrical_projection', '_radial_plane_plan'): (
        '4733fa3e98aea57e1fef235f2da26754ef44228a60af9f5e3552dea7acc26144',
        'Share one Future per concurrent exact plane key, complete outside the cache lock, preserve original build errors and retry eligibility, keep unrelated keys parallel, and retain readonly bounded LRU ownership.',
    ),
    ('cylindrical_projection', '_radial_projection_metadata'): (
        'f961fe2aab01cd9ab17778436351c78cbec4176bf881afc8f963235b5d94f8af',
        'Evaluate the original NumPy sampled-shear equations in bounded multi-shell batches; preserve exact processing maps, ideal shear, float64 rounding and tilt signs.',
    ),
    ('cylindrical_projection', '_gather_radial_pixel'): (
        '865c22ec8d83336bc596b6dae0dce1ea08d4f9277d3fee27e9707382cbe18e5a',
        'Gather nearest mapped mask pixels with original height ties and periodic OR; valid caller-supplied bounds only skip known-zero reads.',
    ),
    ('cylindrical_projection', '_project_radial_block'): (
        'acc24813912d8759bca03f78fc7b3903e15c0e61ba2615ea77110305aa5d0601',
        'Apply the same factored pixel gather across each axis orientation in bounded source-output blocks.',
    ),
    ('cylindrical_projection', '_radial_block_schedule'): (
        '00299ca499d5d419dc25f26762132750698cd4e38ced577d5fe02ad58363892f',
        'Bound output block depth and concurrency by requested/allocated CPUs and the explicit in-flight byte budget.',
    ),
    ('cylindrical_projection', '_ordered_radial_blocks'): (
        'd6c2d1e36d7ff36676f1e6e97f6cf3a15d3ed422f7c4edc2b848985aba3adcca',
        'Deliver concurrent projection blocks in source order, cancelling/draining remaining work before borrowed-input retirement on failure.',
    ),
    ('cylindrical_projection', 'backproject_radial_volume_to_volume'): (
        'ef799fa756fe233a89229c297b8a078fb57a37a83f3c42de9b7270cdd3a8b74b',
        'Preserve direct-CUDA production dispatch and setup partitions; report opt-in graph and empty-packet experiments separately without enabling them at the pipeline boundary.',
    ),
    ('cylindrical_projection', '_nearest_global_shell'): (
        'cb42c5d91613549ba8ad3fd2d644ed2f4e7ff67ebfd526371b00c162266cd07f',
        'Unchanged v20.0.1 reference nearest-global-shell ownership, including inner-shell midpoint ties.',
    ),
    ('cylindrical_projection', '_processing_index'): (
        '0370c22c421c00617b6434802420e7b57029cfad1c21d21513b2125d4a9c55b1',
        'Unchanged v20.0.1 nearest processing-mask index mapping.',
    ),
    ('cylindrical_projection', '_occurrence_rows'): (
        'a30e06657361bc2dfd8686b405ca0810177a9565bf6628833c3e95226f964ea8',
        'Unchanged v20.0.1 per-occurrence inverse-shear reference and global height rounding.',
    ),
    ('cylindrical_projection', '_pull_radial_chunk'): (
        '0425fba0d265589fd0500e2a865b17dc623a5e74a19dc7563173137015d8d09f',
        'Unchanged v20.0.1 NumPy scalar-grid reference retained as the independent oracle and oversized-plan fallback.',
    ),
    ('cylindrical_projection', 'radial_cuda_backproject_enabled'): (
        '3a1b0fba08221b9f0796c968770c7d99739fba94d6bf1fbf695ec9498967d32c',
        'Explicit opt-out for Radial CUDA projection while retaining the unchanged CPU projector.',
    ),
    ('cylindrical_projection', '_RadialCudaStage'): (
        '559197c333d917c55321816e00d2bc8ea48840f86b7978bba2f450fb86ef6b64',
        'Hold the device lease across dense or encoded block production and consumption; forward encoded format explicitly and preserve unsafe fence quarantine.',
    ),
    ('cylindrical_projection', '_close_radial_cuda_resources'): (
        'c7101d75a1756a95a86bf12666620e66d4f8049c068eca0f3cea4bd3c4fe7894',
        'Fence and release resources uniformly for normal exits and constructor/wrapper failure; unsafe fences preserve projector and lease instead of re-admitting the device.',
    ),
    ('cylindrical_projection', '_try_radial_cuda_stage'): (
        '1ed7bea44466397b00deb2e6f945f395e90b802201fd936904740bd709549772',
        'Respect ordinary inference-priority/retirement admission; fallback to CPU only after safe prepublication construction cleanup.',
    ),
    ('cylindrical_projection', '_ordered_radial_cuda_blocks'): (
        'aa0f49b0efda3841a9f9bd47860d2230ce982d5045c31dfd87ca7dba89e4accf',
        'Prefetch dense or encoded blocks with one GPU producer and independent host ownership; settle pending work before cleanup on consumer failure.',
    ),
    ('cylindrical_cuda_projection', 'RadialCudaProjectionUnavailable'): (
        '327436e46a1747c7d221807e7e58762cf0cdc7d21743d5f023ac5b9bb2db08b0',
        'Recoverable prepublication CUDA admission failure distinct from unsafe device ownership.',
    ),
    ('cylindrical_cuda_projection', 'RadialCudaProjectionUnsafeFailure'): (
        'e1225e7a8ab44ffbab8fe12d069404123f8f47fb92fda547b5917e3afd9eb7a5',
        'Fatal BaseException retains the unfenced projector and bypasses ordinary retry handlers.',
    ),
    ('cylindrical_cuda_projection', '_ProjectionContract'): (
        'b7dd54823355c0e35f7680495de00c4ab99f2d644b9b57a54964c65a13facf55',
        'Validate dense or bbox-cropped source addressing under the existing output and geometry admission contract.',
    ),
    ('cylindrical_cuda_projection', '_positive_shape'): (
        'f2e346180fb107cbfb7ec1cba507825c83caac4571aa7ea38c0b9d49e587a341',
        'Require integer positive int32 dimensions before CUDA address construction.',
    ),
    ('cylindrical_cuda_projection', '_contract_array'): (
        'bbae841e3f206f9ddd72ca2dc60e917ef515a8c403330fe98283ab351ce33772',
        'Require exact contiguous dtype/shape for every borrowed host lookup array.',
    ),
    ('cylindrical_cuda_projection', '_validate_projection_contract'): (
        '2954b1598a06fbb4184bcc86f24b147aff63bc6dd592712b1d44854251c330d1',
        'Prefix known source rectangles with uint64 offsets and select cropped addressing only when it reduces storage; preserve gather validation.',
    ),
    ('cylindrical_cuda_projection', 'RadialCudaProjector'): (
        '123acc79a54c42266679f2ac69c175ab22a4f4884b47f5b83c10129cc073ca29',
        'Keep graph replay and empty-packet elision disabled by default after mixed throughput results; explicit experiments retain bounded capture, preflight, direct fallback and fatal ownership fencing.',
    ),
    ('cylindrical_cuda_projection', 'RadialEncodedSlice'): (
        'bc52167c2887163a857a68a9bb873901c53ba29312558d3d56100868717620e4',
        'Immutable per-slice crop bounds, exact foreground count and payload offset/size metadata.',
    ),
    ('cylindrical_cuda_projection', 'RadialEncodedBlock'): (
        '471701be9affe1da43e554088140a728865b7c5fed88b637d3ef13272ed1987f',
        'Immutable encoded block identity/format with independently owned readonly concatenated host payload.',
    ),
    ('cylindrical_cuda_projection', '_encoded_records'): (
        '8b984daf20133cf42ad805c476925031d5b1c87fb3bb7cac81eb9ddbfc52d4e8',
        'Validate device crop bounds/counts, normalize empty slices, and prefix bounded raw or little-endian row-packbits payload offsets before device encoding.',
    ),
}

# Compile policy and bounded resource constants live outside function bodies.
# Pin their complete AST statements so helper hashes cannot mask policy drift.
REVIEWED_V20_ADDED_STATEMENTS = {
    ('packed_publication', 'optional_compiled_kernels'): (
        '832ee0081e592b680682d6a7182c671420ed965717ed8c673a35bdd0b2932bd5',
        'Direct source-bitset bbox/count and row-packbits encoding with exact addressing, padding and bounded output blocks.',
    ),
    ('topology_runs', '_TOPOLOGY_RUNS_DISABLED'): (
        '914bd52d8607bfd7d187e0e84a1b98fe50e5de2536576ee006ee079f564eaf71',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),
    ('topology_runs', 'optional_compiled_kernels'): (
        '6824f9609888b53198dac49de1d9ce860e0894d8c36a2d16c926c7a75a872ddc',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),

    ('cylindrical_owner', '_OWNER_KERNEL_SOURCE'): (
        '4d833c54ff172cfc150730b1ed57f574aac648051198623231368939af4b0877',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_bucket_shell_pixels_compiled'): (
        '8ba550259e851dc5379807dd87dc1867f613202d431e039a280d854829d7229e',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_RADIAL_OWNER_LOCK'): (
        '22f5cde9003cdd04ef1057fe0f92b1552cb53764c95cece589a11564fdc97f93',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_RADIAL_OWNER_STATES'): (
        '23eb5b965d00c50f4f6e2868d42a6a8b605baa7f148c68c43491ff5908393cd9',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_RESERVE_BYTES'): (
        'b1dcdc89cbca2b1e25e13f67a7844a474b129eb0d81ea3bc53806dd1a1745fb2',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_WORK_ITEMS'): (
        '1f26b2c99d35da871b02c17234d80d5bea31cf753604b75ff4cb45b2ea8cfcf9',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', 'RADIAL_OWNER_CONTRACT'): (
        '07d4d26a31de759bf9b189f01f331f1e1a946b405ca62c2d0d0c73814cedd024',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_projection', '_RADIAL_METADATA_CHUNK_VALUES'): (
        'cf19116408f9aa4dc22e7eb1308055641a96ec0eebb2df7ed17a3ceb5c9d8e7c',
        'Bound sampled-shear intermediate arrays at one million values per batch, with one native row as the minimum.',
    ),
    ('cylindrical_cuda_projection', '_pack_radial_source_block_compiled'): (
        '765802ea54a2b542140955b926906800d9fc54561b7e564f210855851ecb3746',
        'Keep source crop packing optional, cached and nogil; compilation failure before upload retains the NumPy path.',
    ),
    ('cylindrical_projection', '_PLANE_BUILD_CHUNK_PIXELS'): (
        '4afc438f9808e94475fae6bf58c44b4e2b9c6177bd0761924ba3ca533dba3a0c',
        'Bound single-pass NumPy coordinate strips at one million pixels to reduce interpreter handoffs during concurrent output work.',
    ),
    ('cylindrical_projection', 'numba_compile_policy'): (
        'a00fd59c3a0ef57a76f093e34ec03927219de47f7f0ad44fe7c22b21bc713347',
        'Numba compilation retains cache=True/nogil=True/fastmath=False and inlines the exact pixel gather; no relaxed floating-point reassociation.',
    ),
    ('cylindrical_projection', '_PULL_CHUNK_VOXELS'): (
        '92570f47d7a6bc732f8519a8171b4a369d8b4565c378b819e7f6e4d37f913191',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE_BYTES'): (
        '06c9fc845d716743a26e1300257b98422676643431557777b3df8345cb237591',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_MAX_BYTES'): (
        '074d4ff4b7a21a7b54aca28e3418535238ba6ab5807b15c33cb93009c94e842c',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_OUTPUT_BLOCK_BYTES'): (
        'bf8f6f6165a78206c9c57c5d280b4154c1a81a74f2ef70db2c1bb8318887c002',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_INFLIGHT_OUTPUT_BYTES'): (
        '61a8484d4af12b9ffcdea6c7d29178336c7c1e93c5e590adefff6db39892b91c',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_cuda_projection', '_BLOCK_BYTES'): (
        '2f39654da65e61d24adab7838ee405a85d095361e889e4e73826b29fb7f02d16',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_UPLOAD_BYTES'): (
        'e40900fa2a6f0fc6a68b8f654bf86342ab930f214b05a494c8bdaadff982f81c',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_RESERVE_BYTES'): (
        'b1dcdc89cbca2b1e25e13f67a7844a474b129eb0d81ea3bc53806dd1a1745fb2',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_SETUP_BYTES'): (
        'c06d5b317b175243352d3c3519e3ddfa7e0ae74e090087d86086836d213d8aec',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_KERNEL_SOURCE'): (
        '87efaa993cf5c91cd6999f39f3f8938b19b1d1f89bc2e4bb91a0cd9337a51244',
        'Preserve exact cropped/dense source addressing and float64 geometry; captured projection optionally reads first-Z from an owned device scalar without changing gather math.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE'): (
        'dda54f3a989503868661b93cc95518d3728061aae4a0c40bcf152e5fa990d316',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE_SIZE'): (
        '17894c39f5d5d8acfa6c95744f1fcac6d246e182cd559f8d5f63cd0fde3bfa1d',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_INFLIGHT'): (
        '581530db643eaabdcf4f9772aec89497a9b01a927a43cdefb7274dfaa815bfb7',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE_GENERATION'): (
        '10e839a0d670a2f9adff107705216624be23d07018b4448754806389f1d3b808',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_LOCK'): (
        'bda5024f36dde3b97e50ccb2aee349e0decab14bccec493d2401329d3854ff7b',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_cuda_projection', '_MAX_ENCODED_SLICES'): (
        'c5a08a0217c3c75befe725137cbc60a52a3fcd01f6bc6c7a8ae7be45f2673a01',
        'Bound compact metadata to 4096 slices with the exact aligned 24-byte crop/count device ABI.',
    ),
    ('cylindrical_cuda_projection', '_CROP_METADATA_DTYPE'): (
        '49ee0788784009e994d9a3be35539f969e78601744e0590b4568c789d36c9e62',
        'Bound compact metadata to 4096 slices with the exact aligned 24-byte crop/count device ABI.',
    ),
    ('cylindrical_projection', 'shared_future_import'): (
        '7785747cb9647444f6bf0c7aae1df3a2a18fc2b09299d488ca78744db5dfecc7',
        'Use the standard concurrent Future and executor for shared plan-build ownership and ordered producers.',
    ),
    ('interpolation', 'encoded_operator_import'): (
        '60e489325a19bee8be6a450945dd253634f01297b803d2869e489219288b5372',
        'Use integer-index validation for device-encoded records before reserving writer state.',
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


def reviewed_v21_contract(manifest: dict[str, object]) -> dict[str, object]:
    """Authenticate the additive QSC review without rebaselining history."""
    review = manifest.get('v21_review', {})
    encoded = json.dumps(review, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V21_SHA256:
        raise RuntimeError('v21 review digest mismatch; require explicit review of changed contracts')
    if review.get('release') != '21.0.0':
        raise RuntimeError('v21 review has an unexpected release identity')
    for category, identity in (
        ('definitions', 'name'), ('statements', 'label'),
        ('local_import_seams', 'name'), ('preserved_radial_modules', 'module'),
        ('preserved_radial_definitions', 'qualified_name'),
    ):
        keys = [(item['module'], item[identity]) for item in review[category]]
        if len(keys) != len(set(keys)) or any(not item.get('reason') for item in review[category]):
            raise RuntimeError(f'v21 review has duplicate or unexplained {category}')
    return review


def _reviewed_v21_patch_contract(
    manifest: dict[str, object], v21: dict[str, object], *, key: str,
    release: str, expected_digest: str, previous_digest: str,
    earlier_patches: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Authenticate patch additions and check every available predecessor pin."""
    review = manifest.get(key, {})
    encoded = json.dumps(review, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        raise RuntimeError(f'v{release} review digest mismatch; require explicit review of changed contracts')
    if review.get('release') != release or review.get('previous_review_sha256') != previous_digest:
        raise RuntimeError(f'v{release} review has an unexpected release or predecessor identity')
    historical_definitions = {
        key: record[0] for key, record in REVIEWED_V20_ADDED_DEFINITIONS.items()
    }
    historical_definitions.update({
        (module, name): replacement_hash
        for (module, _baseline), (name, replacement_hash, _reason)
        in REVIEWED_V20_STATEMENT_REPLACEMENTS.items()
    })
    historical_definitions.update({
        key: record[0] for key, record in REVIEWED_LOCAL_IMPORT_SEAMS.items()
    })
    historical_definitions.update({
        (item['module'], item['name']): item['sha256'] for item in v21['definitions']
    })
    historical_statements = {
        (item['module'], item['label']): item['sha256'] for item in v21['statements']
    }
    for earlier in earlier_patches:
        historical_definitions.update({
            (item['module'], item['name']): item['sha256'] for item in earlier['definitions']
        })
        historical_statements.update({
            (item['module'], item['label']): item['sha256'] for item in earlier['statements']
        })
    for category, identity, historical in (
        ('definitions', 'name', historical_definitions),
        ('statements', 'label', historical_statements),
    ):
        keys = [(item['module'], item[identity]) for item in review[category]]
        if len(keys) != len(set(keys)) or any(not item.get('reason') for item in review[category]):
            raise RuntimeError(f'v{release} review has duplicate or unexplained {category}')
        for item in review[category]:
            key = (item['module'], item[identity])
            if key in historical and item.get('previous_sha256') != historical[key]:
                raise RuntimeError(f'v{release} supersession does not match its historical pin: {key[0]}.{key[1]}')
            for field in ('sha256', 'previous_sha256'):
                value = item.get(field)
                if field == 'previous_sha256' and value is None:
                    continue
                if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                    raise RuntimeError(f'v{release} review has an invalid {field}: {key[0]}.{key[1]}')
    return review


def reviewed_v21_patch_contract(manifest: dict[str, object], v21: dict[str, object]) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_1_review', release='21.0.1',
        expected_digest=REVIEWED_V21_0_1_SHA256, previous_digest=REVIEWED_V21_SHA256,
    )


def reviewed_v21_0_2_contract(
    manifest: dict[str, object], v21: dict[str, object], patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_2_review', release='21.0.2',
        expected_digest=REVIEWED_V21_0_2_SHA256, previous_digest=REVIEWED_V21_0_1_SHA256,
        earlier_patches=(patch,),
    )


def reviewed_v21_0_3_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_3_review', release='21.0.3',
        expected_digest=REVIEWED_V21_0_3_SHA256, previous_digest=REVIEWED_V21_0_2_SHA256,
        earlier_patches=(first_patch, second_patch),
    )


def reviewed_v21_0_4_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
    third_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_4_review', release='21.0.4',
        expected_digest=REVIEWED_V21_0_4_SHA256, previous_digest=REVIEWED_V21_0_3_SHA256,
        earlier_patches=(first_patch, second_patch, third_patch),
    )


def reviewed_v21_0_5_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
    third_patch: dict[str, object], fourth_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_5_review', release='21.0.5',
        expected_digest=REVIEWED_V21_0_5_SHA256, previous_digest=REVIEWED_V21_0_4_SHA256,
        earlier_patches=(first_patch, second_patch, third_patch, fourth_patch),
    )


def reviewed_v21_0_6_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
    third_patch: dict[str, object], fourth_patch: dict[str, object],
    fifth_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_6_review', release='21.0.6',
        expected_digest=REVIEWED_V21_0_6_SHA256, previous_digest=REVIEWED_V21_0_5_SHA256,
        earlier_patches=(first_patch, second_patch, third_patch, fourth_patch, fifth_patch),
    )


def reviewed_v21_1_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_1_review', release='21.1.0',
        expected_digest=REVIEWED_V21_1_SHA256, previous_digest=REVIEWED_V21_0_6_SHA256,
        earlier_patches=earlier_patches,
    )


def reviewed_radial_module_hashes(v21, patches):
    """Require explicit authenticated successors for preserved full modules."""
    expected = {item['module']: item['sha256'] for item in v21['preserved_radial_modules']}
    for patch in patches:
        seen = set()
        for item in patch.get('preserved_radial_module_updates', ()):
            module = item.get('module')
            if module in seen or module not in expected or not item.get('reason'):
                raise RuntimeError('unknown, duplicate or unexplained Radial module review')
            seen.add(module)
            if item.get('previous_sha256') != expected[module]:
                raise RuntimeError('Radial module review does not match its preserved predecessor')
            value = item.get('sha256')
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise RuntimeError('Radial module review has an invalid digest')
            expected[module] = value
    return expected


def reviewed_radial_definition_hashes(v21, patches):
    """Permit only authenticated, predecessor-pinned updates to preserved methods."""
    expected = {(item['module'], item['qualified_name']): item['sha256']
                for item in v21['preserved_radial_definitions']}
    expected[('cylindrical_cuda_projection', 'RadialCudaProjector._upload_cropped_source')] = (
        REVIEWED_PRESERVED_RADIAL_UPLOAD_SHA256)
    for patch in patches:
        seen = set()
        for item in patch.get('preserved_radial_definition_updates', ()):
            key = (item['module'], item['qualified_name'])
            if key in seen or key not in expected or not item.get('reason'):
                raise RuntimeError('unknown, duplicate or unexplained Radial method review')
            seen.add(key)
            if item.get('previous_sha256') != expected[key]:
                raise RuntimeError('Radial method review does not match its preserved predecessor')
            value = item.get('sha256')
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise RuntimeError('Radial method review has an invalid digest')
            expected[key] = value
    return expected


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    baseline_digest = hashlib.sha256(json.dumps(
        manifest['statements'], sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    if baseline_digest != IMMUTABLE_INVENTORY_STATEMENTS_SHA256:
        raise RuntimeError('immutable inventory digest mismatch; retain historical statement records and add explicit reviews')
    rename_replacements = azimuthal_rename_replacements(manifest)
    v21 = reviewed_v21_contract(manifest)
    patch = reviewed_v21_patch_contract(manifest, v21)
    second_patch = reviewed_v21_0_2_contract(manifest, v21, patch)
    third_patch = reviewed_v21_0_3_contract(manifest, v21, patch, second_patch)
    fourth_patch = reviewed_v21_0_4_contract(manifest, v21, patch, second_patch, third_patch)
    fifth_patch = reviewed_v21_0_5_contract(manifest, v21, patch, second_patch, third_patch, fourth_patch)
    sixth_patch = reviewed_v21_0_6_contract(manifest, v21, patch, second_patch, third_patch, fourth_patch, fifth_patch)
    patches = (patch, second_patch, third_patch, fourth_patch, fifth_patch, sixth_patch)
    lta_release = reviewed_v21_1_contract(manifest, v21, *patches)
    patches = (*patches, lta_release)
    patch_definitions = {
        (item['module'], item['name']): item for review in patches for item in review['definitions']
    }
    v21_definitions = {(item['module'], item['name']): item for item in v21['definitions']}
    v21_seams = {
        (item['module'], item['name']): (item['definition_sha256'], item['seam_sha256'])
        for item in v21['local_import_seams']
    }

    def patched_definition_hash(module: str, name: str, previous_hash: str) -> str:
        for review in patches:
            record = next((item for item in review['definitions']
                           if (item['module'], item['name']) == (module, name)), None)
            if record is not None:
                if record['previous_sha256'] != previous_hash:
                    raise RuntimeError(f'v{review["release"]} supersession does not match its historical pin: {module}.{name}')
                previous_hash = str(record['sha256'])
        return previous_hash

    def reviewed_definition_hash(module: str, name: str, previous_hash: str) -> str:
        record = v21_definitions.get((module, name))
        if record is None:
            return patched_definition_hash(module, name, previous_hash)
        if record['previous_sha256'] != previous_hash:
            raise RuntimeError(f'v21 supersession does not match its historical pin: {module}.{name}')
        return patched_definition_hash(module, name, str(record['sha256']))

    available: dict[str, Counter[str]] = {}
    trees: dict[str, ast.Module] = {}
    top_level: dict[str, list[ast.stmt]] = {}
    local_import_seams: dict[tuple[str, str], tuple[str, str]] = {}
    audited_modules = (
        {str(item["module"]) for item in manifest["statements"]}
        | {module for module, _name in REVIEWED_V20_ADDED_DEFINITIONS}
        | {module for module, _label in REVIEWED_V20_ADDED_STATEMENTS}
        | {item['module'] for item in v21['definitions']}
        | {item['module'] for item in v21['statements']}
        | {item['module'] for item in v21['preserved_radial_modules']}
        | {item['module'] for review in patches for item in review['definitions'] + review['statements']}
    )
    for module in audited_modules:
        module_path = PACKAGE / f"{module}.py"
        module_source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(module_source, filename=str(module_path))
        trees[module] = tree
        top_level[module] = list(tree.body)
        available[module] = Counter(digest(node) for node in tree.body)
        local_import_seams.update(reviewed_local_import_seams(module, module_source, tree))

    for (module, name), (expected_hash, reason) in REVIEWED_V20_ADDED_DEFINITIONS.items():
        expected_hash = reviewed_definition_hash(module, name, expected_hash)
        matches = [node for node in top_level.get(module, ()) if getattr(node, 'name', None) == name]
        if not reason or len(matches) != 1 or digest(matches[0]) != expected_hash:
            raise RuntimeError(f'v20 reviewed added definition changed or is missing: {module}.{name}')
    for (module, label), (expected_hash, reason) in REVIEWED_V20_ADDED_STATEMENTS.items():
        if not reason or available.get(module, Counter())[expected_hash] != 1:
            raise RuntimeError(f'v20 reviewed added statement changed or is missing: {module}.{label}')

    for (module, name), record in v21_definitions.items():
        matches = [node for node in top_level[module] if getattr(node, 'name', None) == name]
        if len(matches) != 1 or digest(matches[0]) != patched_definition_hash(module, name, record['sha256']):
            raise RuntimeError(f'v21 reviewed definition changed or is missing: {module}.{name}')
    for (module, name), record in patch_definitions.items():
        matches = [node for node in top_level[module] if getattr(node, 'name', None) == name]
        if len(matches) != 1 or digest(matches[0]) != record['sha256']:
            raise RuntimeError(f'v21 patch reviewed definition changed or is missing: {module}.{name}')
    effective_statements = list(v21['statements'])
    for review in patches:
        replaced_statements = {(item['module'], item['previous_sha256']) for item in review['statements']}
        effective_statements = [
            item for item in effective_statements if (item['module'], item['sha256']) not in replaced_statements
        ] + review['statements']
    for item in effective_statements:
        if available[item['module']][item['sha256']] != 1:
            raise RuntimeError(f'v21 reviewed statement changed or is missing: {item["module"]}.{item["label"]}')
    effective_definitions = {**v21_definitions, **patch_definitions}
    complete_modules = set(v21['complete_modules'])
    complete_modules.update(module for review in patches for module in review.get('complete_modules', ()))
    for module in complete_modules:
        expected = Counter(item['sha256'] for item in list(effective_definitions.values()) + effective_statements if item['module'] == module)
        if available[module] != expected:
            raise RuntimeError(f'v21 complete-module statement coverage differs: {module}')
    for module, expected_hash in reviewed_radial_module_hashes(v21, patches).items():
        source = (PACKAGE / f'{module}.py').read_text(encoding='utf-8')
        if hashlib.sha256(source.encode('utf-8')).hexdigest() != expected_hash:
            raise RuntimeError(f'v21 changed a preserved Radial module: {module}')
    radial_method_hashes = reviewed_radial_definition_hashes(v21, patches)
    for (module, qualified_name), expected_hash in radial_method_hashes.items():
        scope = trees[module]
        for name in qualified_name.split('.'):
            matches = [node for node in scope.body if getattr(node, 'name', None) == name]
            if len(matches) != 1:
                raise RuntimeError(f'v21 preserved Radial definition is missing: {qualified_name}')
            scope = matches[0]
        if digest(scope) != expected_hash:
            raise RuntimeError(f'v21 changed preserved Radial arithmetic: {qualified_name}')

    expected_local_import_seams = {**REVIEWED_LOCAL_IMPORT_SEAMS, **v21_seams}
    for review in patches:
        seen_seams = set()
        for item in review.get('local_import_seam_updates', ()):
            key = (item['module'], item['name'])
            if key in seen_seams or key not in expected_local_import_seams or not item.get('reason'):
                raise RuntimeError('unknown, duplicate or unexplained local-import seam update')
            seen_seams.add(key)
            previous = (item.get('previous_definition_sha256'), item.get('previous_seam_sha256'))
            if previous != expected_local_import_seams[key]:
                raise RuntimeError('local-import seam update does not match its predecessor')
            current = (item.get('definition_sha256'), item.get('seam_sha256'))
            if any(not isinstance(value, str) or len(value) != 64 or
                   any(char not in '0123456789abcdef' for char in value) for value in current):
                raise RuntimeError('local-import seam update has an invalid digest')
            expected_local_import_seams[key] = current
    for key, (previous_hash, _previous_seam) in REVIEWED_LOCAL_IMPORT_SEAMS.items():
        if key in v21_seams:
            reviewed_definition_hash(*key, previous_hash)

    unexpected_local_import_seams = sorted(
        set(local_import_seams) - set(expected_local_import_seams)
    )
    missing_local_import_seams = sorted(
        set(expected_local_import_seams) - set(local_import_seams)
    )
    changed_local_import_seams = sorted(
        key
        for key in set(local_import_seams) & set(expected_local_import_seams)
        if local_import_seams[key] != expected_local_import_seams[key]
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
        replacement_hash = reviewed_definition_hash(module, name, replacement_hash)
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
            replacement_hash = reviewed_definition_hash(module, _name, replacement_hash)
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
        if (destination, name) in patch_definitions:
            expected_hash = reviewed_definition_hash(destination, name, expected_hash)
            if available[destination][expected_hash] < 1:
                missing.append(item)
                continue
            available[destination][expected_hash] -= 1
            changed += 1
            continue
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
