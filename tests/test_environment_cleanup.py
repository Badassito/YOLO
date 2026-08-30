from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.smoke_import import install_stubs

install_stubs()

from XTA import geometry, inference, media, runtime, workspace


class EnvironmentCleanupTests(unittest.TestCase):
    def tearDown(self) -> None:
        workspace.configure_pipeline_modes(
            fast_bundle_active=False,
            d1_pipeline_active=False,
        )

    def test_default_scratch_is_output_temp_and_explicit_root_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            output = root / 'output'
            explicit_root = root / 'explicit'

            with mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_SCRATCH_PREFER_SHM': 'force',
                    'YOLO_TTA_SCRATCH_SHM_DIR': str(explicit_root),
                    'YOLO_TTA_SCRATCH_SHM_MIN_FREE_GIB': '0',
                },
                clear=False,
            ):
                default_scratch = runtime.choose_scratch_dir(None, output, 'scan')
                self.assertEqual(default_scratch, output / 'temp')

                explicit_scratch = runtime.choose_scratch_dir(
                    str(explicit_root), output, 'scan',
                )
                self.assertEqual(
                    explicit_scratch,
                    explicit_root / f'scan_{os.getpid()}_temp',
                )

    def test_fast_bundle_defaults_follow_process_local_resolved_state(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                'YOLO_TTA_V1613_BUNDLE_ACTIVE': '1',
                'YOLO_TTA_V1613_D1_PIPELINE_ACTIVE': '0',
            },
            clear=False,
        ):
            for name in (
                'YOLO_TTA_PROTO_HOLE_TREATMENT',
                'YOLO_TTA_PROTO_HOLE_RADIUS',
                'YOLO_TTA_GPU_UNION_RETIREMENT_LANES',
            ):
                os.environ.pop(name, None)
            workspace.configure_pipeline_modes(
                fast_bundle_active=False,
                d1_pipeline_active=True,
            )
            self.assertFalse(workspace.v1613_fast_bundle_active())
            self.assertFalse(workspace.v1613_d1_pipeline_active())
            self.assertEqual(workspace.proto_hole_treatment_mode(), 'off')
            self.assertEqual(workspace.proto_hole_treatment_radius(), 0)
            self.assertEqual(inference.gpu_union_retirement_lane_count(), 2)

            workspace.configure_pipeline_modes(
                fast_bundle_active=True,
                d1_pipeline_active=True,
            )
            self.assertTrue(workspace.v1613_fast_bundle_active())
            self.assertTrue(workspace.v1613_d1_pipeline_active())
            self.assertEqual(workspace.proto_hole_treatment_mode(), 'close')
            self.assertEqual(workspace.proto_hole_treatment_radius(), 2)
            self.assertEqual(inference.gpu_union_retirement_lane_count(), 3)

    def test_consolidated_streaming_defaults_preserve_unset_behavior(self) -> None:
        names = (
            'YOLO_TTA_GPU_INPUT_STAGING_AHEAD_SOURCES',
            'YOLO_TTA_STREAMING_SOURCE_WARMUP_SOURCES',
            'YOLO_TTA_STREAMING_SOURCE_PREFETCH_FRAMES',
            'YOLO_TTA_STREAMING_SOURCE_WORKERS',
        )
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(geometry, '_cpu_count', return_value=16),
        ):
            for name in names:
                os.environ.pop(name, None)
            self.assertEqual(geometry.gpu_input_staging_ahead_sources(4), 4)
            self.assertEqual(geometry.queued_streaming_source_cpu_warmup_slots(20), 8)
            self.assertEqual(geometry.streaming_prediction_source_prefetch_frames(64), 2048)
            self.assertEqual(geometry.streaming_prediction_source_workers(4, 100), 16)

    def test_retired_runtime_toggles_preserve_default_behavior(self) -> None:
        with (
            mock.patch.object(
                workspace,
                '_read_meminfo_bytes',
                return_value={'MemAvailable': 120, 'SwapFree': 30},
            ),
            mock.patch.object(
                workspace,
                '_cgroup_memory_headroom_bytes',
                return_value=80,
            ),
        ):
            self.assertEqual(workspace.available_anon_work_bytes(), 80)

        self.assertFalse(runtime.raw_store_memfd_enabled())
        self.assertFalse(runtime.prediction_volume_build_flush_enabled())
        self.assertFalse(runtime.prediction_hot_path_flush_enabled())
        self.assertEqual(media.processing_volume_mode(), 'cube')
        self.assertEqual(media._cube_t_axis_resize_backend(), 'slab')

if __name__ == '__main__':
    unittest.main()
