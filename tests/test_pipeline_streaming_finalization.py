from __future__ import annotations

import ast
import inspect
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from volume_tta import pipeline
from volume_tta.geometry import ViewInfo


def _physical_view(name: str = 'transverse', *, family: str = 'orthogonal') -> ViewInfo:
    return ViewInfo(
        name=str(name),
        num_slices=2,
        src_h=3,
        src_w=4,
        pad_mode='clamp',
        family=str(family),
        summary_family=str(name),
        display_name=str(name),
        full_t=2,
        full_h=3,
        full_w=4,
        radial_base_view='transverse' if family == 'radial' else '',
    )


def _variant(physical: ViewInfo, suffix: str) -> ViewInfo:
    values = dict(physical.__dict__)
    values.update({
        'name': f'{physical.name}__tta_{suffix}',
        'physical_view_name': str(physical.name),
        'tta_aug_id': str(suffix),
    })
    return ViewInfo(**values)


class StreamingPhysicalViewFinalizationTests(unittest.TestCase):
    def test_dense_handoff_stop_releases_credit_and_closes_detached_variants(self) -> None:
        physical = _physical_view()
        variants = (
            (_variant(physical, 'a'), np.zeros((1, 1, 1), dtype=np.uint8)),
            (_variant(physical, 'b'), np.zeros((1, 1, 1), dtype=np.uint8)),
        )
        credit = threading.BoundedSemaphore(1)
        stop = threading.Event()
        stop.set()
        finalize = mock.Mock()

        with mock.patch.object(
            pipeline, 'close_memmap_array_without_flush',
        ) as close_volume, self.assertRaisesRegex(RuntimeError, 'was stopped'):
            pipeline._run_physical_view_finalization_with_handoff(
                handoff_credit=credit,
                stop_event=stop,
                variant_volumes=variants,
                finalize=finalize,
            )

        finalize.assert_not_called()
        self.assertEqual(close_volume.call_count, 2)
        self.assertTrue(credit.acquire(blocking=False))
        credit.release()

    def test_streaming_union_futures_are_scheduler_dependencies(self) -> None:
        source = inspect.getsource(pipeline._main_impl)
        self.assertIn(
            'waitables.extend(list(physical_view_union_futures.keys()))', source,
        )
        self.assertIn('not physical_view_union_futures', source)
        self.assertIn('missing_terminal_variants', source)
        self.assertIn('missing_physical_unions', source)
        self.assertIn('threading.BoundedSemaphore(1)', source)
        self.assertIn('dense_handoff_credit_held=True', source)
        tree = ast.parse(textwrap.dedent(source))
        terminal_publications = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == '_mark_view_variant_terminal'
        )
        self.assertEqual(terminal_publications, 4)

    def test_orthogonal_variants_collapse_before_global_tail(self) -> None:
        physical = _physical_view()
        first = np.zeros((2, 3, 4), dtype=np.uint8)
        second = np.zeros_like(first)
        first[0, 0, 0] = 1
        second[1, 2, 3] = 1

        with tempfile.TemporaryDirectory() as tmp:
            model_name, view_name, result = pipeline._finalize_physical_view_volume_group(
                model_name='model',
                physical_view=physical,
                variant_volumes=(
                    (_variant(physical, 'a'), first),
                    (_variant(physical, 'b'), second),
                ),
                out_path=Path(tmp) / 'unused.dat',
                out_shape_tyx=(2, 3, 4),
                workers=2,
            )

        self.assertEqual((model_name, view_name), ('model', 'transverse'))
        expected = np.zeros((2, 3, 4), dtype=np.uint8)
        expected[0, 0, 0] = 1
        expected[1, 2, 3] = 1
        np.testing.assert_array_equal(result, expected)

    def test_radial_variants_project_once_after_native_collapse(self) -> None:
        physical = _physical_view('radial_transverse', family='radial')
        first = np.zeros((2, 3, 4), dtype=np.uint8)
        second = np.zeros_like(first)
        first[0, 1, 1] = 1
        second[1, 1, 2] = 1
        projected = np.full((2, 3, 4), 7, dtype=np.uint8)
        captured_jobs: list[object] = []

        class _Queue:
            def run(self, jobs: object) -> list[tuple[str, str, np.ndarray]]:
                job_list = list(jobs)  # type: ignore[arg-type]
                self_outer.assertEqual(len(job_list), 1)
                job = job_list[0]
                captured_jobs.append(job)
                self_outer.assertEqual(int(np.count_nonzero(job.native_source)), 2)
                return [('model', 'radial_transverse', projected)]

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            pipeline, 'HybridBackprojectionQueue', return_value=_Queue(),
        ) as queue_factory:
            result = pipeline._finalize_physical_view_volume_group(
                model_name='model',
                physical_view=physical,
                variant_volumes=(
                    (_variant(physical, 'a'), first),
                    (_variant(physical, 'b'), second),
                ),
                out_path=Path(tmp) / 'projected.dat',
                out_shape_tyx=(2, 3, 4),
                workers=2,
            )

        queue_factory.assert_called_once_with(cpu_workers=2)
        self.assertEqual(result[:2], ('model', 'radial_transverse'))
        self.assertIs(result[2], projected)
        self.assertEqual(captured_jobs[0].out_shape_tyx, (2, 3, 4))

    def test_empty_dense_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'no dense variant volumes'):
            pipeline._finalize_physical_view_volume_group(
                model_name='model',
                physical_view=_physical_view(),
                variant_volumes=(),
                out_path=Path('unused.dat'),
                out_shape_tyx=(2, 3, 4),
                workers=1,
            )


if __name__ == '__main__':
    unittest.main()
