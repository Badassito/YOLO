"""Streaming terminal fusion for completed TTA physical-view groups."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np

from .backprojection import HybridBackprojectionQueue, ViewBackprojectionQueueJob
from .finalization import collapse_tta_variant_volumes_to_physical_views
from .geometry import ViewInfo, is_tilted_view, physical_view_name
from .runtime import close_memmap_array, close_memmap_array_without_flush


def finalize_physical_view_volume_group(
    *,
    model_name: str,
    physical_view: ViewInfo,
    variant_volumes: Sequence[Tuple[ViewInfo, np.ndarray]],
    out_path: Path,
    out_shape_tyx: Tuple[int, int, int],
    workers: int,
    collapse_variants: Callable[..., object] = collapse_tta_variant_volumes_to_physical_views,
    queue_factory: Callable[..., object] = HybridBackprojectionQueue,
    queue_job_type: Callable[..., object] = ViewBackprojectionQueueJob,
    close_volume: Callable[[object], None] = close_memmap_array,
    close_volume_without_flush: Callable[[object], None] = close_memmap_array_without_flush,
) -> Tuple[str, str, np.ndarray]:
    """Collapse one completed TTA group and perform its terminal projection."""

    if not variant_volumes:
        raise ValueError(
            f"{model_name}/{physical_view_name(physical_view)} has no dense variant volumes"
        )

    runtime_views = [view for view, _volume in variant_volumes]
    runtime_volumes = {str(view.name): volume for view, volume in variant_volumes}
    input_volumes = list(runtime_volumes.values())
    try:
        collapsed = collapse_variants(
            {str(model_name): runtime_volumes},
            runtime_views,
            workers=max(1, int(workers)),
        )
        physical_name = str(physical_view_name(physical_view))
        physical_volumes = collapsed.get(str(model_name), {})  # type: ignore[union-attr]
        native_volume = physical_volumes.get(physical_name)
        if native_volume is None or len(physical_volumes) != 1:
            raise RuntimeError(
                f"{model_name}/{physical_name}: TTA collapse produced "
                f"{sorted(physical_volumes)} instead of one physical volume"
            )

        if physical_view.family != "radial" and not is_tilted_view(physical_view):
            return str(model_name), physical_name, native_volume

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            queue_runner = queue_factory(cpu_workers=max(1, int(workers)))
            projected_results = queue_runner.run(  # type: ignore[attr-defined]
                (
                    queue_job_type(
                        model_name=str(model_name),
                        view=physical_view,
                        native_source=native_volume,
                        out_path=out_path,
                        desc=(
                            "Backprojecting completed physical view "
                            f"{model_name}/{physical_name}"
                        ),
                        min_radius=0.0,
                        workers=max(1, int(workers)),
                        out_shape_tyx=tuple(int(value) for value in out_shape_tyx),
                    ),
                )
            )
            if len(projected_results) != 1:
                raise RuntimeError(
                    f"{model_name}/{physical_name}: terminal backprojection returned "
                    f"{len(projected_results)} result(s)"
                )
            result_model, result_view, projected = projected_results[0]
            return str(result_model), str(result_view), projected
        finally:
            close_volume(native_volume)
    except BaseException:
        # Collapse retires all but its first accumulator. Error paths can stop before that
        # ownership transfer completes, so close every distinct input defensively.
        closed_ids: set[int] = set()
        for volume in input_volumes:
            if id(volume) in closed_ids:
                continue
            closed_ids.add(id(volume))
            try:
                close_volume_without_flush(volume)
            except Exception:
                pass
        raise


def run_physical_view_finalization_with_handoff(
    *,
    handoff_credit: threading.Semaphore,
    stop_event: threading.Event,
    variant_volumes: Sequence[Tuple[ViewInfo, np.ndarray]],
    finalize: Callable[[], Tuple[str, str, np.ndarray]],
    close_volume_without_flush: Callable[[object], None] = close_memmap_array_without_flush,
) -> Tuple[str, str, np.ndarray]:
    """Run one dense finalizer while transferring exactly one reducer credit."""

    credit_acquired = False
    try:
        while not handoff_credit.acquire(timeout=0.25):
            if stop_event.is_set():
                raise RuntimeError(
                    "streaming physical-view finalization stopped while waiting "
                    "for the dense union handoff credit"
                )
        credit_acquired = True
        if stop_event.is_set():
            raise RuntimeError("streaming physical-view finalization was stopped")
        return finalize()
    except BaseException:
        if credit_acquired:
            handoff_credit.release()
        retired_ids: set[int] = set()
        for _view, volume in variant_volumes:
            if id(volume) in retired_ids:
                continue
            retired_ids.add(id(volume))
            try:
                close_volume_without_flush(volume)
            except Exception:
                pass
        raise


__all__ = [
    "finalize_physical_view_volume_group",
    "run_physical_view_finalization_with_handoff",
]
