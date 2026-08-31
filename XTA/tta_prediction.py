"""Prediction-source preparation, caching, staging, and bounded queue ownership."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import (
    AugJob,
    DenseTileJob,
    PredictionVolumeRef,
    StreamingYoloVolumeSource,
    ViewInfo,
    _prediction_ref_has_gpu_input_staging,
    build_dense_tile_raster_plan,
    build_fullframe_raster_plan,
    build_view_frame_cache,
    close_prediction_volume_ref,
    make_dense_tile_channel_renderer,
    make_fullframe_channel_renderer,
    materialize_dense_tile_prediction_volume_for_job,
    materialize_fullframe_prediction_volume_for_job,
    maybe_eager_stage_prediction_ref_on_gpu,
    physical_view_name,
    should_cache_view_frames,
    streaming_prediction_source_autostart_enabled,
    streaming_prediction_source_prefetch_frames,
    streaming_prediction_source_workers,
    streaming_prediction_sources_enabled,
    write_aug_job_meta,
    write_dense_tile_job_meta,
)
from .media import wait_for_volume_ready
from .outputs import CanonicalRenderImageSink


@dataclass(frozen=True)
class PredictionSourceOperations:
    """Injected lower-layer operations used by prediction-source coordination."""

    canonical_image_sink_type: Callable[..., object] = CanonicalRenderImageSink
    write_aug_job_meta: Callable[..., object] = write_aug_job_meta
    write_dense_tile_job_meta: Callable[..., object] = write_dense_tile_job_meta
    build_fullframe_raster_plan: Callable[..., object] = build_fullframe_raster_plan
    build_dense_tile_raster_plan: Callable[..., object] = build_dense_tile_raster_plan
    streaming_source_workers: Callable[..., int] = streaming_prediction_source_workers
    streaming_source_prefetch_frames: Callable[..., int] = (
        streaming_prediction_source_prefetch_frames
    )
    streaming_source_autostart_enabled: Callable[[], bool] = (
        streaming_prediction_source_autostart_enabled
    )
    streaming_sources_enabled: Callable[[], bool] = streaming_prediction_sources_enabled
    make_fullframe_renderer: Callable[..., object] = make_fullframe_channel_renderer
    make_dense_tile_renderer: Callable[..., object] = make_dense_tile_channel_renderer
    streaming_source_type: Callable[..., object] = StreamingYoloVolumeSource
    prediction_ref_type: Callable[..., PredictionVolumeRef] = PredictionVolumeRef
    materialize_fullframe: Callable[..., PredictionVolumeRef] = (
        materialize_fullframe_prediction_volume_for_job
    )
    materialize_dense_tile: Callable[..., PredictionVolumeRef] = (
        materialize_dense_tile_prediction_volume_for_job
    )
    prediction_ref_has_gpu_input_staging: Callable[[PredictionVolumeRef], bool] = (
        _prediction_ref_has_gpu_input_staging
    )
    eager_stage_prediction_ref: Callable[..., PredictionVolumeRef] = (
        maybe_eager_stage_prediction_ref_on_gpu
    )
    close_prediction_ref: Callable[..., None] = close_prediction_volume_ref


class ViewFrameCache:
    """Thread-safe lazy ownership of reusable native view-frame arrays."""

    def __init__(
        self,
        *,
        dense_tiling_active: bool,
        volume_rgb: object,
        temp_dir: Path,
        augmentation_workers: int,
        cache_policy: Callable[[ViewInfo, bool], bool] = should_cache_view_frames,
        wait_for_volume: Callable[[object], None] = wait_for_volume_ready,
        build_cache: Callable[..., np.ndarray] = build_view_frame_cache,
    ) -> None:
        self.dense_tiling_active = bool(dense_tiling_active)
        self.volume_rgb = volume_rgb
        self.temp_dir = Path(temp_dir)
        self.augmentation_workers = max(1, int(augmentation_workers))
        self.arrays: Dict[str, np.ndarray] = {}
        self.paths: Dict[str, Path] = {}
        self._lock = threading.Lock()
        self._cache_policy = cache_policy
        self._wait_for_volume = wait_for_volume
        self._build_cache = build_cache

    def get(self, view: ViewInfo) -> Optional[np.ndarray]:
        if not self._cache_policy(view, self.dense_tiling_active):
            return None
        cache_key = physical_view_name(view)
        cached = self.arrays.get(cache_key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self.arrays.get(cache_key)
            if cached is not None:
                return cached
            self._wait_for_volume(self.volume_rgb)
            cache_path = self.temp_dir / "view_frames" / f"{cache_key}.gray8.dat"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_mm = self._build_cache(
                volume_rgb=self.volume_rgb,
                view=view,
                out_path=cache_path,
                desc=f"{view.name} native frame cache",
                prefer_memory=True,
                workers=self.augmentation_workers,
            )
            self.arrays[cache_key] = cache_mm
            self.paths[cache_key] = cache_path
            return cache_mm


class PredictionSourceCoordinator:
    """Own the bounded prediction-source build queue and ready-source staging."""

    def __init__(
        self,
        *,
        initial_build_jobs: Iterable[Tuple[str, ViewInfo, object]],
        prediction_volume_executor: ThreadPoolExecutor,
        prediction_render_executor: Optional[ThreadPoolExecutor],
        prediction_volume_queue_slots: int,
        per_prediction_volume_workers: int,
        eager_gpu_input_staging_ahead_sources: int,
        queued_streaming_cpu_warmup_sources: int,
        gpu_worker_process_active: bool,
        cpu_worker_process_active: bool,
        temp_dir: Path,
        input_path: Path,
        canonical_images_stage_root: Optional[Path],
        volume_rgb: object,
        channel_format: object,
        args: object,
        pred_cfg: object,
        yolo_models: Sequence[Tuple[str, Optional[object]]],
        keep_temp_artifacts: bool,
        get_view_frame_cache: Callable[[ViewInfo], Optional[np.ndarray]],
        operations: Optional[PredictionSourceOperations] = None,
    ) -> None:
        self.pending_build_jobs: deque[Tuple[str, ViewInfo, object]] = deque(
            initial_build_jobs
        )
        self.prediction_volume_futures: Dict[
            Future, Tuple[str, ViewInfo, object]
        ] = {}
        self.pending_prediction_volume_futures: set[Future] = set()
        self.ready_fullframe: deque[
            Tuple[ViewInfo, AugJob, PredictionVolumeRef]
        ] = deque()
        self.ready_tile_infer: deque[
            Tuple[str, ViewInfo, DenseTileJob, Optional[PredictionVolumeRef]]
        ] = deque()
        self.streaming_cpu_warmup_started_refs: set[int] = set()

        self.prediction_volume_executor = prediction_volume_executor
        self.prediction_render_executor = prediction_render_executor
        self.prediction_volume_queue_slots = int(prediction_volume_queue_slots)
        self.per_prediction_volume_workers = int(per_prediction_volume_workers)
        self.eager_gpu_input_staging_ahead_sources = int(
            eager_gpu_input_staging_ahead_sources
        )
        self.queued_streaming_cpu_warmup_sources = int(
            queued_streaming_cpu_warmup_sources
        )
        self.gpu_worker_process_active = bool(gpu_worker_process_active)
        self.cpu_worker_process_active = bool(cpu_worker_process_active)
        self.temp_dir = Path(temp_dir)
        self.input_path = Path(input_path)
        self.canonical_images_stage_root = canonical_images_stage_root
        self.volume_rgb = volume_rgb
        self.channel_format = channel_format
        self.args = args
        self.pred_cfg = pred_cfg
        self.yolo_models = tuple(yolo_models)
        self.keep_temp_artifacts = bool(keep_temp_artifacts)
        self.get_view_frame_cache = get_view_frame_cache
        self.operations = operations or PredictionSourceOperations()

    def queue_depth(self) -> int:
        return int(
            len(self.pending_prediction_volume_futures)
            + len(self.ready_fullframe)
            + len(self.ready_tile_infer)
        )

    def canonical_image_sink(
        self,
        view: ViewInfo,
        job: AugJob | DenseTileJob,
        kind: str,
        *,
        model_name: str = "shared",
        backend: str = "inprocess_cpu",
    ) -> Optional[CanonicalRenderImageSink]:
        if self.canonical_images_stage_root is None:
            return None
        tile_job = job if isinstance(job, DenseTileJob) else None
        return self.operations.canonical_image_sink_type(
            stage_root=self.canonical_images_stage_root,
            stem=self.input_path.stem,
            model_name=str(model_name),
            view_name=str(view.name),
            kind=str(kind),
            aug_id=str(job.aug_id),
            channel_count=int(self.channel_format.channel_count),
            config_id=(str(tile_job.config_id) if tile_job is not None else None),
            tile_id=(str(tile_job.tile_id) if tile_job is not None else None),
            backend=str(backend),
        )

    def make_streaming_fullframe_ref(
        self, view: ViewInfo, aug_job: AugJob
    ) -> PredictionVolumeRef:
        self.operations.write_aug_job_meta(aug_job, view, self.channel_format)
        raster_plan = self.operations.build_fullframe_raster_plan(
            view, aug_job, self.channel_format
        )
        image_sink = self.canonical_image_sink(view, aug_job, "fullframe")
        render_workers = self.operations.streaming_source_workers(
            self.per_prediction_volume_workers, int(view.num_slices)
        )
        prefetch_frames = self.operations.streaming_source_prefetch_frames(
            max(
                1,
                int(
                    max(
                        self.args.gpu_batch
                        if self.gpu_worker_process_active
                        else 1,
                        self.args.cpu_batch
                        if self.cpu_worker_process_active
                        else 1,
                    )
                ),
            )
        )
        renderer = self.operations.make_fullframe_renderer(
            self.volume_rgb,
            view,
            aug_job,
            channel_format=self.channel_format,
            view_frames=self.get_view_frame_cache(view),
        )
        name = f"Streaming full-frame prediction source {view.name}/{aug_job.aug_id}"
        source = self.operations.streaming_source_type(
            renderer,
            num_frames=int(view.num_slices),
            name=name,
            batch_size=max(1, int(self.args.batch)),
            out_size=int(aug_job.aff.out_size),
            render_workers=int(render_workers),
            prefetch_frames=int(prefetch_frames),
            autostart=bool(self.operations.streaming_source_autostart_enabled()),
            shared_executor=self.prediction_render_executor,
            channel_format=self.channel_format,
            view=view,
            render_batch_sink=image_sink,
            raster_plan=raster_plan,
        )
        return self.operations.prediction_ref_type(
            array=None,
            path=None,
            name=name,
            view_name=str(view.name),
            job_id=str(aug_job.aug_id),
            kind="fullframe",
            source=source,
            channel_format=self.channel_format,
            view=view,
            render_batch_sink=image_sink,
            raster_plan=raster_plan,
        )

    def make_streaming_tile_ref(
        self, view: ViewInfo, tile_job: DenseTileJob
    ) -> PredictionVolumeRef:
        self.operations.write_dense_tile_job_meta(tile_job, self.channel_format)
        raster_plan = self.operations.build_dense_tile_raster_plan(
            view, tile_job, self.channel_format
        )
        image_sink = self.canonical_image_sink(view, tile_job, "tile")
        render_workers = self.operations.streaming_source_workers(
            self.per_prediction_volume_workers, int(view.num_slices)
        )
        prefetch_frames = self.operations.streaming_source_prefetch_frames(
            max(1, int(self.args.batch))
        )
        renderer = self.operations.make_dense_tile_renderer(
            self.volume_rgb,
            view,
            tile_job,
            channel_format=self.channel_format,
            view_frames=self.get_view_frame_cache(view),
        )
        name = f"Streaming tile prediction source {view.name}/{tile_job.tile_id}"
        source = self.operations.streaming_source_type(
            renderer,
            num_frames=int(view.num_slices),
            name=name,
            batch_size=max(1, int(self.args.batch)),
            out_size=int(tile_job.out_size),
            render_workers=int(render_workers),
            prefetch_frames=int(prefetch_frames),
            autostart=bool(self.operations.streaming_source_autostart_enabled()),
            shared_executor=self.prediction_render_executor,
            channel_format=self.channel_format,
            view=view,
            render_batch_sink=image_sink,
            raster_plan=raster_plan,
        )
        return self.operations.prediction_ref_type(
            array=None,
            path=None,
            name=name,
            view_name=str(view.name),
            job_id=str(tile_job.tile_id),
            kind="tile",
            source=source,
            channel_format=self.channel_format,
            view=view,
            render_batch_sink=image_sink,
            raster_plan=raster_plan,
        )

    def submit_prediction_volume_build(
        self, kind: str, view: ViewInfo, job_obj: object
    ) -> None:
        if str(kind) == "fullframe":
            aug_job = job_obj
            assert isinstance(aug_job, AugJob)
            if self.operations.streaming_sources_enabled():
                fut = self.prediction_volume_executor.submit(
                    self.make_streaming_fullframe_ref, view, aug_job
                )
            else:
                out_path = (
                    self.temp_dir
                    / "prediction_volumes"
                    / "fullframe"
                    / view.name
                    / f"{view.name}_{aug_job.aug_id}.u8.dat"
                )
                fut = self.prediction_volume_executor.submit(
                    self.operations.materialize_fullframe,
                    self.volume_rgb,
                    view,
                    aug_job,
                    out_path=out_path,
                    view_frames=self.get_view_frame_cache(view),
                    workers=self.per_prediction_volume_workers,
                    show_progress=False,
                    channel_format=self.channel_format,
                    render_batch_sink=self.canonical_image_sink(
                        view, aug_job, "fullframe"
                    ),
                )
        elif str(kind) == "tile":
            tile_job = job_obj
            assert isinstance(tile_job, DenseTileJob)
            if self.operations.streaming_sources_enabled():
                fut = self.prediction_volume_executor.submit(
                    self.make_streaming_tile_ref, view, tile_job
                )
            else:
                out_path = (
                    self.temp_dir
                    / "prediction_volumes"
                    / "tiles"
                    / view.name
                    / str(tile_job.config_id)
                    / f"{tile_job.tile_id}.u8.dat"
                )
                fut = self.prediction_volume_executor.submit(
                    self.operations.materialize_dense_tile,
                    self.volume_rgb,
                    view,
                    tile_job,
                    out_path=out_path,
                    view_frames=self.get_view_frame_cache(view),
                    workers=self.per_prediction_volume_workers,
                    show_progress=False,
                    channel_format=self.channel_format,
                    render_batch_sink=self.canonical_image_sink(
                        view, tile_job, "tile"
                    ),
                )
        else:  # pragma: no cover
            raise ValueError(f"Unknown prediction volume build kind: {kind}")
        self.prediction_volume_futures[fut] = (str(kind), view, job_obj)
        self.pending_prediction_volume_futures.add(fut)

    def pump_prediction_volume_build_queue(self) -> None:
        while (
            self.pending_build_jobs
            and self.queue_depth() < self.prediction_volume_queue_slots
        ):
            kind, view, job_obj = self.pending_build_jobs.popleft()
            self.submit_prediction_volume_build(str(kind), view, job_obj)

    def queued_gpu_staging_ref_count(self) -> int:
        seen: set[int] = set()
        count = 0
        for _view, _job, ref in list(self.ready_fullframe):
            rid = id(ref)
            if (
                rid not in seen
                and self.operations.prediction_ref_has_gpu_input_staging(ref)
            ):
                seen.add(rid)
                count += 1
        for _model_name, _view, _tile_job, ref in list(self.ready_tile_infer):
            if ref is None:
                continue
            rid = id(ref)
            if (
                rid not in seen
                and self.operations.prediction_ref_has_gpu_input_staging(ref)
            ):
                seen.add(rid)
                count += 1
        return int(count)

    def maybe_eager_stage_prediction_ref(
        self, pred_ref: PredictionVolumeRef
    ) -> PredictionVolumeRef:
        if self.eager_gpu_input_staging_ahead_sources <= 0:
            return pred_ref
        if self.operations.prediction_ref_has_gpu_input_staging(pred_ref):
            return pred_ref
        if (
            self.queued_gpu_staging_ref_count()
            >= self.eager_gpu_input_staging_ahead_sources
        ):
            return pred_ref
        return self.operations.eager_stage_prediction_ref(pred_ref, self.pred_cfg)

    def queued_cpu_warmup_ref_count(self) -> int:
        seen: set[int] = set()
        count = 0
        for _view, _job, ref in list(self.ready_fullframe):
            rid = id(ref)
            if rid in seen:
                continue
            if (
                rid in self.streaming_cpu_warmup_started_refs
                or self.operations.prediction_ref_has_gpu_input_staging(ref)
            ):
                seen.add(rid)
                count += 1
        for _model_name, _view, _tile_job, ref in list(self.ready_tile_infer):
            if ref is None:
                continue
            rid = id(ref)
            if rid in seen:
                continue
            if (
                rid in self.streaming_cpu_warmup_started_refs
                or self.operations.prediction_ref_has_gpu_input_staging(ref)
            ):
                seen.add(rid)
                count += 1
        return int(count)

    def maybe_start_cpu_warmup_prediction_ref(
        self, pred_ref: PredictionVolumeRef
    ) -> None:
        if self.queued_streaming_cpu_warmup_sources <= 0:
            return
        if self.operations.prediction_ref_has_gpu_input_staging(pred_ref):
            return
        rid = id(pred_ref)
        if rid in self.streaming_cpu_warmup_started_refs:
            return
        if (
            self.queued_cpu_warmup_ref_count()
            >= self.queued_streaming_cpu_warmup_sources
        ):
            return
        source = getattr(pred_ref, "source", None)
        start_fn = getattr(source, "start", None)
        if not callable(start_fn):
            return
        try:
            start_fn()
            self.streaming_cpu_warmup_started_refs.add(rid)
        except Exception as exc:
            print(
                "Warning: queued CPU render warmup could not start for "
                f"{pred_ref.name} ({exc}); source will start on demand."
            )

    def warmup_ready_prediction_sources(self) -> None:
        if self.queued_streaming_cpu_warmup_sources <= 0:
            return
        for _view, _job, ref in list(self.ready_fullframe):
            self.maybe_start_cpu_warmup_prediction_ref(ref)
            if (
                self.queued_cpu_warmup_ref_count()
                >= self.queued_streaming_cpu_warmup_sources
            ):
                return
        for _model_name, _view, _tile_job, ref in list(self.ready_tile_infer):
            if ref is None:
                continue
            self.maybe_start_cpu_warmup_prediction_ref(ref)
            if (
                self.queued_cpu_warmup_ref_count()
                >= self.queued_streaming_cpu_warmup_sources
            ):
                return

    def drain_completed_prediction_volume_futures(self) -> None:
        for fut in list(self.pending_prediction_volume_futures):
            if not fut.done():
                continue
            self.pending_prediction_volume_futures.remove(fut)
            kind, view, job_obj = self.prediction_volume_futures.pop(fut)
            pred_ref = self.maybe_eager_stage_prediction_ref(fut.result())
            if str(kind) == "fullframe":
                assert isinstance(job_obj, AugJob)
                self.ready_fullframe.append((view, job_obj, pred_ref))
            else:
                assert isinstance(job_obj, DenseTileJob)
                # A StreamingYoloVolumeSource is single-use. Only the first model receives
                # this built source; subsequent model-specific sources are built lazily.
                tile_model_names = [str(name) for name, _ in self.yolo_models]
                for position, model_name in enumerate(tile_model_names):
                    self.ready_tile_infer.append(
                        (
                            str(model_name),
                            view,
                            job_obj,
                            pred_ref if position == 0 else None,
                        )
                    )
                if not tile_model_names:
                    self.operations.close_prediction_ref(
                        pred_ref, keep_temp=self.keep_temp_artifacts
                    )
        self.pump_prediction_volume_build_queue()
        self.warmup_ready_prediction_sources()


__all__ = [
    "PredictionSourceCoordinator",
    "PredictionSourceOperations",
    "ViewFrameCache",
]
