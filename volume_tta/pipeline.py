"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np
from ._deps import cv2

class _PipelineRunResources:
    """Last-resort ownership for resources created anywhere in one pipeline run.

    The scheduler retains its detailed, phase-aware teardown.  This outer registry covers
    construction failures before that ``try`` and failures in the final assembly/output
    tail, where local executors and processes previously escaped their cleanup block.
    """

    def __init__(self) -> None:
        self.executors: List[object] = []
        self.output_managers: List[object] = []
        self.sinks: List[object] = []
        self.processes: List[object] = []
        self.queues: List[object] = []
        self.threads: List[Tuple[threading.Thread, Optional[threading.Event]]] = []
        self._seen: set[int] = set()

    def _add(self, collection: List[object], resource: object) -> object:
        if resource is not None and id(resource) not in self._seen:
            self._seen.add(id(resource))
            collection.append(resource)
        return resource

    def track_executor(self, resource: object) -> object:
        return self._add(self.executors, resource)

    def track_output_manager(self, resource: object) -> object:
        return self._add(self.output_managers, resource)

    def track_sink(self, resource: object) -> object:
        return self._add(self.sinks, resource)

    def track_process(self, resource: object) -> object:
        return self._add(self.processes, resource)

    def track_queue(self, resource: object) -> object:
        return self._add(self.queues, resource)

    def track_thread(
        self,
        thread: threading.Thread,
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        if id(thread) not in self._seen:
            self._seen.add(id(thread))
            self.threads.append((thread, stop_event))
        return thread

    def close(self, *, failed: bool) -> None:
        for _thread, stop_event in self.threads:
            if stop_event is not None:
                stop_event.set()

        # A process left here escaped the scheduler's cooperative sentinel path. Terminate
        # it before closing queues or waiting on parent thread pools that may depend on it.
        for proc in reversed(self.processes):
            try:
                proc.join(timeout=0.0 if failed else 0.25)  # type: ignore[attr-defined]
                if proc.is_alive():  # type: ignore[attr-defined]
                    proc.terminate()  # type: ignore[attr-defined]
            except Exception:
                pass
        for proc in reversed(self.processes):
            try:
                proc.join(timeout=2.0)  # type: ignore[attr-defined]
                if proc.is_alive() and hasattr(proc, 'kill'):  # type: ignore[attr-defined]
                    proc.kill()  # type: ignore[attr-defined]
                    proc.join(timeout=1.0)  # type: ignore[attr-defined]
            except Exception:
                pass

        for manager in reversed(self.output_managers):
            try:
                manager.wait()  # type: ignore[attr-defined]
            except Exception:
                pass
        for sink in reversed(self.sinks):
            try:
                sink.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
        for executor in reversed(self.executors):
            try:
                executor.shutdown(wait=True, cancel_futures=bool(failed))  # type: ignore[attr-defined]
            except TypeError:
                try:
                    executor.shutdown(wait=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass
        for thread, _stop_event in reversed(self.threads):
            try:
                if thread is not threading.current_thread():
                    thread.join(timeout=5.0)
            except Exception:
                pass
        for process_queue in reversed(self.queues):
            if failed:
                try:
                    process_queue.cancel_join_thread()  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                process_queue.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            if not failed:
                try:
                    process_queue.join_thread()  # type: ignore[attr-defined]
                except Exception:
                    pass

_PIPELINE_RUN_LOCK = threading.Lock()

_ACTIVE_PIPELINE_RUN_RESOURCES: Optional[_PipelineRunResources] = None

def _run_resources() -> _PipelineRunResources:
    resources = _ACTIVE_PIPELINE_RUN_RESOURCES
    if resources is None:
        raise RuntimeError('pipeline resource created outside an active main() invocation')
    return resources

def _create_tracked_thread_pool(
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> ThreadPoolExecutor:
    """Construct and register one pool before the next fallible constructor runs."""
    executor = ThreadPoolExecutor(
        max_workers=int(max_workers),
        thread_name_prefix=str(thread_name_prefix),
    )
    _run_resources().track_executor(executor)
    return executor

def main() -> None:
    """Execute one pipeline run with a lifecycle boundary spanning the full function."""
    global _ACTIVE_PIPELINE_RUN_RESOURCES
    if not _PIPELINE_RUN_LOCK.acquire(blocking=False):
        raise RuntimeError('concurrent volume_tta.pipeline.main() invocations are not supported')
    resources = _PipelineRunResources()
    _ACTIVE_PIPELINE_RUN_RESOURCES = resources
    failed = True
    try:
        reset_streaming_state_for_new_run()
        reset_runtime_state_for_new_run()
        _main_impl()
        failed = False
    finally:
        if failed:
            abort_streaming_producers('pipeline run failed')
        aux_pool = gpu_worker_aux_interpolation_pool()
        if aux_pool is not None and failed:
            try:
                aux_pool.mark_failed('pipeline run failed')
            except Exception:
                pass
        set_gpu_worker_aux_interpolation_pool(None)
        set_interpolation_process_executor(None, 0)
        resources.close(failed=bool(failed))
        active_sink = nrrd_layer_sink()
        if active_sink is not None:
            try:
                active_sink.shutdown()
            except Exception:
                pass
            set_nrrd_layer_sink(None)
        producer_timeout = 30.0 if failed else None
        producers_settled = wait_for_streaming_producers(timeout=producer_timeout)
        if not producers_settled:
            print(
                'Warning: streaming producer teardown exceeded 30 seconds; abort state '
                'remains set and a repeated embedded run will be rejected.'
            )
        else:
            # Native QAT/QPL sessions are thread-owned. Close them on the compressor
            # workers only after every sink/producer has stopped issuing members.
            try:
                shutdown_nrrd_gzip_executors()
            except BaseException as exc:
                # Compressor teardown is best-effort at this outer lifecycle boundary.
                # It must not replace the pipeline failure or strand _PIPELINE_RUN_LOCK.
                try:
                    runtime_telemetry().fallback('nrrd.compression.shutdown', exc)
                except BaseException:
                    pass
                try:
                    print(
                        'Warning: NRRD compressor shutdown failed '
                        f'({type(exc).__name__}: {exc}).'
                    )
                except BaseException:
                    pass
        shutdown_parallel_pool_cache()
        try:
            _set_main_process_gpu_stage_wake_callback(None)
            _set_main_process_gpu_pending_inference(False)
            _set_main_process_gpu_inference_priority_active(False)
            _reset_main_process_gpu_stage_coordinator()
        except Exception:
            pass
        _ACTIVE_PIPELINE_RUN_RESOURCES = None
        _PIPELINE_RUN_LOCK.release()

def _main_impl() -> None:
    initialize_runtime_observability()
    parser = build_argparser()
    args = parser.parse_args()
    try:
        save_request = resolve_save_request(args.save)
        postprocessing_request = resolve_postprocessing_options(args.postprocessing)
    except ValueError as exc:
        parser.error(str(exc))
    save_options = list(save_request.options)
    low_quality_downbin_values: Optional[List[str]] = (
        list(save_request.low_quality_downbins)
        if save_request.low_quality_downbins else None
    )
    # Internal compatibility attributes keep the established post-union implementation
    # unchanged.  The retired individual CLI flags are not registered with argparse and
    # therefore cannot be selected or used as a rollback path.
    args.keep_objects = int(postprocessing_request.keep_objects)
    args.enable_3d_void_fill = bool(postprocessing_request.enable_3d_void_fill)
    args.gaussian_smoothing = (
        float(postprocessing_request.gaussian_sigma)
        if postprocessing_request.gaussian_smoothing_enabled else None
    )
    args.gaussian_smoothing_passes = (
        int(postprocessing_request.gaussian_passes)
        if postprocessing_request.gaussian_smoothing_enabled else None
    )
    save_option_set = set(save_options)
    save_images_enabled = 'images' in save_option_set
    save_labels_enabled = 'labels' in save_option_set
    save_binary_enabled = 'binary' in save_option_set
    save_low_quality_enabled = 'low_quality' in save_option_set
    save_nrrd_enabled = 'nrrd' in save_option_set
    save_voxel_volume_enabled = 'voxel_volume' in save_option_set
    save_high_quality_enabled = 'high_quality' in save_option_set
    save_summary_enabled = 'summary' in save_option_set
    channel_format = resolve_channel_format(args.channel_format)
    print(
        f'[v{SCRIPT_VERSION}] every --angle value is an independent view variant through cleanup, '
        'interpolation, tile gating, NRRD decomposition, and final per-view union. Tile '
        'components are gated individually against their same-angle parent YOLO mask, then '
        'only the residual components are re-gated against same-angle parent bridges. The '
        'retired unified-angle and configuration-canvas tile paths are not present. Radial '
        'interpolation wrapping is mandatory. '
        'View selection uses structured Radial, Tilted, and Tile groups; '
        'interpolation flags use the interpolation_* names; component-NRRD streaming no '
        'longer pauses for topology; and dead telemetry-detail, CPU-retina override, NRRD-yield, '
        'and scratch-msync environment controls are removed. Existing Cartesian, Tilted, and '
        'Radial geometry builders are retained unchanged. The v17.0.10 channel/output behavior '
        'is active: Radial channel stacks wrap across the angular seam and reverse '
        'radial-u after odd 0°/180° crossings, C>=5 saved '
        'view inputs use multi-page TIFF, and unified --save selection controls images, labels, '
        'binary, low_quality[:DOWNBIN], nrrd, voxel_volume, high_quality, and summary, while '
        '--postprocessing selects keep_objects, 3d_void_fill, and gaussian_smoothing. The retained '
        'runtime set includes hardware-linear Radial texture sampling, bilinear in-plane Tilted '
        'forward inputs with exact nearest mask backprojection, retried/loud-failing ffmpeg '
        'launches, dense-prefaulted native sparse final union, header-free D1 NVRTC preflight, '
        'geometry-safe fast-bundle initialization, bounded memfd direct-union windows, sparse '
        'cvol retirement, persistent TensorRT contexts, and parallel atomic outputs.'
    )
    print(
        f'Model input channel format: {channel_format.token} '
        f'(kind={channel_format.kind}, channels={int(channel_format.channel_count)}, '
        f'stride={int(channel_format.stride)}, offsets={list(channel_format.offsets)}; '
        'boundary=radial-wrap+mirror-u/cartesian-clamp; result=center slice N only).'
    )

    print(
        'Save outputs: '
        + (', '.join(save_options) if save_options else '<none>')
        + (
            '; low_quality downbins=' + ','.join(low_quality_downbin_values)
            if save_low_quality_enabled and low_quality_downbin_values else ''
        )
    )

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    try:
        backend_models = resolve_backend_models(args.model)
        backend_devices = resolve_backend_devices(args.device)
        backend_precisions = resolve_backend_precisions(args.quantize, backend_devices)
        backend_batches = resolve_backend_batches(args.batch, backend_devices)
        cpu_instances_requested = resolve_auto_positive_int(
            args.cpu_instances, flag_name='--cpu_instances',
        )
        cpu_threads_requested = resolve_auto_positive_int(
            args.cpu_threads, flag_name='--cpu_threads',
        )
        cpu_streams_requested = resolve_auto_positive_int(
            args.cpu_streams, flag_name='--cpu_streams',
        )
        cpu_infer_requests_requested = resolve_auto_positive_int(
            args.cpu_infer_requests, flag_name='--cpu_infer_requests',
        )
    except ValueError as exc:
        parser.error(str(exc))

    gpu_model_path: Optional[str] = None
    cpu_model_path: Optional[str] = None
    for backend_name, requested_path in (
        ('gpu', backend_models.gpu), ('cpu', backend_models.cpu),
    ):
        if requested_path is None:
            continue
        resolved_path = str(Path(requested_path).expanduser().resolve())
        if not Path(resolved_path).exists():
            raise FileNotFoundError(resolved_path)
        if backend_name == 'gpu':
            gpu_model_path = resolved_path
        else:
            cpu_model_path = resolved_path

    gpu_inference_enabled = bool(backend_devices.gpu_devices)
    cpu_inference_enabled = bool(backend_devices.cpu)
    if gpu_inference_enabled and gpu_model_path is None:
        parser.error('--device selected GPU inference, but --model has no gpu: entry')
    if cpu_inference_enabled and cpu_model_path is None:
        parser.error('--device selected CPU inference, but --model has no cpu: entry')
    if gpu_model_path is not None and not gpu_inference_enabled:
        print(f'GPU model supplied but no GPU backend selected; it will not be loaded: {gpu_model_path}')
    if cpu_model_path is not None and not cpu_inference_enabled:
        print(f'CPU model supplied but cpu was not selected in --device; it will not be loaded: {cpu_model_path}')
    if gpu_inference_enabled and cpu_inference_enabled:
        print(
            'Warning: hybrid inference does not verify that the supplied GPU and CPU artifacts '
            'were exported from identical weights. Their predictions are treated as one logical '
            f'segmentation model. GPU={gpu_model_path}; CPU={cpu_model_path}'
        )

    inference_devices = list(backend_devices.gpu_devices)
    if cpu_inference_enabled:
        inference_devices.append('cpu')
    gpu_device_count = len(backend_devices.gpu_devices)
    gpu_worker_process_active = bool(gpu_inference_enabled)
    cpu_worker_process_active = bool(cpu_inference_enabled)
    inference_worker_process_active = bool(gpu_worker_process_active or cpu_worker_process_active)
    try:
        _allowed_main_cpus = {int(cpu) for cpu in os.sched_getaffinity(0)}
    except Exception:
        _allowed_main_cpus = set(range(max(1, int(_cpu_count()))))
    gpu_logical_indices = [
        int(str(device).split(':')[-1]) for device in backend_devices.gpu_devices
    ]
    _inherited_cvd_for_affinity = os.environ.get('CUDA_VISIBLE_DEVICES')
    if _inherited_cvd_for_affinity is not None and gpu_logical_indices:
        _visible_tokens_for_affinity = [
            token.strip() for token in str(_inherited_cvd_for_affinity).split(',')
            if token.strip()
        ]
        _bad_logical_indices = [
            index for index in gpu_logical_indices
            if int(index) < 0 or int(index) >= len(_visible_tokens_for_affinity)
        ]
        if _bad_logical_indices:
            parser.error(
                f'--device logical CUDA index(es) {_bad_logical_indices} are out of range for '
                f'CUDA_VISIBLE_DEVICES={_inherited_cvd_for_affinity!r} '
                f'({len(_visible_tokens_for_affinity)} visible device(s))'
            )
    try:
        pinned_gpu_tokens = [
            _pin_cuda_visible_device_token(int(idx)) for idx in gpu_logical_indices
        ]
    except RuntimeError as exc:
        parser.error(str(exc))
    gpu_feeder_core_plan: List[List[int]] = (
        plan_gpu_feeder_core_reservations(pinned_gpu_tokens)
        if gpu_worker_process_active else []
    )
    gpu_feeder_reserved_cpus = {
        int(cpu) for values in gpu_feeder_core_plan for cpu in values
    }
    cpu_instance_plans: List[CpuInferenceInstancePlan] = (
        plan_openvino_cpu_instances(
            cpu_instances_requested,
            cpu_threads_requested,
            excluded_cpus=gpu_feeder_reserved_cpus,
        )
        if cpu_worker_process_active else []
    )
    cpu_inference_reserved_cpus = {
        int(cpu) for plan in cpu_instance_plans for cpu in plan.cpus
    }
    hybrid_cpu_affinity_overlap_active = bool(
        gpu_worker_process_active
        and cpu_worker_process_active
        and hybrid_cpu_affinity_overlap_enabled()
    )
    # Dedicated feeder CPUs are always exclusive. OpenVINO remains socket-local; hybrid
    # overlap only permits the parent and CUDA helper pools to share the non-feeder CPUs
    # occupied by OpenVINO.
    feeder_safe_parent_cpus = sorted(_allowed_main_cpus - gpu_feeder_reserved_cpus)
    main_process_reserved_cpus = sorted(
        (_allowed_main_cpus - gpu_feeder_reserved_cpus)
        if hybrid_cpu_affinity_overlap_active else
        (_allowed_main_cpus - gpu_feeder_reserved_cpus - cpu_inference_reserved_cpus)
    )
    if inference_worker_process_active and not main_process_reserved_cpus:
        # An explicit OpenVINO thread request can consume every non-feeder CPU. Preserve
        # feeder exclusivity and allow parent/OpenVINO overlap rather than leaving the parent
        # with an invalid empty mask.
        main_process_reserved_cpus = list(feeder_safe_parent_cpus)
        print(
            'Warning: [affinity] no CPU remained exclusively for parent work after the '
            'OpenVINO request; parent threads will share non-feeder CPUs with OpenVINO.'
        )
    parent_affinity_monitor_stop = threading.Event()
    parent_affinity_monitor_thread: Optional[threading.Thread] = None
    parent_affinity_monitor_errors: List[BaseException] = []
    parent_affinity_lock = threading.Lock()
    parent_applied_affinity_cpus: set[int] = set(_allowed_main_cpus)

    def _apply_parent_cpu_mask(
        cpus: Sequence[int],
        *,
        fail_fast: bool,
        phase_label: str,
    ) -> bool:
        nonlocal parent_applied_affinity_cpus
        desired = {int(cpu) for cpu in cpus if int(cpu) in _allowed_main_cpus}
        if not inference_worker_process_active:
            return False
        if not desired:
            exc = RuntimeError(
                f'v17.0.5 cannot apply an empty parent CPU mask during {phase_label}'
            )
            if fail_fast:
                raise exc
            parent_affinity_monitor_errors.append(exc)
            return False
        with parent_affinity_lock:
            if desired == parent_applied_affinity_cpus:
                return True
            if not _sched_setaffinity_all_threads(sorted(desired)):
                exc = RuntimeError(
                    'v17.0.5 could not apply the parent scheduler/render/output affinity '
                    f'during {phase_label}; requested mask={sorted(desired)}'
                )
                if fail_fast:
                    raise exc
                parent_affinity_monitor_errors.append(exc)
                return False
            parent_applied_affinity_cpus = set(desired)
        print(
            f'[affinity] Parent scheduler/render/output threads use {len(desired)} logical '
            f'CPU(s) during {phase_label}; {len(gpu_feeder_reserved_cpus)} logical CPU(s) '
            'remain exclusive to GPU feeders.'
        )
        return True

    def _apply_parent_inference_affinity(*, fail_fast: bool) -> bool:
        return _apply_parent_cpu_mask(
            main_process_reserved_cpus,
            fail_fast=bool(fail_fast),
            phase_label='steady-state inference',
        )

    def _restore_parent_post_inference_affinity() -> None:
        nonlocal parent_applied_affinity_cpus
        parent_affinity_monitor_stop.set()
        monitor = parent_affinity_monitor_thread
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=2.0)
        with parent_affinity_lock:
            if parent_applied_affinity_cpus == set(_allowed_main_cpus):
                return
            if not _sched_setaffinity_all_threads(sorted(_allowed_main_cpus)):
                print(
                    'Warning: could not restore the parent process to the full allocated CPU '
                    f'mask after inference; requested mask={sorted(_allowed_main_cpus)}'
                )
                return
            parent_applied_affinity_cpus = set(_allowed_main_cpus)
        print(
            f'[affinity] Parent CPU affinity restored to all {len(_allowed_main_cpus)} '
            'allocated logical CPU(s) immediately after inference drain.'
        )

    if gpu_worker_process_active:
        print(
            f'[affinity] Dedicated GPU feeder reservation: '
            f'{gpu_feeder_reserved_physical_cores()} physical core(s) per GPU requested; '
            f'{len(gpu_feeder_reserved_cpus)} logical CPU(s) reserved across '
            f'{len(gpu_feeder_core_plan)} worker(s). '
            'YOLO_TTA_GPU_FEEDER_PHYSICAL_CORES adjusts the per-GPU reservation.'
        )

    if cpu_worker_process_active:
        for plan in cpu_instance_plans:
            print(
                f'[intel] CPU inference instance {plan.instance_id}: nodes={list(plan.numa_nodes)}, '
                f'physical_cores={plan.physical_cores}, logical_threads={plan.inference_threads}, '
                f'cpu_mask={list(plan.cpus)}'
            )
        if hybrid_cpu_affinity_overlap_active:
            print(
                f'[intel] Hybrid CPU-affinity overlap active: OpenVINO remains socket-local on '
                f'{len(cpu_inference_reserved_cpus)} logical CPU(s), while parent/CUDA helper '
                f'threads may share the {len(main_process_reserved_cpus)} non-feeder CPU(s). GPU helper '
                "masks are still derived from each device's discovered NUMA node; no fixed node "
                'is assumed. YOLO_TTA_HYBRID_CPU_AFFINITY_OVERLAP=0 restores exclusive masks.'
            )
        else:
            print(
                f'[intel] OpenVINO reserves {len(cpu_inference_reserved_cpus)} logical CPU(s) '
                f'outside the {len(gpu_feeder_reserved_cpus)} dedicated GPU-feeder logical CPU(s); '
                f'{len(main_process_reserved_cpus)} shared logical CPU(s) remain for parent work.'
            )

    if gpu_worker_process_active and gpu_feeder_reserved_cpus:
        # Apply feeder exclusivity before decode, render-pool creation, or worker startup. The
        # later steady-state mask may narrow this further to exclude OpenVINO, but no parent
        # thread can run on another worker's four dedicated physical cores from this point on.
        _apply_parent_cpu_mask(
            feeder_safe_parent_cpus,
            fail_fast=True,
            phase_label='GPU worker startup and preprocessing',
        )

    args.gpu_model = gpu_model_path
    args.cpu_model = cpu_model_path
    args.gpu_batch = int(backend_batches.gpu)
    args.cpu_batch = int(backend_batches.cpu)
    args.gpu_quantize = backend_precisions.gpu
    args.cpu_precision = str(backend_precisions.cpu)
    # Compatibility values used by the established CUDA path and shared task builders.
    args.batch = int(args.gpu_batch if gpu_inference_enabled else args.cpu_batch)
    args.quantize = args.gpu_quantize
    model_paths: List[str] = []
    if gpu_model_path is not None:
        model_paths.append(f'gpu:{gpu_model_path}')
    if cpu_model_path is not None:
        model_paths.append(f'cpu:{cpu_model_path}')
    model_name = Path(str(gpu_model_path or cpu_model_path)).stem

    # Resolve the complete view request before model loading or volume decode. With no
    # implicit Cartesian/Tilted/Radial defaults, a missing or self-disabling view request must
    # fail immediately rather than decode a multi-terabyte logical run and discover it later.
    angles = resolve_tta_angles(args.angle)
    angle_variant_streaming_cleanup_active = True
    set_inference_batch_size(int(args.batch))
    try:
        enabled_cartesian_views = resolve_cartesian_views(args.enable_cartesian)
        tilt_groups = resolve_tilted_view_groups(args.enable_tilted)
        radial_requests = resolve_radial_view_requests(args.enable_radial)
        tile_configs = resolve_tile_configs(args.enable_tile)
    except ValueError as exc:
        parser.error(str(exc))

    # Concrete Tilted assembly uses the group associations directly so unrelated slots never
    # form an accidental cross-product. The flattened base list is only for Radial eligibility.
    tilt_views = tilted_group_base_views(tilt_groups)
    radial_targets = [request.view for request in radial_requests]
    if radial_requests and cpu_inference_enabled and not gpu_inference_enabled:
        skipped_targets = ', '.join(radial_targets)
        print(
            'Warning: CPU-only inference does not support Radial or Tilted-Radial views; '
            f'the following requests will be skipped: {skipped_targets}'
        )
        radial_requests = []
        radial_targets = []
    elif radial_requests and cpu_inference_enabled and gpu_inference_enabled:
        print(
            'Warning: OpenVINO CPU workers do not process Radial or Tilted-Radial views. '
            'Those enabled views remain active and will be processed exclusively by GPU workers.'
        )

    concrete_tilt_requested = bool(tilt_groups)
    active_radial_request = any(
        (not str(radial_target).startswith('tilted_'))
        or radial_target_base_view(radial_target) in tilt_views
        for radial_target in radial_targets
    )
    if not (enabled_cartesian_views or concrete_tilt_requested or active_radial_request):
        for radial_target in radial_targets:
            if str(radial_target).startswith('tilted_'):
                radial_base = radial_target_base_view(radial_target)
                if radial_base not in tilt_views:
                    print(
                        f'Radial target {radial_target!r} skipped: no {radial_base} Tilted '
                        'variants are enabled.'
                    )
        raise ValueError(
            'No inference views are active. Enable at least one view with --enable_cartesian, '
            '--enable_tilted VIEW[:TILT_ANGLE[:TILT_DIRECTION]], or '
            '--enable_radial VIEWS[:AZIMUTH_ANGLE]. A tilted_* Radial target requires '
            'a matching --enable_tilted base.'
        )

    # Resolve post-union behavior before starting the background model loader.
    # Invalid settings therefore cannot leave a heavyweight loader thread running.
    if int(args.centerline_filter_passes) < 0:
        raise ValueError('--centerline_filter_passes must be >= 0; use 0 to disable')
    if float(args.centerline_radius_factor) <= 1.0:
        raise ValueError('--centerline_radius_factor must be > 1.0')
    if int(args.centerline_temporal_context) < 0:
        raise ValueError('--centerline_temporal_context must be >= 0')
    if int(args.centerline_surface_max_dim) < 64:
        raise ValueError('--centerline_surface_max_dim must be >= 64')
    if int(args.centerline_surface_points) < 1000:
        raise ValueError('--centerline_surface_points must be >= 1000')
    if float(args.centerline_timeout) <= 0.0:
        raise ValueError('--centerline_timeout must be > 0')
    centerline_filter_enabled = bool(int(args.centerline_filter_passes) > 0)
    # Backends are process-local. Every CUDA device and every populated CPU socket owns
    # one persistent model process; the parent retains only path and scheduling metadata.
    configure_gpu_slice_labeling_devices(list(backend_devices.gpu_devices))
    model_load_executor: Optional[ThreadPoolExecutor] = None
    model_load_future: Optional[Future] = None
    if gpu_worker_process_active:
        print(
            f'Parent GPU model load skipped: {gpu_device_count} CUDA worker process(es) will '
            f'load {model_name} from {gpu_model_path} independently.'
        )
    if cpu_worker_process_active:
        print(
            'Parent CPU model load skipped: socket-local OpenVINO worker process(es) will '
            f'compile {cpu_model_path} after applying their CPU affinity.'
        )

    # Validate logical CUDA indices and resolve retina-mask placement before decode begins.
    # device uses torch LOGICAL indices into any inherited
    # CUDA_VISIBLE_DEVICES list. Validate BEFORE decode/model-load work so a submit script
    # still using the accidental physical-id convention (e.g. --device 2,3 under
    # CUDA_VISIBLE_DEVICES=2,3) fails immediately with the correction instead of hours in.
    _inherited_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if _inherited_cvd is not None and gpu_device_count > 0:
        _visible_tokens = [tok.strip() for tok in str(_inherited_cvd).split(',') if tok.strip()]
        _logical = [int(str(d).split(':')[-1]) for d in inference_devices if str(d).startswith('cuda')]
        _bad = [idx for idx in _logical if idx < 0 or idx >= len(_visible_tokens)]
        if _bad:
            hint = (
                f'use --device {",".join(str(i) for i in range(len(_visible_tokens)))} to run on all allocated GPUs'
                if _visible_tokens else 'no GPUs are visible to this job'
            )
            raise SystemExit(
                f'--device index(es) {_bad} are out of range for the inherited '
                f'CUDA_VISIBLE_DEVICES={_inherited_cvd!r} ({len(_visible_tokens)} visible device(s)). '
                f'--device uses torch LOGICAL indices into that list (v13.2.2): {hint}.'
            )
    # Mask/proto postprocessing follows the inference backend automatically. The parent
    # selects GPU semantics when CUDA is active; each OpenVINO worker explicitly selects CPU.
    retina_processor = 'gpu' if gpu_worker_process_active else 'cpu'
    retina_processor_reason = (
        'automatic: GPU inference owns GPU proto/mask processing'
        if gpu_worker_process_active else
        'automatic: OpenVINO inference owns CPU proto/mask processing'
    )
    set_retina_mask_processor(retina_processor)
    # Angle-variant + GPU-retina fast path. Every requested angle is now a separate
    # accumulator, so the per-frame radius cleanup is eligible for every variant: instances below
    # min_conf are dropped before the union/flatten, the union+confidence planes are warped to
    # view-native space on the GPU, and positive --min_radius runs with CuPy. Hole filling is a
    # completed-view or eligible task-end operation. Only the finished view-native plane crosses
    # PCIe. Every TTA angle is an independent variant, so the confidence/radius ordering
    # is variant-local regardless of how many --angle values were requested.
    angle_variant_gpu_fastpath_active = bool(
        angle_variant_streaming_cleanup_active and str(retina_processor).strip().lower() == 'gpu'
    )
    angle_variant_gpu_fastpath_min_conf_value = (
        float(args.min_conf) if angle_variant_gpu_fastpath_active else None
    )
    angle_variant_gpu_fastpath_min_radius_value = (
        float(args.min_radius) if angle_variant_gpu_fastpath_active else 0.0
    )
    set_angle_variant_gpu_fastpath(
        angle_variant_gpu_fastpath_min_conf_value, angle_variant_gpu_fastpath_min_radius_value
    )
    print(
        f'Inference devices: {inference_devices}; '
        f'GPU precision={quantize_display(args.gpu_quantize) if gpu_worker_process_active else "disabled"}, '
        f'CPU precision={args.cpu_precision if cpu_worker_process_active else "disabled"}; '
        f'GPU batch={args.gpu_batch if gpu_worker_process_active else "disabled"}, '
        f'CPU batch={args.cpu_batch if cpu_worker_process_active else "disabled"}; '
        f'mask/proto processing follows each inference backend automatically.'
    )
    if angle_variant_gpu_fastpath_active:
        print(
            'v13.1.0 angle-variant GPU retina fast path active: retina masks are flattened, '
            f'confidence-filtered (--min_conf={float(args.min_conf):.3f}), warped to view-native space '
            f'(v13.3.0 R8: identity warps skipped, grids cached), and --min_radius={float(args.min_radius):g} '
            'is applied on the GPU before the PCIe copy when positive (cupy required; CPU fallback '
            'otherwise). v13.3.0 (R8): the per-frame retina GPU 2D hole fill is removed; a '
            'completed-view pass or eligible task-end device-union pass performs it once in spec order.'
        )
    elif str(retina_processor).strip().lower() == 'gpu':
        print(
            'v13.1.0 GPU retina flatten active: the (n,H,W) retina-mask stack is reduced to union + '
            'max-confidence planes and warped to view-native space on the GPU before the PCIe copy.'
        )
    # Each CUDA inference worker is pinned through CUDA_VISIBLE_DEVICES and sees its assigned
    # device as cuda:0. Worker slice windows are disjoint within each angle variant. Workers
    # write those slices into one variant-owned scheduler mapping: memfd plus descriptor
    # transfer when available, and a real pathname only as fallback.
    gpu_worker_direct_union_active = bool(
        gpu_worker_process_active and gpu_worker_direct_union_enabled()
    )
    # Any full-frame task that CPU workers may claim must use the common view-local union
    # boundary. GPU-only views may retain the D1 owner pipeline.
    worker_direct_union_active = bool(
        gpu_worker_direct_union_active or cpu_worker_process_active
    )
    # Collect geometry-independent eligibility conditions for the v16.1.3 fast bundle.
    # Source and processing geometry are not known until ffprobe and processing-shape
    # resolution complete, so geometry-dependent activation is finalized below after T/H/W
    # are assigned. Unsupported commands retain the dense compatibility paths.
    v1613_bundle_reasons: List[str] = []
    if not v1613_fast_bundle_requested():
        v1613_bundle_reasons.append('YOLO_TTA_V1613_FAST_BUNDLE=0')
    if not gpu_worker_process_active:
        v1613_bundle_reasons.append('CUDA worker processes inactive')
    # v17.0.5 keeps D1 eligible with interpolation by retaining an exact packed
    # view-native shadow for bridge planning while source-space publication remains authoritative.
    if float(args.min_conf) > 0.0:
        v1613_bundle_reasons.append('requires --min_conf 0')
    if float(args.min_radius) > 0.0:
        v1613_bundle_reasons.append('requires --min_radius 0')
    if int(args.batch) != 1:
        v1613_bundle_reasons.append('requires --batch 1')
    if str(retina_processor).strip().lower() != 'gpu':
        v1613_bundle_reasons.append('requires GPU retina processing')
    # Dense tiles no longer disqualify D1: their parent/bridge gates consume the same exact
    # packed view-native shadow used by interpolation.
    if str(channel_format.kind) != 'gray' or int(channel_format.channel_count) != 1:
        v1613_bundle_reasons.append('requires one-channel gray input')
    if not bool(save_nrrd_enabled):
        v1613_bundle_reasons.append('requires --save nrrd for per-view source-space layers')
    if not raw_bbox_nrrd_layers_enabled():
        v1613_bundle_reasons.append('requires raw-bbox/cvol NRRD layers')
    if not gpu_device_union_enabled():
        v1613_bundle_reasons.append('requires task-local GPU device unions')


    if args.min_conf > 0 and args.min_conf < args.conf:
        raise ValueError('--min_conf must be equal to or greater than --conf')
    if int(args.interpolation_distance) < 0:
        raise ValueError('--interpolation_distance must be >= 0')
    if int(args.interpolation_walk_back) < 0:
        raise ValueError('--interpolation_walk_back must be >= 0')
    if int(args.interpolation_candidates) < 1:
        raise ValueError('--interpolation_candidates must be >= 1')
    if int(args.interpolation_passes) < 1:
        raise ValueError('--interpolation_passes must be >= 1')
    gaussian_smoothing_cli_requested = bool(args.gaussian_smoothing is not None or args.gaussian_smoothing_passes is not None)
    gaussian_smoothing_disabled_by_zero = False
    gaussian_smoothing_enabled, gaussian_smoothing_sigma, gaussian_smoothing_passes = resolve_gaussian_smoothing_settings(
        args.gaussian_smoothing,
        args.gaussian_smoothing_passes,
    )
    if float(args.interpolation_min_radius) < 0:
        raise ValueError('--interpolation_min_radius must be >= 0')
    if float(args.min_radius) < 0:
        raise ValueError('--min_radius must be >= 0')
    if not (-90.0 < float(args.interpolation_search_angle) < 90.0):
        raise ValueError('--interpolation_search_angle must be greater than -90 and less than 90')
    low_quality_requested = bool(save_low_quality_enabled)
    # NRRD component layers are produced only for --save nrrd, so requesting only
    # low-quality outputs writes the low-quality videos but no NRRD output.
    # when both --save nrrd and a low-quality request are active, the low-quality
    # NRRD decomposition mirrors the full-quality layers (one downbinned single-layer NRRD per
    # component layer) on the same view-completion schedule, rather than one combined tail volume.
    # Keep these booleans separate. ``nrrd_layers_needed`` controls the established,
    # expensive per-view decomposition and projected-layer behavior. audit-only
    # runs need a sink but must not broaden any of those paths.
    nrrd_layers_needed = bool(save_nrrd_enabled)
    centerline_audit_nrrd_needed = bool(centerline_filter_enabled)
    nrrd_sink_needed = bool(nrrd_layers_needed or centerline_audit_nrrd_needed)
    keep_temp_artifacts = bool(_env_flag('YOLO_TTA_KEEP_TEMP', False))

    out_dir = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / input_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = choose_scratch_dir(args.temp, out_dir, input_path.stem)
    register_unique_run_scratch_cleanup(
        temp_dir,
        keep_temp=bool(keep_temp_artifacts),
    )
    expose_scratch_in_output(out_dir, temp_dir)
    # Say which kind of scratch this is. Memory-backed scratch keeps tile residuals,
    # per-angle accumulators, and the shared source volume off persistent storage; disk-backed
    # scratch remains the safe default.
    print(
        f"Bulk scratch dir: {temp_dir} "
        f"[{'MEMORY-backed (' + str(_mount_fstype_for_path(temp_dir) or 'tmpfs') + ')' if scratch_dir_is_memory_backed() else 'disk-backed'}, "
        f"free={_filesystem_free_bytes(temp_dir) / GIB:.1f} GiB]"
    )
    output_memory_backed = bool(path_is_memory_backed(out_dir))
    print(
        f"Final output dir: {out_dir} "
        f"[{'MEMORY-backed (' + str(_mount_fstype_for_path(out_dir) or 'tmpfs') + ')' if output_memory_backed else 'disk/network-backed'}, "
        f"free={_filesystem_free_bytes(out_dir) / GIB:.1f} GiB]"
    )
    if bool(args.temp) and scratch_dir_is_memory_backed() and not output_memory_backed:
        print(
            'Important: --temp redirects scratch work only. Final MKV/NRRD/manifest files are '
            f'written to --output ({out_dir}); benchmark runs must use the same --output filesystem '
            'to compare output-stage wall time fairly.'
        )

    info = ffprobe_info(input_path)
    input_W = int(info['width'])
    input_H = int(info['height'])
    input_T = int(info['num_frames'])
    fps = float(info['fps'])

    low_quality_downbin_specs, low_quality_downbin_warnings = resolve_low_quality_downbin_specs(
        low_quality_downbin_values,
        bool(low_quality_requested),
        (input_T, input_H, input_W),
    )

    # --save nrrd writes one single-layer NRRD per component layer (in source output
    # geometry X,Y,t) as the layers are produced during the intermediate pipeline steps. The sink
    # is configured here (output geometry is now known) and torn down in the tail after the final
    # global layers are materialized.
    # when --save low_quality is also active, the sink mirrors every component layer into
    # one downbinned single-layer NRRD per spec under low_quality/<token>/nrrd/ on the same
    # view-completion schedule as the full-quality layers (the tail no longer writes a combined
    # low-quality NRRD).
    # record the final source output geometry so radial/tilted
    # NRRD layer projections and the final backprojection queue target it directly (one resample).
    set_final_source_output_shape((input_T, input_H, input_W))

    nrrd_dir = out_dir / 'nrrd'
    if bool(nrrd_sink_needed):
        sink_low_quality_specs = (
            list(low_quality_downbin_specs)
            if bool(save_nrrd_enabled) and bool(low_quality_requested)
            else []
        )
        layer_sink_for_run = NrrdLayerSink(
            nrrd_dir=nrrd_dir,
            stem=input_path.stem,
            output_shape_tyx=(input_T, input_H, input_W),
            max_workers=nrrd_layer_sink_workers(),
            low_quality_specs=sink_low_quality_specs,
            low_quality_root=(out_dir / 'low_quality'),
        )
        _run_resources().track_sink(layer_sink_for_run)
        set_nrrd_layer_sink(layer_sink_for_run)
    else:
        set_nrrd_layer_sink(None)

    preprocess_streaming_active = bool(streaming_preprocess_enabled())
    vol_path = temp_dir / 'input_volume.gray8.dat'
    # Resolve cube geometry before decode. Inference worker processes need a reopenable source, so the
    # decoded native volume uses a transferred memfd whenever it is their source. A real
    # pathname is retained only as the fallback when memfd is unavailable.
    input_processing_shape = (int(input_T), int(input_H), int(input_W))
    legacy_cube_shape = compute_cube_resize_shape(input_T, input_H, input_W, tolerance=0.05)
    processing_mode = processing_volume_mode()
    cube_resize_will_apply = bool(should_resize_to_processing_cube(input_processing_shape, legacy_cube_shape))
    # Inference workers open the shared source volume as a file. Without a cube resize that
    # file is the decode target itself; extends the same rule to cube runs so
    # workers can resident-upload the NATIVE decoded volume (t-resizing on device) without
    # a copy pass first.
    decode_prefer_memory = not bool(inference_worker_process_active)
    if preprocess_streaming_active:
        print(
            'v12.2.15 streaming preprocessing active: ffmpeg decode returns its destination array immediately; '
            'Transverse/native consumers wait only for the needed decoded slice. Legacy cube resize, if explicitly enabled, also streams.'
        )
        input_volume_rgb = decode_video_to_memmap_gray8_streaming(
            input_video=input_path,
            out_dat=vol_path,
            num_frames=input_T,
            width=input_W,
            height=input_H,
            overwrite=False,
            prefer_memory=decode_prefer_memory,
            prefer_memfd=bool(inference_worker_process_active and not decode_prefer_memory),
        )
    else:
        input_volume_rgb = decode_video_to_memmap_gray8(
            input_video=input_path,
            out_dat=vol_path,
            num_frames=input_T,
            width=input_W,
            height=input_H,
            overwrite=False,
            prefer_memory=decode_prefer_memory,
            prefer_memfd=bool(inference_worker_process_active and not decode_prefer_memory),
        )
    (temp_dir / 'input_volume.meta.json').write_text(
        json.dumps({
            'shape': [input_T, input_H, input_W],
            'dtype': 'uint8',
            'channels': 1,
            'source_channels': 1,
            'model_channel_format': channel_format.token,
            'model_input_channels': int(channel_format.channel_count),
            'fps': fps,
            'streaming_preprocess': bool(preprocess_streaming_active),
        }, indent=2)
    )

    if cube_resize_will_apply:
        processing_shape = legacy_cube_shape
        print(
            'v13.0.0 processing geometry: approximately-cubic working volume (default). '
            f'input shape (t,Y,X)=({input_T},{input_H},{input_W}) -> '
            f'processing shape (t,Y,X)={processing_shape} (within 5% of the longest source axis).'
        )
        # The native-t resident CUDA path needs only the logical cube shape; workers upload
        # the logical cube shape. They upload the decoded native volume and compose the
        # t map on device, so constructing the 25+ GiB host cube eagerly is wasted work
        # unless residency/CPU/tile rendering actually falls back.
        lazy_cube_eligible = bool(
            gpu_worker_process_active
            and gpu_cube_resize_enabled()
            and tuple(int(v) for v in processing_shape[1:])
            == tuple(int(v) for v in input_processing_shape[1:])
            and _cube_t_axis_resize_backend() == 'slab'
        )
        if lazy_cube_eligible:
            lazy_cube_path = temp_dir / 'input_volume.v950_cube.gray8.dat'
            volume_rgb = LazyProcessingCube(
                input_volume_rgb,
                processing_shape,
                lazy_cube_path,
                workers=max(1, default_worker_budget()),
                request_path=temp_dir / 'gpu_worker_source_volume.cube_request.sentinel',
                ready_path=temp_dir / 'gpu_worker_source_volume.cube_ready.sentinel',
                failed_path=temp_dir / 'gpu_worker_source_volume.cube_failed.txt',
                streaming_backend=bool(preprocess_streaming_active),
            )
            volume_rgb.start_request_watcher()
            print(
                'v13.3.17 C10: host processing cube deferred; native-t GPU residency '
                f'will avoid {volume_rgb.nbytes / GIB:.2f} GiB of host construction unless '
                'a file-backed fallback requests it.'
            )
        elif preprocess_streaming_active:
            volume_rgb = resize_volume_to_processing_cube_gray8_streaming(
                input_volume_rgb,
                processing_shape,
                temp_dir / 'input_volume.v950_cube.gray8.dat',
                workers=max(1, default_worker_budget()),
                prefer_memory=not bool(inference_worker_process_active),
            )
        else:
            volume_rgb = resize_volume_to_processing_cube_gray8(
                input_volume_rgb,
                processing_shape,
                temp_dir / 'input_volume.v950_cube.gray8.dat',
                workers=max(1, default_worker_budget()),
                prefer_memory=not bool(inference_worker_process_active),
            )
    else:
        processing_shape = input_processing_shape
        volume_rgb = input_volume_rgb
        if processing_mode == 'cube' and legacy_cube_shape == input_processing_shape:
            print(f'v13.0.0 processing geometry: approximately-cubic (default), input already within 5% cube tolerance ({processing_shape}).')
        else:
            print(
                'v13.0.0 processing geometry: native decoded volume (YOLO_TTA_PROCESSING_VOLUME_MODE=native), no cube resize. '
                f'input/processing shape (t,Y,X)={processing_shape}; approximately-cubic target would have been {legacy_cube_shape}. '
                'Unset YOLO_TTA_PROCESSING_VOLUME_MODE to use the default approximately-cubic working geometry.'
            )

    T, H, W = (int(processing_shape[0]), int(processing_shape[1]), int(processing_shape[2]))

    # Finalize the command-specialized bundle only after both source and processing
    # geometry are resolved. This ordering prevents startup from reading H/W or input_H/W
    # before assignment while preserving the original eligibility contract.
    if int(H) != int(input_H) or int(W) != int(input_W):
        v1613_bundle_reasons.append(
            'requires processing XY to equal source XY (only the T axis may be cube-rescaled)'
        )
    v1613_bundle_active = bool(not v1613_bundle_reasons)
    os.environ['YOLO_TTA_V1613_BUNDLE_ACTIVE'] = '1' if v1613_bundle_active else '0'
    # This variable is a parent-to-worker resolved-state publication, not a user input.
    # Clear inherited worker state before resolving this run's canonical request.
    os.environ.pop('YOLO_TTA_V1613_D1_PIPELINE_ACTIVE', None)
    if v1613_bundle_active:
        # v16.1.8: the fast bundle keeps hardware-linear texture sampling. The pointer
        # mode (nearest_xy_linear_t) benchmarked no faster on the standard command, so it
        # is opt-in for VRAM-constrained volumes rather than the bundle default.
        os.environ.setdefault('YOLO_TTA_RADIAL_SOURCE_MODE', 'texture_linear')
        os.environ.setdefault('YOLO_TTA_PROTO_HOLE_TREATMENT', 'close')
        os.environ.setdefault('YOLO_TTA_PROTO_HOLE_RADIUS', '2')
        os.environ.setdefault('YOLO_TTA_GPU_UNION_RETIREMENT_LANES', '3')
    v1613_d1_owner_active = bool(v1613_bundle_active and d1_owner_pipeline_enabled())
    os.environ['YOLO_TTA_V1613_D1_PIPELINE_ACTIVE'] = (
        '1' if v1613_d1_owner_active else '0'
    )
    if v1613_d1_owner_active:
        # D1 supersedes the 25-39 GiB host direct-union workspace entirely.
        gpu_worker_direct_union_active = False
        print(
            'v16.1.8 fast bundle active: hardware-linear Radial texture sampling, B1 sparse '
            'slice metadata, D3 resident-proto closing, C1 runtime-sized leases, C2 '
            'compute/publication credit separation, C3 predicted-cost scheduling, and D1 '
            'project -> infer -> proto-close -> immediate owner-GPU backprojection -> '
            'source-space sparse cvol publication. Commands with interpolation and/or tiles '
            'retain an exact packed view-native shadow and materialize it only in the asynchronous '
            'parent postprocess stage; the scheduler never owns a dense inference result. '
            'YOLO_TTA_V1613_FAST_BUNDLE=0 restores the compatibility paths.'
        )
    elif v1613_bundle_active:
        print(
            'v16.1.8 fast bundle active with D1 disabled by '
            'YOLO_TTA_V1613_D1_OWNER_PIPELINE=0: '
            'hardware-linear Radial texture sampling, B1/D3, and C1-C3 remain active; the '
            'dense direct-union compatibility path is retained.'
        )
    elif v1613_fast_bundle_requested():
        print(
            'v16.1.8 fast bundle not eligible for this command; compatibility paths retained: '
            + '; '.join(v1613_bundle_reasons)
        )
    # D1 may disable dense GPU unions after the initial backend resolution. Recompute the
    # common process-worker requirement so GPU-only D1 runs stay owner-only while hybrid
    # runs can allocate a shareable direct union only for a view actually claimed by OpenVINO.
    worker_direct_union_active = bool(
        gpu_worker_direct_union_active or cpu_worker_process_active
    )
    if cpu_worker_process_active and not gpu_worker_direct_union_active:
        print(
            '[intel] v17.0.3 hybrid ownership: OpenVINO receives an ordered, bounded '
            'reservation sequence and opens one process-shareable Cartesian/Tilted direct union '
            'at a time. Unreserved eligible views remain owner-local CUDA D1. ETA-driven CUDA '
            'assistance applies only to the active CPU view and never globally disables later reservations.'
        )
    if gpu_worker_direct_union_active:
        print(
            'GPU-worker direct union writes active: angle-variant worker tasks write '
            'their disjoint slice windows straight into a bounded variant-owned union '
            '(memfd preferred; no per-task result files or scheduler-side OR pass). '
            'Set YOLO_TTA_GPU_WORKER_DIRECT_UNION=0 to select per-task result files.'
        )

    radial_diameters = [
        radial_target_diameter(target, int(T), int(H), int(W))
        for target in radial_targets
    ]
    resolved_azimuth_angles = resolve_radial_azimuth_angles(
        radial_requests,
        diameters=radial_diameters,
    )
    for request, diameter, spacing in zip(
        radial_requests, radial_diameters, resolved_azimuth_angles,
    ):
        if request.azimuth_angle is None:
            print(
                f'Radial azimuth default [{request.view}]: using full-coverage spacing '
                f'{float(spacing):.8g}° for projected-plane diameter {int(diameter)}'
            )

    (temp_dir / 'processing_volume.meta.json').write_text(
        json.dumps({
            'input_shape_t_y_x': [input_T, input_H, input_W],
            'processing_shape_t_y_x': [T, H, W],
            'processing_volume_mode': processing_mode,
            'legacy_cube_shape_t_y_x': [int(legacy_cube_shape[0]), int(legacy_cube_shape[1]), int(legacy_cube_shape[2])],
            'cube_resize_applied': bool(tuple(int(x) for x in processing_shape) != tuple(int(x) for x in input_processing_shape)),
            'dtype': 'uint8',
            'channels': 1,
            'source_channels': 1,
            'model_channel_format': channel_format.token,
            'model_input_channels': int(channel_format.channel_count),
            'model_channel_stride': int(channel_format.stride),
            'model_channel_offsets': [int(v) for v in channel_format.offsets],
            'model_channel_boundary_policy': 'radial_wrap_mirror_u_cartesian_edge_clamp',
            'model_prediction_slice_policy': 'center_N_only',
            'fps': fps,
            'enable_cartesian': list(enabled_cartesian_views),
            # Retain the prior flattened metadata keys for readers that only need summaries;
            # the structured group records below are authoritative for associations.
            'enable_tilted': list(tilt_views),
            'tilt_angles_deg': list(dict.fromkeys(
                float(angle)
                for group in tilt_groups
                for angle in group.tilt_angles
            )),
            'tilt_directions': list(dict.fromkeys(
                str(direction)
                for group in tilt_groups
                for direction in group.tilt_directions
            )),
            'enable_tilted_groups': [
                {
                    'views': list(group.views),
                    'tilt_angles_deg': [float(v) for v in group.tilt_angles],
                    'tilt_directions': list(group.tilt_directions),
                }
                for group in tilt_groups
            ],
            'enable_radial': list(radial_targets),
            'radial_diameters': [int(v) for v in radial_diameters],
            'azimuth_angles_deg': [float(v) for v in resolved_azimuth_angles],
            'enable_radial_groups': [
                {
                    'view': request.view,
                    'requested_azimuth_angle_deg': (
                        'auto' if request.azimuth_angle is None else float(request.azimuth_angle)
                    ),
                    'diameter': int(diameter),
                    'resolved_azimuth_angle_deg': float(spacing),
                }
                for request, diameter, spacing in zip(
                    radial_requests, radial_diameters, resolved_azimuth_angles,
                )
            ],
            'enable_tile': [
                {
                    'tile_size': int(config.tile_size),
                    'tile_stride': int(config.tile_stride),
                }
                for config in tile_configs
            ],
        }, indent=2)
    )

    # Fold each Radial transformed stack/diameter to --imgsz unless explicitly disabled.
    radial_fold_raster = int(args.imgsz) if _env_flag('YOLO_TTA_RADIAL_FOLD_IMGSZ', True) else 0
    physical_views = get_view_infos(
        T=T,
        H=H,
        W=W,
        cartesian_views=enabled_cartesian_views,
        radial_views=radial_targets,
        radial_azimuth_angles=resolved_azimuth_angles,
        tilt_groups=tilt_groups,
        radial_native_raster=int(radial_fold_raster),
    )
    if not physical_views:
        raise ValueError(
            'No inference views are active. Enable at least one view with --enable_cartesian, '
            '--enable_tilted VIEW[:TILT_ANGLE[:TILT_DIRECTION]], or '
            '--enable_radial VIEWS[:AZIMUTH_ANGLE]. A tilted_* Radial target is skipped '
            'when its matching Tilted base is not enabled.'
        )

    # v16.4.0: TTA rotations are first-class view variants. Every runtime view owns exactly
    # one augmentation, one full-frame accumulator, one tile gate graph, one interpolation
    # graph, and its own NRRD namespace. The physical view list is retained only for the final
    # post-variant OR/backprojection stage.
    views = expand_views_into_tta_variants(physical_views, angles)
    # Reserve the second resident allocation only for runs that actually contain a
    # Radial task. This keeps Cartesian/Tilted-only runs from losing GPU residency merely
    # because the variant supports a lazily-created hardware texture.
    radial_texture_required = bool(
        radial_source_mode() == 'texture_linear'
        and any(is_radial_view(view) for view in physical_views)
    )
    cartesian_views = orthogonal_views_only(physical_views)
    inference_views = list(views)
    interpolating_views = [v for v in inference_views if _view_uses_interpolation(v, int(args.interpolation_distance))]

    # A tilted_* Radial token expands to every concrete signed/directional Tilted variant.
    # Print the physical workload and the expanded TTA-variant workload separately.
    radial_concrete_views = [v for v in physical_views if is_radial_view(v)]
    tilted_radial_concrete_views = [v for v in radial_concrete_views if is_tilted_radial_view(v)]
    upright_radial_concrete_views = [v for v in radial_concrete_views if not is_tilted_radial_view(v)]
    upright_tilted_views = [v for v in physical_views if is_tilted_view(v)]
    source_frames_per_angle = int(sum(int(v.num_slices) for v in physical_views))
    radial_frames_per_angle = int(sum(int(v.num_slices) for v in radial_concrete_views))
    tilted_radial_frames_per_angle = int(sum(int(v.num_slices) for v in tilted_radial_concrete_views))
    radial_expansion_counts = Counter(
        str(v.radial_request_token or radial_base_view_name(v))
        for v in radial_concrete_views
    )
    radial_expansion_note = ', '.join(
        f'{token}=>{int(count)} concrete view(s)'
        for token, count in radial_expansion_counts.items()
    ) or 'none'
    print(
        'Concrete view workload: '
        f'{len(physical_views)} physical view(s) = {len(cartesian_views)} Cartesian + '
        f'{len(upright_tilted_views)} Tilted + {len(upright_radial_concrete_views)} upright Radial + '
        f'{len(tilted_radial_concrete_views)} tilted-Radial; '
        f'{source_frames_per_angle} source frame(s)/--angle, '
        f'{len(inference_views)} independent view-angle variant(s), and '
        f'{source_frames_per_angle * max(1, len(angles))} total model frame(s) across '
        f'{max(1, len(angles))} TTA angle(s).'
    )
    if radial_concrete_views:
        print(
            'Radial concrete expansion: '
            f'{radial_expansion_note}; radial={radial_frames_per_angle} frame(s)/angle, '
            f'tilted-Radial={tilted_radial_frames_per_angle} frame(s)/angle.'
        )
    if tilted_radial_concrete_views:
        print(
            'Tilted-Radial resident CUDA rendering is enabled '
            '(YOLO_TTA_GPU_TILTED_RADIAL_RENDER=0 selects the CPU fallback). '
            'A worker that cannot admit the source volume to VRAM will still use the completed-cube CPU path.'
        )
    spec_notes: List[str] = []
    spec_notes.append(
        'v17.0.5 GPU feed/D1 continuation patch: four whole physical feeder cores per CUDA '
        'worker are topology-local and exclusive during inference; parent/OpenVINO/helper masks '
        "exclude another worker's feeder cores and the parent reclaims the full allocation at "
        'global drain. Auxiliary interpolation cannot borrow a CUDA worker interpreter before '
        'that drain. D1 remains active with interpolation and dense tiles by publishing its '
        'source-space base immediately, retaining one exact packed view-native shadow for parent '
        'support, and backprojecting only interpolation/tile additions thereafter. The D1 '
        'dispatch-window default remains two and topology backend auto-selection remains enabled.'
    )
    spec_notes.append(
        'v17.0.4 interpolation resource patch: sparse-label packing releases source crops '
        'incrementally; interpolation planning and its component/SDF caches are byte-bounded; '
        'accepted plans render in bounded batches; adjacent-slice endpoint continuation uses '
        'exact bbox label reads instead of quadratic component-pair scans; parent and '
        'consolidated-tile interpolation share one global pass boundary; '
        'tile outer/inner worker defaults divide the job CPU budget instead of multiplying it; '
        'process-reopenable tile accumulators avoid anonymous-to-path full-volume copies; and '
        'completed dense views retire through immutable terminal refs (private no-NRRD refs '
        'use row-wise packbits); per-pass telemetry reports real plan/cache/label/backing usage.'
    )
    spec_notes.append(
        'v17.0.3 hybrid scheduler patch: CPU lease target remains 10 seconds and seed ranges '
        'use the larger eligible backend target. Hybrid full-frame views are partitioned into an '
        'ordered CPU reservation sequence (default three views) and immediately CUDA-owned D1 '
        'views. OpenVINO opens one reserved direct-union view at a time, may open the next after '
        'the prior view drains, and is no longer disabled by a global GPU-claim latch. Stealback '
        'uses only the active CPU view ETA, waits for measured CPU samples while mandatory GPU work '
        'remains, and borrows only the proportional CUDA capacity required to converge with that horizon. '
        'After the mandatory backlog drains, all CUDA workers may finish the active reservation. Frame totals '
        'and per-view backend splits are logged.'
    )
    spec_notes.append(
        'v17.0.3 Intel inference update: --model accepts gpu:/PATH and cpu:/PATH; '
        '--device selects GPU indexes, cpu, or GPU_INDEXES:cpu; --quantize and --batch '
        'accept backend-qualified gpu:/cpu: values. OpenVINO runs in persistent socket-local '
        'processes, automatically claims the reserved Cartesian sequence before any reserved '
        'Tilted Cartesian fallback, and never claims Radial/Tilted-Radial work. Unreserved '
        'eligible views remain immediately available to CUDA D1; already-running OpenVINO '
        'requests are never preempted. OpenVINO remains socket-local, low-duty parent/CUDA '
        'helper overlap is enabled by default, and GPU affinity remains per-device topology-discovered. '
        'Mask/proto processing follows the inference backend automatically; --retina_mask_processor '
        'and manual CPU-offload/GPU-steal switches do not exist. Hybrid model identity is warning-only, '
        'and BF16 output quality is accepted as a v17 specification assumption.'
    )
    effective_save_options = list(save_options)
    if bool(low_quality_requested) and 'low_quality' not in effective_save_options:
        effective_save_options.append('low_quality')
    spec_notes.append(
        'v16.2.0 unified output selection: --save=' + (
            ', '.join(effective_save_options) if effective_save_options else '<none>'
        ) + '. high_quality controls the native-resolution final overlay; summary controls the '
        'summary text file; labels, binary, images, low_quality, and nrrd remain independently '
        'selectable with their established paths and filenames.'
    )
    spec_notes.append(
        'v17.0.10 channel seam handling: Radial and Tilted Radial channel offsets wrap '
        'modulo their angular frame count and reverse radial-u after each odd 0°/180° seam '
        'crossing, while Cartesian and Tilted Cartesian offsets clamp '
        'at the stack ends. With --save images, C>=5 channel inputs are written as multi-page '
        'TIFFs containing one uint8 grayscale page per channel in model-input order. The retired '
        '--troubleshooting CLI and legacy environment-variable aliases are removed.'
    )
    spec_notes.append(
        'v16.3.0 CLI maintenance: --enable_radial uses VIEWS:AZIMUTH_ANGLE groups, '
        '--enable_tilted uses VIEW:TILT_ANGLE:TILT_DIRECTION groups, and --enable_tile uses '
        'TILE_SIZE:TILE_STRIDE groups. Spaces separate groups and commas select multiple values '
        'inside a slot. The interpolation flags are --interpolation_distance, '
        '--interpolation_passes, and --interpolation_min_radius. Component-NRRD decode/deflate '
        'continues during topology instead of pausing at member boundaries. Dead telemetry-detail, '
        'CPU-retina override, NRRD-yield, and synchronous scratch-msync environment controls are removed.'
    )
    if tilted_radial_concrete_views:
        spec_notes.append(
            'Concrete tilted-Radial views use the resident CUDA source '
            'renderer and persistent batch-1 TensorRT ring instead of silently materializing the '
            'deferred host cube and running the active-filter/sheared frame renderer on the CPU. '
            'YOLO_TTA_GPU_TILTED_RADIAL_RENDER=0 selects the compatibility CPU path.'
        )
    # spec<->implementation conflict notes (see header docstring and suggested spec edits).
    spec_notes.append(
        'CONFLICT NOTE 1 (--min_radius): the task says --min_radius is now applied per view in each '
        'prediction set\'s own native 2D slice plane before backprojection. Spec flag #8 and item 5 '
        'already agree, but earlier prose ("transverse-plane radius", deferred Sagittal/Coronal pass, '
        'post-backprojection Radial/Tilted pass) conflicted. Per the task, --min_radius is now applied '
        'on every view\'s analysis slices during per-view cleanup (native raster in compatibility '
        'mode; radius-scaled canonical inference raster under v13.3.12 D6) and is NOT re-applied after '
        'backprojection. Suggested spec edit: delete every "transverse plane" reference for --min_radius '
        'and state "measured on the YOLO output masks in each prediction set\'s own native 2D slice '
        'plane, before backprojection, independently per active view".'
    )
    if str(retina_processor).strip().lower() == 'gpu':
        spec_notes.append(
            'CUDA-task path (v13.1.0 #2.2): GPU retina-mask flatten + warp. The (n,H,W) retina-mask stack is reduced on '
            'the GPU to a union plane and a max-confidence plane, and both are warped to the view analysis grid '
            'on the GPU (torch grid_sample), so only those reduced planes cross PCIe (O(2*H*W)); no '
            'affine warp and no per-instance loop run on the CPU.'
        )
    if angle_variant_gpu_fastpath_active:
        spec_notes.append(
            'CUDA-task path (v13.1.0 #2.3 + v13.3.0 R8): angle-variant GPU fast path. YOLO -> proto-resolution union '
            '(R9) -> --min_conf -> warp (identity-skipped/grid-cached) -> --min_radius (cupy, only when '
            'positive) run on the GPU, then one finished view-native plane is sent to the CPU. The '
            'per-frame retina GPU 2D hole fill is removed: a completed-view pass or eligible task-end '
            'device-union pass fills once, preserving --min_conf -> --min_radius -> hole fill.'
        )
    if str(retina_processor).strip().lower() == 'gpu':
        spec_notes.append(
            'CUDA-task path (v13.3.0 R9/R18/R1/R21): GPU retina unions are reduced at PROTO resolution inside a patched '
            'construct_result (one plane upsampled per frame instead of an (n, imgsz, imgsz) retina stack; '
            'YOLO_TTA_GPU_PROTO_UNION=0 restores the native path); the GPU postprocess tail runs on '
            'per-thread side CUDA streams with pinned D2H staging (YOLO_TTA_GPU_POSTPROCESS_STREAM / '
            'YOLO_TTA_GPU_POSTPROCESS_PINNED); CUDA workers render full-frame views on their own '
            'GPU when the source volume fits resident (YOLO_TTA_GPU_RENDER / YOLO_TTA_GPU_RENDER_RESIDENT '
            '/ YOLO_TTA_GPU_RENDER_RESERVE_GIB), with upright radial tasks GPU-prerendered from '
            'orientation-aware logical-stack blocks '
            'otherwise (YOLO_TTA_GPU_RENDER_TBLOCK_SLICES).'
        )
    spec_notes.append(
        'v13.3.1 (R3/R4/R7b/R12/R16/R19/R24/R25): same-host shared-mapping coherence needs '
        'no synchronous scratch flushes; v16.3.0 removes the obsolete opt-in msync path because '
        'ephemeral mappings are retired rather than treated as durability records. Angle-variant '
        'CUDA workers write disjoint angle-variant result windows directly into bounded memfd-backed variant unions '
        '(YOLO_TTA_GPU_WORKER_DIRECT_UNION); NRRD payload writes stream bounded double-buffered blocks '
        '(YOLO_TTA_NRRD_Z_CHUNK_SLICES) filled by a GIL-releasing pool (YOLO_TTA_NRRD_FILL_WORKERS) '
        'that overlaps the selected gzip backend; CUDA workers receive the native source through descriptor-transferred memfd storage '
        '(no post-decode copy to scratch); foreground scans and the raw-bbox encoder no longer make '
        'full-slice cast copies, and interpolation bridge-delta layers reuse the pass added_voxels '
        'stat instead of rescanning; the interpolation merge visits only schedule-touched slices; '
        'the NRRD member-gzip level defaults to 3 (YOLO_TTA_NRRD_GZIP_LEVEL), and low-quality '
        'MP4s use x264 '
        'preset slow.'
    )
    spec_notes.append(
        'Capable views remain on the canonical angle-0 inference raster through '
        'union, component labeling, interpolation/SDF work, tile gating, and sparse-layer encoding; '
        'Cartesian, Tilted, and Radial contributors receive one terminal mapped restore to source geometry. '
        'Radius and interpolation-cone thresholds are scaled by the smaller in-plane axis factor; '
        'intermediate added/accepted voxel counters are therefore processing-grid pixel units. '
        'YOLO_TTA_DELAY_NATIVE_EXPANSION=0 restores native-view accumulation.'
    )
    spec_notes.append(
        'v13.3.14 D6/D7/G8/P4: CUDA slice labeling emits exact adjacent-slice pair codes before '
        'releasing each device block and distributes blocks across the selected devices '
        '(YOLO_TTA_GPU_SLICE_LABELING_PAIRS / YOLO_TTA_GPU_SLICE_LABELING_DEVICES); CUDA '
        'workers overlap one sealed device-union flush with the next task '
        '(YOLO_TTA_GPU_UNION_FLUSH_OVERLAP); and resident-ring Radial/Tilted frames can render '
        'directly from the uint8 source volume into the TensorRT binding '
        '(YOLO_TTA_FUSED_DIRECT_RENDER and the per-family renderer gates), with stable launches '
        'captured by YOLO_TTA_FUSED_RENDER_CUDA_GRAPHS.'
    )
    spec_notes.append(
        'v17.0.8 / N17/N15: complete-member NRRD compression prefers validated hardware-only '
        'QAT, then libdeflate, ISA-L, or zlib via YOLO_TTA_NRRD_MEMBER_CODEC; `cpu` opts out '
        'of QAT while retaining the CPU chain, and explicit `iaa` remains hardware-only; '
        'global z-shard counts are resolved '
        'at sink execution against the shared band capacity, and the ordered shard queue defaults '
        'to 32 items.'
    )
    spec_notes.append(
        'v13.3.18 C11/C12/C13/C14/N21: the process-per-GPU scheduler keeps only a bounded issued '
        'window, prioritizes parent-unlocking work, and splits a full-frame lease only at the '
        'actual dispatch tail; deferred device unions publish '
        'as soon as their D2H Future retires; the angle-variant Radial path pipelines projection '
        'and fuses YOLO+bridge layers; sparse topology unions are compiled in larger batches and '
        'G5 may use idle GPUs; FFV1 shards remain in scratch and videos enter publication only '
        'after an atomic replace; '
        'all single-layer NRRDs use full reference rasters; compact per-layer NRRD cropping is removed.'
    )
    spec_notes.append(
        'v13.3.2 (R5/R6/R10/R14/R15/R21): CPU tilted rendering hoists all frame-invariant shear '
        'math into the render plan (contiguous row fast path where separable, single clipped flat '
        'gather otherwise); radial backprojection skips empty cross-sections and streams through '
        'the GPU when available (YOLO_TTA_GPU_BACKPROJECT); the interpolation compact relabel is '
        'bbox-restricted and numba-nogil when available; keep_objects harvests component areas '
        'during 2D labeling (no full-volume bincount, no compact relabel, untouched slices are '
        'never rewritten); tilted scatter and CUDA input staging drop redundant casts/copies '
        '(staging now ships pinned uint8 and normalizes on device); coronal frame reads and '
        'projection writes go through K-column transposed blocks (YOLO_TTA_CORONAL_BLOCK_COLS / '
        'YOLO_TTA_CORONAL_BLOCK_CACHE); global/transverse NRRD layers encode straight from the '
        'source volume (no pre-encode copy); and compatible GPU-retina tasks accumulate raw '
        'task-local unions in an on-device volume with one chunked pinned D2H per task '
        '(YOLO_TTA_GPU_DEVICE_UNION) when VRAM allows. Angle-variant confidence/radius cleanup '
        'is variant-local; every angle is cleaned before physical-view union.'
    )
    if preprocess_streaming_active:
        spec_notes.append(
            'v12.2.15 streaming preprocessing is active: decoded native slices become available as ffmpeg produces them. Transverse readers wait only for the needed decoded slice; stack-sampling view families wait for the completed decoded volume. Legacy cube resize, when explicitly enabled, still streams its output slices.'
        )
    else:
        spec_notes.append(
            'v12.2.15 streaming preprocessing is disabled by YOLO_TTA_STREAMING_PREPROCESS=0; decode finishes before inference scheduling begins, and legacy cube resize runs only when explicitly requested.'
        )
    spec_notes.append(
        'v17.0.3 process-local model startup: the parent loads no inference model. Each persistent '
        'CUDA worker deserializes its GPU engine, and each socket-local CPU worker compiles its '
        'OpenVINO model only after applying its CPU affinity. Startup overlaps the default streaming '
        'decode producer; YOLO_TTA_STREAMING_PREPROCESS=0 makes decode synchronous.'
    )
    spec_notes.append(
        'v16.4.0 angle-variant streaming cleanup is mandatory: every --angle variant is '
        'confidence-filtered and analysis-grid min_radius-filtered as it streams. D6 scales '
        'the threshold on a reduced canonical raster, and 2D hole filling runs after '
        '--min_conf and --min_radius before that variant is interpolated.'
    )
    spec_notes.append(
        'Input-channel handling: RGB/YUV video is flattened to one gray/luma source volume; '
        f'--channel_format {channel_format.token} then constructs H×W×{int(channel_format.channel_count)} '
        f'model inputs with offsets {list(channel_format.offsets)} in each active view. '
        'Radial and Tilted Radial neighbors wrap modulo the view slice count and reverse '
        'radial-u after odd 0°/180° seam crossings; Cartesian and '
        'Tilted Cartesian neighbors edge-clamp. Channel order is preserved, each result is assigned '
        'only to center slice N, and no reverse-order inference set is generated.'
    )
    if bool(save_images_enabled):
        spec_notes.append(
            f'Active-view image saving uses the inference channel layout {channel_format.token}: '
            + (
                'one multi-page TIFF per center, with one grayscale page per channel in input order.'
                if int(channel_format.channel_count) >= 5 else
                'PNG output for each center.'
            )
        )
    spec_notes.append('Voxel-volume reporting, when enabled, counts the final binary mask after restoration to native input geometry, not imgsz or cubic working geometry.')
    spec_notes.append('v12.2.0 tilt-angle validation follows the specification: values must be greater than 0 and less than or equal to 45 degrees.')
    if low_quality_downbin_warnings:
        for warning in low_quality_downbin_warnings:
            print(f'Warning: {warning}')
            spec_notes.append(warning)
    if low_quality_downbin_specs:
        spec_notes.append(
            'Low-quality outputs use isotropic X/Y/t downbinning in native input space; frame count is resampled with the same scale as XY, rather than preserving the original frame count. '
            + '; '.join(
                f'{spec.raw_value}->(t,Y,X)={spec.output_shape_t_y_x}'
                for spec in low_quality_downbin_specs
            )
        )
        spec_notes.append(
            'v13.2.1 (bug #2): each low-quality downbin is submitted as an independent background job whose '
            'overlay and binary videos always run. When --save nrrd is also enabled, the low-quality NRRD now '
            'follows the full-quality format and schedule: one downbinned single-layer NRRD per component layer '
            'under low_quality/<token>/nrrd/, written by the NrrdLayerSink as each view completes (sharing the '
            'full-quality layer suffixes and a per-downbin manifest), replacing the single combined tail volume. '
            'Requesting only low-quality outputs still produces the low-quality videos but no NRRD.'
        )
    if (int(T), int(H), int(W)) != (int(input_T), int(input_H), int(input_W)):
        spec_notes.append(
            f'Working volume resized to v12.2.0 approximately-cubic processing geometry '
            f'(t,Y,X)=({int(T)},{int(H)},{int(W)}). v13.2.4 (ruling A1): the final stage runs in the '
            f'original source geometry (t,Y,X)=({int(input_T)},{int(input_H)},{int(input_W)}) — '
            'Radial/Tilted results are backprojected directly to source dimensions in a single resample, '
            'Cartesian view stacks are restored with one resample during union assembly, and the global '
            'union / optional 3D void fill / Gaussian smoothing (sigma in source voxels) / postprocessing keep_objects '
            'all execute at source dimensions. No tail restore resample occurs.'
        )
    spec_notes.append(
        f'v16.0.2 Cartesian selection: --enable_cartesian={list(enabled_cartesian_views)}. '
        'No Cartesian view is implicit; non-90 degree Cartesian augmentations use clamp-to-frame '
        'black fill rather than expanded padding.'
    )
    concrete_tilted = [v for v in views if is_tilted_view(v)]
    if concrete_tilted:
        active_tilt_labels = ', '.join(pretty_view_name(v) for v in concrete_tilted)
        requested_tilt_groups = ' '.join(
            f'{",".join(group.views)}:'
            f'{",".join(f"{float(angle):g}" for angle in group.tilt_angles)}:'
            f'{",".join(group.tilt_directions)}'
            for group in tilt_groups
        )
        spec_notes.append(
            f'v16.3.0 Tilted Views active from --enable_tilted {requested_tilt_groups}. '
            'No Tilted base is implicit; each group preserves its own view/angle/direction '
            'associations, and every positive angle expands to independent positive and negative '
            'variants before final union. Active tilted configurations: '
            f'{active_tilt_labels}'
        )
    else:
        spec_notes.append('Tilted Views disabled because --enable_tilted was not supplied.')
    concrete_radial = [v for v in views if is_radial_view(v)]
    if concrete_radial:
        radial_notes = ', '.join(
            f'{v.radial_request_token}->{v.name}@{_radial_view_nominal_spacing_deg(v):.8g}° '
            f'(diameter={int(v.diameter)})'
            for v in concrete_radial
        )
        spec_notes.append(
            'Radial transforms active: ' + radial_notes + '. Cartesian Radial bases do '
            'not require upright Cartesian views. Each tilted_* target expands across every matching '
            'enabled signed-angle/direction Tilted variant. Circles are constructed in working projected '
            'view space; source-geometry t restoration naturally produces sagittal/coronal and Tilted '
            f'ellipses. {RADIAL_TEXTURE_VARIANT_LABEL} sampling and wraparound interpolation are retained. Upright bases can '
            'use fused resident rendering, orientation-aware streamed GPU prerender, and reduced-grid '
            'processing when admitted.'
        )
    if any((v.family == 'radial' or is_tilted_view(v)) for v in views):
        spec_notes.append(
            'Current final backprojection: upright Radial bases use orientation-aware dense or '
            'sink-only mapping into source-space bands; transverse can additionally use the GPU '
            'backprojector. Tilted Radial views reconstruct their concrete Tilted stack before the '
            'Tilted shear backprojection. The sequential queue preserves the full CPU worker budget.'
        )
    spec_notes.append(
        'v13.2.0 NRRD export (--save nrrd) writes one single-layer 3-axis NRRD (X,Y,t) per component layer to '
        f'nrrd/, named {OUTPUT_NRRD_PREFIX}{{Filestem}}_{{ViewToken|Global}}_{{layer}}.seg.nrrd (model name dropped; v13.2.3 tags each file '
        'with the 3D Slicer segmentation header fields — segment named after the file, deterministic per-layer '
        'palette color). Layer families: full-frame '
        'YOLO masks, full-frame interpolation bridges per pass, tiled masks accepted by parent YOLO masks, tiled '
        'masks accepted by parent bridges, consolidated tile bridges per pass, optional Global_union_presmoothing, '
        'Global_smoothing_pass<N>, and Global_final_output. Each layer is restored to source output geometry while '
        'streaming, gzip-compressed by the selected validated backend, and written by a background sink as the layer is produced '
        'during the intermediate pipeline steps (so the Transverse layer compresses while Tiled Transverse is still '
        'inferencing, and the global union layer is written while smoothing runs). A single '
        f'{OUTPUT_NRRD_PREFIX}{{Filestem}}_nrrd_manifest.json lists every written layer. '
        'The default member codec policy is QAT-first `auto`; set YOLO_TTA_NRRD_MEMBER_CODEC=cpu '
        'to opt out without losing the CPU fallback chain. Tune YOLO_TTA_NRRD_MEMBER_CODEC, '
        'YOLO_TTA_NRRD_MEMBER_GZIP_WINDOW_MIB, YOLO_TTA_NRRD_GZIP_CHUNK_MIB, and '
        'YOLO_TTA_NRRD_LAYER_SINK_WORKERS for member-parallel compression. The previous mega '
        f'4D decomposed NRRD (one file, trailing list axis) was removed. space={NRRD_SPACE}.'
    )
    spec_notes.append(
        'CONFLICT NOTE 3 (low-quality NRRD form): spec --save low_quality says "low bitrate output videos and '
        'NRRDs". v13.2.1 (bug #2) makes the low-quality NRRD follow the full-quality NRRD format and scheduling: '
        'one downbinned single-layer NRRD per component layer under low_quality/<token>/nrrd/, restored from the '
        'same NrrdLayerRef and written as each view completes, with a per-downbin manifest whose layer suffixes '
        'match the full-quality nrrd/ folder. This supersedes the v13.2.0 single combined volume. It remains gated '
        'behind --save nrrd (the decomposition is only meaningful when --save nrrd is set). Suggested spec edit: '
        'state that low-quality NRRDs are emitted as one single-layer NRRD per component layer (downbinned) per '
        '--save low_quality[:LOW_QUALITY_DOWNBIN] specification, only when --save nrrd is enabled.'
    )
    spec_notes.append(
        'v17 backend-local result processing: CUDA tasks retain the proto-union/flatten, warp, '
        '--min_conf, and positive --min_radius GPU path before their reduced result is published. '
        'OpenVINO tasks keep raw head/prototype outputs on the CPU and reconstruct bbox-local masks. '
        'Within an OpenVINO-committed direct-union view, CPU and assisting CUDA tasks publish '
        'disjoint view-union windows; CUDA-committed views instead retain the D1 source-space sparse '
        'contract. Progress bars outside inference therefore describe independent '
        'render/postprocess/output stages rather than cross-device mask migration.'
    )
    inference_backend_details: List[str] = []
    if gpu_worker_process_active:
        inference_backend_details.append(
            f'{gpu_device_count} CUDA worker process(es), one per logical GPU, with '
            'persistent model/context ownership and central short-lease dispatch'
        )
    if cpu_worker_process_active:
        inference_backend_details.append(
            f'{len(cpu_instance_plans)} socket-local OpenVINO worker process(es), each with '
            'one compiled model and a shallow asynchronous infer-request pool'
        )
    if gpu_worker_process_active and cpu_worker_process_active:
        inference_routing_note = (
            'OpenVINO opens one ordered reserved shared-union view at a time and advances to '
            'the next reservation after completion. Unreserved Cartesian/Tilted views are '
            'ordinary CUDA D1 work. CUDA assists only the active CPU view when its measured ETA '
            'exceeds the mandatory-GPU horizon; already-running OpenVINO requests are never preempted.'
        )
    elif gpu_worker_process_active:
        inference_routing_note = (
            'CUDA workers own every selected view and prioritize Radial/Tilted-Radial work '
            'within the central short-lease scheduler.'
        )
    else:
        inference_routing_note = (
            'OpenVINO workers own Cartesian/Tilted-Cartesian work; unsupported Radial '
            'requests were removed before scheduling.'
        )
    spec_notes.append(
        f'v17.0.3 inference backends: {inference_devices}; ' + '; '.join(inference_backend_details) + '. '
        'All backends claim atomically from one central queue; each full-frame view is committed '
        f'to exactly one result contract before its first task leaves that queue. {inference_routing_note}'
    )
    if gpu_worker_process_active and cpu_worker_process_active:
        spec_notes.append(
            'Mask/prototype processing is backend-local in hybrid mode: CUDA tasks reduce and '
            'warp their proto/mask unions on the GPU, while OpenVINO tasks reconstruct '
            'bbox-ROI retina-quality masks on the CPU from raw head/prototype outputs. CPU-owned '
            'views use one shared direct-union binary/confidence contract for both backends; '
            'CUDA-owned eligible views use D1 and are no longer CPU-claimable. There is no global '
            '--retina_mask_processor selection.'
        )
    elif cpu_worker_process_active:
        spec_notes.append(
            'Mask/prototype processing follows the OpenVINO CPU backend: raw segmentation head '
            'and prototype outputs are converted to compact CPU payloads, and bbox-ROI masks are '
            'bilinearly reconstructed and thresholded on the CPU before view-union publication.'
        )
    else:
        spec_notes.append(
            'Mask/prototype processing follows the CUDA backend: proto-resolution unions, '
            'confidence handling, affine warp, and eligible radius cleanup remain GPU-local '
            'before reduced result publication.'
        )
    native_ffv1_outputs = []
    if bool(save_high_quality_enabled):
        native_ffv1_outputs.append('overlay')
    if bool(save_binary_enabled):
        native_ffv1_outputs.append('binary')
    if native_ffv1_outputs:
        spec_notes.append(
            f'v13.3.12 final FFV1 encode sharding: segments={int(ffv1_segment_count(int(input_T)))} '
            '(YOLO_TTA_FFV1_SEGMENTS; automatic default min(6, ceil(allocated_cpus/32))). '
            f'Selected native-resolution {" and ".join(native_ffv1_outputs)} MKV output(s) encode '
            'contiguous t segments concurrently and losslessly concat their FFV1 packets; value 1 '
            'restores one encoder.'
        )
    if bool(keep_temp_artifacts):
        spec_notes.append('YOLO_TTA_KEEP_TEMP=1 active: temporary scratch artifacts are retained.')
    if bool(gaussian_smoothing_enabled):
        spec_notes.append(
            f'Gaussian smoothing active by v12.2.0 explicit-flag rule: sigma={float(gaussian_smoothing_sigma):g} voxel(s), '
            f'passes={int(gaussian_smoothing_passes)}; applied after final union/optional 3D void fill and before postprocessing keep_objects. '
            'The default smoothing backend attempts chunked GPU execution through CuPy/cupyx.scipy.ndimage with halo/core writes, then falls back to scipy.ndimage on CPU if the GPU backend is unavailable.'
        )
    else:
        if not bool(gaussian_smoothing_cli_requested):
            spec_notes.append('Gaussian smoothing disabled by v12.2.0 activation rule because neither Gaussian flag was explicitly set.')
        elif bool(gaussian_smoothing_disabled_by_zero):
            spec_notes.append('Gaussian smoothing disabled because at least one explicitly supplied Gaussian flag was set to 0.')
        else:
            spec_notes.append('Gaussian smoothing disabled because the resolved sigma or pass count was not positive.')
    spec_notes.append(
        'Interpolation endpoint discovery uses the per-slice connected-component scan backed by cached per-slice component tables. '
        'Projection candidate search runs on source-component local SDF crops, and variable-cost seed planning is consumed through a bounded unordered completion queue. '
        'v17.0.9 planner seeds are stably cost-balanced within four worker-wave slice-local windows by default; '
        'YOLO_TTA_INTERPOLATION_SEED_SCHEDULE_WINDOW_FACTOR=0 restores strict slice-major submission for A/B verification. '
        'v13.0.0 removed optional skeletonization entirely; interpolation never used skeletonization.'
    )
    spec_notes.append(
        f'Processing volume mode={processing_mode}; cube_resize_applied={bool(tuple(int(x) for x in processing_shape) != tuple(int(x) for x in input_processing_shape))}. '
        'v13.0.0 (bug fix) defaults to approximately-cubic virtual working geometry so the working dimensions '
        'stay within 5% of the longest source axis (spec item 2); the t/Transverse stacking axis is no longer left short. '
        f'Approximately-cubic target = {tuple(int(x) for x in legacy_cube_shape)}. Set YOLO_TTA_PROCESSING_VOLUME_MODE=native to opt back into v12.2.15 decoded-native geometry for regression. '
        f'Cube T-axis backend={_cube_t_axis_resize_backend()} (YOLO_TTA_CUBE_T_RESIZE_BACKEND=slice_exact restores the endpoint-aligned per-slice interpolation path).'
    )
    yolo_model: Optional[object] = None
    if not inference_worker_process_active:
        raise RuntimeError('v17 requires at least one process-local GPU or OpenVINO CPU backend')
    # The parent retains scheduling metadata only. CUDA workers deserialize TensorRT locally;
    # socket-local CPU workers compile the OpenVINO IR after applying affinity.
    yolo_models: List[Tuple[str, Optional[object]]] = [(model_name, yolo_model)]
    yolo_by_model_name: Dict[str, object] = (
        {model_name: yolo_model} if yolo_model is not None else {}
    )

    # Metadata-only compatibility config. Each worker receives its backend-specific batch and
    # precision below; no model is loaded in the parent.
    pred_cfg = PredictConfig(
        imgsz=args.imgsz,
        conf=args.conf,
        device=(str(backend_devices.gpu_devices[0]) if gpu_worker_process_active else 'cpu'),
        quantize=(resolve_quantize(args.gpu_quantize) if gpu_worker_process_active else None),
        batch=max(1, int(args.gpu_batch if gpu_worker_process_active else args.cpu_batch)),
        input_channels=int(channel_format.channel_count),
        channel_token=str(channel_format.token),
    )
    print(
        'v17 process-local inference: '
        f'{gpu_device_count} CUDA worker(s), {len(cpu_instance_plans)} OpenVINO CPU worker(s); '
        'the parent loaded no inference model.'
    )


    # Size main-process pools around live inference subprocess reservations.
    # inference subprocesses already consume ~visible_cpu render threads (the GPU-blocking pools), so
    # the main-process GPU-non-blocking pools take only the remaining headroom of the 2x box target.
    worker_budget = int(main_process_worker_budget(int(gpu_device_count), bool(gpu_worker_process_active)))
    if cpu_worker_process_active and not hybrid_cpu_affinity_overlap_active:
        # Exclusive mode bounds parent pools to CPUs outside OpenVINO masks. Hybrid overlap
        # mode intentionally preserves the ordinary CUDA-side worker budget instead.
        parent_logical = max(1, int(len(main_process_reserved_cpus)))
        worker_budget = max(1, min(int(worker_budget), 2 * int(parent_logical)))
    # Parent affinity is already narrowed to the non-feeder mask at this point, so
    # ``default_worker_budget()`` would size the later tail from that temporary mask and
    # fail to reclaim the four dedicated physical cores per GPU after inference drains.
    # Preserve the original full SLURM/cpuset allocation explicitly for tail expansion.
    whole_box_worker_budget = max(1, 2 * int(len(_allowed_main_cpus)))
    tail_budget_expansion_active = bool(tail_worker_budget_expansion_enabled())
    tail_worker_budget = int(whole_box_worker_budget if tail_budget_expansion_active else worker_budget)
    augmentation_workers = resolve_worker_count(
        0,
        'YOLO_TTA_AUG_WORKERS',
        worker_budget,
        max_tasks=max(1, max((v.num_slices for v in inference_views), default=1)),
    )
    interpolation_workers = resolve_worker_count(
        0,
        'YOLO_TTA_INTERPOLATION_WORKERS',
        worker_budget,
        max_tasks=max(1, len(yolo_models) * max(1, len(interpolating_views))),
    )
    output_workers = resolve_worker_count(
        0,
        'YOLO_TTA_OUTPUT_WORKERS',
        worker_budget,
    )
    output_frame_workers = max(1, _env_int('YOLO_TTA_OUTPUT_FRAME_WORKERS', max(1, min(_cpu_count(), output_workers))))
    slice_postprocess_workers = max(1, int(augmentation_workers))
    predict_postprocess_cap = max(1, _env_int('YOLO_TTA_PREDICT_POSTPROCESS_MAX_WORKERS', max(1, int(worker_budget))))
    predict_postprocess_workers = max(
        1,
        min(
            int(predict_postprocess_cap),
            _env_int('YOLO_TTA_PREDICT_POSTPROCESS_WORKERS', slice_postprocess_workers),
        ),
    )

    interpolation_process_backend_active = bool(
        interpolation_process_backend_enabled() and len(interpolating_views) > 0
    )
    interpolation_global_pass_limit = max(
        1,
        _env_int('YOLO_TTA_INTERPOLATION_GLOBAL_PASSES', 1),
    )
    configure_interpolation_pass_admission(int(interpolation_global_pass_limit))
    (
        parent_postprocess_workers,
        parent_slice_postprocess_workers,
        parent_postprocess_estimated_bytes,
        parent_postprocess_memory_cap,
        parent_postprocess_default_workers,
    ) = resolve_parent_postprocess_worker_allocation(
        worker_budget=int(worker_budget),
        views=inference_views,
        nrrd_layers_enabled=bool(nrrd_layers_needed),
        interpolation_enabled=bool(len(interpolating_views) > 0),
    )
    (
        parent_interpolation_overlap,
        parent_interpolation_task_workers_default,
        parent_interpolation_task_workers,
    ) = resolve_parent_interpolation_worker_allocation(
        worker_budget=int(worker_budget),
        parent_postprocess_workers=int(parent_postprocess_workers),
        interpolation_process_backend_active=bool(interpolation_process_backend_active),
    )

    # Gate tasks are themselves slice-parallel.  The previous worker_budget x
    # worker_budget defaults could create 25,600 runnable tasks on a 160-CPU node.
    # Keep a small outer fan-out and divide the box budget across its inner pools.
    tile_postprocess_workers_default = max(1, min(4, int(worker_budget)))
    tile_postprocess_workers = max(
        1,
        min(
            int(worker_budget),
            _env_int('YOLO_TTA_TILE_POSTPROCESS_WORKERS', tile_postprocess_workers_default),
        ),
    )
    # Dense GPU-worker tile results must never wait behind parent interpolation or sparse
    # gate work in the shared tile executor. A small dedicated outer pool performs cleanup
    # plus CTILE conversion; each task still uses the configured slice-parallel workers.
    tile_dense_retirement_workers_default = max(
        1,
        min(
            8,
            int(tile_postprocess_workers),
            max(2, 2 * max(1, int(gpu_device_count))),
        ),
    )
    tile_dense_retirement_workers = max(
        1,
        _env_int(
            'YOLO_TTA_TILE_DENSE_RETIREMENT_WORKERS',
            int(tile_dense_retirement_workers_default),
        ),
    )
    tile_dense_retirement_slice_workers_default = max(
        1,
        int(worker_budget) // max(1, int(tile_dense_retirement_workers)),
    )
    tile_dense_retirement_slice_workers = max(
        1,
        _env_int(
            'YOLO_TTA_TILE_DENSE_RETIREMENT_SLICE_WORKERS',
            int(tile_dense_retirement_slice_workers_default),
        ),
    )
    tile_slice_postprocess_workers_default = max(
        1,
        int(worker_budget) // max(1, int(tile_postprocess_workers)),
    )
    tile_slice_postprocess_workers = max(
        1,
        min(
            int(worker_budget),
            _env_int('YOLO_TTA_TILE_SLICE_WORKERS', tile_slice_postprocess_workers_default),
        ),
    )
    # Consolidation runs in the same outer executor as gate tasks; give it one outer
    # share rather than another full-box inner pool.
    tile_interpolation_task_workers_default = int(tile_slice_postprocess_workers)
    tile_interpolation_task_workers = max(
        1,
        min(
            int(worker_budget),
            _env_int('YOLO_TTA_TILE_INTERPOLATION_TASK_WORKERS', tile_interpolation_task_workers_default),
        ),
    )

    # Interpolation workers provide independent Python interpreters for GIL-heavy planning.
    # Each child has a separately admitted pass workspace, so the cap balances CPU parallelism
    # against anonymous-memory headroom and the disk-backed fallback.
    interpolation_process_workers_default = max(
        1,
        min(
            int(INTERPOLATION_PROCESS_WORKER_DEFAULT_CAP),
            max(1, int(parent_postprocess_workers)) + (1 if len(tile_configs) > 0 else 0),
        ),
    )
    interpolation_process_workers = (
        max(
            1,
            min(
                int(interpolation_global_pass_limit),
                _env_int(
                    'YOLO_TTA_INTERPOLATION_PROCESS_WORKERS',
                    interpolation_process_workers_default,
                ),
            ),
        )
        if bool(interpolation_process_backend_active) else 0
    )

    print(f'Allocated CPU count: {_cpu_count()}')
    print(f'Worker budget (main process): {worker_budget}')
    print(
        'Post-inference tail worker budget: '
        f'{int(tail_worker_budget)} '
        f'(YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND={int(tail_budget_expansion_active)})'
    )
    if bool(gpu_worker_process_active):
        _gpu_share = int(gpu_worker_cpu_share(int(gpu_device_count)))
        if hybrid_cpu_affinity_overlap_active:
            print(
                'Hybrid helper overlap is intentional: CUDA workers retain topology-local helper '
                f'pools (~{_gpu_share} thread(s)/GPU before the measured NUMA split) and the main '
                f'process retains {worker_budget} workers while OpenVINO stays pinned to its '
                'socket-local masks. The helper pools are low-duty in the measured GPU-only run; '
                'YOLO_TTA_HYBRID_CPU_AFFINITY_OVERLAP=0 restores exclusive accounting.'
            )
        else:
            print(
                'Worker oversubscription is intentional (whole-box target = 2x visible CPUs = '
                f'{whole_box_worker_budget}). CUDA workers ({gpu_device_count} GPU(s)): each inference '
                f'subprocess uses ~{_gpu_share} render thread(s) ({_gpu_share * int(gpu_device_count)} total '
                'for GPU-blocking work), so the GPU-non-blocking main-process pools are sized to the '
                f'remaining {worker_budget} to keep the box near 2:1 instead of oversubscribing it.'
            )
    else:
        print('Inference-phase worker pools are bounded against live GPU/OpenVINO process reservations; tail-only pools may expand after inference drains.')
    print(f'Augmentation workers: {augmentation_workers}')
    print(f'Slice-parallel postprocess workers: {slice_postprocess_workers}')
    print(f'Inference postprocess workers: {predict_postprocess_workers}')
    print(
        'Parent full-frame postprocess workers: '
        f'{parent_postprocess_workers} (independent of interpolation; default={parent_postprocess_default_workers}, '
        f'memory_cap={parent_postprocess_memory_cap}, estimated_live_view={parent_postprocess_estimated_bytes / GIB:.2f} GiB, '
        f'slice_workers/view={parent_slice_postprocess_workers}; expected interpolation overlap: '
        f'{parent_interpolation_overlap}, per-parent interpolation workers: {parent_interpolation_task_workers})'
    )
    print(
        'Tile postprocess workers: '
        f'{tile_postprocess_workers} (dedicated dense cleanup/CTILE retirement workers: '
        f'{tile_dense_retirement_workers} x {tile_dense_retirement_slice_workers} slice workers, '
        f'gate/consolidation per-tile slice workers: '
        f'{tile_slice_postprocess_workers}, consolidated-tile interpolation workers: '
        f'{tile_interpolation_task_workers})'
    )
    if bool(interpolation_process_backend_active):
        print(
            'Interpolation process backend: enabled '
            f'(process workers: {int(interpolation_process_workers)}, start_method={interpolation_process_start_method()}, '
            f'global pass limit={int(interpolation_global_pass_limit)}, '
            f'child cv2_threads={interpolation_process_cv2_threads()}, compiled kernels: {interpolation_compiled_kernels_status()})'
        )
    else:
        reason = 'no interpolation-enabled views' if len(interpolating_views) <= 0 else 'backend disabled'
        print(f'Interpolation process backend: inactive for this command ({reason}).')
    print(f'Background output workers: {output_workers} (frame workers per labels/TIFF task: {output_frame_workers})')
    max_predict_video_frames = max(1, max((int(v.num_slices) for v in inference_views), default=1))
    example_cpu_mask_workers = max(1, min(int(predict_postprocess_workers), int(max_predict_video_frames)))
    example_cpu_mask_pending = cpu_mask_postprocess_pending_limit(int(example_cpu_mask_workers), int(max_predict_video_frames))
    spec_notes.append(
        'YOLO result accumulation is bounded per in-memory prediction source by the number of pending CPU postprocess futures. ' 
        'worker_count=max(1, min(YOLO_TTA_PREDICT_POSTPROCESS_WORKERS, num_frames)) for the live v12 in-memory source path; ' 
        'v12.2.15 keeps the v12.2.14 hard 32-worker ceiling removed and defaults YOLO_TTA_PREDICT_POSTPROCESS_MAX_WORKERS to the oversubscribed worker budget so CPU result processing can queue/drain behind the GPU instead of throttling inference. '
        'pending_limit=cpu_mask_postprocess_pending_limit(worker_count, num_frames) = max(worker_count, min(num_frames, max(YOLO_TTA_CPU_MASK_PENDING_FRAMES, worker_count*2))). ' 
        'The default YOLO_TTA_CPU_MASK_PENDING_FRAMES is 0, so the GPU-facing iterator can buffer all CPU result work for that prediction source; set it positive only to reintroduce a RAM cap. ' 
        f'For this run, the largest active prediction source has {int(max_predict_video_frames)} frame(s), ' 
        f'example worker_count={int(example_cpu_mask_workers)}, pending_limit={int(example_cpu_mask_pending)}.'
    )
    spec_notes.append(
        f'Backend-specific batching: GPU batch={int(args.gpu_batch) if gpu_worker_process_active else "disabled"}, '
        f'CPU batch={int(args.cpu_batch) if cpu_worker_process_active else "disabled"}. Each backend '
        f'independently pads its final source batch by repeating the last complete H×W×{int(channel_format.channel_count)} '
        'channel stack and discards synthetic results. StreamingYoloVolumeSource remains the bounded '
        'render/prefetch source; CUDA tasks may additionally stage complete BCHW batches in VRAM.'
    )
    spec_notes.append(
        'v16.4 per-tile semantics remain active: every original tile stays crop-local and is '
        'component-gated independently against its same-angle parent mask, then only its failed '
        'components are re-gated against the same-angle parent interpolation bridge. No raw '
        'configuration-wide tile canvas or cross-tile component authorization path exists. '
        f'v16.4.3 bounds live dense tile-result workspaces to approximately '
        f'{tile_dense_worker_result_limit_bytes() / GIB:.1f} GiB and '
        f'{tile_dense_worker_result_limit_tasks()} task(s) '
        '(YOLO_TTA_TILE_DENSE_RESULT_MAX_GIB / _MAX_TASKS) when YOLO_TTA_KEEP_TEMP is disabled, '
        'preferring parent-owned memfd shared RAM and using gpu_worker_results pathnames only '
        'when anonymous-memory admission cannot preserve the configured reserve; stale '
        'gpu_worker_results from interrupted older runs are purged before dispatch. '
        'Full-frame parent tasks and tiles whose same-angle parent mask P is already published are '
        'scheduled ahead of P-not-ready tiles. Every nonempty tile is cleaned and converted to '
        'crop-local ctile-mask-v2-raw on a dedicated retirement pool before it enters either '
        'parent/bridge gate; CTILE publication immediately closes the parent-owned memfd/pathname '
        'and returns its dense-result scheduling credit. Gate workers open sparse descriptors lazily, '
        'so parent interpolation cannot extend dense uint8 result lifetime. '
        f'Dense-retirement concurrency={tile_dense_retirement_workers} task(s) x '
        f'{tile_dense_retirement_slice_workers} slice worker(s); '
        f'Tile-set/category accumulators prefer process-reopenable memfd RAM={int(tile_intermediate_accumulators_prefer_memory())}; '
        f'the cgroup-corrected accumulator reserve is {tile_intermediate_accumulator_reserve_bytes() / GIB:.1f} GiB. '
        f'CTILE/CVOL payload memfd opt-in={int(raw_store_memfd_enabled())} '
        '(YOLO_TTA_CVOL_MEMFD; pathname-backed by default because raw-store queues are not RAM-admitted). '
        'External CTILE/CVOL support and NRRD stores elide empty slices and retain raw uint8 '
        'bbox payloads; private no-NRRD final-view retention uses row-wise packbits (up to 8x '
        'smaller) and no store uses LZ4. YOLO_TTA_KEEP_TEMP=1 intentionally retains the original '
        'dense artifacts for diagnostics and therefore disables the live-result storage guarantee.'
    )
    spec_notes.append(
        f'Fused cleanup backend={cleanup_backend()}; set YOLO_TTA_CLEANUP_BACKEND=scipy to use the previous scipy.ndimage cleanup path.'
    )
    spec_notes.append(
        'Interpolation labeling uses parallel 2D per-slice connected-component labeling, parallel adjacent-slice pair extraction, '
        'row-blocked parallel compact relabeling, lazy byte-bounded per-slice component tables, and exact bbox label-window '
        'continuation tests that avoid quadratic same-label component-pair scans. '
        f'Per-parent interpolation task workers default to worker_budget / expected_live_parent_overlap = {int(parent_interpolation_task_workers_default)} '
        f'using YOLO_TTA_INTERPOLATION_CONCURRENT_PARENTS={int(parent_interpolation_overlap)}; '
        'YOLO_TTA_INTERPOLATION_TASK_WORKERS still overrides the exact per-parent worker count. '
        'Tune YOLO_TTA_INTERPOLATION_LABEL_WORKERS, YOLO_TTA_INTERPOLATION_PAIR_WORKERS, YOLO_TTA_INTERPOLATION_COMPACT_WORKERS, '
        'YOLO_TTA_INTERPOLATION_COMPACT_RELABEL_ROWS, and YOLO_TTA_INTERPOLATION_CONCURRENT_PARENTS if needed.'
    )
    spec_notes.append(
        'v12.2.7 interpolation process isolation active by default: full-frame and consolidated-tile interpolation passes reopen uint8 mask volumes from disk-backed memmaps in a ProcessPoolExecutor worker and return only small stats. '
        f'Process backend enabled={bool(interpolation_process_backend_active)}, process_workers={int(interpolation_process_workers)}, start_method={interpolation_process_start_method()}, '
        f'global_pass_limit={int(interpolation_global_pass_limit)}, fallback_on_worker_failure={bool(interpolation_process_fallback_enabled())}. '
        'The global lease covers backing conversion plus auxiliary/dedicated execution, so queued passes cannot create full-volume process inputs concurrently. Consolidated-tile accumulators are process-reopenable at creation; other anonymous in-memory mask arrays are copied once to a process memmap before interpolation, avoiding multi-GiB pickle payloads.'
    )
    spec_notes.append(
        f'v12.2.7 compiled interpolation kernels: {interpolation_compiled_kernels_status()}. '
        'The compiled kernel accelerates projection-candidate discovery in seed planning with Numba nogil=True when numba is installed; the exact Python candidate search remains the fallback.'
    )

    output_manager = BackgroundOutputManager(max_workers=output_workers)
    _run_resources().track_output_manager(output_manager)

    if augmentation_workers > 1 or interpolation_workers > 1 or slice_postprocess_workers > 1 or output_workers > 1:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    interpolation_process_executor: Optional[ProcessPoolExecutor] = None
    if bool(interpolation_process_backend_active):
        interpolation_process_executor = create_interpolation_process_executor(int(interpolation_process_workers))
        if interpolation_process_executor is not None:
            _run_resources().track_executor(interpolation_process_executor)
        set_interpolation_process_executor(interpolation_process_executor, int(interpolation_process_workers))
    else:
        set_interpolation_process_executor(None, 0)

    dense_tiling_active = len(tile_configs) > 0

    view_infos_by_name: Dict[str, ViewInfo] = {view.name: view for view in views}
    view_infos_by_name.update({view.name: view for view in physical_views})
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    native_view_support_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    parent_mask_support_by_model: Dict[str, Dict[str, object]] = {model_name: {} for model_name, _ in yolo_models}
    parent_bridge_support_by_model: Dict[str, Dict[str, object]] = {model_name: {} for model_name, _ in yolo_models}
    radial_native_output_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    tilted_native_output_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    nrrd_layer_refs: List[NrrdLayerRef] = []
    view_volume_locks: Dict[Tuple[str, str], threading.Lock] = {
        (model_name, view.name): threading.Lock()
        for model_name, _ in yolo_models
        for view in views
    }

    aug_jobs_by_view: Dict[str, List[AugJob]] = {}
    aug_job_lookup_by_view: Dict[str, Dict[str, AugJob]] = {}
    tile_jobs_by_view_config: Dict[str, Dict[str, List[DenseTileJob]]] = {view.name: {} for view in views}
    tile_jobs_by_aug: Dict[Tuple[str, str], List[DenseTileJob]] = {}
    view_prediction_stats: Dict[str, int] = {}
    view_prediction_labels: Dict[str, str] = {}
    for _view_for_stats in views:
        key = str(_view_for_stats.summary_family)
        view_prediction_stats.setdefault(key, 0)
        view_prediction_labels[key] = pretty_view_name(_view_for_stats)
    interpolation_stats: List[Dict[str, object]] = []

    inference_view_names = {v.name for v in inference_views}
    for view in views:
        jobs = [build_aug_job_for_variant(
            view=view,
            out_size=args.imgsz,
            temp_dir=temp_dir,
        )]
        aug_jobs_by_view[view.name] = jobs
        aug_job_lookup_by_view[view.name] = {job.aug_id: job for job in jobs}
        if dense_tiling_active and view.name in inference_view_names:
            jobs_by_config: Dict[str, List[DenseTileJob]] = {}
            for tile_cfg in tile_configs:
                cfg_jobs: List[DenseTileJob] = []
                for aug_job in jobs:
                    built_jobs = build_dense_tile_jobs_for_aug(
                        view=view,
                        aug_job=aug_job,
                        tile_cfg=tile_cfg,
                        out_size=int(args.imgsz),
                        temp_dir=temp_dir,
                    )
                    cfg_jobs.extend(built_jobs)
                    tile_jobs_by_aug.setdefault((view.name, aug_job.aug_id), []).extend(built_jobs)
                if cfg_jobs:
                    jobs_by_config[tile_cfg.config_id] = cfg_jobs
            if jobs_by_config:
                tile_jobs_by_view_config[view.name] = jobs_by_config

    tile_expected_by_parent: Dict[Tuple[str, str], int] = {}
    tile_expected_by_set: Dict[Tuple[str, str, str], int] = {}
    tile_config_ids_by_parent: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    if dense_tiling_active:
        for view in inference_views:
            jobs_by_config = tile_jobs_by_view_config.get(view.name, {})
            expected_for_variant = int(sum(len(jobs) for jobs in jobs_by_config.values()))
            if expected_for_variant <= 0:
                continue
            for model_name, _ in yolo_models:
                parent_key = (str(model_name), str(view.name))
                tile_expected_by_parent[parent_key] = int(expected_for_variant)
                tile_config_ids_by_parent[parent_key] = tuple(str(v) for v in jobs_by_config)
                for config_id, config_jobs in jobs_by_config.items():
                    tile_expected_by_set[(
                        str(model_name), str(view.name), str(config_id),
                    )] = int(len(config_jobs))

    view_frame_caches: Dict[str, np.ndarray] = {}
    view_frame_cache_paths: Dict[str, Path] = {}
    view_frame_cache_lock = threading.Lock()

    def _get_view_frame_cache(view: ViewInfo) -> Optional[np.ndarray]:
        if not should_cache_view_frames(view, dense_tiling_active):
            return None
        cache_key = physical_view_name(view)
        cached = view_frame_caches.get(cache_key)
        if cached is not None:
            return cached
        with view_frame_cache_lock:
            cached = view_frame_caches.get(cache_key)
            if cached is not None:
                return cached
            wait_for_volume_ready(volume_rgb)
            cache_path = temp_dir / 'view_frames' / f'{cache_key}.gray8.dat'
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_mm = build_view_frame_cache(
                volume_rgb=volume_rgb,
                view=view,
                out_path=cache_path,
                desc=f'{view.name} native frame cache',
                prefer_memory=True,
                workers=max(1, int(augmentation_workers)),
            )
            view_frame_caches[cache_key] = cache_mm
            view_frame_cache_paths[cache_key] = cache_path
            return cache_mm

    baseline_union_by_model_view: Dict[Tuple[str, str], np.ndarray] = {}
    baseline_confmap_by_model_view: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    baseline_union_paths: Dict[Tuple[str, str], Path] = {}
    baseline_confmap_paths: Dict[Tuple[str, str], Optional[Path]] = {}
    baseline_slice_locks_by_model_view: Dict[Tuple[str, str], List[threading.Lock]] = {}
    direct_union_inference_views: set[Tuple[str, str]] = set()
    direct_union_postprocess_views: set[Tuple[str, str]] = set()
    direct_union_inference_bytes: Dict[Tuple[str, str], int] = {}
    direct_union_postprocess_bytes: Dict[Tuple[str, str], int] = {}
    direct_union_backing_leases: Dict[Tuple[str, str], _DirectUnionBackingLease] = {}
    # Requested NRRD components already form an immutable terminal-union backing. Without
    # --save nrrd, one private pathname-backed final-view cvol is materialized instead.
    # Either representation lets the dense canvas retire; KEEP_TEMP preserves the legacy
    # dense diagnostics contract. Activate direct-union byte/view admission whenever that
    # bounded retirement path is authoritative.
    component_ref_dense_retirement_active = bool(not keep_temp_artifacts)
    direct_union_sparse_retirement_active = bool(component_ref_dense_retirement_active)

    _direct_headroom = max(0, int(available_anon_work_bytes()) - 64 * GIB)
    _default_inference_view_limit = max(2, min(4, max(1, int(gpu_device_count))))
    direct_union_inference_view_limit = max(
        1,
        _env_int('YOLO_TTA_DIRECT_UNION_INFERENCE_VIEWS', int(_default_inference_view_limit)),
    )
    # Keep the live dense inference window deliberately below the interpolation heap.
    # The former 512/768 GiB defaults could legally retain almost the entire 1 TiB job
    # allocation before the one active interpolation pass allocated labels/bridges.  An
    # individual oversize view still has an emergency lane, so these conservative caps never
    # deadlock a 100+ GiB logical tiled parent.
    _default_inference_bytes = max(
        64 * GIB, min(128 * GIB, int(max(1, _direct_headroom) * 0.25)),
    )
    direct_union_inference_byte_limit = int(
        max(
            1.0,
            _env_float(
                'YOLO_TTA_DIRECT_UNION_INFERENCE_GIB',
                _default_inference_bytes / GIB,
            ),
        ) * GIB
    )
    _default_total_dense_bytes = max(
        128 * GIB, min(256 * GIB, int(max(1, _direct_headroom) * 0.40)),
    )
    direct_union_total_dense_byte_limit = int(
        max(
            direct_union_inference_byte_limit / GIB,
            _env_float(
                'YOLO_TTA_DIRECT_UNION_TOTAL_GIB',
                _default_total_dense_bytes / GIB,
            ),
        ) * GIB
    )
    _parent_transient_default = max(
        64 * GIB,
        min(192 * GIB, int(max(1, available_anon_work_bytes() - 64 * GIB) * 0.25)),
    )
    parent_transient_admission = _ByteAdmissionPool(
        int(max(1.0, _env_float('YOLO_TTA_PARENT_TRANSIENT_GIB', _parent_transient_default / GIB)) * GIB),
        'Parent view postprocess admission',
    )
    if direct_union_sparse_retirement_active:
        print(
            'v16.1.3 split direct-union leases active: '
            f'inference_views={int(direct_union_inference_view_limit)}, '
            f'inference_dense={direct_union_inference_byte_limit / GIB:.1f} GiB, '
            f'total_inference+postprocess_dense={direct_union_total_dense_byte_limit / GIB:.1f} GiB. '
            'The inference lease is released when the final chunk is committed; hole fill, '
            'projection, NRRD/cvol work, and sparse retirement continue under a separate '
            'postprocess lease.'
        )
    fullframe_remaining: Dict[Tuple[str, str], int] = {}
    # slices per (model, view) hole-filled on device by the GPU workers; when the
    # sum reaches the view's slice count, the CPU per-view "2D hole fill" pass is skipped.
    view_device_hole_filled_slices: Dict[Tuple[str, str], int] = {}
    # per-(model, view) aggregation of the workers' device-union slice metadata
    # (any flags / bboxes / bit-packed row occupancy). A single task without metadata marks the
    # whole view invalid, and every downstream consumer falls back to scanning.
    view_slice_meta: Dict[Tuple[str, str], Dict[str, object]] = {}

    for view in inference_views:
        for model_name, _ in yolo_models:
            # lazy allocation: do not zero every full-view accumulator before the
            # first prediction. On 20+ view runs those eager zeros can touch hundreds of GiB
            # and dominate time-to-first-prediction. Allocate a view's union/confidence
            # workspaces only when its first full-frame prediction is about to run.
            fullframe_remaining[(model_name, view.name)] = int(len(aug_jobs_by_view[view.name]))

    total_fullframe_jobs = sum(len(aug_jobs_by_view.get(view.name, [])) for view in inference_views)
    total_tile_prediction_jobs = sum(
        len(tile_jobs_by_aug.get((view.name, aug_job.aug_id), []))
        for view in inference_views
        for aug_job in aug_jobs_by_view[view.name]
    )
    total_prediction_volume_build_tasks = int(total_fullframe_jobs + total_tile_prediction_jobs)
    streaming_sources_active = bool(streaming_prediction_sources_enabled())
    max_prediction_source_frames = max(1, max((int(v.num_slices) for v in inference_views), default=1))
    prediction_render_workers = resolve_prediction_render_workers(
        max(1, int(worker_budget)),
        max_prediction_source_frames,
    )
    prediction_volume_queue_slots = resolve_prediction_source_queue_slots(
        total_prediction_volume_build_tasks,
        streaming_sources=bool(streaming_sources_active),
    )
    eager_gpu_input_staging_ahead_sources = (
        gpu_input_staging_ahead_sources(int(prediction_volume_queue_slots))
        if gpu_input_staging_enabled(pred_cfg)
        else 0
    )
    queued_streaming_cpu_warmup_sources = (
        queued_streaming_source_cpu_warmup_slots(int(prediction_volume_queue_slots))
        if bool(streaming_sources_active)
        else 0
    )
    active_build_slot_default = max(
        1,
        min(
            int(augmentation_workers),
            int(prediction_volume_queue_slots),
            max(1, int(total_prediction_volume_build_tasks)),
        ),
    )
    requested_build_workers = max(
        1,
        _env_int('YOLO_TTA_VOLUME_BUILD_WORKERS', int(active_build_slot_default)),
    )
    prediction_volume_builder_workers = max(
        1,
        min(
            int(augmentation_workers),
            int(prediction_volume_queue_slots),
            max(1, int(total_prediction_volume_build_tasks)),
            int(requested_build_workers),
        ),
    )
    legacy_per_prediction_volume_workers = max(1, int(max(1, augmentation_workers) // max(1, prediction_volume_builder_workers)))
    # Streaming sources submit only a bounded prefetch window, so a single active source should
    # see the full render pool instead of a divided-by-builder slice of the CPU allocation.
    per_prediction_volume_workers = int(prediction_render_workers) if bool(streaming_sources_active) else int(legacy_per_prediction_volume_workers)
    async_prediction_accumulation_active = bool(async_predict_postprocess_enabled())
    # Queue slots are unbounded by default in, so do not size the lightweight join
    # executor from the total source count. The join tasks mostly wait on result futures;
    # a CPU-count-sized pool is enough to overlap drains without spawning thousands of threads.
    async_prediction_join_worker_count = async_predict_join_workers(max(2, min(max(1, _cpu_count()), int(worker_budget))))
    default_async_result_workers = max(1, int(predict_postprocess_workers))
    async_prediction_result_worker_count = max(
        1,
        min(
            int(predict_postprocess_workers),
            _env_int('YOLO_TTA_ASYNC_PREDICT_RESULT_WORKERS', int(default_async_result_workers)),
        ),
    )
    if bool(async_prediction_accumulation_active):
        print(
            'Async prediction accumulation: enabled '
            f'(angles={len(angles)}, result workers={int(async_prediction_result_worker_count)}, '
            f'join workers={int(async_prediction_join_worker_count)}, '
            'each angle variant owns an independent full-frame accumulator)'
        )
        spec_notes.append(
            'v16.4.0 async prediction accumulation queues YOLO result detach/copy, native inverse-mapping, and variant-local streaming cleanup to a shared prediction-result executor. Each angle variant owns its union/confidence accumulator, so no cross-angle writer locks or unified-angle drain barrier exists.'
        )
    else:
        print('Async prediction accumulation: disabled by YOLO_TTA_ASYNC_PREDICT_POSTPROCESS=0; prediction sources are drained synchronously.')
        spec_notes.append(
            'v12.2.12 async prediction accumulation was disabled by configuration; prediction sources are drained synchronously.'
        )
    spec_notes.append(
        'v15 streaming prediction sources are active by default: full-frame and tiled YOLO inputs render only a bounded CPU prefetch window before model.predict starts, rather than materializing a complete (slice,--imgsz,--imgsz[,channels]) volume first. '
        'Set YOLO_TTA_STREAMING_PREDICTION_SOURCES=0 to restore the legacy dense prediction-volume path; per-source prediction memmap flushes are still skipped by default unless YOLO_TTA_FLUSH_PREDICTION_VOLUME_ON_BUILD=1 or YOLO_TTA_PREDICT_FLUSH_EACH_VOLUME=1 is set.'
    )
    print(
        f'Streaming prediction-source preparers: {prediction_volume_builder_workers} '
        f'(per-source render workers: {per_prediction_volume_workers}, shared render workers: {prediction_render_workers}, '
        f'source tasks: {total_prediction_volume_build_tasks}, queued-source bound: {prediction_volume_queue_slots}, '
        f'CPU-warmed queued sources: {int(queued_streaming_cpu_warmup_sources)}, '
        f'eager CUDA-staged queued sources: {int(eager_gpu_input_staging_ahead_sources)})'
    )
    spec_notes.append(
        f'v15 prediction scheduler active: full-frame and tiled YOLO sources stream H×W×{int(channel_format.channel_count)} frames through StreamingYoloVolumeSource for --channel_format {channel_format.token}. Streaming prediction-source creation is unbounded by default, so cheap source refs for every remaining view/tile can be queued immediately; expensive CPU rendering is separately bounded by the shared render pool and each source prefetch window. '
        f'The resolved queued-source bound is {int(prediction_volume_queue_slots)} source(s); set YOLO_TTA_PREDICTION_VOLUME_QUEUE_SLOTS to a positive value to cap it or 0 to force all remaining sources in legacy dense mode too. '
        f'Each active streaming source can submit enough work to use the full shared render pool ({int(prediction_render_workers)} worker thread(s)); legacy dense materialization still uses {int(legacy_per_prediction_volume_workers)} worker(s) per builder. '
        f'Up to {int(queued_streaming_cpu_warmup_sources)} ready queued source(s) are CPU-warmed at a time (YOLO_TTA_STREAMING_SOURCE_WARMUP_SOURCES), so source creation can be unbounded without every future source enqueueing a prefetch window. '
        f'Up to {int(eager_gpu_input_staging_ahead_sources)} queued source(s) are eagerly CUDA-staged before they become the active model.predict source (YOLO_TTA_GPU_INPUT_STAGING_AHEAD_SOURCES), with no fixed four-source default. '
        'v12.2.11 lazily allocates each full-view union/confidence workspace only when that view first reaches inference, avoiding an eager all-views zero-fill before the first prediction.'
    )
    if dense_tiling_active:
        spec_notes.append(
            'Tiled prediction sources follow the deterministic tile footprint, stride order, angle variant, '
            'and inverse-mapping rules from the v12.2.0 specification.'
        )

    prediction_render_executor: Optional[ThreadPoolExecutor] = None
    if bool(streaming_sources_active) and shared_streaming_render_pool_enabled():
        prediction_render_executor = _create_tracked_thread_pool(
            max_workers=int(prediction_render_workers),
            thread_name_prefix='prediction-render',
        )
        print(f'Shared streaming render pool: enabled ({int(prediction_render_workers)} worker thread(s))')
    elif bool(streaming_sources_active):
        print('Shared streaming render pool: disabled; each source owns its render executor.')

    prediction_volume_executor = _create_tracked_thread_pool(
        max_workers=int(prediction_volume_builder_workers),
        thread_name_prefix='prediction-volume',
    )
    prediction_result_executor = _create_tracked_thread_pool(
        max_workers=int(async_prediction_result_worker_count),
        thread_name_prefix='predict-result',
    )
    prediction_join_executor = _create_tracked_thread_pool(
        max_workers=int(async_prediction_join_worker_count),
        thread_name_prefix='predict-join',
    )
    parent_postprocess_executor = _create_tracked_thread_pool(
        max_workers=int(parent_postprocess_workers),
        thread_name_prefix='parent-postprocess',
    )
    tile_dense_retirement_executor = _create_tracked_thread_pool(
        max_workers=int(tile_dense_retirement_workers),
        thread_name_prefix='tile-dense-retire',
    )
    tile_postprocess_executor = _create_tracked_thread_pool(
        max_workers=int(tile_postprocess_workers),
        thread_name_prefix='tile-postprocess',
    )

    pending_prediction_build_jobs: deque[Tuple[str, ViewInfo, object]] = deque()
    for view, aug_job in iter_aug_jobs_round_robin(inference_views, aug_jobs_by_view):
        pending_prediction_build_jobs.append(('fullframe', view, aug_job))
        if dense_tiling_active:
            for tile_job in tile_jobs_by_aug.get((view.name, aug_job.aug_id), []):
                pending_prediction_build_jobs.append(('tile', view, tile_job))

    prediction_volume_futures: Dict[Future, Tuple[str, ViewInfo, object]] = {}
    pending_prediction_volume_futures: set[Future] = set()
    ready_fullframe: deque[Tuple[ViewInfo, AugJob, PredictionVolumeRef]] = deque()
    # the ref is Optional because with several models only the first entry for a tile
    # carries the source that was already built; the rest build their own at pop time.
    ready_tile_infer: deque[Tuple[str, ViewInfo, DenseTileJob, Optional[PredictionVolumeRef]]] = deque()
    tile_inference_done: set[Tuple[str, str, str]] = set()
    prediction_accumulation_futures: Dict[Future, Dict[str, object]] = {}
    streaming_cpu_warmup_started_refs: set[int] = set()

    view_processing_futures: Dict[Future, Tuple[str, str]] = {}
    view_processing_submitted: set[Tuple[str, str]] = set()
    tile_cleanup_futures: Dict[Future, Tuple[str, str, str, str]] = {}
    # P (cleaned parent YOLO) is published before parent interpolation through this queue.
    # B (parent-only interpolation delta) becomes ready when the parent future completes.
    parent_mask_ready_events = queue.SimpleQueue()
    parent_bridge_ready: set[Tuple[str, str]] = set()
    parent_tile_supports_retired: set[Tuple[str, str]] = set()
    postprocessed_tiles_waiting_by_parent: Dict[Tuple[str, str], Dict[str, object]] = {}
    residual_tiles_waiting_by_parent: Dict[Tuple[str, str], Dict[str, object]] = {}
    tile_parent_gate_futures: Dict[Future, Tuple[str, str, str, str]] = {}
    tile_bridge_gate_futures: Dict[Future, Tuple[str, str, str, str]] = {}
    tile_consolidation_futures: Dict[Future, Tuple[str, str, str]] = {}
    tile_parent_finalization_futures: Dict[Future, Tuple[str, str]] = {}
    tile_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray] = {}
    tile_accumulator_paths: Dict[Tuple[str, str, str], Path] = {}
    tile_accumulator_locks_by_set: Dict[Tuple[str, str, str], List[threading.Lock]] = {}
    tile_parent_mask_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray] = {}
    tile_parent_bridge_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray] = {}
    tile_category_accumulator_locks: Dict[Tuple[str, str, str, str], List[threading.Lock]] = {}
    tile_completed_by_parent: Dict[Tuple[str, str], set[str]] = {}
    tile_completed_by_set: Dict[Tuple[str, str, str], set[str]] = {}
    tile_consolidation_submitted: set[Tuple[str, str, str]] = set()
    tile_consolidation_completed: set[Tuple[str, str, str]] = set()
    tile_parent_finalization_submitted: set[Tuple[str, str]] = set()
    def _prediction_volume_queue_depth() -> int:
        return int(len(pending_prediction_volume_futures) + len(ready_fullframe) + len(ready_tile_infer))

    def _make_streaming_fullframe_ref(view: ViewInfo, aug_job: AugJob) -> PredictionVolumeRef:
        write_aug_job_meta(aug_job, view, channel_format)
        render_workers = streaming_prediction_source_workers(int(per_prediction_volume_workers), int(view.num_slices))
        prefetch_frames = streaming_prediction_source_prefetch_frames(
            max(1, int(max(args.gpu_batch if gpu_worker_process_active else 1, args.cpu_batch if cpu_worker_process_active else 1)))
        )

        renderer = make_fullframe_channel_renderer(
            volume_rgb,
            view,
            aug_job,
            channel_format=channel_format,
            view_frames=_get_view_frame_cache(view),
        )

        name = f'Streaming full-frame prediction source {view.name}/{aug_job.aug_id}'
        source = StreamingYoloVolumeSource(
            renderer,
            num_frames=int(view.num_slices),
            name=name,
            batch_size=max(1, int(args.batch)),
            out_size=int(aug_job.aff.out_size),
            render_workers=int(render_workers),
            prefetch_frames=int(prefetch_frames),
            autostart=bool(streaming_prediction_source_autostart_enabled()),
            shared_executor=prediction_render_executor,
            channel_format=channel_format,
        )
        return PredictionVolumeRef(
            array=None,
            path=None,
            name=name,
            view_name=str(view.name),
            job_id=str(aug_job.aug_id),
            kind='fullframe',
            source=source,
            channel_format=channel_format,
        )

    def _make_streaming_tile_ref(view: ViewInfo, tile_job: DenseTileJob) -> PredictionVolumeRef:
        write_dense_tile_job_meta(tile_job, channel_format)
        render_workers = streaming_prediction_source_workers(int(per_prediction_volume_workers), int(view.num_slices))
        prefetch_frames = streaming_prediction_source_prefetch_frames(max(1, int(args.batch)))

        renderer = make_dense_tile_channel_renderer(
            volume_rgb,
            view,
            tile_job,
            channel_format=channel_format,
            view_frames=_get_view_frame_cache(view),
        )

        name = f'Streaming tile prediction source {view.name}/{tile_job.tile_id}'
        source = StreamingYoloVolumeSource(
            renderer,
            num_frames=int(view.num_slices),
            name=name,
            batch_size=max(1, int(args.batch)),
            out_size=int(tile_job.out_size),
            render_workers=int(render_workers),
            prefetch_frames=int(prefetch_frames),
            autostart=bool(streaming_prediction_source_autostart_enabled()),
            shared_executor=prediction_render_executor,
            channel_format=channel_format,
        )
        return PredictionVolumeRef(
            array=None,
            path=None,
            name=name,
            view_name=str(view.name),
            job_id=str(tile_job.tile_id),
            kind='tile',
            source=source,
            channel_format=channel_format,
        )

    def _submit_prediction_volume_build(kind: str, view: ViewInfo, job_obj: object) -> None:
        if str(kind) == 'fullframe':
            aug_job = job_obj
            assert isinstance(aug_job, AugJob)
            if streaming_prediction_sources_enabled():
                fut = prediction_volume_executor.submit(_make_streaming_fullframe_ref, view, aug_job)
            else:
                out_path = temp_dir / 'prediction_volumes' / 'fullframe' / view.name / f'{view.name}_{aug_job.aug_id}.u8.dat'
                fut = prediction_volume_executor.submit(
                    materialize_fullframe_prediction_volume_for_job,
                    volume_rgb,
                    view,
                    aug_job,
                    out_path=out_path,
                    view_frames=_get_view_frame_cache(view),
                    workers=int(per_prediction_volume_workers),
                    show_progress=False,
                    channel_format=channel_format,
                )
        elif str(kind) == 'tile':
            tile_job = job_obj
            assert isinstance(tile_job, DenseTileJob)
            if streaming_prediction_sources_enabled():
                fut = prediction_volume_executor.submit(_make_streaming_tile_ref, view, tile_job)
            else:
                out_path = temp_dir / 'prediction_volumes' / 'tiles' / view.name / str(tile_job.config_id) / f'{tile_job.tile_id}.u8.dat'
                fut = prediction_volume_executor.submit(
                    materialize_dense_tile_prediction_volume_for_job,
                    volume_rgb,
                    view,
                    tile_job,
                    out_path=out_path,
                    view_frames=_get_view_frame_cache(view),
                    workers=int(per_prediction_volume_workers),
                    show_progress=False,
                    channel_format=channel_format,
                )
        else:  # pragma: no cover
            raise ValueError(f'Unknown prediction volume build kind: {kind}')
        prediction_volume_futures[fut] = (str(kind), view, job_obj)
        pending_prediction_volume_futures.add(fut)

    def _pump_prediction_volume_build_queue() -> None:
        while pending_prediction_build_jobs and _prediction_volume_queue_depth() < int(prediction_volume_queue_slots):
            kind, view, job_obj = pending_prediction_build_jobs.popleft()
            _submit_prediction_volume_build(str(kind), view, job_obj)

    def _queued_gpu_staging_ref_count() -> int:
        seen: set[int] = set()
        count = 0
        for _view, _job, ref in list(ready_fullframe):
            rid = id(ref)
            if rid not in seen and _prediction_ref_has_gpu_input_staging(ref):
                seen.add(rid)
                count += 1
        for _model_name, _view, _tile_job, ref in list(ready_tile_infer):
            if ref is None:  # a queued tile whose source is not built yet
                continue
            rid = id(ref)
            if rid not in seen and _prediction_ref_has_gpu_input_staging(ref):
                seen.add(rid)
                count += 1
        return int(count)

    def _maybe_eager_stage_prediction_ref(pred_ref: PredictionVolumeRef) -> PredictionVolumeRef:
        if int(eager_gpu_input_staging_ahead_sources) <= 0:
            return pred_ref
        if _prediction_ref_has_gpu_input_staging(pred_ref):
            return pred_ref
        if _queued_gpu_staging_ref_count() >= int(eager_gpu_input_staging_ahead_sources):
            return pred_ref
        return maybe_eager_stage_prediction_ref_on_gpu(pred_ref, pred_cfg)

    def _queued_cpu_warmup_ref_count() -> int:
        seen: set[int] = set()
        count = 0
        for _view, _job, ref in list(ready_fullframe):
            rid = id(ref)
            if rid in seen:
                continue
            if rid in streaming_cpu_warmup_started_refs or _prediction_ref_has_gpu_input_staging(ref):
                seen.add(rid)
                count += 1
        for _model_name, _view, _tile_job, ref in list(ready_tile_infer):
            if ref is None:  # a queued tile whose source is not built yet
                continue
            rid = id(ref)
            if rid in seen:
                continue
            if rid in streaming_cpu_warmup_started_refs or _prediction_ref_has_gpu_input_staging(ref):
                seen.add(rid)
                count += 1
        return int(count)

    def _maybe_start_cpu_warmup_prediction_ref(pred_ref: PredictionVolumeRef) -> None:
        if int(queued_streaming_cpu_warmup_sources) <= 0:
            return
        if _prediction_ref_has_gpu_input_staging(pred_ref):
            return
        rid = id(pred_ref)
        if rid in streaming_cpu_warmup_started_refs:
            return
        if _queued_cpu_warmup_ref_count() >= int(queued_streaming_cpu_warmup_sources):
            return
        source = getattr(pred_ref, 'source', None)
        start_fn = getattr(source, 'start', None)
        if not callable(start_fn):
            return
        try:
            start_fn()
            streaming_cpu_warmup_started_refs.add(rid)
        except Exception as exc:
            print(f'Warning: queued CPU render warmup could not start for {pred_ref.name} ({exc}); source will start on demand.')

    def _warmup_ready_prediction_sources() -> None:
        if int(queued_streaming_cpu_warmup_sources) <= 0:
            return
        for _view, _job, ref in list(ready_fullframe):
            _maybe_start_cpu_warmup_prediction_ref(ref)
            if _queued_cpu_warmup_ref_count() >= int(queued_streaming_cpu_warmup_sources):
                return
        for _model_name, _view, _tile_job, ref in list(ready_tile_infer):
            if ref is None:  # a queued tile whose source is not built yet
                continue
            _maybe_start_cpu_warmup_prediction_ref(ref)
            if _queued_cpu_warmup_ref_count() >= int(queued_streaming_cpu_warmup_sources):
                return

    def _drain_completed_prediction_volume_futures() -> None:
        for fut in list(pending_prediction_volume_futures):
            if not fut.done():
                continue
            pending_prediction_volume_futures.remove(fut)
            kind, view, job_obj = prediction_volume_futures.pop(fut)
            pred_ref = _maybe_eager_stage_prediction_ref(fut.result())
            if str(kind) == 'fullframe':
                assert isinstance(job_obj, AugJob)
                ready_fullframe.append((view, job_obj, pred_ref))
            else:
                assert isinstance(job_obj, DenseTileJob)
                # StreamingYoloVolumeSource is single-use — __next__ closes it on
                # exhaustion and start then raises — and the in-process tile path also
                # closes the ref in its finally. Queueing the SAME ref once per model
                # therefore handed every model after the first a closed source. Each model
                # now gets its own, built lazily at pop time so N models do not spin up N
                # prefetch pipelines for a tile that is still sitting in the queue.
                tile_model_names = [str(name) for name, _ in yolo_models]
                for position, model_name in enumerate(tile_model_names):
                    ready_tile_infer.append(
                        (str(model_name), view, job_obj, pred_ref if position == 0 else None)
                    )
                if not tile_model_names:
                    close_prediction_volume_ref(pred_ref, keep_temp=bool(keep_temp_artifacts))
        _pump_prediction_volume_build_queue()
        _warmup_ready_prediction_sources()

    def _ensure_baseline_workspaces(model_name: str, view: ViewInfo) -> None:
        key = (str(model_name), str(view.name))
        if key in baseline_union_by_model_view:
            return
        union_path = temp_dir / 'union' / str(model_name) / f'{view.name}.union.u8.dat'
        confmap_path = temp_dir / 'union' / str(model_name) / f'{view.name}.confmap.u8.dat'
        union_path.parent.mkdir(parents=True, exist_ok=True)

        # Interpolated views retain a real pathname because the isolated interpolation
        # backend reopens them by path. Every process-worker direct-union view (CUDA and/or
        # OpenVINO) must instead have reopenable shared backing; ordinary in-process views
        # may prefer anonymous RAM. v17.0.1 fixed hybrid D1 runs where GPU direct union was
        # disabled for GPU-only views but CPU-eligible views still write a common direct union.
        process_worker_direct_union = bool(worker_direct_union_active)
        union_prefer_memory = not (
            (
                interpolation_process_backend_enabled()
                and _view_uses_interpolation(view, int(args.interpolation_distance))
            )
            or process_worker_direct_union
        )
        processing_shape = view_processing_volume_shape(view, int(args.imgsz))
        union_mm: Optional[np.ndarray] = None
        conf_mm: Optional[np.ndarray] = None
        try:
            union_mm = allocate_workspace_array(
                shape=processing_shape,
                dtype=np.uint8,
                path=union_path,
                desc=f'{model_name}/{view.name} baseline union workspace',
                prefer_memory=bool(union_prefer_memory),
                prefer_memfd=bool(process_worker_direct_union),
            )
            if float(args.min_conf) > 0.0:
                conf_mm = allocate_workspace_array(
                    shape=processing_shape,
                    dtype=np.uint8,
                    path=confmap_path,
                    desc=f'{model_name}/{view.name} baseline confidence workspace',
                    prefer_memory=not process_worker_direct_union,
                    prefer_memfd=bool(process_worker_direct_union),
                )
        except BaseException:
            close_memmap_array_without_flush(conf_mm)
            close_memmap_array_without_flush(union_mm)
            for failed_path in (union_path, confmap_path):
                try:
                    failed_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        assert union_mm is not None
        union_backing = _memmap_backing_path(union_mm)
        conf_backing = _memmap_backing_path(conf_mm) if conf_mm is not None else None
        backing_error: Optional[str] = None
        if process_worker_direct_union and union_backing is None:
            backing_error = (
                f'{model_name}/{view.name} direct-union workspace is not process-shareable'
            )
        elif process_worker_direct_union and conf_mm is not None and conf_backing is None:
            backing_error = (
                f'{model_name}/{view.name} direct-union confidence workspace is not process-shareable'
            )
        if backing_error is not None:
            close_memmap_array_without_flush(conf_mm)
            close_memmap_array_without_flush(union_mm)
            for failed_path in (union_path, confmap_path):
                try:
                    failed_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise RuntimeError(backing_error)
        baseline_union_by_model_view[key] = union_mm
        baseline_union_paths[key] = union_backing or union_path
        baseline_confmap_by_model_view[key] = conf_mm
        baseline_confmap_paths[key] = (
            (conf_backing or confmap_path) if conf_mm is not None else None
        )
        baseline_slice_locks_by_model_view[key] = [threading.Lock() for _ in range(int(view.num_slices))]
        if process_worker_direct_union:
            if (
                key in direct_union_backing_leases
                or key in direct_union_inference_views
                or key in direct_union_postprocess_views
            ):
                raise RuntimeError(f'direct-union backing {key} was admitted more than once')
            dense_volume_bytes = int(array_nbytes(processing_shape, np.uint8))
            # Reserve the complete deterministic per-parent dense set before inference
            # starts. Tiling adds the consolidated accumulator and NRRD decomposition adds
            # two category accumulators; charging only the parent union let those canvases
            # appear later outside admission.
            dense_volume_count = (2 if conf_mm is not None else 1)
            if bool(dense_tiling_active):
                dense_volume_count += 1
                if bool(nrrd_layers_needed):
                    dense_volume_count += 2
            backing_bytes = int(dense_volume_bytes) * int(dense_volume_count)
            direct_union_backing_leases[key] = _DirectUnionBackingLease(
                key=key, nbytes=int(backing_bytes), phase='inference', owner_count=1,
            )
            direct_union_inference_views.add(key)
            direct_union_inference_bytes[key] = int(backing_bytes)

    def _publish_parent_mask_ready(model_name: str, view_name: str, support_mm: object) -> None:
        parent_mask_ready_events.put((str(model_name), str(view_name), support_mm))
        # In push-drain mode the scheduler may be sleeping on a shared event while the
        # parent future remains incomplete. Wake it as soon as P is publishable.
        try:
            scheduler_wake.set()
        except (NameError, UnboundLocalError):
            pass


    def _submit_view_prepare(model_name: str, view: ViewInfo) -> None:
        key = (str(model_name), str(view.name))
        if key in view_processing_submitted:
            return
        view_processing_submitted.add(key)
        d1_shadow_path = d1_view_shadow_path_by_parent.pop(key, None)
        preinterpolation_layer_already_published = bool(d1_shadow_path is not None)
        if d1_shadow_path is None:
            union_mm: Optional[np.ndarray] = baseline_union_by_model_view.pop(key)
            confmap_mm = baseline_confmap_by_model_view.pop(key)
            union_path = baseline_union_paths.pop(key)
            confmap_path = baseline_confmap_paths.pop(key)
            baseline_slice_locks_by_model_view.pop(key, None)
        else:
            union_mm = None
            confmap_mm = None
            union_path = (
                Path(temp_dir) / 'd1_view_shadow_dense' / str(model_name)
                / f'{str(view.name)}.fullframe.u8.dat'
            )
            confmap_path = None
        # every slice of this view already hole-filled on device by the GPU
        # workers -> the CPU "2D hole fill (<view>)" pass is a no-op recompute; skip it.
        hole_fill_done_on_device = bool(
            int(view_device_hole_filled_slices.get(key, 0)) >= int(view.num_slices)
        )
        # hand the aggregated device-union slice metadata to the view prepare
        # (None when any task lacked it — consumers fall back to scanning).
        slice_meta_holder = view_slice_meta.pop(key, None)
        if slice_meta_holder is not None and not bool(slice_meta_holder.get('valid', False)):
            slice_meta_holder = None
        processing_bytes = int(array_nbytes(view_processing_volume_shape(view, int(args.imgsz)), np.uint8))
        source_bytes = int(array_nbytes((int(input_T), int(input_H), int(input_W)), np.uint8))
        view_name_lower = str(view.name).lower()
        transient_bytes = 2 * GIB
        if 'radial_tilted' in view_name_lower or ('tilted' in view_name_lower and str(view.family) == 'radial'):
            # v16.1.3 D2 writes the final source destination directly and no longer owns a
            # processing-sized tilted base stack in addition to that destination.
            transient_bytes = int(source_bytes) + 4 * GIB
        elif 'tilted' in view_name_lower:
            transient_bytes = int(source_bytes) + 4 * GIB
        elif _view_uses_interpolation(view, int(args.interpolation_distance)):
            transient_bytes = int(processing_bytes) * 2 + 4 * GIB

        def _run_admitted_view_prepare() -> PreparedViewResult:
            with parent_transient_admission.reserve(
                int(transient_bytes), f'{model_name}/{view.name}',
            ):
                local_union_mm = union_mm
                if local_union_mm is None:
                    if d1_shadow_path is None:
                        raise RuntimeError(f'{model_name}/{view.name}: missing D1 view shadow')
                    local_union_mm = materialize_raw_bbox_mask_store_workspace(
                        d1_shadow_path,
                        union_path,
                        desc=f'D1 view-native shadow materialization {model_name}/{view.name}',
                        workers=int(parent_slice_postprocess_workers),
                    )
                    if not bool(keep_temp_artifacts):
                        try:
                            shutil.rmtree(d1_shadow_path, ignore_errors=True)
                        except Exception:
                            pass
                return prepare_view_volume_after_fullframe(
                    model_name=str(model_name),
                    view=view,
                    union_mm=local_union_mm,
                    confmap_mm=confmap_mm,
                    union_path=union_path,
                    confmap_path=confmap_path,
                    temp_dir=temp_dir,
                    dense_tiling_active=bool(dense_tiling_active),
                    min_conf=float(args.min_conf),
                    min_radius=float(args.min_radius),
                    interpolate=int(args.interpolation_distance),
                    interpolation_walk_back=int(args.interpolation_walk_back),
                    interpolation_candidates=int(args.interpolation_candidates),
                    interpolate_passes=int(args.interpolation_passes),
                    interpolate_min_radius=float(args.interpolation_min_radius),
                    interpolation_search_angle=float(args.interpolation_search_angle),
                    keep_temp=bool(keep_temp_artifacts),
                    slice_workers=int(parent_slice_postprocess_workers),
                    interpolation_task_workers=int(parent_interpolation_task_workers),
                    nrrd_layers_enabled=bool(nrrd_layers_needed),
                    precleaned_slice_cleanup=bool(angle_variant_streaming_cleanup_active),
                    hole_fill_done_on_device=bool(hole_fill_done_on_device),
                    slice_meta=slice_meta_holder,
                    fuse_radial_component_layers=bool(
                        angle_variant_gpu_fastpath_active
                        and fused_angle_variant_radial_component_layer_enabled()
                    ),
                    parent_mask_ready_callback=(
                        _publish_parent_mask_ready if bool(dense_tiling_active) else None
                    ),
                    internal_final_layer_enabled=bool(
                        component_ref_dense_retirement_active
                        and not nrrd_layers_needed
                    ),
                    preinterpolation_layer_already_published=bool(
                        preinterpolation_layer_already_published
                    ),
                )

        lease = direct_union_backing_leases.get(key)
        transitioned = False
        if lease is not None:
            if key not in direct_union_inference_views or key in direct_union_postprocess_views:
                raise RuntimeError(
                    f'direct-union backing {key} is not exclusively inference-owned at handoff'
                )
            lease.transition('inference', 'postprocess')
            direct_union_inference_views.remove(key)
            direct_union_inference_bytes.pop(key, None)
            direct_union_postprocess_views.add(key)
            direct_union_postprocess_bytes[key] = int(lease.nbytes)
            transitioned = True
        try:
            fut = parent_postprocess_executor.submit(_run_admitted_view_prepare)
        except BaseException:
            if transitioned and lease is not None:
                lease.transition('postprocess', 'inference')
                direct_union_postprocess_views.discard(key)
                direct_union_postprocess_bytes.pop(key, None)
                direct_union_inference_views.add(key)
                direct_union_inference_bytes[key] = int(lease.nbytes)
            # Restore ownership registries because the postprocess closure never started.
            if d1_shadow_path is not None:
                d1_view_shadow_path_by_parent[key] = d1_shadow_path
            else:
                assert union_mm is not None
                baseline_union_by_model_view[key] = union_mm
                baseline_confmap_by_model_view[key] = confmap_mm
                baseline_union_paths[key] = union_path
                baseline_confmap_paths[key] = confmap_path
                baseline_slice_locks_by_model_view[key] = [
                    threading.Lock() for _ in range(int(view.num_slices))
                ]
            view_processing_submitted.discard(key)
            raise
        view_processing_futures[fut] = key
        # The last inference chunk has committed and the backing is now owned solely by the
        # postprocess closure. Refill worker queues immediately instead of waiting for cvol/NRRD
        # completion to release a GPU-admission slot.
        if transitioned and gpu_worker_pending_task_ids:
            _dispatch_inference_windows()

    def _tile_gate_lock_shards(view_name: str) -> int:
        view = view_infos_by_name[str(view_name)]
        return max(1, min(64, int(view.num_slices)))

    def _get_tile_accumulator(
        model_name: str,
        view_name: str,
        config_id: str,
    ) -> np.ndarray:
        key = (str(model_name), str(view_name), str(config_id))
        acc = tile_accumulator_by_set.get(key)
        if acc is not None:
            return acc
        view = view_infos_by_name[str(view_name)]
        acc_path = (
            temp_dir / 'tile_consolidated' / str(model_name) /
            str(view_name) / str(config_id) / 'gated_or.u8.dat'
        )
        prefer_shared_ram = bool(tile_intermediate_accumulators_prefer_memory())
        acc = allocate_workspace_array(
            shape=view_processing_volume_shape(view, int(args.imgsz)),
            dtype=np.uint8,
            path=acc_path,
            desc=(
                f'{model_name}/{view_name}/{config_id} consolidated '
                'two-stage gated-tile accumulator'
            ),
            # The consolidated volume is subsequently reopened by a spawned
            # interpolation process.  Prefer a parent-owned memfd, never an anonymous
            # ndarray that would require a second full-volume process_input copy.
            prefer_memory=False,
            prefer_memfd=bool(prefer_shared_ram),
            reserve_bytes=tile_intermediate_accumulator_reserve_bytes(),
        )
        tile_accumulator_by_set[key] = acc
        actual_path = _memmap_backing_path(acc)
        tile_accumulator_paths[key] = Path(actual_path) if actual_path is not None else acc_path
        tile_accumulator_locks_by_set[key] = [
            threading.Lock() for _ in range(_tile_gate_lock_shards(str(view_name)))
        ]
        return acc

    def _get_tile_category_accumulator(
        model_name: str,
        view_name: str,
        config_id: str,
        category: str,
    ) -> np.ndarray:
        key = (str(model_name), str(view_name), str(config_id))
        category_norm = str(category)
        store = (
            tile_parent_mask_accumulator_by_set
            if category_norm == 'parent_mask'
            else tile_parent_bridge_accumulator_by_set
        )
        tile_category_accumulator_locks.setdefault(
            (key[0], key[1], key[2], category_norm),
            [threading.Lock() for _ in range(_tile_gate_lock_shards(str(view_name)))],
        )
        acc = store.get(key)
        if acc is not None:
            return acc
        view = view_infos_by_name[str(view_name)]
        acc_path = (
            temp_dir / 'tile_consolidated' / str(model_name) / str(view_name) /
            str(config_id) / f'gated_or_accepted_by_{category_norm}.u8.dat'
        )
        prefer_shared_ram = bool(tile_intermediate_accumulators_prefer_memory())
        acc = allocate_workspace_array(
            shape=view_processing_volume_shape(view, int(args.imgsz)),
            dtype=np.uint8,
            path=acc_path,
            desc=(
                f'{model_name}/{view_name}/{config_id} consolidated gated-tile '
                f'accumulator accepted by {category_norm}'
            ),
            # Category canvases do not enter interpolation today, but keeping all tile
            # accumulators process-reopenable avoids silently reintroducing anonymous
            # full-volume state into this path.
            prefer_memory=False,
            prefer_memfd=bool(prefer_shared_ram),
            reserve_bytes=tile_intermediate_accumulator_reserve_bytes(),
        )
        store[key] = acc
        return acc


    def _parent_destination_ready(model_name: str, view_name: str) -> bool:
        view = view_infos_by_name[str(view_name)]
        if view.family == 'radial':
            return str(view_name) in radial_native_output_by_model.get(str(model_name), {})
        if is_tilted_view(view):
            return str(view_name) in tilted_native_output_by_model.get(str(model_name), {})
        return str(view_name) in view_volumes_by_model.get(str(model_name), {})

    def _parent_destination_volume(model_name: str, view_name: str) -> np.ndarray:
        view = view_infos_by_name[str(view_name)]
        if view.family == 'radial':
            return radial_native_output_by_model[str(model_name)][str(view_name)]
        if is_tilted_view(view):
            return tilted_native_output_by_model[str(model_name)][str(view_name)]
        return view_volumes_by_model[str(model_name)][str(view_name)]

    def _retire_parent_dense_view(
        model_name: str,
        view_name: str,
        *,
        extra_arrays: Sequence[object] = (),
        reason: str,
    ) -> None:
        """Drop every registry reference to one component-ref-backed dense view."""
        parent_key = (str(model_name), str(view_name))
        candidates: List[object] = list(extra_arrays)
        for registry in (
            native_view_support_by_model,
            radial_native_output_by_model,
            tilted_native_output_by_model,
            view_volumes_by_model,
        ):
            value = registry.get(parent_key[0], {}).pop(parent_key[1], None)
            if value is not None:
                candidates.append(value)

        retired_ids: set[int] = set()
        retired_bytes = 0
        for value in candidates:
            if value is None or id(value) in retired_ids:
                continue
            retired_ids.add(id(value))
            try:
                retired_bytes += int(np.asarray(value).nbytes)
            except Exception:
                pass
            backing_path = _memmap_backing_path(value)
            close_memmap_array_without_flush(value)
            if backing_path is not None and not str(backing_path).startswith('/proc/'):
                try:
                    Path(backing_path).unlink(missing_ok=True)
                except Exception:
                    pass
        if retired_ids:
            print(
                f'Component-ref dense retirement: released {retired_bytes / GIB:.2f} GiB '
                f'for {parent_key[0]}/{parent_key[1]} ({reason}).'
            )

        lease = direct_union_backing_leases.pop(parent_key, None)
        if lease is not None:
            if (
                parent_key not in direct_union_postprocess_views
                or str(lease.phase) != 'postprocess'
            ):
                raise RuntimeError(
                    f'direct-union backing {parent_key} retired outside postprocess ownership'
                )
            lease.release('postprocess')
            direct_union_postprocess_views.remove(parent_key)
            direct_union_postprocess_bytes.pop(parent_key, None)
            # A tiled parent can hold this byte lease well beyond its parent future.
            # Refill inference immediately when consolidation finally retires it.
            try:
                if gpu_worker_pending_task_ids:
                    _dispatch_inference_windows()
            except (NameError, UnboundLocalError):
                pass

    def _retire_parent_tile_supports_if_gates_complete(
        model_name: str,
        view_name: str,
    ) -> None:
        """Release immutable P/B stores immediately after their final tile gate."""
        parent_key = (str(model_name), str(view_name))
        if parent_key in parent_tile_supports_retired:
            return
        expected_tiles = int(tile_expected_by_parent.get(parent_key, 0))
        if expected_tiles <= 0:
            return
        if len(tile_completed_by_parent.get(parent_key, set())) < int(expected_tiles):
            return
        # parent_bridge_ready is published only after the parent future (which also
        # returns the already-published P object) has been drained.  Waiting for it avoids
        # closing P and then accidentally re-registering the same closed object later.
        if parent_key not in parent_bridge_ready:
            return

        released: List[str] = []
        for label, registry in (
            ('P', parent_mask_support_by_model),
            ('B', parent_bridge_support_by_model),
        ):
            support = registry.get(parent_key[0], {}).pop(parent_key[1], None)
            if support is None:
                continue
            close_raw_store_or_memmap_volume(
                support, keep_temp=bool(keep_temp_artifacts),
            )
            released.append(str(label))
        parent_tile_supports_retired.add(parent_key)
        if released:
            print(
                f'Released parent tile support {"/".join(released)} for '
                f'{parent_key[0]}/{parent_key[1]} after all {expected_tiles} tile gate(s).'
            )

    def _maybe_finalize_tile_parent(model_name: str, view_name: str) -> None:
        """Retire one parent only after every configured tile set has been unioned."""
        parent_key = (str(model_name), str(view_name))
        config_ids = tile_config_ids_by_parent.get(parent_key, ())
        if not config_ids or parent_key in tile_parent_finalization_submitted:
            return
        if any(
            (parent_key[0], parent_key[1], str(config_id))
            not in tile_consolidation_completed
            for config_id in config_ids
        ):
            return

        tile_parent_finalization_submitted.add(parent_key)
        if not bool(component_ref_dense_retirement_active):
            return
        if bool(nrrd_layers_needed):
            _retire_parent_dense_view(
                str(model_name),
                str(view_name),
                reason='full-frame and all configured tile-set terminal refs are complete',
            )
            return

        # With no requested component NRRDs, retain one private sparse reference to the
        # destination only after every independently interpolated tile set has entered it.
        view = view_infos_by_name[str(view_name)]
        fut = tile_postprocess_executor.submit(
            finalize_parent_without_tile_contribution_for_sparse_retirement,
            model_name=str(model_name),
            view=view,
            destination_mm=_parent_destination_volume(str(model_name), str(view_name)),
            destination_lock=view_volume_locks[(str(model_name), str(view_name))],
            temp_dir=temp_dir,
            slice_workers=int(tile_slice_postprocess_workers),
        )
        tile_parent_finalization_futures[fut] = parent_key

    def _maybe_submit_tile_consolidation(
        model_name: str,
        view_name: str,
        config_id: str,
    ) -> None:
        parent_key = (str(model_name), str(view_name))
        set_key = (str(model_name), str(view_name), str(config_id))
        if set_key in tile_consolidation_submitted:
            return
        expected_tiles = int(tile_expected_by_set.get(set_key, 0))
        if expected_tiles <= 0:
            return
        if len(tile_completed_by_set.get(set_key, set())) < expected_tiles:
            return
        _retire_parent_tile_supports_if_gates_complete(
            str(model_name), str(view_name),
        )
        if not _parent_destination_ready(str(model_name), str(view_name)):
            return

        tile_consolidation_submitted.add(set_key)
        acc = tile_accumulator_by_set.get(set_key)
        if acc is None:
            # Every tile in this configuration was empty after cleanup. Other configured
            # sets still have their own consolidation and parent-terminal barrier.
            tile_consolidation_completed.add(set_key)
            _maybe_finalize_tile_parent(str(model_name), str(view_name))
            return

        view = view_infos_by_name[str(view_name)]
        fut = tile_postprocess_executor.submit(
            finalize_consolidated_tile_volume_for_parent,
            model_name=str(model_name),
            view=view,
            tile_accumulator_mm=acc,
            destination_mm=_parent_destination_volume(str(model_name), str(view_name)),
            destination_lock=view_volume_locks[(str(model_name), str(view_name))],
            temp_dir=temp_dir,
            interpolate=int(args.interpolation_distance),
            interpolation_walk_back=int(args.interpolation_walk_back),
            interpolation_candidates=int(args.interpolation_candidates),
            interpolate_passes=int(args.interpolation_passes),
            interpolate_min_radius=float(args.interpolation_min_radius),
            interpolation_search_angle=float(args.interpolation_search_angle),
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(tile_slice_postprocess_workers),
            interpolation_task_workers=int(tile_interpolation_task_workers),
            nrrd_layers_enabled=bool(nrrd_layers_needed),
            tile_parent_mask_accumulator_mm=tile_parent_mask_accumulator_by_set.get(set_key),
            tile_parent_bridge_accumulator_mm=tile_parent_bridge_accumulator_by_set.get(set_key),
            # A parent-level terminal task materializes this only after every configured
            # set has entered the destination, avoiding an incomplete private final ref.
            internal_final_layer_enabled=False,
            config_id=str(config_id),
        )
        tile_consolidation_futures[fut] = set_key

    def _maybe_submit_tile_consolidations_for_parent(
        model_name: str,
        view_name: str,
    ) -> None:
        parent_key = (str(model_name), str(view_name))
        for config_id in tile_config_ids_by_parent.get(parent_key, ()):
            _maybe_submit_tile_consolidation(
                str(model_name), str(view_name), str(config_id),
            )

    def _mark_tile_complete(
        model_name: str,
        view_name: str,
        config_id: str,
        tile_id: str,
    ) -> None:
        parent_key = (str(model_name), str(view_name))
        if parent_key not in tile_expected_by_parent:
            return
        completed = tile_completed_by_parent.setdefault(parent_key, set())
        completed.add(str(tile_id))
        set_key = (str(model_name), str(view_name), str(config_id))
        completed_for_set = tile_completed_by_set.setdefault(set_key, set())
        completed_for_set.add(str(tile_id))
        _maybe_submit_tile_consolidation(
            str(model_name), str(view_name), str(config_id),
        )

    def _retire_tile_result(
        result: TilePostprocessResult | DeferredTilePostprocessResult,
    ) -> None:
        if isinstance(result, TilePostprocessResult) and result.tile_mask_mm is not None:
            close_memmap_array(result.tile_mask_mm)
        _delete_tile_result_storage(result, keep_temp=bool(keep_temp_artifacts))
        _release_tile_dense_result_for_key(
            str(result.model_name), str(result.view_name), str(result.tile_id),
            reason='tile result storage retired',
        )


    def _submit_tile_parent_gate(
        result: TilePostprocessResult | DeferredTilePostprocessResult,
    ) -> None:
        """Gate one original tile against immutable same-angle parent YOLO support P."""
        parent_key = (str(result.model_name), str(result.view_name))
        parent_mask_support = parent_mask_support_by_model.get(
            str(result.model_name), {}
        ).get(str(result.view_name))
        if parent_mask_support is None:
            waiting = postprocessed_tiles_waiting_by_parent.setdefault(parent_key, {})
            if isinstance(result, DeferredTilePostprocessResult):
                waiting[str(result.tile_id)] = result
            elif result.tile_mask_store is not None:
                waiting[str(result.tile_id)] = defer_open_tile_result_store(result)
            else:
                waiting[str(result.tile_id)] = spill_waiting_tile_result_to_raw_store(
                    result,
                    temp_dir,
                    workers=int(tile_slice_postprocess_workers),
                    keep_original=bool(keep_temp_artifacts),
                )
            _release_tile_dense_result_for_key(
                str(result.model_name), str(result.view_name), str(result.tile_id),
                reason='sparse retirement while waiting for parent mask',
            )
            return

        tile_accumulator_mm = _get_tile_accumulator(
            str(result.model_name), str(result.view_name), str(result.config_id),
        )
        set_key = (
            str(result.model_name), str(result.view_name), str(result.config_id),
        )
        tile_parent_mask_accumulator_mm = None
        if bool(nrrd_layers_needed):
            tile_parent_mask_accumulator_mm = _get_tile_category_accumulator(
                str(result.model_name), str(result.view_name),
                str(result.config_id), 'parent_mask',
            )

        fut = tile_postprocess_executor.submit(
            gate_tile_result_against_parent_mask,
            result,
            parent_mask_support_mm=parent_mask_support,
            tile_accumulator_mm=tile_accumulator_mm,
            tile_accumulator_locks=tile_accumulator_locks_by_set.get(set_key),
            work_dir=temp_dir / 'tile_parent_gate_residuals',
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(tile_slice_postprocess_workers),
            tile_parent_mask_accumulator_mm=tile_parent_mask_accumulator_mm,
            tile_parent_mask_accumulator_locks=tile_category_accumulator_locks.get(
                (
                    str(result.model_name), str(result.view_name),
                    str(result.config_id), 'parent_mask',
                )
            ),
        )
        tile_parent_gate_futures[fut] = (
            str(result.model_name), str(result.view_name),
            str(result.config_id), str(result.tile_id),
        )


    def _submit_tile_bridge_gate(
        result: TilePostprocessResult | DeferredTilePostprocessResult,
    ) -> None:
        """Re-gate only P-failed components against immutable same-angle parent bridge B."""
        parent_key = (str(result.model_name), str(result.view_name))
        if int(args.interpolation_distance) <= 0:
            # Without parent interpolation there can never be bridge support. Parent-gate
            # residuals are final rejections and should retire immediately instead of waiting
            # for the parent postprocess future to publish an empty bridge milestone.
            _retire_tile_result(result)
            _mark_tile_complete(
                str(result.model_name), str(result.view_name),
                str(result.config_id), str(result.tile_id),
            )
            return
        if parent_key not in parent_bridge_ready:
            waiting = residual_tiles_waiting_by_parent.setdefault(parent_key, {})
            if isinstance(result, DeferredTilePostprocessResult):
                waiting[str(result.tile_id)] = result
            elif result.tile_mask_store is not None:
                # The parent gate emitted a crop-local CTILE. Drop its mmap while B is
                # pending and retain only a lightweight descriptor.
                waiting[str(result.tile_id)] = defer_open_tile_result_store(result)
            else:
                waiting[str(result.tile_id)] = spill_waiting_tile_result_to_raw_store(
                    result,
                    temp_dir,
                    workers=int(tile_slice_postprocess_workers),
                    keep_original=bool(keep_temp_artifacts),
                )
                _release_tile_dense_result_for_key(
                    str(result.model_name), str(result.view_name), str(result.tile_id),
                    reason='sparse retirement while waiting for parent bridge',
                )
            return

        parent_bridge_support = parent_bridge_support_by_model.get(
            str(result.model_name), {}
        ).get(str(result.view_name))
        if parent_bridge_support is None:
            # Parent interpolation produced no bridge voxels (or was disabled), so every
            # remaining whole component is definitively rejected.
            _retire_tile_result(result)
            _mark_tile_complete(
                str(result.model_name), str(result.view_name),
                str(result.config_id), str(result.tile_id),
            )
            return

        tile_accumulator_mm = _get_tile_accumulator(
            str(result.model_name), str(result.view_name), str(result.config_id),
        )
        set_key = (
            str(result.model_name), str(result.view_name), str(result.config_id),
        )
        tile_parent_bridge_accumulator_mm = None
        if bool(nrrd_layers_needed):
            tile_parent_bridge_accumulator_mm = _get_tile_category_accumulator(
                str(result.model_name), str(result.view_name),
                str(result.config_id), 'parent_bridge',
            )

        fut = tile_postprocess_executor.submit(
            gate_tile_residual_against_parent_bridge,
            result,
            parent_bridge_support_mm=parent_bridge_support,
            tile_accumulator_mm=tile_accumulator_mm,
            tile_accumulator_locks=tile_accumulator_locks_by_set.get(set_key),
            work_dir=temp_dir / 'tile_bridge_gate_residuals',
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(tile_slice_postprocess_workers),
            tile_parent_bridge_accumulator_mm=tile_parent_bridge_accumulator_mm,
            tile_parent_bridge_accumulator_locks=tile_category_accumulator_locks.get(
                (
                    str(result.model_name), str(result.view_name),
                    str(result.config_id), 'parent_bridge',
                )
            ),
        )
        tile_bridge_gate_futures[fut] = (
            str(result.model_name), str(result.view_name),
            str(result.config_id), str(result.tile_id),
        )


    def _flush_ready_postprocessed_tiles() -> None:
        ready_results: List[TilePostprocessResult | DeferredTilePostprocessResult] = []
        for parent_key, waiting in list(postprocessed_tiles_waiting_by_parent.items()):
            model_name, view_name = parent_key
            if view_name not in parent_mask_support_by_model.get(model_name, {}):
                continue
            for wait_result in waiting.values():
                if isinstance(wait_result, (DeferredTilePostprocessResult, TilePostprocessResult)):
                    # Keep CTILE descriptors closed in the scheduler. The gate worker opens
                    # the store only when it is actually ready to consume it.
                    ready_results.append(wait_result)
                else:
                    raise TypeError(f'Unsupported waiting tile result type: {type(wait_result)!r}')
            del postprocessed_tiles_waiting_by_parent[parent_key]

        for result in ready_results:
            _submit_tile_parent_gate(result)


    def _flush_ready_residual_tiles() -> None:
        ready_results: List[TilePostprocessResult | DeferredTilePostprocessResult] = []
        for parent_key, waiting in list(residual_tiles_waiting_by_parent.items()):
            if parent_key not in parent_bridge_ready:
                continue
            for wait_result in waiting.values():
                if isinstance(wait_result, (DeferredTilePostprocessResult, TilePostprocessResult)):
                    ready_results.append(wait_result)
                else:
                    raise TypeError(f'Unsupported residual tile result type: {type(wait_result)!r}')
            del residual_tiles_waiting_by_parent[parent_key]

        for result in ready_results:
            _submit_tile_bridge_gate(result)


    def _drain_parent_mask_ready_events() -> None:
        published = False
        while True:
            try:
                model_name, view_name, support_mm = parent_mask_ready_events.get_nowait()
            except queue.Empty:
                break
            existing = parent_mask_support_by_model[str(model_name)].get(str(view_name))
            if existing is not None and existing is not support_mm:
                raise RuntimeError(
                    f'Parent mask support {model_name}/{view_name} was published more than once'
                )
            parent_mask_support_by_model[str(model_name)][str(view_name)] = support_mm
            published = True
        if published:
            _flush_ready_postprocessed_tiles()
            # Parent-ready publication changes tile dispatch priority immediately. Refill
            # idle worker queues now rather than waiting for the much later B/final-parent
            # completion event.
            if gpu_worker_pending_task_ids:
                _dispatch_inference_windows()


    def _submit_prediction_accumulation_join(handle: PredictionAccumulationHandle, context: Dict[str, object]) -> None:
        fut = prediction_join_executor.submit(handle.wait)
        prediction_accumulation_futures[fut] = dict(context)

    def _drain_completed_prediction_accumulation_futures() -> None:
        for fut in list(prediction_accumulation_futures.keys()):
            if not fut.done():
                continue
            context = prediction_accumulation_futures.pop(fut)
            pred_stats = fut.result()
            kind = str(context.get('kind', ''))

            if kind == 'fullframe':
                model_name = str(context['model_name'])
                view = context['view']
                assert isinstance(view, ViewInfo)
                yolo_obj = context.get('yolo')
                if offload_between_jobs_enabled() and yolo_obj is not None:
                    offload_yolo_from_gpu(yolo_obj)
                view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))
                remaining_key = (model_name, view.name)
                fullframe_remaining[remaining_key] = int(fullframe_remaining.get(remaining_key, 0)) - 1
                if int(fullframe_remaining.get(remaining_key, 0)) == 0:
                    _submit_view_prepare(model_name, view)
                continue

            if kind == 'tile':
                model_name = str(context['model_name'])
                view = context['view']
                tile_job = context['tile_job']
                assert isinstance(view, ViewInfo)
                assert isinstance(tile_job, DenseTileJob)
                tile_mask_mm = context['tile_mask_mm']
                tile_conf_mm = context.get('tile_conf_mm')
                tile_mask_path = Path(context['tile_mask_path'])
                tile_conf_path_obj = context.get('tile_conf_path')
                tile_conf_path = Path(tile_conf_path_obj) if tile_conf_path_obj is not None else None
                ready_key = (str(model_name), str(view.name), str(tile_job.tile_id))
                yolo_obj = context.get('yolo')
                if offload_between_jobs_enabled() and yolo_obj is not None:
                    offload_yolo_from_gpu(yolo_obj)
                tile_inference_done.add(ready_key)
                view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                if int(pred_stats.get('frames_with_predictions', 0)) <= 0:
                    close_memmap_array(tile_mask_mm)
                    close_memmap_array(tile_conf_mm)
                    if not keep_temp_artifacts:
                        try:
                            tile_mask_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        if tile_conf_path is not None:
                            try:
                                tile_conf_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                    _mark_tile_complete(
                        str(model_name), str(view.name),
                        str(tile_job.config_id), str(tile_job.tile_id),
                    )
                    continue

                task = TilePostprocessTask(
                    model_name=str(model_name),
                    view_name=str(view.name),
                    aug_id=str(view.tta_aug_id),
                    angle_deg=float(view.tta_angle_deg),
                    config_id=str(tile_job.config_id),
                    tile_id=str(tile_job.tile_id),
                    parent_crop=tuple(int(v) for v in tile_job.parent_crop),
                    tile_mask_mm=tile_mask_mm,
                    tile_confmap_mm=tile_conf_mm,
                    tile_mask_path=tile_mask_path,
                    tile_confmap_path=tile_conf_path,
                    precleaned_slice_cleanup=bool(angle_variant_streaming_cleanup_active),
                    processing_shape=tuple(int(v) for v in np.asarray(tile_mask_mm).shape),
                    threshold_plane_shape=tuple(int(v) for v in context['threshold_plane_shape']),
                )
                tile_fut = tile_dense_retirement_executor.submit(
                    postprocess_tile_volume_after_inference,
                    task,
                    view=view,
                    min_conf=float(args.min_conf),
                    min_radius=float(args.min_radius),
                    keep_temp=bool(keep_temp_artifacts),
                    slice_workers=int(tile_dense_retirement_slice_workers),
                    sparse_retire_dir=temp_dir,
                )
                tile_cleanup_futures[tile_fut] = (str(model_name), str(view.name), str(tile_job.config_id), str(tile_job.tile_id))
                continue

            raise RuntimeError(f'Unknown prediction accumulation kind: {kind!r}')

    def _drain_completed_background_futures() -> None:
        direct_union_capacity_released = False
        _drain_parent_mask_ready_events()
        for fut in list(view_processing_futures.keys()):
            if not fut.done():
                continue
            result = fut.result()
            del view_processing_futures[fut]
            completed_view_key = (str(result.model_name), str(result.view_name))
            retire_completed_non_tiled_view = bool(
                component_ref_dense_retirement_active
                and int(tile_expected_by_parent.get(completed_view_key, 0)) <= 0
            )
            if result.native_support_mm is not None and not retire_completed_non_tiled_view:
                native_view_support_by_model[result.model_name][result.view_name] = result.native_support_mm
            if result.parent_mask_support_mm is not None:
                existing_parent_mask = parent_mask_support_by_model[result.model_name].get(result.view_name)
                if existing_parent_mask is None:
                    parent_mask_support_by_model[result.model_name][result.view_name] = result.parent_mask_support_mm
                elif existing_parent_mask is not result.parent_mask_support_mm:
                    raise RuntimeError(
                        f'Parent mask support identity changed for {result.model_name}/{result.view_name}'
                    )
            if result.parent_bridge_support_mm is not None:
                parent_bridge_support_by_model[result.model_name][result.view_name] = result.parent_bridge_support_mm
            parent_bridge_ready.add((str(result.model_name), str(result.view_name)))
            _flush_ready_residual_tiles()
            interpolation_stats.extend(result.interpolation_stats)
            nrrd_layer_refs.extend(result.nrrd_layers)
            lease = direct_union_backing_leases.get(completed_view_key)
            if lease is not None:
                if completed_view_key not in direct_union_postprocess_views:
                    raise RuntimeError(
                        f'direct-union backing {completed_view_key} completed without a postprocess lease'
                    )
                if not bool(component_ref_dense_retirement_active):
                    direct_union_backing_leases.pop(completed_view_key, None)
                    lease.release('postprocess')
                    direct_union_postprocess_views.remove(completed_view_key)
                    direct_union_postprocess_bytes.pop(completed_view_key, None)
                    direct_union_capacity_released = True

            view_info = view_infos_by_name[result.view_name]
            if result.final_view_volume_mm is not None and not retire_completed_non_tiled_view:
                if view_info.family == 'radial':
                    radial_native_output_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
                elif is_tilted_view(view_info):
                    tilted_native_output_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
                else:
                    view_volumes_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
            if retire_completed_non_tiled_view:
                _retire_parent_dense_view(
                    str(result.model_name),
                    str(result.view_name),
                    extra_arrays=(result.native_support_mm, result.final_view_volume_mm),
                    reason='non-tiled terminal component refs are complete',
                )
            # Parent YOLO and bridge supports are now immutable and same-angle. Release
            # any cleaned tiles that finished inference first, then check whether all original
            # tiles have completed their two-stage component gates.
            _maybe_submit_tile_consolidations_for_parent(
                str(result.model_name), str(result.view_name),
            )

        for fut in list(tile_cleanup_futures.keys()):
            if not fut.done():
                continue
            ready_key = tile_cleanup_futures.pop(fut)
            result = fut.result()
            if result is None:
                _release_tile_dense_result_for_key(
                    str(ready_key[0]), str(ready_key[1]), str(ready_key[3]),
                    reason='tile empty after cleanup',
                )
                _mark_tile_complete(
                    str(ready_key[0]), str(ready_key[1]),
                    str(ready_key[2]), str(ready_key[3]),
                )
                continue
            if isinstance(result, DeferredTilePostprocessResult):
                # v16.4.3: CTILE publication is the dense-result retirement boundary.
                # Release the parent-owned memfd/path mapping before interpolation or either
                # gate can queue behind unrelated CPU work.
                _release_tile_dense_result_for_key(
                    str(ready_key[0]), str(ready_key[1]), str(ready_key[3]),
                    reason='immediate CTILE retirement after tile cleanup',
                )
            _submit_tile_parent_gate(result)

        _flush_ready_postprocessed_tiles()

        for fut in list(tile_parent_gate_futures.keys()):
            if not fut.done():
                continue
            model_name, view_name, config_id, tile_id = tile_parent_gate_futures.pop(fut)
            parent_gate_result = fut.result()
            if parent_gate_result.residual_result is None:
                _release_tile_dense_result_for_key(
                    str(model_name), str(view_name), str(tile_id),
                    reason='parent gate consumed dense tile result',
                )
                _mark_tile_complete(
                    str(model_name), str(view_name), str(config_id), str(tile_id),
                )
            else:
                _submit_tile_bridge_gate(parent_gate_result.residual_result)

        _flush_ready_residual_tiles()

        for fut in list(tile_bridge_gate_futures.keys()):
            if not fut.done():
                continue
            model_name, view_name, config_id, tile_id = tile_bridge_gate_futures.pop(fut)
            fut.result()
            _release_tile_dense_result_for_key(
                str(model_name), str(view_name), str(tile_id),
                reason='bridge gate consumed dense tile residual',
            )
            _mark_tile_complete(
                str(model_name), str(view_name), str(config_id), str(tile_id),
            )

        for fut in list(tile_consolidation_futures.keys()):
            if not fut.done():
                continue
            set_key = tile_consolidation_futures.pop(fut)
            parent_key = (str(set_key[0]), str(set_key[1]))
            result = fut.result()
            interpolation_stats.extend(result.interpolation_stats)
            nrrd_layer_refs.extend(result.nrrd_layers)

            # The interpolation backend may rebind the consolidated accumulator to a
            # fresh memmap containing bridge voxels. Re-point the registry used for
            # keep-temp archiving, then release the superseded pre-interpolation array.
            final_acc = result.final_accumulator_mm
            stale_acc = tile_accumulator_by_set.get(set_key)
            if final_acc is not None and stale_acc is not None and stale_acc is not final_acc:
                tile_accumulator_by_set[set_key] = final_acc
                final_acc_path = _memmap_backing_path(final_acc)
                if final_acc_path is not None:
                    tile_accumulator_paths[set_key] = Path(final_acc_path)
                else:
                    tile_accumulator_paths.pop(set_key, None)
                stale_acc_path = _memmap_backing_path(stale_acc)
                try:
                    close_memmap_array_without_flush(stale_acc)
                except Exception:
                    pass
                if stale_acc_path is not None and not bool(keep_temp_artifacts):
                    try:
                        Path(stale_acc_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            # Once the consolidated tile volume has entered its parent destination and any
            # NRRD layers have been materialized, all tile accumulators can retire.
            for label, store in (
                ('consolidated gated tiles', tile_accumulator_by_set),
                ('tile components accepted by parent mask', tile_parent_mask_accumulator_by_set),
                ('tile components accepted by parent bridge', tile_parent_bridge_accumulator_by_set),
            ):
                acc = store.pop(set_key, None)
                if acc is not None:
                    archive_or_delete_binary_volume_storage(
                        acc,
                        keep_temp=bool(keep_temp_artifacts),
                        workers=int(tile_slice_postprocess_workers),
                        desc=f'{label} {set_key[0]}/{set_key[1]}/{set_key[2]}',
                    )
                    if label == 'consolidated gated tiles':
                        tile_accumulator_paths.pop(set_key, None)

            tile_consolidation_completed.add(set_key)
            _maybe_finalize_tile_parent(str(parent_key[0]), str(parent_key[1]))

        for fut in list(tile_parent_finalization_futures.keys()):
            if not fut.done():
                continue
            parent_key = tile_parent_finalization_futures.pop(fut)
            result = fut.result()
            interpolation_stats.extend(result.interpolation_stats)
            nrrd_layer_refs.extend(result.nrrd_layers)
            _retire_parent_dense_view(
                str(parent_key[0]),
                str(parent_key[1]),
                reason='full-frame and all configured tile-set terminal refs are complete',
            )

        output_manager.reap_completed()
        # Allocation admission can block every otherwise-idle worker while the active
        # view window is full. Completing cvol materialization releases that slot, so
        # immediately refill the worker queues instead of waiting for an unrelated event.
        if direct_union_capacity_released and gpu_worker_pending_task_ids:
            _dispatch_inference_windows()

    last_scheduler_wait_log = 0.0

    def _log_scheduler_wait_state(force: bool = False) -> None:
        nonlocal last_scheduler_wait_log
        now = time.time()
        interval = max(5.0, _env_float('YOLO_TTA_SCHEDULER_STATUS_INTERVAL_SEC', 30.0))
        if not bool(force) and (now - float(last_scheduler_wait_log)) < float(interval):
            return
        last_scheduler_wait_log = float(now)
        waiting_parent_tiles = sum(len(v) for v in postprocessed_tiles_waiting_by_parent.values())
        waiting_bridge_tiles = sum(len(v) for v in residual_tiles_waiting_by_parent.values())
        gpu_stage_state = _MAIN_PROCESS_GPU_STAGE_COORDINATOR.snapshot()
        print(
            'Scheduler wait: no inference-ready in-memory volume; '
            f'gpu_inference_inflight={gpu_stage_state.get("inference_inflight", {})}, '
            f'gpu_stage_leases={gpu_stage_state.get("stage_leases", {})}, '
            f'inference_priority={bool(gpu_stage_state.get("inference_priority_active", False))}, '
            f'pending_volume_builds={len(pending_prediction_volume_futures)}, '
            f'queued_build_jobs={len(pending_prediction_build_jobs)}, '
            f'prediction_accumulation={len(prediction_accumulation_futures)}, '
            f'parent_postprocess={len(view_processing_futures)}, '
            f'direct_union_inference={len(direct_union_inference_views)}/'
            f'{sum(direct_union_inference_bytes.values()) / GIB:.1f}GiB, '
            f'direct_union_postprocess={len(direct_union_postprocess_views)}/'
            f'{sum(direct_union_postprocess_bytes.values()) / GIB:.1f}GiB, '
            f'tile_dense_results={len(gpu_worker_tile_dense_result_reservations)}/'
            f'{gpu_worker_tile_dense_result_task_limit} task(s), '
            f'{gpu_worker_tile_dense_result_bytes_reserved / GIB:.1f}/'
            f'{gpu_worker_tile_dense_result_limit / GIB:.1f}GiB, '
            f'tile_cleanup={len(tile_cleanup_futures)}, '
            f'tile_parent_gate={len(tile_parent_gate_futures)}, '
            f'tile_bridge_gate={len(tile_bridge_gate_futures)}, '
            f'tile_consolidation={len(tile_consolidation_futures)}, '
            f'tile_parent_finalization={len(tile_parent_finalization_futures)}, '
            f'waiting_tiles_for_parent={waiting_parent_tiles}, '
            f'waiting_residuals_for_bridge={waiting_bridge_tiles}, '
            f'ready_fullframe={len(ready_fullframe)}, ready_tiles={len(ready_tile_infer)}'
        )

    #
    # Process-per-GPU scheduler. This path is active for every CUDA run, including one GPU.
    #
    gpu_worker_processes: List[object] = []
    cpu_worker_processes: List[object] = []
    _reset_main_process_gpu_stage_coordinator()
    gpu_task_queues: Dict[int, object] = {}
    cpu_task_queues: Dict[int, object] = {}
    cpu_worker_dispatched_by_id: Dict[int, int] = {}
    cpu_worker_results_by_id: Dict[int, int] = {}
    cpu_worker_seconds_per_frame_ewma: Dict[Tuple[object, ...], float] = {}
    cpu_worker_predicted_load_by_id: Dict[int, float] = {}
    cpu_worker_task_predicted_seconds_by_id: Dict[int, float] = {}
    cpu_worker_ready_details_by_id: Dict[int, Dict[str, object]] = {}
    cpu_worker_dispatch_cursor = 0
    gpu_result_queue: object = None
    gpu_worker_tasks_by_id: Dict[int, Dict[str, object]] = {}
    gpu_worker_results_collected = 0
    gpu_worker_total_tasks = 0
    gpu_worker_dispatched_tasks = 0
    gpu_worker_dispatched_by_id: Dict[int, int] = {}
    gpu_worker_results_by_id: Dict[int, int] = {}
    # Compute credits are released as soon as render/TRT/post kernels have handed the
    # task to an asynchronous retirement lane. Final result publication is tracked
    # independently so D2H/cvol bookkeeping cannot starve the next GPU lease.
    gpu_worker_compute_completed_by_id: Dict[int, int] = {}
    gpu_worker_compute_released_task_ids: set[int] = set()
    gpu_worker_seconds_per_frame_ewma: Dict[Tuple[object, ...], float] = {}
    gpu_worker_predicted_load_by_id: Dict[int, float] = {}
    gpu_worker_task_predicted_seconds_by_id: Dict[int, float] = {}
    gpu_worker_dispatch_cursor = 0
    gpu_worker_next_dynamic_task_id = 0
    gpu_worker_pending_task_ids: deque = deque()
    gpu_worker_result_dir = temp_dir / 'gpu_worker_results'
    if not bool(keep_temp_artifacts) and gpu_worker_result_dir.exists():
        # Worker-result files are never resumable inputs. Remove leftovers from an interrupted
        # older run before admission starts, otherwise stale v16.4.0/16.4.1 tile files can make
        # a correctly bounded v16.4.3 run appear to retain terabytes it did not create.
        release_memfd_owners_under(gpu_worker_result_dir)
        shutil.rmtree(gpu_worker_result_dir, ignore_errors=True)
        if gpu_worker_result_dir.exists():
            print(
                f'Warning: stale inference-worker result scratch could not be fully removed: '
                f'{gpu_worker_result_dir}'
            )
        else:
            print(f'Purged stale inference-worker result scratch before this run: {gpu_worker_result_dir}')
    # v16.4.3: every tile needs one independent dense crop only through GPU publication and
    # cleanup. Those files must not form a producer/consumer queue behind interpolation or
    # component gates. Reserve their logical uint8 bytes at dispatch and release the credit
    # immediately when the dedicated cleanup pool publishes a crop-local CTILE store.
    gpu_worker_tile_dense_result_limit = int(tile_dense_worker_result_limit_bytes())
    gpu_worker_tile_dense_result_task_limit = int(tile_dense_worker_result_limit_tasks())
    gpu_worker_tile_dense_result_memory_safe_limit: Optional[int] = None
    if scratch_dir_is_memory_backed():
        # A pathname fallback on tmpfs consumes the same cgroup/anonymous-memory budget as
        # memfd. Keep the live dense cap below current headroom after a fixed safety reserve;
        # unlike the old max(1 GiB, ...) floor, zero headroom must reject every positive tile
        # before dispatch instead of knowingly risking an uncatchable cgroup OOM kill.
        safe_headroom = max(0, int(available_anon_work_bytes()) - 16 * GIB)
        explicit_anon_cap = int(workspace_anon_cap_bytes())
        if int(explicit_anon_cap) > 0:
            safe_headroom = min(int(safe_headroom), int(explicit_anon_cap))
        gpu_worker_tile_dense_result_memory_safe_limit = int(safe_headroom)
        gpu_worker_tile_dense_result_limit = min(
            int(gpu_worker_tile_dense_result_limit),
            int(gpu_worker_tile_dense_result_memory_safe_limit),
        )
    gpu_worker_tile_dense_result_bytes_reserved = 0
    gpu_worker_tile_dense_result_memfd_bytes_reserved = 0
    gpu_worker_tile_dense_result_max_retention_seconds = 0.0
    gpu_worker_tile_dense_result_reservations: Dict[int, int] = {}
    gpu_worker_tile_dense_result_memfd_reservations: Dict[int, int] = {}
    gpu_worker_tile_dense_result_reserved_at: Dict[int, float] = {}
    gpu_worker_tile_dense_result_workspaces: Dict[
        int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    ] = {}
    gpu_worker_tile_task_id_by_key: Dict[Tuple[str, str, str], int] = {}
    gpu_inference_drained_at: Optional[float] = None
    gpu_inference_drain_announced = False
    # D1 source-space bitsets are deliberately view-owned. One worker may own exactly one
    # unfinished view at a time; completed views release this compute ownership before their
    # path-backed cvol publication finishes on the worker's CPU publication pool.
    d1_owner_by_parent: Dict[Tuple[str, str], int] = {}
    d1_active_parent_by_worker: Dict[int, Tuple[str, str]] = {}
    d1_layer_ref_by_parent: Dict[Tuple[str, str], NrrdLayerRef] = {}
    d1_view_shadow_path_by_parent: Dict[Tuple[str, str], Path] = {}
    # Hybrid full-frame views are committed as a whole. A bounded, ordered parent list is
    # reserved for sequential OpenVINO direct-union ownership; every other CPU-compatible
    # parent remains immediately available to CUDA D1. Only the active CPU parent may be
    # assisted by CUDA, and the next reserved parent can open after the prior one drains.
    hybrid_view_mode_by_parent: Dict[Tuple[str, str], str] = {}
    fullframe_task_ids_by_parent: Dict[Tuple[str, str], List[int]] = {}
    hybrid_cpu_reserved_parents: List[Tuple[str, str]] = []
    hybrid_cpu_reserved_parent_set: set[Tuple[str, str]] = set()
    hybrid_cpu_reservation_rank_by_parent: Dict[Tuple[str, str], int] = {}
    gpu_worker_cpu_assist_inflight_task_ids: set[int] = set()
    gpu_worker_cpu_assist_completed_task_ids: set[int] = set()
    hybrid_stealback_announced_parents: set[Tuple[str, str]] = set()
    hybrid_cpu_idle_reason_counts: Counter[str] = Counter()
    hybrid_cpu_idle_reason_last = ''
    hybrid_cpu_idle_active = False
    hybrid_cpu_idle_since: Optional[float] = None
    gpu_frames_completed_total = 0
    cpu_frames_completed_total = 0
    hybrid_gpu_frames_completed_total = 0
    hybrid_cpu_frames_completed_total = 0
    hybrid_view_frames_by_backend: Dict[Tuple[str, str], Counter[str]] = {}
    hybrid_view_tasks_by_backend: Dict[Tuple[str, str], Counter[str]] = {}
    gpu_worker_seed_task_count = 0

    def _tile_dense_result_task_bytes(task: Dict[str, object]) -> int:
        if str(task.get('kind', '')) != 'tile':
            return 0
        shape = tuple(int(v) for v in task.get('processing_shape', ()))
        if len(shape) != 3:
            return 0
        planes = 2 if task.get('result_conf_path') else 1
        return int(array_nbytes(shape, np.uint8)) * int(planes)

    def _tile_dense_result_task_admissible(task: Dict[str, object]) -> bool:
        if bool(keep_temp_artifacts) or str(task.get('kind', '')) != 'tile':
            return True
        task_id = int(task.get('task_id', -1))
        if task_id in gpu_worker_tile_dense_result_reservations:
            return True
        need = int(_tile_dense_result_task_bytes(task))
        if need <= 0:
            return True
        # Always admit one oversized tile when the live set is empty; otherwise a valid
        # geometry whose crop exceeds the configured budget could deadlock forever.
        if not gpu_worker_tile_dense_result_reservations:
            return True
        if len(gpu_worker_tile_dense_result_reservations) >= int(
            gpu_worker_tile_dense_result_task_limit
        ):
            return False
        return bool(
            int(gpu_worker_tile_dense_result_bytes_reserved) + int(need)
            <= int(gpu_worker_tile_dense_result_limit)
        )

    def _tile_parent_mask_ready_for_task(task: Dict[str, object]) -> bool:
        if str(task.get('kind', '')) != 'tile':
            return False
        view_obj = task.get('view')
        view_name = getattr(view_obj, 'name', None)
        if view_name is None:
            return False
        return str(view_name) in parent_mask_support_by_model.get(
            str(task.get('model_name', '')), {}
        )

    def _inference_storage_priority_rank(task: Dict[str, object]) -> int:
        """Prefer parent work, then immediately gateable tiles, then early tiles."""
        if str(task.get('kind', '')) != 'tile':
            return 0
        return 1 if _tile_parent_mask_ready_for_task(task) else 2

    def _reserve_tile_dense_result_task(task: Dict[str, object]) -> bool:
        nonlocal gpu_worker_tile_dense_result_bytes_reserved
        if bool(keep_temp_artifacts) or str(task.get('kind', '')) != 'tile':
            return False
        task_id = int(task.get('task_id', -1))
        if task_id < 0 or task_id in gpu_worker_tile_dense_result_reservations:
            return False
        if not _tile_dense_result_task_admissible(task):
            raise RuntimeError(
                f'tile dense-result admission raced its {gpu_worker_tile_dense_result_limit / GIB:.1f} GiB budget'
            )
        need = int(_tile_dense_result_task_bytes(task))
        gpu_worker_tile_dense_result_reservations[task_id] = int(need)
        gpu_worker_tile_dense_result_reserved_at[task_id] = float(time.monotonic())
        gpu_worker_tile_dense_result_bytes_reserved += int(need)
        runtime_telemetry().gauge(
            'tile.dense_worker_result_bytes_reserved',
            int(gpu_worker_tile_dense_result_bytes_reserved),
        )
        runtime_telemetry().gauge(
            'tile.dense_worker_result_tasks_reserved',
            int(len(gpu_worker_tile_dense_result_reservations)),
        )
        return True

    def _prepare_tile_dense_result_workspaces(task: Dict[str, object]) -> None:
        """Allocate one task's dense tile result in shared RAM when possible.

        The parent owns each mapping and transfers duplicate memfd descriptors to the selected
        CUDA worker. Pathname files remain a bounded fallback when cgroup/RAM headroom is too
        small. A logical memfd reservation ledger covers not-yet-faulted pages, preventing a
        burst of dispatches from all passing the same stale memory-headroom snapshot. Holding
        the parent mapping until sparse retirement also prevents asynchronous D2H publication
        from outliving its backing object.
        """
        nonlocal gpu_worker_tile_dense_result_memfd_bytes_reserved
        if bool(keep_temp_artifacts) or str(task.get('kind', '')) != 'tile':
            return
        task_id = int(task.get('task_id', -1))
        if task_id < 0:
            raise ValueError('tile result workspace requires a nonnegative task_id')
        if task_id in gpu_worker_tile_dense_result_workspaces:
            return
        shape = tuple(int(v) for v in task.get('processing_shape', ()))
        if len(shape) != 3:
            raise ValueError(f'tile task {task_id} has invalid processing_shape={shape}')
        task.setdefault('result_mask_fallback_path', str(task['result_mask_path']))
        task.setdefault('result_conf_fallback_path', task.get('result_conf_path'))
        original_mask_path = Path(str(task['result_mask_fallback_path']))
        original_conf_path = (
            Path(str(task['result_conf_fallback_path']))
            if task.get('result_conf_fallback_path') else None
        )
        mask_mm: Optional[np.ndarray] = None
        conf_mm: Optional[np.ndarray] = None
        total_need = int(_tile_dense_result_task_bytes(task))
        anon_cap = int(workspace_anon_cap_bytes())
        projected_memfd = int(gpu_worker_tile_dense_result_memfd_bytes_reserved) + int(total_need)
        prefer_shared_memfd = bool(
            memfd_workspace_enabled()
            and int(available_anon_work_bytes()) >= int(projected_memfd) + 16 * GIB
            and (int(anon_cap) <= 0 or int(projected_memfd) <= int(anon_cap))
        )
        try:
            mask_mm = allocate_workspace_array(
                shape=shape,
                dtype=np.uint8,
                path=original_mask_path,
                desc=f'GPU-worker tile result mask task {task_id}',
                prefer_memory=False,
                prefer_memfd=bool(prefer_shared_memfd),
                reserve_bytes=16 * GIB,
                initialize_zero=True,
            )
            mask_backing = _memmap_backing_path(mask_mm)
            if mask_backing is None:
                raise RuntimeError(f'tile task {task_id} mask workspace has no reopenable backing')
            task['result_mask_path'] = str(mask_backing)

            if original_conf_path is not None:
                conf_mm = allocate_workspace_array(
                    shape=shape,
                    dtype=np.uint8,
                    path=original_conf_path,
                    desc=f'GPU-worker tile result confidence task {task_id}',
                    prefer_memory=False,
                    prefer_memfd=bool(prefer_shared_memfd),
                    reserve_bytes=16 * GIB,
                    initialize_zero=True,
                )
                conf_backing = _memmap_backing_path(conf_mm)
                if conf_backing is None:
                    raise RuntimeError(
                        f'tile task {task_id} confidence workspace has no reopenable backing'
                    )
                task['result_conf_path'] = str(conf_backing)

            task['result_workspace_preallocated'] = True
            gpu_worker_tile_dense_result_workspaces[task_id] = (mask_mm, conf_mm)
            actual_memfd_bytes = sum(
                int(np.asarray(mm).nbytes)
                for mm in (mask_mm, conf_mm)
                if mm is not None and _memfd_owner_key_from_array(mm) is not None
            )
            if int(actual_memfd_bytes) > 0:
                gpu_worker_tile_dense_result_memfd_reservations[task_id] = int(actual_memfd_bytes)
                gpu_worker_tile_dense_result_memfd_bytes_reserved += int(actual_memfd_bytes)
                runtime_telemetry().add(
                    'tile.dense_worker_result_memfd_bytes', int(actual_memfd_bytes),
                )
                runtime_telemetry().gauge(
                    'tile.dense_worker_result_memfd_bytes_reserved',
                    int(gpu_worker_tile_dense_result_memfd_bytes_reserved),
                )
            path_bytes = max(0, int(total_need) - int(actual_memfd_bytes))
            if int(path_bytes) > 0:
                runtime_telemetry().add(
                    'tile.dense_worker_result_path_fallback_bytes', int(path_bytes),
                )
        except BaseException:
            task['result_mask_path'] = str(original_mask_path)
            task['result_conf_path'] = (
                str(original_conf_path) if original_conf_path is not None else None
            )
            task.pop('result_workspace_preallocated', None)
            for mm in (conf_mm, mask_mm):
                if mm is None:
                    continue
                backing = _memmap_backing_path(mm)
                is_memfd = _memfd_owner_key_from_array(mm) is not None
                try:
                    close_memmap_array_without_flush(mm)
                except Exception:
                    pass
                if not is_memfd and backing is not None:
                    try:
                        Path(backing).unlink(missing_ok=True)
                    except Exception:
                        pass
            raise

    def _release_tile_dense_result_task_id(
        task_id: int, *, reason: str = '', refill: bool = True,
    ) -> bool:
        nonlocal gpu_worker_tile_dense_result_bytes_reserved
        nonlocal gpu_worker_tile_dense_result_memfd_bytes_reserved
        nonlocal gpu_worker_tile_dense_result_max_retention_seconds
        task_id_i = int(task_id)
        released = gpu_worker_tile_dense_result_reservations.pop(task_id_i, None)
        released_memfd = gpu_worker_tile_dense_result_memfd_reservations.pop(task_id_i, None)
        reserved_at = gpu_worker_tile_dense_result_reserved_at.pop(task_id_i, None)
        workspaces = gpu_worker_tile_dense_result_workspaces.pop(task_id_i, None)
        task_obj = gpu_worker_tasks_by_id.get(task_id_i)
        cleanup_paths: set[Path] = set()
        if isinstance(task_obj, dict):
            for field_name in (
                'result_mask_path', 'result_conf_path',
                'result_mask_fallback_path', 'result_conf_fallback_path',
            ):
                raw_path = task_obj.get(field_name)
                if raw_path:
                    try:
                        cleanup_paths.add(Path(str(raw_path)))
                    except Exception:
                        pass
        if workspaces is not None:
            for mm in workspaces:
                if mm is None:
                    continue
                backing = _memmap_backing_path(mm)
                if backing is not None:
                    cleanup_paths.add(Path(backing))
                is_memfd = _memfd_owner_key_from_array(mm) is not None
                try:
                    close_memmap_array_without_flush(mm)
                except Exception:
                    pass
                if not is_memfd and not bool(keep_temp_artifacts) and backing is not None:
                    try:
                        Path(backing).unlink(missing_ok=True)
                    except Exception:
                        pass
        if not bool(keep_temp_artifacts):
            # Also remove a fallback pathname that may have been created before a memfd
            # handoff or survived a worker-side error. The guard keeps cleanup confined to
            # this run's non-resumable GPU result directory.
            for cleanup_path in cleanup_paths:
                try:
                    if (
                        cleanup_path.suffix.lower() == '.dat'
                        and _path_is_relative_to(cleanup_path, gpu_worker_result_dir)
                    ):
                        cleanup_path.unlink(missing_ok=True)
                except Exception:
                    pass
        if isinstance(task_obj, dict):
            if task_obj.get('result_mask_fallback_path') is not None:
                task_obj['result_mask_path'] = task_obj.get('result_mask_fallback_path')
            task_obj['result_conf_path'] = task_obj.get('result_conf_fallback_path')
            task_obj.pop('result_workspace_preallocated', None)
        if released_memfd is not None:
            gpu_worker_tile_dense_result_memfd_bytes_reserved = max(
                0,
                int(gpu_worker_tile_dense_result_memfd_bytes_reserved) - int(released_memfd),
            )
            runtime_telemetry().gauge(
                'tile.dense_worker_result_memfd_bytes_reserved',
                int(gpu_worker_tile_dense_result_memfd_bytes_reserved),
            )
        if (
            released is None
            and released_memfd is None
            and reserved_at is None
            and workspaces is None
        ):
            return False
        if reserved_at is not None:
            retention_seconds = max(0.0, float(time.monotonic()) - float(reserved_at))
            gpu_worker_tile_dense_result_max_retention_seconds = max(
                float(gpu_worker_tile_dense_result_max_retention_seconds),
                float(retention_seconds),
            )
            runtime_telemetry().add(
                'tile.dense_worker_result_retention_seconds_total',
                float(retention_seconds),
            )
            runtime_telemetry().add('tile.dense_worker_result_retirements', 1)
            runtime_telemetry().gauge(
                'tile.dense_worker_result_last_retention_seconds',
                float(retention_seconds),
            )
            runtime_telemetry().gauge(
                'tile.dense_worker_result_max_retention_seconds',
                float(gpu_worker_tile_dense_result_max_retention_seconds),
            )
            if reason:
                runtime_telemetry().add(
                    f'tile.dense_worker_result_retired_reason.'
                    f'{_sanitize_filesystem_token(reason)}',
                    1,
                )
            warn_seconds = float(tile_dense_worker_result_warn_seconds())
            if warn_seconds > 0.0 and retention_seconds >= warn_seconds:
                print(
                    f'Warning: dense tile worker result task {task_id_i} remained live for '
                    f'{retention_seconds:.1f}s before {reason or "retirement"}; '
                    'the backing has now been closed and deleted. '
                    'YOLO_TTA_TILE_DENSE_RESULT_WARN_SECONDS adjusts this diagnostic.'
                )
        if released is not None:
            gpu_worker_tile_dense_result_bytes_reserved = max(
                0, int(gpu_worker_tile_dense_result_bytes_reserved) - int(released),
            )
            runtime_telemetry().add('tile.dense_worker_result_bytes_retired', int(released))
            runtime_telemetry().gauge(
                'tile.dense_worker_result_bytes_reserved',
                int(gpu_worker_tile_dense_result_bytes_reserved),
            )
            runtime_telemetry().gauge(
                'tile.dense_worker_result_tasks_reserved',
                int(len(gpu_worker_tile_dense_result_reservations)),
            )
        if bool(refill) and gpu_worker_pending_task_ids:
            _dispatch_inference_windows()
        return True

    def _release_tile_dense_result_for_key(
        model_name_s: str, view_name_s: str, tile_id_s: str, *, reason: str = '',
    ) -> bool:
        key = (str(model_name_s), str(view_name_s), str(tile_id_s))
        task_id = gpu_worker_tile_task_id_by_key.get(key)
        if task_id is None:
            return False
        did_release = _release_tile_dense_result_task_id(
            int(task_id), reason=str(reason), refill=True,
        )
        if did_release:
            gpu_worker_tile_task_id_by_key.pop(key, None)
        return bool(did_release)

    def _gpu_worker_task_seconds(task: Dict[str, object]) -> float:
        view_obj = task.get('view')
        count = max(1, int(task.get('slice_count', 1)))
        key = gpu_worker_task_cost_key(task)
        sec_per_frame = gpu_worker_seconds_per_frame_ewma.get(key)
        if sec_per_frame is None:
            sec_per_frame = (
                gpu_worker_default_seconds_per_frame(view_obj)
                if isinstance(view_obj, ViewInfo) else 0.05
            )
        return max(1e-4, float(sec_per_frame) * float(count))

    def _update_gpu_worker_cost(task: Dict[str, object], stats: Dict[str, object]) -> None:
        elapsed = float(stats.get('worker_compute_seconds', 0.0) or 0.0)
        count = max(1, int(task.get('slice_count', 1)))
        units = max(1, int(count))
        if elapsed <= 0.0:
            return
        observed = max(1e-5, float(elapsed) / float(units))
        key = gpu_worker_task_cost_key(task)
        prior = gpu_worker_seconds_per_frame_ewma.get(key)
        alpha = min(0.8, max(0.05, _env_float('YOLO_TTA_GPU_WORKER_COST_EWMA_ALPHA', 0.30)))
        gpu_worker_seconds_per_frame_ewma[key] = (
            observed if prior is None else (1.0 - alpha) * float(prior) + alpha * observed
        )

    def _split_gpu_worker_task_to_runtime_target(task_id: int) -> int:
        """Repeatedly split the selected full-frame lease to the current measured target."""
        nonlocal gpu_worker_total_tasks, gpu_worker_next_dynamic_task_id
        current_id = int(task_id)
        task = gpu_worker_tasks_by_id[current_id]
        if str(task.get('kind', '')) != 'fullframe' or bool(task.get('disable_runtime_split', False)):
            return current_id
        min_slices = max(1, int(gpu_worker_min_lease_slices()))
        align = max(1, int(args.gpu_batch))
        while True:
            count = int(task.get('slice_count', 0))
            if count < 2 * min_slices:
                break
            view_obj = task.get('view')
            key = gpu_worker_task_cost_key(task)
            sec_per_frame = gpu_worker_seconds_per_frame_ewma.get(key)
            if sec_per_frame is None:
                sec_per_frame = (
                    gpu_worker_default_seconds_per_frame(view_obj)
                    if isinstance(view_obj, ViewInfo) else 0.05
                )
            target_count = int(round(gpu_worker_target_lease_seconds() / max(1e-5, float(sec_per_frame))))
            target_count = max(min_slices, min(gpu_worker_max_lease_slices(), target_count))
            target_count = max(align, (int(target_count) // align) * align)
            if count <= max(2 * min_slices - 1, int(math.ceil(target_count * 1.25))):
                break
            start = int(task.get('slice_start', 0))
            stop = int(start + count)
            midpoint = min(stop - min_slices, start + max(min_slices, target_count))
            midpoint = start + ((midpoint - start) // align) * align
            if midpoint <= start or stop - midpoint < min_slices:
                break
            child_id = int(gpu_worker_next_dynamic_task_id)
            gpu_worker_next_dynamic_task_id += 1
            child = dict(task)
            task['slice_count'] = int(midpoint - start)
            task['render_workers'] = max(1, min(int(task.get('render_workers', 1)), int(midpoint - start)))
            child['task_id'] = child_id
            child['slice_start'] = int(midpoint)
            child['slice_count'] = int(stop - midpoint)
            child['render_workers'] = max(1, min(int(child.get('render_workers', 1)), int(stop - midpoint)))
            child['runtime_split_parent_task_id'] = current_id
            task['runtime_split_child_task_id'] = child_id
            if str(child.get('result_mode', 'file')) != 'direct_union':
                for field_name in ('result_mask_path', 'result_conf_path'):
                    raw_path = child.get(field_name)
                    if raw_path:
                        path_obj = Path(str(raw_path))
                        child[field_name] = str(path_obj.with_name(f'{path_obj.name}.rt{child_id}'))
            gpu_worker_tasks_by_id[child_id] = child
            gpu_worker_pending_task_ids.append(child_id)
            gpu_worker_total_tasks += 1
            parent_key = _gpu_worker_fullframe_parent_key(task)
            if parent_key is not None:
                fullframe_remaining[parent_key] = int(fullframe_remaining.get(parent_key, 0)) + 1
                fullframe_task_ids_by_parent.setdefault(parent_key, []).append(int(child_id))
            runtime_telemetry().add('scheduler.runtime_lease_splits', 1)
            # The selected front lease is now target-sized; leave the remainder central so
            # C3 can place it on the least-loaded eligible worker/owner.
            break
        return current_id

    def _gpu_worker_fullframe_parent_key(task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        if str(task.get('kind', '')) != 'fullframe':
            return None
        view_obj = task.get('view')
        view_name = getattr(view_obj, 'name', None)
        if view_name is None:
            return None
        return (str(task.get('model_name', '')), str(view_name))

    def _hybrid_task_parent_key(
        task: Dict[str, object],
    ) -> Optional[Tuple[str, str]]:
        if not bool(task.get('hybrid_cpu_eligible_origin', False)):
            return None
        return _gpu_worker_fullframe_parent_key(task)

    def _hybrid_parent_state(parent: Optional[Tuple[str, str]]) -> str:
        if parent is None:
            return 'not_hybrid'
        return str(hybrid_view_mode_by_parent.get(parent, 'unclaimed'))

    def _active_cpu_shared_parent() -> Optional[Tuple[str, str]]:
        active = [
            parent for parent, mode in hybrid_view_mode_by_parent.items()
            if str(mode) == 'direct_union'
            and int(fullframe_remaining.get(parent, 0)) > 0
        ]
        if len(active) > 1:
            raise RuntimeError(
                f'hybrid scheduler opened more than one CPU direct-union view: {active}'
            )
        return active[0] if active else None

    def _hybrid_parent_is_cpu_reserved(
        parent: Optional[Tuple[str, str]],
    ) -> bool:
        return bool(parent is not None and parent in hybrid_cpu_reserved_parent_set)

    def _next_cpu_reserved_parent() -> Optional[Tuple[str, str]]:
        """Return the first unfinished reserved parent that OpenVINO may still own."""
        active = _active_cpu_shared_parent()
        if active is not None:
            return active
        for parent in hybrid_cpu_reserved_parents:
            if int(fullframe_remaining.get(parent, 0)) <= 0:
                continue
            state = _hybrid_parent_state(parent)
            if state in {'unclaimed', 'direct_union'}:
                return parent
        return None

    def _hybrid_task_is_active_cpu_assist(
        task: Dict[str, object],
        active_parent: Optional[Tuple[str, str]] = None,
    ) -> bool:
        parent = _hybrid_task_parent_key(task)
        active = _active_cpu_shared_parent() if active_parent is None else active_parent
        return bool(
            parent is not None
            and active is not None
            and parent == active
            and str(task.get('result_mode', 'file')) == 'direct_union'
        )

    def _hybrid_task_is_gpu_mandatory(task: Dict[str, object]) -> bool:
        """True for ordinary GPU work and unreserved hybrid views assigned to CUDA D1."""
        parent = _hybrid_task_parent_key(task)
        if parent is None:
            return True
        state = _hybrid_parent_state(parent)
        if state == 'd1_owner':
            return True
        if state == 'direct_union':
            return False
        if state == 'unclaimed':
            return not _hybrid_parent_is_cpu_reserved(parent)
        return True

    def _set_hybrid_cpu_idle_reason(reason: str) -> None:
        """Record OpenVINO idle-state transitions without repeating one reason every lease."""
        nonlocal hybrid_cpu_idle_reason_last, hybrid_cpu_idle_active, hybrid_cpu_idle_since
        normalized = str(reason).strip()
        now = float(time.monotonic())
        if not normalized:
            if hybrid_cpu_idle_active:
                elapsed = max(0.0, now - float(hybrid_cpu_idle_since or now))
                runtime_telemetry().add('hybrid.cpu_idle_seconds', float(elapsed))
            hybrid_cpu_idle_active = False
            hybrid_cpu_idle_since = None
            return
        if hybrid_cpu_idle_active and normalized == hybrid_cpu_idle_reason_last:
            return
        if hybrid_cpu_idle_active:
            elapsed = max(0.0, now - float(hybrid_cpu_idle_since or now))
            runtime_telemetry().add('hybrid.cpu_idle_seconds', float(elapsed))
        hybrid_cpu_idle_active = True
        hybrid_cpu_idle_since = now
        if normalized != hybrid_cpu_idle_reason_last:
            hybrid_cpu_idle_reason_last = normalized
            hybrid_cpu_idle_reason_counts[normalized] += 1
            runtime_telemetry().add(
                f'hybrid.cpu_idle_reason.{_sanitize_filesystem_token(normalized)}', 1,
            )
            print(f'[hybrid] OpenVINO idle: {normalized}.')

    def _describe_hybrid_cpu_idle_reason() -> str:
        active = _active_cpu_shared_parent()
        pending_ids = [int(value) for value in gpu_worker_pending_task_ids]
        if active is not None:
            pending_active = [
                task_id for task_id in pending_ids
                if _hybrid_task_parent_key(gpu_worker_tasks_by_id[int(task_id)]) == active
                and bool(gpu_worker_tasks_by_id[int(task_id)].get('cpu_eligible', False))
                and str(gpu_worker_tasks_by_id[int(task_id)].get('result_mode', 'file')) == 'direct_union'
            ]
            if pending_active:
                return (
                    f'active reserved view {active[0]}/{active[1]} has no currently '
                    'admissible CPU lease'
                )
            cpu_inflight = sum(_cpu_worker_inflight(worker_id) for worker_id in cpu_task_queues)
            gpu_assist = sum(
                1 for task_id in gpu_worker_cpu_assist_inflight_task_ids
                if _hybrid_task_parent_key(gpu_worker_tasks_by_id.get(int(task_id), {})) == active
            )
            return (
                f'waiting for active reserved view {active[0]}/{active[1]} to drain '
                f'({cpu_inflight} CPU lease(s), {gpu_assist} CUDA assist lease(s) in flight)'
            )
        next_parent = _next_cpu_reserved_parent()
        if next_parent is not None:
            next_pending = [
                task_id for task_id in pending_ids
                if _hybrid_task_parent_key(gpu_worker_tasks_by_id[int(task_id)]) == next_parent
                and bool(gpu_worker_tasks_by_id[int(task_id)].get('cpu_eligible', False))
            ]
            if next_pending:
                return (
                    f'next reserved view {next_parent[0]}/{next_parent[1]} is waiting '
                    'for direct-union admission'
                )
            return (
                f'next reserved view {next_parent[0]}/{next_parent[1]} has no central '
                'CPU-claimable lease'
            )
        if any(
            bool(gpu_worker_tasks_by_id[int(task_id)].get('hybrid_cpu_eligible_origin', False))
            for task_id in pending_ids
        ):
            return (
                'CPU reservation sequence exhausted; remaining CPU-compatible views are '
                'assigned to CUDA D1'
            )
        if pending_ids:
            return 'no CPU-compatible task remains in the central inference queue'
        return 'central inference queue is empty'

    def _record_backend_frame_completion(
        task: Dict[str, object], backend: str,
    ) -> None:
        nonlocal gpu_frames_completed_total, cpu_frames_completed_total
        nonlocal hybrid_gpu_frames_completed_total, hybrid_cpu_frames_completed_total
        backend_name = str(backend).strip().lower()
        count = max(0, int(task.get('slice_count', 0)))
        if backend_name == 'cpu':
            cpu_frames_completed_total += int(count)
        else:
            gpu_frames_completed_total += int(count)
        parent = _hybrid_task_parent_key(task)
        if parent is None:
            return
        if backend_name == 'cpu':
            hybrid_cpu_frames_completed_total += int(count)
        else:
            hybrid_gpu_frames_completed_total += int(count)
        holder = hybrid_view_frames_by_backend.get(parent)
        if holder is None:
            holder = Counter()
            hybrid_view_frames_by_backend[parent] = holder
        holder[backend_name] += int(count)
        task_holder = hybrid_view_tasks_by_backend.get(parent)
        if task_holder is None:
            task_holder = Counter()
            hybrid_view_tasks_by_backend[parent] = task_holder
        task_holder[backend_name] += 1

    def _commit_hybrid_fullframe_mode(
        task: Dict[str, object], requested_mode: str, *, backend_label: str,
    ) -> str:
        """Commit every lease of one CPU-eligible view to one result contract."""
        requested = str(requested_mode)
        if requested not in {'d1_owner', 'direct_union'}:
            raise ValueError(f'invalid hybrid result mode {requested!r}')
        parent = _hybrid_task_parent_key(task)
        if parent is None:
            return str(task.get('result_mode', 'file'))
        existing = hybrid_view_mode_by_parent.get(parent)
        if existing is not None:
            if str(existing) != requested:
                raise RuntimeError(
                    f'hybrid view {parent} was already committed to {existing}, '
                    f'cannot recommit it to {requested}'
                )
            return str(existing)
        if str(task.get('result_mode', 'file')) != HYBRID_DEFERRED_RESULT_MODE:
            raise RuntimeError(
                f'hybrid view {parent} reached first claim with result_mode='
                f'{task.get("result_mode")!r}'
            )
        task_ids = list(fullframe_task_ids_by_parent.get(parent, ()))
        if not task_ids:
            raise RuntimeError(f'hybrid view {parent} has no indexed full-frame tasks')
        if requested == 'direct_union' and not _hybrid_parent_is_cpu_reserved(parent):
            raise RuntimeError(
                f'OpenVINO attempted to claim unreserved hybrid view {parent}; '
                'only the ordered CPU reservation sequence may open dense unions'
            )
        hybrid_view_mode_by_parent[parent] = requested
        changed = 0
        for indexed_task_id in task_ids:
            candidate = gpu_worker_tasks_by_id[int(indexed_task_id)]
            candidate_mode = str(candidate.get('result_mode', 'file'))
            if candidate_mode == HYBRID_DEFERRED_RESULT_MODE:
                candidate['result_mode'] = requested
                candidate['hybrid_committed_backend'] = str(backend_label)
                candidate['device_hole_fill'] = False
                if requested == 'd1_owner':
                    candidate['cpu_eligible'] = False
                changed += 1
            elif candidate_mode != requested:
                raise RuntimeError(
                    f'hybrid view {parent} contains mixed result contracts: '
                    f'{candidate_mode} vs {requested}'
                )
        runtime_telemetry().add(f'hybrid.view_commits.{requested}', 1)
        reservation_note = (
            f', CPU reservation #{hybrid_cpu_reservation_rank_by_parent[parent] + 1}'
            if parent in hybrid_cpu_reservation_rank_by_parent else
            ', unreserved CUDA view'
        )
        print(
            f'[hybrid] first claim committed {parent[0]}/{parent[1]} to {requested} '
            f'via {backend_label}{reservation_note}; {changed} lease descriptor(s) updated.'
        )
        return requested

    def _hybrid_gpu_selection_rank(task: Dict[str, object]) -> int:
        parent = _hybrid_task_parent_key(task)
        mode = str(task.get('result_mode', 'file'))
        if parent is not None and mode == 'direct_union':
            return 0
        if mode == 'd1_owner' and _d1_task_parent_key(task) in d1_owner_by_parent:
            return 1
        if parent is None:
            return 2
        if mode == 'd1_owner':
            return 3
        if mode == HYBRID_DEFERRED_RESULT_MODE and not _hybrid_parent_is_cpu_reserved(parent):
            return 4
        if mode == HYBRID_DEFERRED_RESULT_MODE:
            return 5
        return 6

    def _hybrid_gpu_stealback_quota(
        mandatory_gpu_pairs: Sequence[Tuple[int, int]],
        active_cpu_pairs: Sequence[Tuple[int, int]],
    ) -> int:
        """Return a proportional concurrent-task quota for the active CPU-owned view.

        Unopened reserved views are excluded from the CPU ETA. Unreserved hybrid views are
        mandatory CUDA D1 work. Only central leases from the one active direct-union view
        are assistable. The quota is expressed as live assist tasks, which maps much more
        closely to GPU-worker equivalents than multiplying by each worker's publication
        overlap depth.
        """
        active_parent = _active_cpu_shared_parent()
        if (
            not hybrid_gpu_stealback_enabled()
            or active_parent is None
            or not active_cpu_pairs
            or not cpu_task_queues
            or not gpu_task_queues
        ):
            return 0
        active_pairs = [
            (int(position), int(task_id))
            for position, task_id in active_cpu_pairs
            if _hybrid_task_parent_key(gpu_worker_tasks_by_id[int(task_id)]) == active_parent
            and str(gpu_worker_tasks_by_id[int(task_id)].get('result_mode', 'file')) == 'direct_union'
        ]
        if not active_pairs:
            return 0

        gpu_workers = max(1, len(gpu_task_queues))
        cpu_workers = max(1, len(cpu_task_queues))
        cpu_committed = float(sum(
            float(predicted)
            for task_id, predicted in cpu_worker_task_predicted_seconds_by_id.items()
            if _hybrid_task_parent_key(gpu_worker_tasks_by_id.get(int(task_id), {})) == active_parent
        ))
        cpu_pending = float(sum(
            _cpu_worker_task_seconds(gpu_worker_tasks_by_id[int(task_id)])
            for _position, task_id in active_pairs
        ))
        cpu_work = max(0.0, float(cpu_committed) + float(cpu_pending))
        cpu_eta = float(cpu_work) / float(cpu_workers)

        # Count the complete central mandatory backlog, not only tasks feasible on the
        # particular free worker subset used by this one dispatch iteration. Otherwise
        # owner-affined D1 work can disappear from the horizon and trigger premature assist.
        gpu_committed = float(sum(
            float(predicted)
            for task_id, predicted in gpu_worker_task_predicted_seconds_by_id.items()
            if _hybrid_task_is_gpu_mandatory(gpu_worker_tasks_by_id.get(int(task_id), {}))
        ))
        gpu_pending = float(sum(
            _gpu_worker_task_seconds(gpu_worker_tasks_by_id[int(task_id)])
            for task_id in list(gpu_worker_pending_task_ids)
            if bool(gpu_worker_tasks_by_id[int(task_id)].get('gpu_eligible', gpu_worker_process_active))
            and _hybrid_task_is_gpu_mandatory(gpu_worker_tasks_by_id[int(task_id)])
        ))
        gpu_mandatory_work = max(0.0, float(gpu_committed) + float(gpu_pending))
        gpu_horizon = float(gpu_mandatory_work) / float(gpu_workers)

        completed_cpu_samples = int(
            hybrid_view_tasks_by_backend.get(active_parent, Counter()).get('cpu', 0)
        )
        minimum_samples = int(hybrid_gpu_stealback_min_cpu_samples())
        if gpu_mandatory_work > 0.0 and completed_cpu_samples < minimum_samples:
            runtime_telemetry().gauge(
                'hybrid.gpu_assist_waiting_for_cpu_samples',
                {
                    'parent': f'{active_parent[0]}/{active_parent[1]}',
                    'completed': int(completed_cpu_samples),
                    'required': int(minimum_samples),
                },
            )
            return 0

        ratio = float(hybrid_gpu_stealback_eta_ratio())
        threshold = (
            float(gpu_horizon) * float(ratio)
            + float(hybrid_gpu_stealback_min_lead_seconds())
        )
        runtime_telemetry().gauge('hybrid.active_cpu_eta_seconds', float(cpu_eta))
        runtime_telemetry().gauge('hybrid.mandatory_gpu_eta_seconds', float(gpu_horizon))
        runtime_telemetry().gauge('hybrid.active_cpu_pending_seconds', float(cpu_pending))
        runtime_telemetry().gauge('hybrid.active_cpu_committed_seconds', float(cpu_committed))
        runtime_telemetry().gauge('hybrid.active_cpu_samples', int(completed_cpu_samples))
        if cpu_eta <= threshold or cpu_pending <= 0.0:
            runtime_telemetry().gauge('hybrid.gpu_assist_task_quota', 0)
            return 0

        # Transfer only the still-central CPU seconds needed to make the active view finish
        # near the mandatory-GPU horizon. Already-running OpenVINO leases are non-preemptive.
        target_cpu_eta = max(0.0, float(threshold))
        excess_cpu_seconds = min(
            float(cpu_pending),
            max(0.0, float(cpu_work) - float(target_cpu_eta) * float(cpu_workers)),
        )
        if excess_cpu_seconds <= 0.0:
            return 0
        active_gpu_seconds = float(sum(
            _gpu_worker_task_seconds(gpu_worker_tasks_by_id[int(task_id)])
            for _position, task_id in active_pairs
        ))
        gpu_per_cpu_second = (
            float(active_gpu_seconds) / float(cpu_pending)
            if cpu_pending > 0.0 else 1.0
        )
        gpu_seconds_needed = max(
            0.0, float(excess_cpu_seconds) * max(1e-6, float(gpu_per_cpu_second)),
        )

        max_fraction = float(hybrid_gpu_stealback_max_fraction())
        if max_fraction <= 0.0:
            return 0
        # The fraction cap protects mandatory GPU throughput during early assistance. Once
        # that backlog is empty, every otherwise-idle GPU may drain the active CPU view.
        max_assist_tasks = (
            int(gpu_workers)
            if gpu_mandatory_work <= 0.0 else
            max(1, min(
                int(gpu_workers),
                int(math.floor(float(gpu_workers) * float(max_fraction) + 1e-9)),
            ))
        )
        capacity_per_gpu = max(
            float(gpu_horizon),
            float(gpu_worker_target_lease_seconds()),
            1e-3,
        )
        required_gpu_workers = max(
            1,
            int(math.ceil(float(gpu_seconds_needed) / float(capacity_per_gpu))),
        )
        quota = max(1, min(int(max_assist_tasks), int(required_gpu_workers)))
        runtime_telemetry().gauge('hybrid.gpu_assist_task_quota', int(quota))
        runtime_telemetry().gauge('hybrid.gpu_assist_seconds_needed', float(gpu_seconds_needed))
        if active_parent not in hybrid_stealback_announced_parents:
            hybrid_stealback_announced_parents.add(active_parent)
            print(
                'v17.0.3 active-view ETA GPU assist enabled for '
                f'{active_parent[0]}/{active_parent[1]}: active CPU ETA={cpu_eta:.1f}s, '
                f'mandatory-GPU ETA={gpu_horizon:.1f}s, '
                f'estimated GPU assist={gpu_seconds_needed:.1f}s, '
                f'CUDA assist quota={quota}/{gpu_workers} concurrent task(s), '
                f'CPU samples={completed_cpu_samples}.'
            )
        return int(quota)

    def _d1_task_parent_key(task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        if not bool(v1613_d1_owner_active):
            return None
        if str(task.get('result_mode', 'file')) != 'd1_owner':
            return None
        return _gpu_worker_fullframe_parent_key(task)

    def _d1_feasible_workers(
        task: Dict[str, object], candidate_workers: Sequence[int],
    ) -> List[int]:
        workers = [int(value) for value in candidate_workers]
        if not bool(v1613_d1_owner_active):
            return workers
        parent = _d1_task_parent_key(task)
        if parent is None:
            return [
                worker for worker in workers
                if int(worker) not in d1_active_parent_by_worker
            ]
        owner = d1_owner_by_parent.get(parent)
        if owner is not None:
            return [int(owner)] if int(owner) in workers else []
        return [
            worker for worker in workers
            if int(worker) not in d1_active_parent_by_worker
        ]

    def _claim_d1_owner(task: Dict[str, object], worker_id: int) -> bool:
        parent = _d1_task_parent_key(task)
        if parent is None:
            return False
        worker = int(worker_id)
        owner = d1_owner_by_parent.get(parent)
        active = d1_active_parent_by_worker.get(worker)
        if owner is not None:
            if int(owner) != worker or active != parent:
                raise RuntimeError(
                    f'D1 owner registry mismatch for {parent}: owner={owner}, '
                    f'worker={worker}, active={active}'
                )
            return False
        if active is not None:
            raise RuntimeError(
                f'D1 worker {worker} cannot claim {parent}; it still owns {active}'
            )
        d1_owner_by_parent[parent] = worker
        d1_active_parent_by_worker[worker] = parent
        runtime_telemetry().add('d1.owner_claims', 1)
        return True

    def _release_d1_owner_if_complete(
        task: Dict[str, object], worker_id: int, stats: Dict[str, object],
    ) -> None:
        if not bool(stats.get('d1_view_complete', False)):
            return
        parent = _d1_task_parent_key(task)
        if parent is None:
            return
        worker = int(worker_id)
        owner = d1_owner_by_parent.get(parent)
        # A deferred publication sends compute_released first and the final result later.
        # The second notification is intentionally idempotent.
        if owner is None:
            return
        if int(owner) != worker or d1_active_parent_by_worker.get(worker) != parent:
            raise RuntimeError(
                f'D1 completion registry mismatch for {parent}: owner={owner}, '
                f'worker={worker}, active={d1_active_parent_by_worker.get(worker)}'
            )
        d1_owner_by_parent.pop(parent, None)
        d1_active_parent_by_worker.pop(worker, None)
        runtime_telemetry().add('d1.owner_releases', 1)

    def _split_one_gpu_worker_dispatch_tail(
        issue_slots: int,
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> bool:
        """Split one still-central full-frame lease only when issue reaches the tail."""
        nonlocal gpu_worker_total_tasks, gpu_worker_next_dynamic_task_id
        if bool(v1613_d1_owner_active):
            return False
        if int(gpu_device_count) <= 1 or int(issue_slots) <= 0:
            return False
        pending_ids = [int(v) for v in gpu_worker_pending_task_ids]
        # This is the actual central-queue tail, not a per-view construction-time guess:
        # every pending descriptor would otherwise be issued by this refill.
        if not pending_ids or len(pending_ids) > int(issue_slots):
            return False
        parent_pending_counts: Dict[Tuple[str, str], int] = {}
        for task_id in pending_ids:
            parent_key = _gpu_worker_fullframe_parent_key(gpu_worker_tasks_by_id[int(task_id)])
            if parent_key is not None:
                parent_pending_counts[parent_key] = int(parent_pending_counts.get(parent_key, 0)) + 1
        eligible: List[Tuple[int, int, int, int, int]] = []
        for position, task_id in enumerate(pending_ids):
            task = gpu_worker_tasks_by_id[int(task_id)]
            if str(task.get('kind', '')) != 'fullframe' or bool(task.get('tail_adapted', False)):
                continue
            midpoint = gpu_worker_tail_split_point(
                int(task.get('slice_start', 0)),
                int(task.get('slice_count', 0)),
                int(args.gpu_batch),
            )
            if midpoint is None:
                continue
            parent_key = _gpu_worker_fullframe_parent_key(task)
            remaining = int(fullframe_remaining.get(parent_key, 2 ** 31 - 1)) if parent_key is not None else 2 ** 31 - 1
            # Preserve the lease most likely to unlock parent postprocessing. In
            # particular, splitting preferred_parent here would turn its sole
            # pending lease into two and make _pop_gpu_worker_pending_task_id skip it.
            eligible.append((
                1 if parent_key == preferred_parent else 0,
                1 if parent_key is not None and int(parent_pending_counts.get(parent_key, 0)) == 1 else 0,
                -int(remaining), int(position), int(midpoint),
            ))
        if not eligible:
            return False
        _preferred, _sole_pending, _negative_remaining, position, midpoint = min(eligible)
        original_id = int(pending_ids[int(position)])
        original = gpu_worker_tasks_by_id[original_id]
        original_start = int(original.get('slice_start', 0))
        original_stop = int(original_start) + int(original.get('slice_count', 0))
        child_id = int(gpu_worker_next_dynamic_task_id)
        gpu_worker_next_dynamic_task_id += 1
        child = dict(original)
        original['slice_count'] = int(midpoint - original_start)
        original['render_workers'] = max(
            1, min(int(original.get('render_workers', 1)), int(midpoint - original_start)),
        )
        original['tail_adapted'] = True
        original['tail_child_task_id'] = int(child_id)
        child['task_id'] = int(child_id)
        child['slice_start'] = int(midpoint)
        child['slice_count'] = int(original_stop - midpoint)
        child['render_workers'] = max(
            1, min(int(child.get('render_workers', 1)), int(original_stop - midpoint)),
        )
        child['tail_adapted'] = True
        child['tail_parent_task_id'] = int(original_id)
        if str(child.get('result_mode', 'file')) != 'direct_union':
            for field_name in ('result_mask_path', 'result_conf_path'):
                raw_path = child.get(field_name)
                if raw_path:
                    path_obj = Path(str(raw_path))
                    child[field_name] = str(path_obj.with_name(
                        f'{path_obj.name}.tail{int(child_id)}'
                    ))
        gpu_worker_tasks_by_id[int(child_id)] = child
        pending_ids.insert(int(position) + 1, int(child_id))
        gpu_worker_pending_task_ids.clear()
        gpu_worker_pending_task_ids.extend(pending_ids)
        gpu_worker_total_tasks += 1
        parent_key = _gpu_worker_fullframe_parent_key(original)
        if parent_key is not None:
            fullframe_remaining[parent_key] = int(fullframe_remaining.get(parent_key, 0)) + 1
            fullframe_task_ids_by_parent.setdefault(parent_key, []).append(int(child_id))
        print(
            f'v13.3.18 (C11): dispatch tail split task {original_id} '
            f'[{original_start}:{original_stop}] -> [{original_start}:{midpoint}] + '
            f'[{midpoint}:{original_stop}] as task {child_id}.'
        )
        return True

    def _direct_union_task_key(task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        if str(task.get('kind', '')) != 'fullframe' or str(task.get('result_mode', 'file')) != 'direct_union':
            return None
        view_obj = task.get('view')
        if view_obj is None:
            return None
        return (str(task.get('model_name', '')), str(getattr(view_obj, 'name', '')))

    def _direct_union_task_bytes(task: Dict[str, object]) -> int:
        shape = tuple(int(v) for v in task.get('processing_shape', ()))
        if len(shape) != 3:
            view_obj = task['view']
            shape = view_processing_volume_shape(view_obj, int(task.get('out_size', args.imgsz)))
        dense_volume_count = 2 if float(args.min_conf) > 0.0 else 1
        if bool(dense_tiling_active):
            dense_volume_count += 1
            if bool(nrrd_layers_needed):
                dense_volume_count += 2
        return int(array_nbytes(shape, np.uint8)) * int(dense_volume_count)

    def _direct_union_task_admissible(task: Dict[str, object]) -> bool:
        key = _direct_union_task_key(task)
        if key is None or not direct_union_sparse_retirement_active:
            return True
        if key in direct_union_inference_views:
            lease = direct_union_backing_leases.get(key)
            if lease is None or lease.phase != 'inference':
                raise RuntimeError(f'direct-union inference registry is inconsistent for {key}')
            return True
        if key in direct_union_postprocess_views:
            # A task for a view whose final chunk already handed ownership to postprocess is
            # a scheduler lifecycle error; never write into a buffer now read by CPU/NRRD work.
            raise RuntimeError(f'inference task targeted postprocess-owned direct union {key}')
        if len(direct_union_inference_views) >= int(direct_union_inference_view_limit):
            return False
        need = int(_direct_union_task_bytes(task))
        inference_active = int(sum(direct_union_inference_bytes.values()))
        postprocess_active = int(sum(direct_union_postprocess_bytes.values()))
        total_active = int(inference_active + postprocess_active)
        inference_ok = bool(
            not direct_union_inference_views
            or int(inference_active) + int(need) <= int(direct_union_inference_byte_limit)
        )
        total_ok = bool(
            not direct_union_backing_leases
            or int(total_active) + int(need) <= int(direct_union_total_dense_byte_limit)
        )
        return bool(inference_ok and total_ok)

    def _activate_direct_union_task(task: Dict[str, object]) -> None:
        key = _direct_union_task_key(task)
        if key is None:
            return
        view_obj = task['view']
        _ensure_baseline_workspaces(str(key[0]), view_obj)
        task['result_mask_path'] = str(baseline_union_paths[key])
        conf_path = baseline_confmap_paths.get(key)
        task['result_conf_path'] = str(conf_path) if conf_path is not None else None

    def _pop_gpu_worker_pending_task_id(
        preferred_parent: Optional[Tuple[str, str]] = None,
        candidate_workers: Optional[Sequence[int]] = None,
    ) -> Optional[Tuple[int, List[int]]]:
        """Pick an admissible GPU task and the worker subset allowed to own it.

        Unreserved hybrid parents are ordinary mandatory CUDA D1 work. Future CPU-reserved
        parents remain protected. The one active direct-union parent enters the candidate
        pool only when its active-view ETA quota has an open CUDA-assist slot.
        """
        pending_ids = [int(v) for v in gpu_worker_pending_task_ids]
        candidates = [int(v) for v in (candidate_workers or tuple(gpu_task_queues))]
        feasible_by_id: Dict[int, List[int]] = {}
        eligible: List[Tuple[int, int]] = []
        for position, task_id in enumerate(pending_ids):
            task = gpu_worker_tasks_by_id[int(task_id)]
            if not bool(task.get('gpu_eligible', gpu_worker_process_active)):
                continue
            if not _direct_union_task_admissible(task):
                continue
            if not _tile_dense_result_task_admissible(task):
                continue
            feasible = _d1_feasible_workers(task, candidates)
            if not feasible:
                continue
            feasible_by_id[int(task_id)] = feasible
            eligible.append((int(position), int(task_id)))
        if not eligible:
            return None

        active_parent = _active_cpu_shared_parent()
        mandatory_gpu: List[Tuple[int, int]] = []
        active_cpu_assist: List[Tuple[int, int]] = []
        for pair in eligible:
            task = gpu_worker_tasks_by_id[int(pair[1])]
            if _hybrid_task_is_active_cpu_assist(task, active_parent):
                active_cpu_assist.append(pair)
            elif _hybrid_task_is_gpu_mandatory(task):
                mandatory_gpu.append(pair)

        assist_ids: set[int] = set()
        quota = _hybrid_gpu_stealback_quota(mandatory_gpu, active_cpu_assist)
        assist_slots_open = max(
            0,
            int(quota) - int(len(gpu_worker_cpu_assist_inflight_task_ids)),
        )
        if assist_slots_open > 0 and active_cpu_assist:
            selected_pool = list(active_cpu_assist)
            assist_ids = {int(task_id) for _position, task_id in active_cpu_assist}
        elif mandatory_gpu:
            selected_pool = list(mandatory_gpu)
        else:
            return None

        parent_pending_counts: Dict[Tuple[str, str], int] = {}
        for _position, task_id in selected_pool:
            parent_key = _gpu_worker_fullframe_parent_key(gpu_worker_tasks_by_id[int(task_id)])
            if parent_key is not None:
                parent_pending_counts[parent_key] = int(parent_pending_counts.get(parent_key, 0)) + 1
        unlock_candidates: List[Tuple[int, int, int, int, int]] = []
        for position, task_id in selected_pool:
            task = gpu_worker_tasks_by_id[int(task_id)]
            parent_key = _gpu_worker_fullframe_parent_key(task)
            if parent_key is None or int(parent_pending_counts.get(parent_key, 0)) != 1:
                continue
            unlock_candidates.append((
                _hybrid_gpu_selection_rank(task),
                0 if parent_key == preferred_parent else 1,
                int(fullframe_remaining.get(parent_key, 2 ** 31 - 1)),
                int(position),
                int(task_id),
            ))
        if unlock_candidates:
            _hybrid_rank, _preferred, _remaining, _position, selected_id = min(unlock_candidates)
        else:
            parent_seconds: Dict[Optional[Tuple[str, str]], float] = {}
            for _position_i, task_id_i in selected_pool:
                task_i = gpu_worker_tasks_by_id[int(task_id_i)]
                parent_i = _gpu_worker_fullframe_parent_key(task_i)
                parent_seconds[parent_i] = float(parent_seconds.get(parent_i, 0.0)) + _gpu_worker_task_seconds(task_i)
            selected_id = min(
                selected_pool,
                key=lambda pair: (
                    _hybrid_gpu_selection_rank(gpu_worker_tasks_by_id[int(pair[1])]),
                    0 if _d1_task_parent_key(gpu_worker_tasks_by_id[int(pair[1])]) in d1_owner_by_parent else 1,
                    0 if (_direct_union_task_key(gpu_worker_tasks_by_id[int(pair[1])]) in direct_union_inference_views) else 1,
                    _inference_storage_priority_rank(gpu_worker_tasks_by_id[int(pair[1])]),
                    -float(parent_seconds.get(_gpu_worker_fullframe_parent_key(gpu_worker_tasks_by_id[int(pair[1])]), 0.0)),
                    -float(_gpu_worker_task_seconds(gpu_worker_tasks_by_id[int(pair[1])])),
                    int(pair[0]),
                ),
            )[1]
        selected_task = gpu_worker_tasks_by_id[int(selected_id)]
        selected_task['hybrid_gpu_assist_dispatch'] = bool(int(selected_id) in assist_ids)
        gpu_worker_pending_task_ids.remove(int(selected_id))
        return int(selected_id), list(feasible_by_id[int(selected_id)])

    def _publish_gpu_worker_admissible_backlog() -> None:
        """Keep D1 from winning a device while dispatchable inference remains central."""
        admissible = False
        for pending_task_id in list(gpu_worker_pending_task_ids):
            pending_task = gpu_worker_tasks_by_id[int(pending_task_id)]
            gpu_policy_eligible = bool(
                _hybrid_task_is_gpu_mandatory(pending_task)
                or _hybrid_task_is_active_cpu_assist(pending_task)
            )
            if (
                gpu_policy_eligible
                and _direct_union_task_admissible(pending_task)
                and _tile_dense_result_task_admissible(pending_task)
            ):
                admissible = True
                break
        _set_main_process_gpu_pending_inference(bool(admissible))

    def _gpu_worker_inflight(worker_id: int) -> int:
        worker = int(worker_id)
        return max(
            0,
            int(gpu_worker_dispatched_by_id.get(worker, 0))
            - int(gpu_worker_compute_completed_by_id.get(worker, 0)),
        )

    def _refresh_gpu_aux_interpolation_leases() -> None:
        """Lease warm CUDA worker interpreters only after global inference drain."""
        aux_pool = gpu_worker_aux_interpolation_pool()
        worker_ids = sorted(int(worker_id) for worker_id in gpu_task_queues)
        if aux_pool is None or not worker_ids:
            return
        inference_tail_drained = bool(
            int(gpu_worker_total_tasks) > 0
            and int(gpu_worker_results_collected) >= int(gpu_worker_total_tasks)
        )
        if not inference_tail_drained:
            for worker_id in worker_ids:
                aux_pool.revoke_worker(worker_id)
            return
        for worker_id in worker_ids:
            if (
                _gpu_worker_inflight(worker_id) == 0
                and _main_process_gpu_stage_can_dispatch_inference(worker_id)
            ):
                # Feeder exclusivity ends at global drain; post-inference interpolation may
                # reclaim the worker's full inherited allocation.
                aux_pool.enable_worker(worker_id, allow_full_cpu_affinity=True)
            else:
                aux_pool.revoke_worker(worker_id)

    def _dispatch_gpu_worker_inference_window(
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Issue bounded, targeted worker leases with D1 owner affinity."""
        nonlocal gpu_worker_dispatched_tasks, gpu_worker_dispatch_cursor
        worker_ids = sorted(int(worker_id) for worker_id in gpu_task_queues)
        if not worker_ids:
            _set_main_process_gpu_pending_inference(False)
            return
        _publish_gpu_worker_admissible_backlog()
        per_gpu = (
            max(1, min(4, _env_int('YOLO_TTA_V1613_D1_DISPATCH_WINDOW_PER_GPU', 2)))
            if bool(v1613_d1_owner_active) else max(
                1, _env_int('YOLO_TTA_GPU_WORKER_DISPATCH_WINDOW_PER_GPU', 2),
            )
        )
        aux_pool = gpu_worker_aux_interpolation_pool()
        while gpu_worker_pending_task_ids:
            candidates: List[int] = []
            for worker_id in worker_ids:
                if _gpu_worker_inflight(worker_id) >= int(per_gpu):
                    continue
                if not _main_process_gpu_stage_can_dispatch_inference(worker_id):
                    continue
                if aux_pool is not None and not aux_pool.revoke_worker(worker_id):
                    continue
                candidates.append(int(worker_id))
            if not candidates:
                break
            issue_slots = sum(
                max(0, int(per_gpu) - _gpu_worker_inflight(worker_id))
                for worker_id in candidates
            )
            _split_one_gpu_worker_dispatch_tail(int(issue_slots), preferred_parent)
            selected = _pop_gpu_worker_pending_task_id(preferred_parent, candidates)
            if selected is None:
                break
            task_id, _precommit_feasible_workers = selected
            task_to_dispatch = gpu_worker_tasks_by_id[int(task_id)]
            cpu_assist_dispatch = bool(
                task_to_dispatch.get('hybrid_gpu_assist_dispatch', False)
            )
            if (
                bool(task_to_dispatch.get('hybrid_cpu_eligible_origin', False))
                and str(task_to_dispatch.get('result_mode', 'file')) == HYBRID_DEFERRED_RESULT_MODE
            ):
                _commit_hybrid_fullframe_mode(
                    task_to_dispatch, 'd1_owner', backend_label='CUDA',
                )
            task_id = _split_gpu_worker_task_to_runtime_target(int(task_id))
            task_to_dispatch = gpu_worker_tasks_by_id[int(task_id)]
            if str(task_to_dispatch.get('result_mode', 'file')) == HYBRID_DEFERRED_RESULT_MODE:
                raise RuntimeError('GPU dispatch retained an unresolved hybrid result contract')
            feasible_workers = _d1_feasible_workers(task_to_dispatch, candidates)
            if not feasible_workers:
                gpu_worker_pending_task_ids.appendleft(int(task_id))
                break
            start_rank = int(gpu_worker_dispatch_cursor) % len(worker_ids)
            position = {
                worker_id: (worker_ids.index(worker_id) - start_rank) % len(worker_ids)
                for worker_id in feasible_workers
            }
            predicted_seconds = float(_gpu_worker_task_seconds(task_to_dispatch))
            worker_id = min(
                feasible_workers,
                key=lambda value: (
                    float(gpu_worker_predicted_load_by_id.get(int(value), 0.0)) + predicted_seconds,
                    _gpu_worker_inflight(value),
                    position[value],
                ),
            )
            gpu_worker_dispatch_cursor = (worker_ids.index(worker_id) + 1) % len(worker_ids)
            if not _main_process_gpu_stage_begin_inference(worker_id):
                gpu_worker_pending_task_ids.appendleft(int(task_id))
                continue
            owner_claimed = False
            tile_storage_reserved = False
            try:
                owner_claimed = _claim_d1_owner(task_to_dispatch, int(worker_id))
                _activate_direct_union_task(task_to_dispatch)
                tile_storage_reserved = _reserve_tile_dense_result_task(task_to_dispatch)
                if tile_storage_reserved:
                    _prepare_tile_dense_result_workspaces(task_to_dispatch)
                dispatch_task = dict(task_to_dispatch)
                dispatch_task.pop('hybrid_gpu_assist_dispatch', None)
                _attach_memfd_transfers_to_task(dispatch_task)
                preflight_multiprocessing_payload(dispatch_task)
                gpu_task_queues[int(worker_id)].put(dispatch_task)
            except BaseException:
                if tile_storage_reserved:
                    _release_tile_dense_result_task_id(
                        int(task_id), reason='dispatch failure', refill=False,
                    )
                if owner_claimed:
                    parent = _d1_task_parent_key(task_to_dispatch)
                    if parent is not None:
                        d1_owner_by_parent.pop(parent, None)
                    d1_active_parent_by_worker.pop(int(worker_id), None)
                gpu_worker_pending_task_ids.appendleft(int(task_id))
                _main_process_gpu_stage_finish_inference(worker_id)
                raise
            task_to_dispatch.pop('hybrid_gpu_assist_dispatch', None)
            if cpu_assist_dispatch:
                task_to_dispatch['hybrid_gpu_assist_dispatched'] = True
                gpu_worker_cpu_assist_inflight_task_ids.add(int(task_id))
                runtime_telemetry().add('hybrid.gpu_assist_tasks_dispatched', 1)
                runtime_telemetry().add(
                    'hybrid.gpu_assist_frames_dispatched',
                    int(task_to_dispatch.get('slice_count', 0)),
                )
            gpu_worker_dispatched_tasks += 1
            gpu_worker_dispatched_by_id[int(worker_id)] = int(
                gpu_worker_dispatched_by_id.get(int(worker_id), 0)
            ) + 1
            gpu_worker_task_predicted_seconds_by_id[int(task_id)] = float(predicted_seconds)
            gpu_worker_predicted_load_by_id[int(worker_id)] = float(
                gpu_worker_predicted_load_by_id.get(int(worker_id), 0.0)
            ) + float(predicted_seconds)
        _publish_gpu_worker_admissible_backlog()
        _refresh_gpu_aux_interpolation_leases()

    def _cpu_worker_task_seconds(task: Dict[str, object]) -> float:
        view_obj = task.get('view')
        count = max(1, int(task.get('slice_count', 1)))
        key = ('cpu',) + tuple(gpu_worker_task_cost_key(task))
        sec_per_frame = cpu_worker_seconds_per_frame_ewma.get(key)
        if sec_per_frame is None:
            sec_per_frame = (
                cpu_worker_default_seconds_per_frame(view_obj)
                if isinstance(view_obj, ViewInfo) else 0.25
            )
        return max(1e-4, float(sec_per_frame) * float(count))

    def _update_cpu_worker_cost(task: Dict[str, object], stats: Dict[str, object]) -> None:
        elapsed = float(stats.get('worker_compute_seconds', 0.0) or 0.0)
        count = max(1, int(task.get('slice_count', 1)))
        if elapsed <= 0.0:
            return
        observed = max(1e-5, float(elapsed) / float(count))
        key = ('cpu',) + tuple(gpu_worker_task_cost_key(task))
        prior = cpu_worker_seconds_per_frame_ewma.get(key)
        alpha = min(0.8, max(0.05, _env_float('YOLO_TTA_CPU_WORKER_COST_EWMA_ALPHA', 0.30)))
        cpu_worker_seconds_per_frame_ewma[key] = (
            observed if prior is None else (1.0 - alpha) * float(prior) + alpha * observed
        )

    def _split_cpu_worker_task_to_runtime_target(task_id: int) -> int:
        """Split an oversized seed only when OpenVINO actually claims it."""
        nonlocal gpu_worker_total_tasks, gpu_worker_next_dynamic_task_id
        current_id = int(task_id)
        task = gpu_worker_tasks_by_id[current_id]
        if str(task.get('kind', '')) != 'fullframe' or bool(task.get('disable_runtime_split', False)):
            return current_id
        count = int(task.get('slice_count', 0))
        if count <= 1:
            return current_id
        view_obj = task.get('view')
        key = ('cpu',) + tuple(gpu_worker_task_cost_key(task))
        sec_per_frame = cpu_worker_seconds_per_frame_ewma.get(key)
        if sec_per_frame is None:
            sec_per_frame = (
                cpu_worker_default_seconds_per_frame(view_obj)
                if isinstance(view_obj, ViewInfo) else 0.25
            )
        align = max(1, int(args.cpu_batch))
        target_count = int(round(cpu_worker_target_lease_seconds() / max(1e-5, float(sec_per_frame))))
        target_count = max(
            cpu_worker_min_lease_slices(),
            min(cpu_worker_max_lease_slices(), int(target_count)),
        )
        target_count = max(align, int(math.ceil(float(target_count) / float(align))) * align)
        if count <= target_count:
            return current_id
        remainder = int(count - target_count)
        # Do not manufacture a tiny tail merely to hit the target exactly. The supplied
        # workload's 57- and 44-frame GPU seeds therefore remain intact for OpenVINO,
        # filling its 18/20 asynchronous request pools without recreating 7/10-slice tasks.
        min_useful_remainder = max(
            cpu_worker_min_lease_slices(),
            int(math.ceil(float(target_count) * 0.50)),
        )
        if remainder < int(min_useful_remainder):
            return current_id
        start = int(task.get('slice_start', 0))
        stop = int(start + count)
        split_at = int(start + target_count)
        child_id = int(gpu_worker_next_dynamic_task_id)
        gpu_worker_next_dynamic_task_id += 1
        child = dict(task)
        task['slice_count'] = int(split_at - start)
        task['render_workers'] = max(
            1, min(int(task.get('render_workers', 1)), int(split_at - start)),
        )
        child['task_id'] = int(child_id)
        child['slice_start'] = int(split_at)
        child['slice_count'] = int(stop - split_at)
        child['render_workers'] = max(
            1, min(int(child.get('render_workers', 1)), int(stop - split_at)),
        )
        child['cpu_runtime_split_parent_task_id'] = int(current_id)
        task['cpu_runtime_split_child_task_id'] = int(child_id)
        if str(child.get('result_mode', 'file')) != 'direct_union':
            for field_name in ('result_mask_path', 'result_conf_path'):
                raw_path = child.get(field_name)
                if raw_path:
                    path_obj = Path(str(raw_path))
                    child[field_name] = str(path_obj.with_name(f'{path_obj.name}.cpu{child_id}'))
        gpu_worker_tasks_by_id[int(child_id)] = child
        gpu_worker_pending_task_ids.append(int(child_id))
        gpu_worker_total_tasks += 1
        parent_key = _gpu_worker_fullframe_parent_key(task)
        if parent_key is not None:
            fullframe_remaining[parent_key] = int(fullframe_remaining.get(parent_key, 0)) + 1
            fullframe_task_ids_by_parent.setdefault(parent_key, []).append(int(child_id))
        runtime_telemetry().add('scheduler.cpu_claim_lease_splits', 1)
        return current_id

    def _cpu_worker_inflight(worker_id: int) -> int:
        worker = int(worker_id)
        return max(
            0,
            int(cpu_worker_dispatched_by_id.get(worker, 0))
            - int(cpu_worker_results_by_id.get(worker, 0)),
        )

    def _pop_cpu_worker_pending_task_id(
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> Optional[int]:
        """Select work from the active or next ordered CPU reservation only."""
        active_cpu_parent = _active_cpu_shared_parent()
        next_reserved_parent = _next_cpu_reserved_parent()
        reservation_policy_active = bool(hybrid_cpu_reserved_parents)
        eligible: List[Tuple[int, int]] = []
        for position, task_id in enumerate(list(gpu_worker_pending_task_ids)):
            task = gpu_worker_tasks_by_id[int(task_id)]
            if not bool(task.get('cpu_eligible', False)):
                continue
            if str(task.get('result_mode', 'file')) == 'd1_owner':
                continue
            hybrid_parent = _hybrid_task_parent_key(task)
            hybrid_state = _hybrid_parent_state(hybrid_parent)
            if hybrid_parent is not None:
                if hybrid_state == 'd1_owner':
                    continue
                if active_cpu_parent is not None:
                    if hybrid_parent != active_cpu_parent or hybrid_state != 'direct_union':
                        continue
                else:
                    if not reservation_policy_active or next_reserved_parent is None:
                        continue
                    if hybrid_parent != next_reserved_parent:
                        continue
                    if hybrid_state not in {'unclaimed', 'direct_union'}:
                        continue
            elif active_cpu_parent is not None or next_reserved_parent is not None:
                continue
            if not _direct_union_task_admissible(task):
                continue
            if not _tile_dense_result_task_admissible(task):
                continue
            eligible.append((int(position), int(task_id)))
        if not eligible:
            return None
        selected = min(
            eligible,
            key=lambda pair: (
                0 if _hybrid_task_parent_key(
                    gpu_worker_tasks_by_id[int(pair[1])]
                ) == active_cpu_parent and active_cpu_parent is not None else 1,
                hybrid_cpu_reservation_rank_by_parent.get(
                    _hybrid_task_parent_key(gpu_worker_tasks_by_id[int(pair[1])]),
                    2 ** 31 - 1,
                ),
                0 if _gpu_worker_fullframe_parent_key(
                    gpu_worker_tasks_by_id[int(pair[1])]
                ) == preferred_parent else 1,
                cpu_inference_task_priority(gpu_worker_tasks_by_id[int(pair[1])]),
                0 if _direct_union_task_key(
                    gpu_worker_tasks_by_id[int(pair[1])]
                ) in direct_union_inference_views else 1,
                _inference_storage_priority_rank(gpu_worker_tasks_by_id[int(pair[1])]),
                -int(gpu_worker_tasks_by_id[int(pair[1])].get('slice_count', 0)),
                int(pair[0]),
            ),
        )[1]
        gpu_worker_pending_task_ids.remove(int(selected))
        return int(selected)

    def _dispatch_cpu_worker_inference_window(
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Issue one claim-time-sized range per socket from the CPU-opened view."""
        nonlocal cpu_worker_dispatch_cursor
        worker_ids = sorted(int(worker_id) for worker_id in cpu_task_queues)
        if not worker_ids:
            return
        while gpu_worker_pending_task_ids:
            available = [
                worker_id for worker_id in worker_ids
                if _cpu_worker_inflight(worker_id) < 1
            ]
            if not available:
                _set_hybrid_cpu_idle_reason('')
                break
            task_id = _pop_cpu_worker_pending_task_id(preferred_parent)
            if task_id is None:
                _set_hybrid_cpu_idle_reason(_describe_hybrid_cpu_idle_reason())
                break
            task = gpu_worker_tasks_by_id[int(task_id)]
            if (
                bool(task.get('hybrid_cpu_eligible_origin', False))
                and str(task.get('result_mode', 'file')) == HYBRID_DEFERRED_RESULT_MODE
            ):
                _commit_hybrid_fullframe_mode(
                    task, 'direct_union', backend_label='OpenVINO',
                )
            task = gpu_worker_tasks_by_id[int(task_id)]
            if str(task.get('result_mode', 'file')) == HYBRID_DEFERRED_RESULT_MODE:
                raise RuntimeError('CPU dispatch retained an unresolved hybrid result contract')
            if not _direct_union_task_admissible(task):
                gpu_worker_pending_task_ids.appendleft(int(task_id))
                _set_hybrid_cpu_idle_reason(_describe_hybrid_cpu_idle_reason())
                break
            task_id = _split_cpu_worker_task_to_runtime_target(int(task_id))
            task = gpu_worker_tasks_by_id[int(task_id)]
            start_rank = int(cpu_worker_dispatch_cursor) % len(worker_ids)
            position = {
                worker_id: (worker_ids.index(worker_id) - start_rank) % len(worker_ids)
                for worker_id in available
            }
            predicted_seconds = float(_cpu_worker_task_seconds(task))
            worker_id = min(
                available,
                key=lambda value: (
                    float(cpu_worker_predicted_load_by_id.get(int(value), 0.0))
                    + predicted_seconds,
                    position[value],
                ),
            )
            cpu_worker_dispatch_cursor = (worker_ids.index(worker_id) + 1) % len(worker_ids)
            tile_storage_reserved = False
            try:
                _activate_direct_union_task(task)
                tile_storage_reserved = _reserve_tile_dense_result_task(task)
                if tile_storage_reserved:
                    _prepare_tile_dense_result_workspaces(task)
                dispatch_task = dict(task)
                _attach_memfd_transfers_to_task(dispatch_task)
                preflight_multiprocessing_payload(dispatch_task)
                cpu_task_queues[int(worker_id)].put(dispatch_task)
            except BaseException:
                if tile_storage_reserved:
                    _release_tile_dense_result_task_id(
                        int(task_id), reason='CPU dispatch failure', refill=False,
                    )
                gpu_worker_pending_task_ids.appendleft(int(task_id))
                raise
            cpu_worker_dispatched_by_id[int(worker_id)] = int(
                cpu_worker_dispatched_by_id.get(int(worker_id), 0)
            ) + 1
            cpu_worker_task_predicted_seconds_by_id[int(task_id)] = float(predicted_seconds)
            cpu_worker_predicted_load_by_id[int(worker_id)] = float(
                cpu_worker_predicted_load_by_id.get(int(worker_id), 0.0)
            ) + float(predicted_seconds)
            _set_hybrid_cpu_idle_reason('')

    def _dispatch_inference_windows(
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> None:
        # Give each idle socket-local OpenVINO worker one claim-time-sized range from the
        # active or next ordered CPU reservation. CUDA fills every remaining slot with
        # mandatory GPU work and assists only the active direct-union view when its measured
        # ETA exceeds the mandatory-GPU horizon.
        # All ownership transitions and range claims run on this main thread.
        _dispatch_cpu_worker_inference_window(preferred_parent)
        _dispatch_gpu_worker_inference_window(preferred_parent)

    def _ensure_source_volume_file_backed() -> Tuple[str, Tuple[int, int, int], str]:
        wait_for_volume_ready(volume_rgb)
        shape = (int(volume_rgb.shape[0]), int(volume_rgb.shape[1]), int(volume_rgb.shape[2]))
        backing = _memmap_backing_path(volume_rgb)
        if backing is not None:
            flush_array(volume_rgb)
            return str(backing), shape, str(np.asarray(volume_rgb).dtype)
        # both this copy and the decode target it usually replaces live under the scratch
        # root, so a memory-backed scratch dir puts the shared source volume in RAM with the
        # tile canvases — one set of physical pages mapped by all GPU workers, no filesystem.
        shared_path = temp_dir / 'gpu_worker_source_volume.gray8.dat'
        shared_mm = copy_workspace_array(
            np.asarray(volume_rgb), shared_path,
            desc='inference-worker shared source volume', prefer_memory=False, prefer_memfd=True,
            workers=int(worker_budget),
        )
        flush_array(shared_mm)
        shared_backing = _memmap_backing_path(shared_mm)
        if shared_backing is None:
            raise RuntimeError('inference-worker shared source volume has no reopenable backing path')
        return str(shared_backing), shape, str(np.asarray(shared_mm).dtype)

    if inference_worker_process_active:
        gpu_worker_result_dir.mkdir(parents=True, exist_ok=True)
        # CUDA workers drive render/postprocess parallelism through their own thread pools.
        # so OpenCV must run each call
        # single-threaded (cv2.setNumThreads(1)); otherwise OpenCV's per-process pool funnels every
        # warpAffine/resize through a handful of threads and the render pool starves the GPUs while
        # most cores sit idle. YOLO_TTA_GPU_WORKER_CPU and YOLO_TTA_GPU_WORKER_CV2_THREADS override.
        per_worker_workers = (
            max(
                1,
                _env_int(
                    'YOLO_TTA_GPU_WORKER_CPU',
                    max(1, int(_cpu_count()) // max(1, gpu_device_count)),
                ),
            )
            if gpu_worker_process_active else 1
        )
        fused_preflight_specs: List[Dict[str, object]] = []
        if gpu_worker_process_active and fused_renderer_preflight_enabled():
            for requested_family in ('radial', 'tilted', 'tilted_radial'):
                selected_view: Optional[ViewInfo] = None
                for candidate in inference_views:
                    if _fused_preflight_family(candidate) == requested_family:
                        selected_view = candidate
                        break
                if selected_view is None:
                    continue
                candidate_jobs = list(aug_jobs_by_view.get(selected_view.name, ()))
                if not candidate_jobs:
                    raise RuntimeError(
                        f'No augmentation job is available for fused-render preflight '
                        f'{requested_family}/{selected_view.name}'
                    )
                selected_job = min(candidate_jobs, key=lambda item: abs(float(item.angle_deg)))
                fused_preflight_specs.append({
                    'view': selected_view,
                    'job': selected_job,
                    'frame_index': max(0, min(int(selected_view.num_slices) - 1, int(selected_view.num_slices) // 2)),
                })
            if fused_preflight_specs:
                print(
                    'v16.1.3 fused-render fail-fast preflight armed for: '
                    + ', '.join(_fused_preflight_family(spec['view']) for spec in fused_preflight_specs)
                )

        worker_init = {
            'imgsz': int(args.imgsz), 'conf': float(args.conf),
            'quantize': resolve_quantize(args.gpu_quantize), 'batch': max(1, int(args.gpu_batch)),
            'channel_format': channel_format,
            'input_channels': int(channel_format.channel_count),
            'channel_token': str(channel_format.token),
            'retina_processor': str(retina_processor),
            'cv2_threads': max(1, _env_int('YOLO_TTA_GPU_WORKER_CV2_THREADS', 1)),
            # angle-variant GPU fast-path (min_conf None when inactive) + min_radius.
            'angle_variant_gpu_fastpath_min_conf': angle_variant_gpu_fastpath_min_conf_value,
            'angle_variant_gpu_fastpath_min_radius': angle_variant_gpu_fastpath_min_radius_value,
            'fused_preflight_specs': tuple(fused_preflight_specs),
        }
        # Torch logical indices and their physical CUDA_VISIBLE_DEVICES tokens were resolved
        # before OpenVINO planning so dedicated feeder cores could be excluded job-wide.
        _configure_main_process_gpu_stage_workers(gpu_logical_indices)
        if gpu_worker_process_active and main_process_gpu_stage_inference_priority_enabled():
            if v1613_d1_owner_active:
                print(
                    'v16.1.3 D1 owner pipeline active inside the persistent CUDA workers: '
                    'backprojection is part of each inference lease, while main-process NRRD, '
                    'downbin, and topology GPU stages remain inference-first until global drain.'
                )
            elif v1613_d1_backprojection_overlap_enabled():
                print(
                    'v16.1.3 compatibility backprojection overlap active: a completed dense view '
                    'may borrow an otherwise-idle worker GPU when no dispatch-admissible inference '
                    'lease remains. YOLO_TTA_V1613_D1_BACKPROJECT_OVERLAP=0 restores strict ownership.'
                )
            else:
                print(
                    'Inference-first GPU ownership active (v16.1.3): main-process NRRD, '
                    'backprojection, downbin, and topology stages cannot seize worker GPUs until '
                    'the global inference queue is permanently drained. '
                    'YOLO_TTA_MAIN_GPU_STAGE_INFERENCE_PRIORITY=0 restores opportunistic leasing.'
                )
        pinned_tokens = list(pinned_gpu_tokens)
        # discovery-driven GPU-worker -> NUMA-node CPU pin plan (None entries
        # stay unpinned). Computed from the PHYSICAL tokens before CUDA_VISIBLE_DEVICES is
        # narrowed inside the workers.
        worker_numa_plan = plan_gpu_worker_affinity(
            pinned_tokens,
            excluded_cpus=(
                () if hybrid_cpu_affinity_overlap_active else cpu_inference_reserved_cpus
            ),
            reserved_cpus_by_worker=gpu_feeder_core_plan,
        )
        explicit_worker_cpu = bool(os.environ.get('YOLO_TTA_GPU_WORKER_CPU', '').strip())
        worker_cpu_budgets: List[int] = []
        for worker_pos, cpus in enumerate(worker_numa_plan):
            if explicit_worker_cpu or not cpus:
                worker_cpu_budgets.append(int(per_worker_workers))
                continue
            cpu_set = {int(c) for c in cpus}
            # The below-minimum-core fallback intentionally gives multiple workers the
            # same whole node. Divide its logical CPUs among those overlapping pools;
            # disjoint whole-core plans have overlap_count=1 and use their full allotment.
            overlap_positions = [
                int(i) for i, other in enumerate(worker_numa_plan)
                if other and cpu_set.intersection(int(c) for c in other)
            ]
            overlap_count = max(1, len(overlap_positions))
            q, r = divmod(len(cpu_set), int(overlap_count))
            rank = overlap_positions.index(int(worker_pos))
            worker_cpu_budgets.append(max(1, int(q) + (1 if int(rank) < int(r) else 0)))

        # spawn the worker processes BEFORE blocking on decode/cube-resize
        # completion — workers idle in task_queue.get while their CUDA context + model load
        # (~30-90 s each) overlaps the decode instead of running strictly after it. Tasks are
        # enqueued below once the shared source volume is ready.
        mp_ctx = mp.get_context('spawn')
        gpu_result_queue = mp_ctx.Queue()
        _run_resources().track_queue(gpu_result_queue)
        for worker_pos, gpu_index in enumerate(gpu_logical_indices):
            worker_id = int(gpu_index)
            worker_queue = mp_ctx.Queue()
            _run_resources().track_queue(worker_queue)
            gpu_task_queues[worker_id] = worker_queue
            gpu_worker_dispatched_by_id[worker_id] = 0
            gpu_worker_results_by_id[worker_id] = 0
            worker_init_i = dict(worker_init)
            worker_init_i['numa_affinity_cpus'] = worker_numa_plan[int(worker_pos)]
            worker_init_i['cpu_workers'] = int(worker_cpu_budgets[int(worker_pos)])
            proc = mp_ctx.Process(
                target=_gpu_inference_worker_main,
                args=(
                    worker_id, str(gpu_model_path), worker_init_i,
                    worker_queue, gpu_result_queue,
                ),
                name=f'gpu-worker-{gpu_index}', daemon=True,
            )
            _run_resources().track_process(proc)
            proc.start()
            gpu_worker_processes.append(proc)
        startup_overlap_note = (
            'model load/CUDA init overlaps streaming decode'
            if bool(preprocess_streaming_active)
            else 'decode was synchronous; worker model load starts after decode'
        )
        if gpu_worker_process_active:
            print(
                f'Started {len(gpu_worker_processes)} GPU worker process(es) for logical devices {gpu_logical_indices} '
                f'(pinned to CUDA_VISIBLE_DEVICES tokens {pinned_tokens}); {startup_overlap_note}; '
                f'per-worker CPU workers={worker_cpu_budgets}, cv2 threads=1. Tasks enqueue once the source volume is ready.'
            )

        if cpu_worker_process_active:
            cpu_render_workers = max(1, _env_int('YOLO_TTA_CPU_RENDER_WORKERS', 2))
            for plan in cpu_instance_plans:
                instance_id = int(plan.instance_id)
                worker_queue = mp_ctx.Queue()
                _run_resources().track_queue(worker_queue)
                cpu_task_queues[instance_id] = worker_queue
                cpu_worker_dispatched_by_id[instance_id] = 0
                cpu_worker_results_by_id[instance_id] = 0
                cpu_init = {
                    'imgsz': int(args.imgsz),
                    'conf': float(args.conf),
                    'batch': max(1, int(args.cpu_batch)),
                    'input_channels': int(channel_format.channel_count),
                    'channel_token': str(channel_format.token),
                    'precision': str(args.cpu_precision),
                    'numa_affinity_cpus': tuple(int(cpu) for cpu in plan.cpus),
                    'inference_threads': int(plan.inference_threads),
                    'physical_cores': int(plan.physical_cores),
                    'streams': cpu_streams_requested,
                    'infer_requests': cpu_infer_requests_requested,
                    'render_workers': int(cpu_render_workers),
                    'cv2_threads': 1,
                }
                proc = mp_ctx.Process(
                    target=_cpu_inference_worker_main,
                    args=(
                        instance_id, str(cpu_model_path), cpu_init,
                        worker_queue, gpu_result_queue,
                    ),
                    name=f'openvino-worker-{instance_id}', daemon=True,
                )
                _run_resources().track_process(proc)
                proc.start()
                cpu_worker_processes.append(proc)
            print(
                f'Started {len(cpu_worker_processes)} socket-local OpenVINO worker process(es); '
                f'precision={args.cpu_precision}, batch={args.cpu_batch}, '
                f'streams={cpu_streams_requested or "auto"}, '
                f'infer_requests={cpu_infer_requests_requested or "auto"}, '
                f'render_workers/instance={cpu_render_workers}. '
                'Cartesian work is preferred, Tilted Cartesian follows, and Radial work is never claimed.'
            )

        # Warm GPU workers double as interpolation hosts whenever their targeted inference queue is idle.
        if gpu_worker_process_active and gpu_worker_aux_interpolation_enabled() and bool(interpolation_process_backend_active):
            set_gpu_worker_aux_interpolation_pool(
                _GpuWorkerAuxInterpolationPool(gpu_task_queues)
            )

        # when the cube resize applies and both volumes are file-backed,
        # enqueue against the cube PATH without waiting for its content — workers go
        # resident straight from the NATIVE decoded volume (device t-resize), and a
        # sentinel written after wait_for_volume_ready+flush gates the file-backed
        # fallbacks (failed residency, tile tasks, CPU renders). Enqueue then waits only
        # for the decode, not the resize tail + cube flush.
        native_resize_task_spec: Optional[Dict[str, object]] = None
        source_volume_ready_async = False
        native_shape_now = tuple(int(x) for x in np.asarray(input_volume_rgb).shape)
        # metadata access must not invoke LazyProcessingCube.__array__.
        working_shape_now = tuple(int(x) for x in volume_rgb.shape)
        native_t_only_resize = bool(
            native_shape_now[1:] == working_shape_now[1:]
            and _cube_t_axis_resize_backend() == 'slab'
        )
        if (
            gpu_worker_process_active
            and gpu_cube_resize_enabled()
            and bool(cube_resize_will_apply)
            and (volume_rgb is not input_volume_rgb)
            and native_t_only_resize
        ):
            native_backing = _memmap_backing_path(input_volume_rgb)
            cube_backing = _memmap_backing_path(volume_rgb)
            if native_backing is not None and cube_backing is not None:
                wait_for_volume_ready(input_volume_rgb)
                flush_array(input_volume_rgb)
                lazy_cube = volume_rgb if isinstance(volume_rgb, LazyProcessingCube) else None
                cube_ready_sentinel = (
                    lazy_cube.ready_path
                    if lazy_cube is not None
                    else temp_dir / 'gpu_worker_source_volume.cube_ready.sentinel'
                )
                if lazy_cube is None:
                    # A reusable --temp_dir may contain a marker from an interrupted/keep-temp run.
                    # Remove it before the eager cube producer can race any file-backed fallback.
                    try:
                        cube_ready_sentinel.unlink()
                    except FileNotFoundError:
                        pass
                native_resize_task_spec = {
                    'path': str(native_backing),
                    'shape': [int(x) for x in np.asarray(input_volume_rgb).shape],
                    'dtype': str(np.asarray(input_volume_rgb).dtype),
                    'sentinel': str(cube_ready_sentinel),
                    'request': (str(lazy_cube.request_path) if lazy_cube is not None else None),
                    'failed': (str(lazy_cube.failed_path) if lazy_cube is not None else None),
                }
                source_volume_path = str(cube_backing)
                source_volume_shape = tuple(int(x) for x in volume_rgb.shape)
                source_volume_dtype = str(volume_rgb.dtype)
                source_volume_ready_async = True

                if lazy_cube is None:
                    def _signal_cube_ready() -> None:
                        try:
                            wait_for_volume_ready(volume_rgb)
                            flush_array(volume_rgb)
                            cube_ready_sentinel.touch()
                            print('Shared cube volume complete; sentinel written for file-backed worker fallbacks.')
                        except BaseException as exc:
                            # The producer failure also surfaces in main; teardown terminates
                            # any workers parked on the (never-written) sentinel.
                            print(f'Warning: cube-ready signaling failed ({exc}).')

                    cube_ready_thread = threading.Thread(
                        target=_signal_cube_ready,
                        name='cube-ready-sentinel',
                        daemon=True,
                    )
                    _run_resources().track_thread(cube_ready_thread)
                    cube_ready_thread.start()
                print(
                    'v13.3.9 (E3): task enqueue gated on the decode only — GPU workers retain the '
                    'NATIVE-t decoded volume and fold t scaling into device renderers '
                    f'({"v13.3.17 C10 host cube is demand-only; " if lazy_cube is not None else ""}'
                    'YOLO_TTA_GPU_CUBE_RESIZE=0 restores cube-gated enqueue).'
                )
        elif gpu_worker_process_active and gpu_cube_resize_enabled() and bool(cube_resize_will_apply) and (volume_rgb is not input_volume_rgb):
            print(
                'v13.3.9 (E3): native-t residency bypassed because the cube resize changes X/Y '
                'or uses YOLO_TTA_CUBE_T_RESIZE_BACKEND=slice_exact; waiting for the exact cube '
                'volume before worker rendering.'
            )
        if not source_volume_ready_async:
            source_volume_path, source_volume_shape, source_volume_dtype = _ensure_source_volume_file_backed()
        # Split each full-frame volume into contiguous slice-range chunks so multiple GPUs can work
        # the SAME (often huge, e.g. full-coverage Radial) volume in parallel, instead of one GPU per
        # whole volume leaving the other GPUs idle at the tail. The chunk is large relative to the
        # render prefetch window so per-chunk render ramp-up stays amortized. Tiles are gated/
        # consolidated as whole volumes, so they are never slice-split.
        slice_chunk = max(
            gpu_worker_min_lease_slices(),
            _env_int('YOLO_TTA_GPU_WORKER_SLICE_CHUNK', gpu_worker_max_lease_slices()),
        )
        prefetch_frames = streaming_prediction_source_prefetch_frames(max(1, int(args.batch)))
        # per-task device-side 2D hole fill is valid only when the spec steps that
        # precede hole fill (--min_conf, --min_radius) have no work, results stream through the
        # angle-variant cleanup path, and the task writes a disjoint direct-union window.
        gpu_worker_device_hole_fill = bool(
            gpu_worker_process_active
            and gpu_device_union_enabled()
            and gpu_device_hole_fill_enabled()
            and bool(angle_variant_streaming_cleanup_active)
            and float(args.min_conf) <= 0.0
            and float(args.min_radius) <= 0.0
        )
        chunk_hole_fill_enabled = bool(gpu_worker_chunk_hole_fill_enabled())
        if gpu_worker_device_hole_fill and not chunk_hole_fill_enabled:
            print(
                'Split-view hole fill moved off the inference handoff (v16.1.3): multi-chunk '
                'full-frame views run one completed-view CPU pass instead of a CuPy label/fill '
                'barrier after every GPU lease. YOLO_TTA_GPU_WORKER_CHUNK_HOLE_FILL=1 restores '
                'the per-chunk device pass.'
            )
        fullframe_subtasks_per_view: Dict[Tuple[str, str], int] = {}
        next_task_id = 0
        for kind, view, job_obj in list(pending_prediction_build_jobs):
            n_slices = int(view.num_slices)
            if str(kind) == 'fullframe':
                aug_job = job_obj
                write_aug_job_meta(aug_job, view, channel_format)
                m_out = np.asarray(aug_job.aff.M_out_to_src, dtype=np.float32)
                out_size = int(aug_job.aff.out_size)
                job_id = str(aug_job.aug_id)
                # construction keeps every profiled steady-state lease untouched.
                # A still-central lease may split later, only when dispatch reaches the
                # actual global inference tail; tiles are never split.
                initial_lease_candidates: List[int] = []
                if gpu_worker_process_active:
                    initial_lease_candidates.append(
                        int(gpu_worker_initial_lease_slices(view, int(args.gpu_batch)))
                    )
                if cpu_worker_process_active and cpu_inference_supports_view(view):
                    initial_lease_candidates.append(
                        int(cpu_worker_initial_lease_slices(view, int(args.cpu_batch)))
                    )
                if not initial_lease_candidates:
                    raise RuntimeError(f'No inference backend is eligible for {view.name}')
                # v17.0.3 retains the v17.0.2 LARGEST-eligible-backend seed target. The previous
                # minimum permanently fragmented every GPU steal into 7/10-slice CPU
                # leases. A backend may split an oversized seed only when it claims it.
                initial_chunk = min(int(slice_chunk), max(initial_lease_candidates))
                ranges = gpu_worker_fullframe_task_ranges(
                    int(n_slices), int(initial_chunk),
                )
                key_fv = (str(model_name), str(view.name))
                fullframe_subtasks_per_view[key_fv] = int(fullframe_subtasks_per_view.get(key_fv, 0)) + len(ranges)
                prefix = f'{view.name}__{job_id}'
                # v16.1.3 allocates the shared direct-union workspace only when the
                # first task for this view is actually dispatched.  This prevents all 30
                # logical 25-39 GiB unions from being created/touched at startup.
            else:
                tile_job = job_obj
                write_dense_tile_job_meta(tile_job, channel_format)
                m_out = np.asarray(tile_job.M_out_to_src, dtype=np.float32)
                out_size = int(tile_job.out_size)
                job_id = str(tile_job.tile_id)
                ranges = [(0, int(n_slices))]
                prefix = f'tile__{view.name}__{job_id}'
            full_parent_processing_shape = view_processing_volume_shape(view, int(out_size))
            if str(kind) == 'tile':
                py0, py1, px0, px1 = (int(v) for v in tile_job.parent_crop)
                task_processing_shape = (
                    int(n_slices), int(py1 - py0), int(px1 - px0),
                )
                m_out_processing = np.asarray(tile_job.M_out_to_crop, dtype=np.float32)
                threshold_plane_shape = tuple(int(v) for v in full_parent_processing_shape[-2:])
            else:
                task_processing_shape = full_parent_processing_shape
                m_out_processing = output_to_view_processing_affine(view, m_out, int(out_size))
                threshold_plane_shape = tuple(int(v) for v in task_processing_shape[-2:])
            processing_min_radius = view_processing_min_radius(
                view, float(args.min_radius), threshold_plane_shape,
            )
            for chunk_idx, (s0, s1) in enumerate(ranges):
                count = int(s1) - int(s0)
                # Size the render pool to this worker's CPU share (NOT streaming_prediction_source_workers,
                # which returns the full machine CPU count and would have each of N workers spin up an
                # N-times-too-large render pool). With cv2.setNumThreads(1) these are real render threads.
                render_workers = max(1, min(int(per_worker_workers), int(count)))
                cpu_eligible = bool(
                    cpu_worker_process_active and cpu_inference_supports_view(view)
                )
                gpu_eligible = bool(gpu_worker_process_active)
                hybrid_deferred = bool(
                    str(kind) == 'fullframe'
                    and v1613_d1_owner_active
                    and worker_direct_union_active
                    and cpu_eligible
                    and gpu_eligible
                )
                if hybrid_deferred:
                    # First claim resolves the whole view: OpenVINO -> shared direct union;
                    # CUDA -> D1. This prevents mere CPU eligibility from disabling D1 for
                    # every Cartesian/Tilted view before either backend performs work.
                    rmask = None
                    rconf = None
                    result_mode = HYBRID_DEFERRED_RESULT_MODE
                elif str(kind) == 'fullframe' and v1613_d1_owner_active and not cpu_eligible:
                    # D1 retains only a task-local device union and one persistent source-space
                    # bitset on the owner GPU. No host result path or dense per-view workspace exists.
                    rmask = None
                    rconf = None
                    result_mode = 'd1_owner'
                elif str(kind) == 'fullframe' and worker_direct_union_active:
                    # The scheduler attaches transferred memfd descriptors for the shared
                    # per-view union; a real result pathname is only the fallback.
                    rmask = None
                    rconf = None
                    result_mode = 'direct_union'
                else:
                    rmask = gpu_worker_result_dir / f'{prefix}__c{chunk_idx}.mask.u8.dat'
                    rconf = (gpu_worker_result_dir / f'{prefix}__c{chunk_idx}.conf.u8.dat') if float(args.min_conf) > 0.0 else None
                    result_mode = 'file'
                task = {
                    'task_id': int(next_task_id), 'kind': str(kind), 'model_name': str(model_name),
                    'cpu_eligible': bool(cpu_eligible), 'gpu_eligible': bool(gpu_eligible),
                    'view': view, 'job': job_obj, 'job_id': job_id, 'out_size': int(out_size),
                    'channel_format': channel_format,
                    'M_out_to_src': m_out,
                    'M_out_to_processing': m_out_processing,
                    'processing_shape': task_processing_shape,
                    'threshold_plane_shape': threshold_plane_shape,
                    'parent_crop': (
                        tuple(int(v) for v in tile_job.parent_crop)
                        if str(kind) == 'tile' else None
                    ),
                    'slice_start': int(s0), 'slice_count': int(count),
                    'source_volume_path': source_volume_path, 'source_shape': list(source_volume_shape),
                    'source_dtype': source_volume_dtype,
                    'radial_texture_required': bool(radial_texture_required),
                    # native-volume residency spec + cube-ready sentinel
                    # (None when the cube gated enqueue synchronously, as before).
                    'native_resize': native_resize_task_spec,
                    'result_mask_path': (str(rmask) if rmask is not None else None),
                    'result_conf_path': (str(rconf) if rconf is not None else None),
                    'result_mode': str(result_mode), 'union_num_slices': int(n_slices),
                    'hybrid_cpu_eligible_origin': bool(hybrid_deferred),
                    'render_workers': int(render_workers), 'prefetch_frames': int(prefetch_frames),
                    'postprocess_workers': int(per_worker_workers),
                    'streaming_cleanup_enabled': bool(angle_variant_streaming_cleanup_active),
                    'streaming_cleanup_min_conf': float(args.min_conf),
                    'streaming_cleanup_min_radius': float(processing_min_radius),
                    'device_hole_fill': bool(
                        gpu_worker_device_hole_fill
                        and str(kind) == 'fullframe'
                        and str(result_mode) == 'direct_union'
                        and (len(ranges) <= 1 or chunk_hole_fill_enabled)
                    ),
                }
                if str(result_mode) in {'d1_owner', HYBRID_DEFERRED_RESULT_MODE}:
                    d1_key = _nrrd_layer_key(
                        view_name=str(view.name), source='fullframe', mask_kind='yolo',
                        pass_index=0, stage='pre_interpolation',
                    )
                    task['d1_output_shape'] = [int(input_T), int(input_H), int(input_W)]
                    task['d1_store_dir'] = str(
                        Path(temp_dir) / 'nrrd_layers' / str(view.name)
                        / f'{d1_key}.orthogonal.cvol'
                    )
                    d1_shadow_required = bool(
                        _view_uses_interpolation(view, int(args.interpolation_distance))
                        or bool(tile_jobs_by_view_config.get(str(view.name)))
                    )
                    task['d1_view_shadow_required'] = bool(d1_shadow_required)
                    if d1_shadow_required:
                        task['d1_view_shadow_shape'] = [
                            int(v) for v in full_parent_processing_shape
                        ]
                        task['d1_view_shadow_store_dir'] = str(
                            Path(temp_dir) / 'd1_view_shadow' / str(model_name)
                            / f'{str(view.name)}.packed.cvol'
                        )
                gpu_worker_tasks_by_id[int(next_task_id)] = task
                if str(kind) == 'fullframe':
                    parent_key = (str(model_name), str(view.name))
                    fullframe_task_ids_by_parent.setdefault(parent_key, []).append(int(next_task_id))
                if str(kind) == 'tile':
                    tile_key = (str(model_name), str(view.name), str(job_id))
                    if tile_key in gpu_worker_tile_task_id_by_key:
                        raise RuntimeError(f'duplicate tile worker task key: {tile_key}')
                    gpu_worker_tile_task_id_by_key[tile_key] = int(next_task_id)
                next_task_id += 1
        gpu_worker_total_tasks = int(next_task_id)
        gpu_worker_seed_task_count = int(next_task_id)
        gpu_worker_next_dynamic_task_id = int(next_task_id)
        # A full-frame view is finalized only after every (angle x slice-chunk) result for it has been
        # unioned, so override fullframe_remaining with the per-view sub-task count.
        for key_fv, cnt in fullframe_subtasks_per_view.items():
            fullframe_remaining[key_fv] = int(cnt)
        # v17.0.3 reserves a small ordered parent sequence for OpenVINO instead of
        # treating all CPU-compatible views as one global backlog. Orthogonal views are ordered
        # transverse -> sagittal -> coronal within the user's TTA-angle order; Tilted views are
        # fallback candidates only when fewer orthogonal parents exist. Every unreserved parent
        # remains immediately available to CUDA D1.
        if gpu_worker_process_active and cpu_worker_process_active:
            angle_order = {
                round(float(angle) % 360.0, 9): int(index)
                for index, angle in enumerate(angles)
            }
            axis_order = {'transverse': 0, 'sagittal': 1, 'coronal': 2}
            hybrid_candidates: List[Tuple[Tuple[object, ...], Tuple[str, str]]] = []
            for parent_key, indexed_ids in fullframe_task_ids_by_parent.items():
                if not indexed_ids:
                    continue
                first_task = gpu_worker_tasks_by_id[int(indexed_ids[0])]
                if not bool(first_task.get('hybrid_cpu_eligible_origin', False)):
                    continue
                view_obj = first_task.get('view')
                if not isinstance(view_obj, ViewInfo):
                    continue
                family_rank = 0 if str(view_obj.family) == 'orthogonal' else 1
                base_axis = str(
                    view_obj.tilt_base_view
                    or physical_view_name(view_obj)
                ).strip().lower()
                angle_key = round(float(view_obj.tta_angle_deg) % 360.0, 9)
                hybrid_candidates.append((
                    (
                        int(family_rank),
                        int(angle_order.get(angle_key, len(angle_order))),
                        int(axis_order.get(base_axis, 99)),
                        abs(float(getattr(view_obj, 'tilt_angle_deg', 0.0))),
                        str(getattr(view_obj, 'tilt_direction', '')),
                        str(view_obj.name),
                    ),
                    parent_key,
                ))
            hybrid_candidates.sort(key=lambda item: item[0])
            reserve_count = min(
                int(hybrid_cpu_reserved_view_count()),
                int(len(hybrid_candidates)),
            )
            hybrid_cpu_reserved_parents.extend(
                parent for _sort_key, parent in hybrid_candidates[:reserve_count]
            )
            hybrid_cpu_reserved_parent_set.update(hybrid_cpu_reserved_parents)
            hybrid_cpu_reservation_rank_by_parent.update({
                parent: int(index)
                for index, parent in enumerate(hybrid_cpu_reserved_parents)
            })
            for _sort_key, parent in hybrid_candidates:
                reserved = bool(parent in hybrid_cpu_reserved_parent_set)
                for indexed_task_id in fullframe_task_ids_by_parent.get(parent, ()):
                    gpu_worker_tasks_by_id[int(indexed_task_id)]['hybrid_cpu_reserved'] = reserved
            if hybrid_cpu_reserved_parents:
                sequence = ' -> '.join(
                    f'{parent[0]}/{parent[1]}' for parent in hybrid_cpu_reserved_parents
                )
                print(
                    '[hybrid] OpenVINO full-frame reservation sequence '
                    f'({len(hybrid_cpu_reserved_parents)}/{len(hybrid_candidates)} eligible view(s)): '
                    f'{sequence}. Unreserved eligible views remain CUDA D1 work. '
                    'YOLO_TTA_HYBRID_CPU_RESERVED_VIEW_COUNT adjusts the reservation count.'
                )
            elif hybrid_candidates:
                print(
                    '[hybrid] OpenVINO full-frame reservation is disabled '
                    '(YOLO_TTA_HYBRID_CPU_RESERVED_VIEW_COUNT=0); all eligible views remain CUDA D1 work.'
                )
        pending_prediction_build_jobs.clear()

        tile_result_sizes = [
            int(_tile_dense_result_task_bytes(task_obj))
            for task_obj in gpu_worker_tasks_by_id.values()
            if str(task_obj.get('kind', '')) == 'tile'
        ]
        if tile_result_sizes:
            largest_tile_result = int(max(tile_result_sizes))
            if bool(keep_temp_artifacts):
                print(
                    'Warning: YOLO_TTA_KEEP_TEMP=1 retains every dense tile worker result for '
                    'diagnostics; v16.4.3 tile-result backpressure and immediate dense-file '
                    f'retirement are intentionally disabled (largest result={largest_tile_result / GIB:.2f} GiB).'
                )
            else:
                if (
                    gpu_worker_tile_dense_result_memory_safe_limit is not None
                    and int(largest_tile_result) > int(gpu_worker_tile_dense_result_memory_safe_limit)
                ):
                    raise RuntimeError(
                        'One dense crop-local tile result requires '
                        f'{largest_tile_result / GIB:.2f} GiB, but the memory-backed scratch '
                        'root has only '
                        f'{gpu_worker_tile_dense_result_memory_safe_limit / GIB:.2f} GiB of '
                        'safe live-result headroom after reserve. Use a disk-backed --temp root, '
                        'reduce the tile footprint, or disable the confidence result by setting '
                        '--min_conf 0.'
                    )
                print(
                    'v16.4.3 cleanup-boundary tile-result retirement active: '
                    f'live dense result budget={gpu_worker_tile_dense_result_limit / GIB:.1f} GiB / '
                    f'{gpu_worker_tile_dense_result_task_limit} task(s), '
                    f'largest tile result={largest_tile_result / GIB:.2f} GiB; '
                    'shared memfd RAM is preferred so gpu_worker_results is a bounded pathname fallback. '
                    'Stale worker-result scratch was purged before dispatch; full-frame parents and '
                    'P-ready tiles are scheduled ahead of P-not-ready tiles. '
                    f'every nonempty tile retires to CTILE on {tile_dense_retirement_workers} dedicated '
                    f'cleanup task(s) x {tile_dense_retirement_slice_workers} slice worker(s) before '
                    'parent interpolation or component gates can retain its dense backing. '
                    'YOLO_TTA_TILE_DENSE_RESULT_MAX_GIB / _MAX_TASKS adjust admission; '
                    'YOLO_TTA_TILE_DENSE_RETIREMENT_WORKERS / _SLICE_WORKERS adjust retirement throughput.'
                )

        gpu_worker_pending_task_ids.extend(range(int(gpu_worker_total_tasks)))
        # Hybrid native-t GPU residency can defer the 25+ GiB host processing cube until a
        # CPU worker asks for it. Let that one-time resize use every NON-FEEDER CPU while the
        # OpenVINO workers are still waiting on the cube sentinel. Dedicated feeder cores are
        # protected immediately; a monitor applies the narrower parent/OpenVINO isolation mask
        # after publication.
        defer_parent_affinity_until_cube = bool(
            cpu_worker_process_active
            and not hybrid_cpu_affinity_overlap_active
            and source_volume_ready_async
            and isinstance(native_resize_task_spec, dict)
            and native_resize_task_spec.get('sentinel')
            and not Path(str(native_resize_task_spec['sentinel'])).exists()
        )
        if defer_parent_affinity_until_cube:
            # GPU inference can begin before the host cube exists, so feeder exclusivity cannot
            # wait for the cube sentinel. The resize may use all remaining non-feeder CPUs.
            _apply_parent_cpu_mask(
                feeder_safe_parent_cpus,
                fail_fast=True,
                phase_label='deferred processing-cube construction',
            )
            sentinel_path = Path(str(native_resize_task_spec['sentinel']))
            failed_path = (
                Path(str(native_resize_task_spec['failed']))
                if native_resize_task_spec.get('failed') else None
            )

            def _restrict_parent_after_cube_ready() -> None:
                while not parent_affinity_monitor_stop.wait(0.10):
                    if failed_path is not None and failed_path.exists():
                        return
                    if not sentinel_path.exists():
                        continue
                    _apply_parent_inference_affinity(fail_fast=False)
                    return

            parent_affinity_monitor_thread = threading.Thread(
                target=_restrict_parent_after_cube_ready,
                name='v17-parent-affinity-after-cube',
                daemon=True,
            )
            _run_resources().track_thread(
                parent_affinity_monitor_thread, parent_affinity_monitor_stop,
            )
            parent_affinity_monitor_thread.start()
            print(
                '[intel] Strict parent/OpenVINO isolation is deferred until the shared processing '
                'cube is materialized; the one-time resize may use every non-feeder CPU while '
                'the four-core-per-GPU feeder reservations remain exclusive.'
            )
        else:
            _apply_parent_inference_affinity(fail_fast=True)
        _dispatch_inference_windows()
        lease_details: List[str] = []
        if gpu_worker_process_active:
            lease_details.append(f'GPU target={gpu_worker_target_lease_seconds():.2f}s')
        if cpu_worker_process_active:
            lease_details.append(f'CPU target={cpu_worker_target_lease_seconds():.2f}s')
        hybrid_policy = (
            ' Hybrid full-frame views use ordered CPU reservations: OpenVINO opens one reserved '
            'direct-union view at a time and advances after completion; unreserved eligible views '
            f'are CUDA D1 work. After {hybrid_gpu_stealback_min_cpu_samples()} completed CPU lease sample(s), '
            'active-view ETA may borrow a proportional number of concurrent CUDA assist tasks; all CUDA '
            'workers may assist after mandatory GPU work drains.'
            if gpu_worker_process_active and cpu_worker_process_active else ''
        )
        dynamic_splits = max(0, int(gpu_worker_total_tasks) - int(gpu_worker_seed_task_count))
        print(
            f'v17 process-local leases: {gpu_worker_seed_task_count} seed inference task(s) retained '
            f'in one bounded central dispatch window; {gpu_worker_total_tasks} task(s) currently '
            f'tracked after {dynamic_splits} claim-time split(s) (hard full-frame chunk cap={slice_chunk}; '
            f'{", ".join(lease_details)}).{hybrid_policy}'
        )
    def _finalize_fullframe_view_after_worker(model_name_s: str, view: ViewInfo) -> None:
        remaining_key = (str(model_name_s), str(view.name))
        fullframe_remaining[remaining_key] = int(fullframe_remaining.get(remaining_key, 0)) - 1
        if int(fullframe_remaining.get(remaining_key, 0)) == 0:
            _submit_view_prepare(str(model_name_s), view)

    def _accumulate_fullframe_slice_metadata(
        task: Dict[str, object], stats: Dict[str, object], view: ViewInfo,
    ) -> None:
        meta_key = (str(task['model_name']), str(view.name))
        holder = view_slice_meta.get(meta_key)
        if holder is None:
            holder = {
                'valid': True,
                'slice_any': np.zeros((int(view.num_slices),), dtype=bool),
                'slice_bboxes': np.zeros((int(view.num_slices), 4), dtype=np.int64),
                'slice_row_any': None,
                'slice_row_count': 0,
            }
            view_slice_meta[meta_key] = holder
        task_meta = stats.get('slice_meta')
        if not isinstance(task_meta, dict):
            holder['valid'] = False
            return
        if not bool(holder['valid']):
            return
        try:
            s0_meta = int(task.get('slice_start', 0))
            any_arr = np.asarray(task_meta['slice_any'], dtype=bool)
            n_meta = int(any_arr.shape[0])
            holder['slice_any'][s0_meta:s0_meta + n_meta] = any_arr
            holder['slice_bboxes'][s0_meta:s0_meta + n_meta] = np.asarray(
                task_meta['slice_bboxes'], dtype=np.int64,
            )
            rows_packed = task_meta.get('slice_row_any')
            if rows_packed is not None:
                rows_packed = np.asarray(rows_packed, dtype=np.uint8)
                if holder['slice_row_any'] is None:
                    holder['slice_row_any'] = np.zeros(
                        (int(view.num_slices), int(rows_packed.shape[1])), dtype=np.uint8,
                    )
                    holder['slice_row_count'] = int(
                        np.asarray(task_meta['slice_row_count']).reshape(-1)[0]
                    )
                holder['slice_row_any'][s0_meta:s0_meta + n_meta] = rows_packed
        except Exception:
            holder['valid'] = False

    def _handle_fullframe_worker_result(task: Dict[str, object], stats: Dict[str, object]) -> None:
        view = task['view']
        model_name_s = str(task['model_name'])
        view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(stats.get('prediction_count', 0))
        # accumulate device-side hole-filled slice counts toward the per-view skip.
        _filled = int(stats.get('device_hole_filled_frames', 0))
        if _filled > 0:
            key_fill = (model_name_s, str(view.name))
            view_device_hole_filled_slices[key_fill] = int(view_device_hole_filled_slices.get(key_fill, 0)) + _filled
        meta_key = (model_name_s, str(view.name))
        _accumulate_fullframe_slice_metadata(task, stats, view)
        if str(task.get('result_mode', 'file')) == 'd1_owner':
            complete = bool(stats.get('d1_view_complete', False))
            layer_ref = stats.get('d1_layer_ref')
            if layer_ref is not None:
                if not isinstance(layer_ref, NrrdLayerRef):
                    raise TypeError(
                        f'D1 result {meta_key} returned {type(layer_ref)!r}, expected NrrdLayerRef'
                    )
                if meta_key in d1_layer_ref_by_parent:
                    raise RuntimeError(f'D1 layer {meta_key} was published more than once')
                d1_layer_ref_by_parent[meta_key] = layer_ref
                nrrd_layer_refs.append(layer_ref)
                sink = nrrd_layer_sink()
                if sink is not None:
                    sink.submit_layer(
                        layer_ref,
                        nrrd_layer_output_suffix(
                            view_token=view_output_token(view), source='fullframe',
                            mask_kind='yolo', pass_index=0, stage='pre_interpolation',
                        ),
                    )
            shadow_path_raw = str(stats.get('d1_view_shadow_path', '') or '').strip()
            if shadow_path_raw:
                shadow_path = Path(shadow_path_raw)
                existing_shadow = d1_view_shadow_path_by_parent.get(meta_key)
                if existing_shadow is not None and existing_shadow != shadow_path:
                    raise RuntimeError(
                        f'D1 view shadow {meta_key} changed {existing_shadow} -> {shadow_path}'
                    )
                d1_view_shadow_path_by_parent[meta_key] = shadow_path
            fullframe_remaining[meta_key] = int(fullframe_remaining.get(meta_key, 0)) - 1
            remaining = int(fullframe_remaining.get(meta_key, 0))
            if remaining < 0:
                raise RuntimeError(f'D1 view {meta_key} completed too many inference tasks')
            if remaining == 0:
                if not complete or meta_key not in d1_layer_ref_by_parent:
                    raise RuntimeError(
                        f'D1 view {meta_key} exhausted its tasks without a finalized source-space cvol'
                    )
                shadow_required = bool(task.get('d1_view_shadow_required', False))
                if shadow_required:
                    if meta_key not in d1_view_shadow_path_by_parent:
                        raise RuntimeError(
                            f'D1 view {meta_key} requires interpolation/tile support but returned no '
                            'view-native sparse shadow'
                        )
                    _submit_view_prepare(model_name_s, view)
                else:
                    view_processing_submitted.add(meta_key)
                    view_slice_meta.pop(meta_key, None)
            return
        if str(task.get('result_mode', 'file')) == 'direct_union':
            # The worker already wrote its disjoint slice window straight into the
            # shared per-view union mapping; nothing to reopen, OR, flush, or unlink here, and
            # the scheduler thread is free to drain the next GPU result immediately.
            _finalize_fullframe_view_after_worker(model_name_s, view)
            return
        _ensure_baseline_workspaces(model_name_s, view)
        s0 = int(task.get('slice_start', 0))
        count = int(task.get('slice_count', int(view.num_slices)))
        # The worker result covers only this task's slice window; union it into that window of the
        # per-view union. Result handlers run only on the main thread, so concurrent sub-tasks of the
        # same view never race here.
        _proc_shape = tuple(int(v) for v in task.get(
            'processing_shape', view_processing_volume_shape(view, int(task.get('out_size', args.imgsz))),
        ))
        result_shape = (int(count), int(_proc_shape[1]), int(_proc_shape[2]))
        res_mask = open_existing_gray_memmap(task['result_mask_path'], result_shape, 'uint8', mode='r')
        res_conf = open_existing_gray_memmap(task['result_conf_path'], result_shape, 'uint8', mode='r') if task.get('result_conf_path') else None
        dst_mask = baseline_union_by_model_view[(model_name_s, str(view.name))]
        dst_conf = baseline_confmap_by_model_view.get((model_name_s, str(view.name)))
        union_conf_volume_into_volume_inplace(
            dst_mask[s0:s0 + count],
            (dst_conf[s0:s0 + count] if dst_conf is not None else None),
            res_mask, res_conf,
            workers=int(slice_postprocess_workers),
            desc=f'Union {view.name}[{s0}:{s0 + count}] worker result',
        )
        close_memmap_array(res_mask)
        if res_conf is not None:
            close_memmap_array(res_conf)
        if not keep_temp_artifacts:
            for pth in (task.get('result_mask_path'), task.get('result_conf_path')):
                if pth:
                    try:
                        Path(str(pth)).unlink(missing_ok=True)
                    except Exception:
                        pass
        _finalize_fullframe_view_after_worker(model_name_s, view)

    def _handle_tile_worker_result(task: Dict[str, object], stats: Dict[str, int]) -> None:
        view = task['view']
        model_name_s = str(task['model_name'])
        tile_job = task['job']
        view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(stats.get('prediction_count', 0))
        tile_inference_done.add((model_name_s, str(view.name), str(tile_job.tile_id)))
        if int(stats.get('frames_with_predictions', 0)) <= 0:
            if not keep_temp_artifacts:
                for pth in (task.get('result_mask_path'), task.get('result_conf_path')):
                    if pth:
                        try:
                            Path(str(pth)).unlink(missing_ok=True)
                        except Exception:
                            pass
            _release_tile_dense_result_task_id(
                int(task.get('task_id', -1)),
                reason='inference worker reported an empty tile result',
                refill=True,
            )
            _mark_tile_complete(
                model_name_s, str(view.name),
                str(tile_job.config_id), str(tile_job.tile_id),
            )
            return
        result_shape = tuple(int(v) for v in task.get(
            'processing_shape', view_processing_volume_shape(view, int(task.get('out_size', args.imgsz))),
        ))
        tile_mask_path = Path(str(task['result_mask_path']))
        tile_mask_mm = open_existing_gray_memmap(tile_mask_path, result_shape, 'uint8', mode='r+')
        tile_conf_mm = None
        tile_conf_path = Path(str(task['result_conf_path'])) if task.get('result_conf_path') else None
        if tile_conf_path is not None:
            tile_conf_mm = open_existing_gray_memmap(tile_conf_path, result_shape, 'uint8', mode='r+')
        ptask = TilePostprocessTask(
            model_name=model_name_s, view_name=str(view.name),
            aug_id=str(view.tta_aug_id), angle_deg=float(view.tta_angle_deg),
            config_id=str(tile_job.config_id), tile_id=str(tile_job.tile_id),
            parent_crop=tuple(int(v) for v in tile_job.parent_crop),
            tile_mask_mm=tile_mask_mm, tile_confmap_mm=tile_conf_mm,
            tile_mask_path=tile_mask_path, tile_confmap_path=tile_conf_path,
            precleaned_slice_cleanup=bool(angle_variant_streaming_cleanup_active),
            processing_shape=result_shape,
            threshold_plane_shape=tuple(int(v) for v in task.get(
                'threshold_plane_shape', view_processing_volume_shape(view, int(task.get('out_size', args.imgsz)))[-2:]
            )),
        )
        fut = tile_dense_retirement_executor.submit(
            postprocess_tile_volume_after_inference, ptask,
            view=view, min_conf=float(args.min_conf), min_radius=float(args.min_radius),
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(tile_dense_retirement_slice_workers),
            sparse_retire_dir=temp_dir,
        )
        tile_cleanup_futures[fut] = (model_name_s, str(view.name), str(tile_job.config_id), str(tile_job.tile_id))

    def _announce_process_inference_drain_if_complete() -> None:
        nonlocal gpu_inference_drained_at, gpu_inference_drain_announced
        if int(gpu_worker_results_collected) < int(gpu_worker_total_tasks):
            return
        if gpu_worker_process_active:
            _set_main_process_gpu_inference_priority_active(False)
        _restore_parent_post_inference_affinity()
        if bool(gpu_inference_drain_announced):
            return
        gpu_inference_drain_announced = True
        gpu_inference_drained_at = time.perf_counter()
        runtime_telemetry().gauge('pipeline.phase', 'inference_drained_scheduler_tail')
        sink_now = nrrd_layer_sink()
        if sink_now is not None:
            nrrd_done_now, nrrd_total_now = sink_now.progress_counts()
        else:
            nrrd_done_now, nrrd_total_now = 0, 0
        waiting_parent_tiles_now = sum(
            len(value) for value in postprocessed_tiles_waiting_by_parent.values()
        )
        waiting_bridge_tiles_now = sum(
            len(value) for value in residual_tiles_waiting_by_parent.values()
        )
        gpu_done = int(sum(gpu_worker_results_by_id.values()))
        cpu_done = int(sum(cpu_worker_results_by_id.values()))
        hybrid_gpu_done = int(len(gpu_worker_cpu_assist_completed_task_ids))
        hybrid_gpu_frames = int(sum(
            int(gpu_worker_tasks_by_id[task_id].get('slice_count', 0))
            for task_id in gpu_worker_cpu_assist_completed_task_ids
            if task_id in gpu_worker_tasks_by_id
        ))
        hybrid_parents = {
            parent
            for task_obj in gpu_worker_tasks_by_id.values()
            if bool(task_obj.get('hybrid_cpu_eligible_origin', False))
            for parent in [_gpu_worker_fullframe_parent_key(task_obj)]
            if parent is not None
        }
        unresolved_hybrid_parents = sorted(
            parent for parent in hybrid_parents
            if parent not in hybrid_view_mode_by_parent
        )
        if unresolved_hybrid_parents:
            raise RuntimeError(
                'inference drained with unresolved hybrid full-frame contracts: '
                f'{unresolved_hybrid_parents}'
            )
        hybrid_contract_counts = Counter(
            str(hybrid_view_mode_by_parent[parent]) for parent in hybrid_parents
        )
        print('\n=== Process-local inference queue drained; scheduler postprocessing continues ===')
        drained_backend_notes: List[str] = []
        if gpu_worker_process_active:
            drained_backend_notes.append(
                'CUDA devices are released for eligible output/backprojection stages'
            )
        if cpu_worker_process_active:
            drained_backend_notes.append('OpenVINO workers are idle until shutdown')
        print(
            f'Inference tasks completed={int(gpu_worker_results_collected)}/'
            f'{int(gpu_worker_total_tasks)} (GPU={gpu_done}, CPU={cpu_done}); '
            f'frames completed={int(gpu_frames_completed_total + cpu_frames_completed_total)} '
            f'(GPU={int(gpu_frames_completed_total)}, CPU={int(cpu_frames_completed_total)}; '
            f'CPU-eligible GPU/CPU={int(hybrid_gpu_frames_completed_total)}/'
            f'{int(hybrid_cpu_frames_completed_total)}; active-view CUDA assist='
            f'{hybrid_gpu_done} task(s)/{hybrid_gpu_frames} frame(s)); '
            f'hybrid contracts D1={int(hybrid_contract_counts.get("d1_owner", 0))}, '
            f'direct_union={int(hybrid_contract_counts.get("direct_union", 0))}; '
            f'parent_postprocess={len(view_processing_futures)}, '
            f'tile_dense_results={len(gpu_worker_tile_dense_result_reservations)}/'
            f'{gpu_worker_tile_dense_result_task_limit} task(s), '
            f'{gpu_worker_tile_dense_result_bytes_reserved / GIB:.1f}/'
            f'{gpu_worker_tile_dense_result_limit / GIB:.1f}GiB, '
            f'tile_cleanup={len(tile_cleanup_futures)}, '
            f'tile_parent_gate={len(tile_parent_gate_futures)}, '
            f'tile_bridge_gate={len(tile_bridge_gate_futures)}, '
            f'tile_consolidation={len(tile_consolidation_futures)}, '
            f'tile_parent_finalization={len(tile_parent_finalization_futures)}, '
            f'waiting_tiles_for_parent={int(waiting_parent_tiles_now)}, '
            f'waiting_residuals_for_bridge={int(waiting_bridge_tiles_now)}, '
            f'NRRD writes={int(nrrd_done_now)}/{int(nrrd_total_now)}. '
            + '; '.join(drained_backend_notes) + '.'
        )
        if hybrid_parents:
            runtime_telemetry().gauge(
                'hybrid.frames.gpu', int(hybrid_gpu_frames_completed_total),
            )
            runtime_telemetry().gauge(
                'hybrid.frames.cpu', int(hybrid_cpu_frames_completed_total),
            )
            print('Hybrid CPU-eligible frame split by view:')
            ordered_hybrid_parents = list(hybrid_cpu_reserved_parents) + sorted(
                parent for parent in hybrid_parents
                if parent not in hybrid_cpu_reserved_parent_set
            )
            for parent in ordered_hybrid_parents:
                counts = hybrid_view_frames_by_backend.get(parent, Counter())
                task_counts = hybrid_view_tasks_by_backend.get(parent, Counter())
                contract = str(hybrid_view_mode_by_parent.get(parent, 'unresolved'))
                reservation = (
                    f'reserved#{hybrid_cpu_reservation_rank_by_parent[parent] + 1}'
                    if parent in hybrid_cpu_reservation_rank_by_parent else
                    'unreserved'
                )
                print(
                    f'  {parent[0]}/{parent[1]}: '
                    f'CPU={int(counts.get("cpu", 0))} frame(s)/'
                    f'{int(task_counts.get("cpu", 0))} task(s), '
                    f'GPU={int(counts.get("gpu", 0))} frame(s)/'
                    f'{int(task_counts.get("gpu", 0))} task(s), '
                    f'contract={contract}, {reservation}'
                )
        if cpu_worker_process_active:
            final_idle_reason = hybrid_cpu_idle_reason_last or _describe_hybrid_cpu_idle_reason()
            print(f'OpenVINO last idle reason: {final_idle_reason}.')

    def _process_one_worker_result(msg: Dict[str, object]) -> None:
        nonlocal gpu_worker_results_collected
        nonlocal gpu_inference_drained_at, gpu_inference_drain_announced
        mtype = str(msg.get('type'))
        worker_kind = str(msg.get('worker_kind', 'gpu')).strip().lower()
        if mtype == 'ready':
            if worker_kind == 'cpu':
                ready_cpu_index = int(msg.get('cpu_index', -1))
                if ready_cpu_index not in cpu_worker_ready_details_by_id:
                    ready_details = {
                        'precision': str(msg.get('precision')),
                        'requests': int(msg.get('requests', 0) or 0),
                        'threads': int(msg.get('threads', 0) or 0),
                        'input_element_type': str(msg.get('input_element_type')),
                        'model_int8_quantized': bool(msg.get('model_int8_quantized')),
                        'class_count': msg.get('class_count'),
                        'amx_tile': bool(msg.get('amx_tile')),
                        'amx_bf16': bool(msg.get('amx_bf16')),
                        'amx_int8': bool(msg.get('amx_int8')),
                        'openvino_capabilities': list(msg.get('openvino_capabilities') or ()),
                        'model_xml': str(msg.get('model_xml')),
                    }
                    cpu_worker_ready_details_by_id[ready_cpu_index] = ready_details
                    spec_notes.append(
                        f'v17 OpenVINO CPU instance {ready_cpu_index}: resolved precision='
                        f'{ready_details["precision"]}, requests={ready_details["requests"]}, '
                        f'threads={ready_details["threads"]}, input={ready_details["input_element_type"]}, '
                        f'INT8-export={ready_details["model_int8_quantized"]}, '
                        f'classes={ready_details["class_count"]}, '
                        f'AMX(tile/bf16/int8)={ready_details["amx_tile"]}/'
                        f'{ready_details["amx_bf16"]}/{ready_details["amx_int8"]}, '
                        f'capabilities={ready_details["openvino_capabilities"]}, '
                        f'model={ready_details["model_xml"]}.'
                    )
                    runtime_telemetry().gauge(
                        f'inference.cpu_instance.{ready_cpu_index}.ready', ready_details,
                    )
                print(
                    f"OpenVINO worker ready: instance {msg.get('cpu_index')} "
                    f"(pid {msg.get('pid')}, precision={msg.get('precision')}, "
                    f"requests={msg.get('requests')}, threads={msg.get('threads')}, "
                    f"input={msg.get('input_element_type')}, INT8-export={msg.get('model_int8_quantized')}, "
                    f"classes={msg.get('class_count')}, "
                    f"AMX(tile/bf16/int8)={msg.get('amx_tile')}/{msg.get('amx_bf16')}/{msg.get('amx_int8')}, "
                    f"OpenVINO capabilities={msg.get('openvino_capabilities')}, "
                    f"model={msg.get('model_xml')})."
                )
            else:
                print(f"GPU worker ready: device cuda:{msg.get('gpu_index')} (pid {msg.get('pid')}).")
            return
        if mtype == 'fatal':
            backend_label = (
                f"OpenVINO CPU instance {msg.get('cpu_index')}"
                if worker_kind == 'cpu' else f"GPU device {msg.get('gpu_index')}"
            )
            raise RuntimeError(
                f"{backend_label} failed to initialize: "
                f"{msg.get('error')}\n{msg.get('traceback')}"
            )
        if worker_kind == 'cpu':
            worker_id = int(msg.get('cpu_index', -1))
            task_id = int(msg.get('task_id', -1))
            predicted = float(cpu_worker_task_predicted_seconds_by_id.pop(task_id, 0.0))
            cpu_worker_predicted_load_by_id[worker_id] = max(
                0.0,
                float(cpu_worker_predicted_load_by_id.get(worker_id, 0.0)) - predicted,
            )
            cpu_worker_results_by_id[worker_id] = int(
                cpu_worker_results_by_id.get(worker_id, 0)
            ) + 1
            gpu_worker_results_collected += 1
            if not bool(msg.get('ok')):
                raise RuntimeError(
                    f"OpenVINO worker task {task_id} failed on CPU instance {worker_id}: "
                    f"{msg.get('error')}\n{msg.get('traceback')}"
                )
            task = gpu_worker_tasks_by_id[int(task_id)]
            stats = dict(msg.get('stats') or {})
            _update_cpu_worker_cost(task, stats)
            _record_backend_frame_completion(task, 'cpu')
            runtime_telemetry().add('inference.cpu_tasks_completed', 1)
            runtime_telemetry().add(
                'inference.cpu_frames_completed', int(task.get('slice_count', 0)),
            )
            # Apply the result first so the final lease can close this reservation and make
            # the next reserved parent visible before the just-freed OpenVINO worker refills.
            if str(task['kind']) == 'fullframe':
                _handle_fullframe_worker_result(task, stats)
            else:
                _handle_tile_worker_result(task, stats)
            _dispatch_inference_windows(_gpu_worker_fullframe_parent_key(task))
            _announce_process_inference_drain_if_complete()
            return

        if mtype == 'compute_released':
            worker_id = int(msg.get('gpu_index', -1))
            task_id = int(msg.get('task_id', -1))
            if task_id not in gpu_worker_compute_released_task_ids:
                gpu_worker_compute_released_task_ids.add(task_id)
                _main_process_gpu_stage_finish_inference(worker_id)
                gpu_worker_compute_completed_by_id[worker_id] = int(
                    gpu_worker_compute_completed_by_id.get(worker_id, 0)
                ) + 1
                predicted = float(gpu_worker_task_predicted_seconds_by_id.pop(task_id, 0.0))
                gpu_worker_predicted_load_by_id[worker_id] = max(
                    0.0, float(gpu_worker_predicted_load_by_id.get(worker_id, 0.0)) - predicted,
                )
                task_for_cost = gpu_worker_tasks_by_id.get(task_id)
                if isinstance(task_for_cost, dict):
                    release_stats = dict(msg.get('stats') or {})
                    _update_gpu_worker_cost(task_for_cost, release_stats)
                    _release_d1_owner_if_complete(task_for_cost, worker_id, release_stats)
            gpu_worker_cpu_assist_inflight_task_ids.discard(int(task_id))
            # Refill immediately; final result publication may still be copying several
            # GiB over PCIe or committing metadata on a retirement lane.
            task = gpu_worker_tasks_by_id.get(task_id)
            preferred = _gpu_worker_fullframe_parent_key(task) if isinstance(task, dict) else None
            _dispatch_inference_windows(preferred)
            _refresh_gpu_aux_interpolation_leases()
            return
        if mtype == 'aux_result':
            # Route the targeted worker result to its waiting interpolation caller.  Once the
            # lease is free, newly ready inference can immediately revoke it and dispatch.
            worker_id = int(msg.get('gpu_index', -1))
            aux_pool = gpu_worker_aux_interpolation_pool()
            if aux_pool is not None:
                aux_pool.complete(
                    int(msg.get('task_id', -1)),
                    worker_id,
                    bool(msg.get('ok')),
                    msg.get('stats') if isinstance(msg.get('stats'), dict) else None,
                    str(msg.get('error') or '') + ('\n' + str(msg.get('traceback')) if msg.get('traceback') else ''),
                )
            _dispatch_inference_windows()
            _refresh_gpu_aux_interpolation_leases()
            return
        worker_id = int(msg.get('gpu_index', -1))
        task_id = int(msg.get('task_id', -1))
        if task_id not in gpu_worker_compute_released_task_ids:
            gpu_worker_compute_released_task_ids.add(task_id)
            _main_process_gpu_stage_finish_inference(worker_id)
            gpu_worker_compute_completed_by_id[worker_id] = int(
                gpu_worker_compute_completed_by_id.get(worker_id, 0)
            ) + 1
            predicted = float(gpu_worker_task_predicted_seconds_by_id.pop(task_id, 0.0))
            gpu_worker_predicted_load_by_id[worker_id] = max(
                0.0, float(gpu_worker_predicted_load_by_id.get(worker_id, 0.0)) - predicted,
            )
            task_for_cost = gpu_worker_tasks_by_id.get(task_id)
            if isinstance(task_for_cost, dict):
                _update_gpu_worker_cost(task_for_cost, dict(msg.get('stats') or {}))
        gpu_worker_cpu_assist_inflight_task_ids.discard(int(task_id))
        gpu_worker_results_collected += 1
        gpu_worker_results_by_id[worker_id] = int(gpu_worker_results_by_id.get(worker_id, 0)) + 1
        if not bool(msg.get('ok')):
            raise RuntimeError(
                f"GPU worker task {msg.get('task_id')} failed on device {msg.get('gpu_index')}: "
                f"{msg.get('error')}\n{msg.get('traceback')}"
            )
        task = gpu_worker_tasks_by_id[int(task_id)]
        stats = dict(msg.get('stats') or {})
        _record_backend_frame_completion(task, 'gpu')
        if bool(task.get('hybrid_gpu_assist_dispatched', False)):
            gpu_worker_cpu_assist_completed_task_ids.add(int(task_id))
            runtime_telemetry().add('hybrid.gpu_assist_tasks_completed', 1)
            runtime_telemetry().add(
                'hybrid.gpu_assist_frames_completed', int(task.get('slice_count', 0)),
            )
        _release_d1_owner_if_complete(task, worker_id, stats)
        # Refill before scheduler-side memmap union/postprocess so a worker does not wait
        # behind CPU handling of the result that just freed its window slot. Prefer the
        # current parent when its last unissued lease can unlock postprocessing.
        _dispatch_inference_windows(_gpu_worker_fullframe_parent_key(task))
        if str(task['kind']) == 'fullframe':
            _handle_fullframe_worker_result(task, stats)
        else:
            _handle_tile_worker_result(task, stats)
        # A GPU may publish the final assisted direct-union lease. Refill once more after
        # result handling so OpenVINO can immediately open the next reservation.
        _dispatch_inference_windows(_gpu_worker_fullframe_parent_key(task))
        _announce_process_inference_drain_if_complete()
        _refresh_gpu_aux_interpolation_leases()

    # push drain — a transport-only daemon thread blocks on the process
    # result queue and hands messages to the main thread through a local deque + wake
    # event. Handlers still run ONLY on the main thread (they mutate scheduler state and
    # raise scheduler-fatal errors); completed futures set the same event through one-time
    # done-callbacks, so one event wait covers both wake sources. In push mode the drainer
    # thread OWNS the mp queue end (a second concurrent get would race it).
    push_drain_active = bool(
        inference_worker_process_active and gpu_result_queue is not None and scheduler_push_drain_enabled()
    )
    scheduler_wake = threading.Event()
    _set_main_process_gpu_stage_wake_callback(scheduler_wake.set)
    push_drain_stop = threading.Event()
    pushed_worker_results: deque = deque()
    _wake_hooked_futures: 'weakref.WeakSet' = weakref.WeakSet()

    def _wake_scheduler(_fut: object = None) -> None:
        scheduler_wake.set()

    def _hook_scheduler_wake(futures_list: Sequence[Future]) -> None:
        # add_done_callback on an already-done future fires synchronously, so hooking is
        # race-free: a completion between collection and wait still sets the event.
        for fut in futures_list:
            if fut not in _wake_hooked_futures:
                _wake_hooked_futures.add(fut)
                fut.add_done_callback(_wake_scheduler)

    def _push_drain_pump() -> None:
        while not push_drain_stop.is_set():
            try:
                msg = gpu_result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            pushed_worker_results.append(msg)
            scheduler_wake.set()

    if push_drain_active:
        push_drain_thread = threading.Thread(
            target=_push_drain_pump,
            name='inference-result-push-drain',
            daemon=True,
        )
        _run_resources().track_thread(push_drain_thread, push_drain_stop)
        push_drain_thread.start()
        print(
            'Scheduler push drain active (v13.3.8 G1; results handled the instant they '
            'arrive; YOLO_TTA_SCHEDULER_PUSH_DRAIN=0 restores polling).'
        )

    def _drain_process_inference_results() -> None:
        if gpu_result_queue is None:
            return
        if push_drain_active:
            while pushed_worker_results:
                _process_one_worker_result(pushed_worker_results.popleft())
            return
        while True:
            try:
                msg = gpu_result_queue.get_nowait()
            except queue.Empty:
                break
            _process_one_worker_result(msg)

    def _wait_for_one_process_result(timeout: float) -> None:
        if gpu_result_queue is None:
            return
        if push_drain_active:
            if not pushed_worker_results:
                scheduler_wake.wait(timeout=float(timeout))
                scheduler_wake.clear()
            _drain_process_inference_results()
            return
        try:
            msg = gpu_result_queue.get(timeout=float(timeout))
        except queue.Empty:
            return
        _process_one_worker_result(msg)

    def _process_inference_outstanding() -> bool:
        return bool(
            inference_worker_process_active
            and gpu_worker_results_collected < gpu_worker_total_tasks
        )

    def _check_inference_workers_alive() -> None:
        """Fail fast when any selected backend process exits before global drain."""
        if parent_affinity_monitor_errors:
            raise RuntimeError(
                f'Parent/OpenVINO CPU-affinity isolation failed: {parent_affinity_monitor_errors[0]}'
            ) from parent_affinity_monitor_errors[0]
        aux_pool = gpu_worker_aux_interpolation_pool()
        aux_outstanding = int(aux_pool.outstanding()) if aux_pool is not None else 0
        if not _process_inference_outstanding() and aux_outstanding <= 0:
            return

        remaining = max(0, int(gpu_worker_total_tasks - gpu_worker_results_collected))
        for backend_label, processes in (
            ('GPU', gpu_worker_processes),
            ('OpenVINO CPU', cpu_worker_processes),
        ):
            for proc in processes:
                if proc.is_alive():
                    continue
                reason = (
                    f'{backend_label} worker {getattr(proc, "name", "?")} exited unexpectedly '
                    f'(exitcode={getattr(proc, "exitcode", None)}) with '
                    f'{remaining} inference result(s) and '
                    f'{aux_outstanding} GPU-worker auxiliary interpolation pass(es) still outstanding.'
                )
                if aux_pool is not None:
                    # Unblock parent-postprocess threads waiting on a GPU auxiliary pass
                    # before raising, so shutdown cannot deadlock on their futures.
                    aux_pool.mark_failed(reason)
                raise RuntimeError(reason)

    def _reap_inference_worker_process(proc: object, backend_label: str) -> None:
        try:
            proc.join(timeout=30)
        except Exception:
            pass
        if not proc.is_alive():
            return
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=10)
        except Exception:
            pass
        if not proc.is_alive() or not hasattr(proc, 'kill'):
            return
        try:
            proc.kill()
            # Never join unbounded after SIGKILL. A worker wedged in uninterruptible
            # kernel sleep may not be reaped, and an unbounded join can escalate a job
            # failure into a drained SLURM node.
            proc.join(timeout=30)
        except Exception:
            pass
        if proc.is_alive():
            print(
                f'WARNING: {backend_label} worker pid={getattr(proc, "pid", "?")} survived '
                f'SIGKILL after 30s and is being abandoned. It is most likely stuck in '
                f'D state; this node may need manual cleanup before it accepts another job.',
                flush=True,
            )

    def _shutdown_inference_worker_processes() -> None:
        processes = list(gpu_worker_processes) + list(cpu_worker_processes)
        if not processes:
            return
        # On abnormal paths, fail any GPU-worker auxiliary interpolation waiters before
        # stopping the backend processes. mark_failed also rejects racing submissions.
        aux_pool = gpu_worker_aux_interpolation_pool()
        if aux_pool is not None:
            aux_pool.mark_failed('Inference worker processes are shutting down')
            set_gpu_worker_aux_interpolation_pool(None)
        for task_queue in list(gpu_task_queues.values()) + list(cpu_task_queues.values()):
            try:
                task_queue.put(None)
            except Exception:
                pass
        for proc in gpu_worker_processes:
            _reap_inference_worker_process(proc, 'GPU')
        for proc in cpu_worker_processes:
            _reap_inference_worker_process(proc, 'OpenVINO CPU')

    try:
        _pump_prediction_volume_build_queue()
        while True:
            _drain_completed_prediction_volume_futures()
            _drain_completed_prediction_accumulation_futures()
            _drain_completed_background_futures()
            _pump_prediction_volume_build_queue()
            _warmup_ready_prediction_sources()

            if inference_worker_process_active:
                # Both backends publish one common result contract. Drain it before CPU-side
                # postprocessing and fail fast if either a CUDA or OpenVINO process disappeared.
                _drain_process_inference_results()
                _check_inference_workers_alive()

            if (not inference_worker_process_active) and ready_fullframe:
                view, job, prediction_ref = ready_fullframe.popleft()
                print(f"Inferencing full-frame in-memory volume: {view.name}/{job.aug_id}")
                view_mask_shape = view_processing_volume_shape(view, int(args.imgsz))
                processing_affine = output_to_view_processing_affine(
                    view, job.aff.M_out_to_src, int(args.imgsz),
                )
                processing_min_radius = view_processing_min_radius(
                    view, float(args.min_radius), view_mask_shape[-2:],
                )
                try:
                    for model_name, yolo in yolo_models:
                        _ensure_baseline_workspaces(str(model_name), view)
                        if bool(async_prediction_accumulation_active):
                            handle = predict_in_memory_volume_and_submit_accumulation(
                                model=yolo,
                                prediction_volume=prediction_ref,
                                num_frames=view.num_slices,
                                out_size=args.imgsz,
                                cfg=pred_cfg,
                                view_union_mm=baseline_union_by_model_view[(model_name, view.name)],
                                view_confmap_mm=baseline_confmap_by_model_view[(model_name, view.name)],
                                M_out_to_native=processing_affine,
                                native_h=int(view_mask_shape[1]),
                                native_w=int(view_mask_shape[2]),
                                postprocess_executor=prediction_result_executor,
                                streaming_cleanup_enabled=bool(angle_variant_streaming_cleanup_active),
                                streaming_cleanup_min_conf=float(args.min_conf),
                                streaming_cleanup_min_radius=float(processing_min_radius),
                                slice_locks=baseline_slice_locks_by_model_view.get((str(model_name), str(view.name))),
                            )
                            _submit_prediction_accumulation_join(handle, {
                                'kind': 'fullframe',
                                'model_name': str(model_name),
                                'view': view,
                                'job': job,
                                'yolo': yolo,
                            })
                        else:
                            pred_stats = predict_in_memory_volume_and_accumulate(
                                model=yolo,
                                prediction_volume=prediction_ref,
                                num_frames=view.num_slices,
                                out_size=args.imgsz,
                                cfg=pred_cfg,
                                view_union_mm=baseline_union_by_model_view[(model_name, view.name)],
                                view_confmap_mm=baseline_confmap_by_model_view[(model_name, view.name)],
                                M_out_to_native=processing_affine,
                                native_h=int(view_mask_shape[1]),
                                native_w=int(view_mask_shape[2]),
                                postprocess_workers=predict_postprocess_workers,
                                streaming_cleanup_enabled=bool(angle_variant_streaming_cleanup_active),
                                streaming_cleanup_min_conf=float(args.min_conf),
                                streaming_cleanup_min_radius=float(processing_min_radius),
                                slice_locks=baseline_slice_locks_by_model_view.get((str(model_name), str(view.name))),
                            )
                            if offload_between_jobs_enabled():
                                offload_yolo_from_gpu(yolo)
                            view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                            remaining_key = (model_name, view.name)
                            fullframe_remaining[remaining_key] = int(fullframe_remaining.get(remaining_key, 0)) - 1
                            if int(fullframe_remaining.get(remaining_key, 0)) == 0:
                                _submit_view_prepare(model_name, view)
                finally:
                    close_prediction_volume_ref(prediction_ref, keep_temp=bool(keep_temp_artifacts))
                    _pump_prediction_volume_build_queue()
                continue

            if (not inference_worker_process_active) and ready_tile_infer:
                model_name, view, tile_job, prediction_ref = ready_tile_infer.popleft()
                ready_key = (str(model_name), str(view.name), str(tile_job.tile_id))
                if ready_key in tile_inference_done:
                    close_prediction_volume_ref(prediction_ref, keep_temp=bool(keep_temp_artifacts))
                    continue
                if prediction_ref is None:
                    # this model's own single-use source for the tile (see the drain).
                    prediction_ref = _maybe_eager_stage_prediction_ref(
                        _make_streaming_tile_ref(view, tile_job)
                    )

                print(f"Inferencing tile in-memory volume: {model_name}/{view.name}/{tile_job.tile_id}")
                parent_processing_shape = view_processing_volume_shape(view, int(args.imgsz))
                py0, py1, px0, px1 = (int(v) for v in tile_job.parent_crop)
                tile_shape = (int(view.num_slices), int(py1 - py0), int(px1 - px0))
                tile_processing_affine = np.asarray(tile_job.M_out_to_crop, dtype=np.float32)
                tile_threshold_plane_shape = tuple(int(v) for v in parent_processing_shape[-2:])
                tile_processing_min_radius = view_processing_min_radius(
                    view, float(args.min_radius), tile_threshold_plane_shape,
                )
                tile_mask_path = temp_dir / 'tile_volumes' / model_name / view.name / f'{tile_job.tile_id}.u8.dat'
                tile_conf_path = temp_dir / 'tile_volumes' / model_name / view.name / f'{tile_job.tile_id}.confmap.u8.dat'
                tile_mask_path.parent.mkdir(parents=True, exist_ok=True)

                tile_mask_mm = allocate_workspace_array(
                    shape=tile_shape,
                    dtype=np.uint8,
                    path=tile_mask_path,
                    desc=f'{model_name}/{view.name}/{tile_job.tile_id} raw tile volume',
                    prefer_memory=True,
                )
                if float(args.min_conf) > 0.0:
                    tile_conf_mm = allocate_workspace_array(
                        shape=tile_shape,
                        dtype=np.uint8,
                        path=tile_conf_path,
                        desc=f'{model_name}/{view.name}/{tile_job.tile_id} raw tile confidence workspace',
                        prefer_memory=True,
                    )
                    tile_conf_store_path: Optional[Path] = tile_conf_path
                else:
                    tile_conf_mm = None
                    tile_conf_store_path = None

                yolo = yolo_by_model_name[str(model_name)]
                try:
                    if bool(async_prediction_accumulation_active):
                        handle = predict_in_memory_volume_and_submit_accumulation(
                            model=yolo,
                            prediction_volume=prediction_ref,
                            num_frames=view.num_slices,
                            out_size=int(args.imgsz),
                            cfg=pred_cfg,
                            view_union_mm=tile_mask_mm,
                            view_confmap_mm=tile_conf_mm,
                            M_out_to_native=tile_processing_affine,
                            native_h=int(tile_shape[1]),
                            native_w=int(tile_shape[2]),
                            postprocess_executor=prediction_result_executor,
                            streaming_cleanup_enabled=bool(angle_variant_streaming_cleanup_active),
                            streaming_cleanup_min_conf=float(args.min_conf),
                            streaming_cleanup_min_radius=float(tile_processing_min_radius),
                        )
                        _submit_prediction_accumulation_join(handle, {
                            'kind': 'tile',
                            'model_name': str(model_name),
                            'view': view,
                            'tile_job': tile_job,
                            'tile_mask_mm': tile_mask_mm,
                            'tile_conf_mm': tile_conf_mm,
                            'tile_mask_path': tile_mask_path,
                            'tile_conf_path': tile_conf_store_path,
                            'threshold_plane_shape': tile_threshold_plane_shape,
                            'yolo': yolo,
                        })
                    else:
                        pred_stats = predict_in_memory_volume_and_accumulate(
                            model=yolo,
                            prediction_volume=prediction_ref,
                            num_frames=view.num_slices,
                            out_size=int(args.imgsz),
                            cfg=pred_cfg,
                            view_union_mm=tile_mask_mm,
                            view_confmap_mm=tile_conf_mm,
                            M_out_to_native=tile_processing_affine,
                            native_h=int(tile_shape[1]),
                            native_w=int(tile_shape[2]),
                            postprocess_workers=predict_postprocess_workers,
                            streaming_cleanup_enabled=bool(angle_variant_streaming_cleanup_active),
                            streaming_cleanup_min_conf=float(args.min_conf),
                            streaming_cleanup_min_radius=float(tile_processing_min_radius),
                        )
                        if offload_between_jobs_enabled():
                            offload_yolo_from_gpu(yolo)
                        tile_inference_done.add(ready_key)
                        view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                        if int(pred_stats.get('frames_with_predictions', 0)) <= 0:
                            close_memmap_array(tile_mask_mm)
                            close_memmap_array(tile_conf_mm)
                            if not keep_temp_artifacts:
                                try:
                                    tile_mask_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                                if tile_conf_path is not None:
                                    try:
                                        tile_conf_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                            _mark_tile_complete(
                                str(model_name), str(view.name),
                                str(tile_job.config_id), str(tile_job.tile_id),
                            )
                            continue

                        task = TilePostprocessTask(
                            model_name=str(model_name),
                            view_name=str(view.name),
                            aug_id=str(view.tta_aug_id),
                            angle_deg=float(view.tta_angle_deg),
                            config_id=str(tile_job.config_id),
                            tile_id=str(tile_job.tile_id),
                            parent_crop=tuple(int(v) for v in tile_job.parent_crop),
                            tile_mask_mm=tile_mask_mm,
                            tile_confmap_mm=tile_conf_mm,
                            tile_mask_path=tile_mask_path,
                            tile_confmap_path=tile_conf_store_path,
                            precleaned_slice_cleanup=bool(angle_variant_streaming_cleanup_active),
                            processing_shape=tile_shape,
                            threshold_plane_shape=tile_threshold_plane_shape,
                        )
                        fut = tile_dense_retirement_executor.submit(
                            postprocess_tile_volume_after_inference,
                            task,
                            view=view,
                            min_conf=float(args.min_conf),
                            min_radius=float(args.min_radius),
                            keep_temp=bool(keep_temp_artifacts),
                            slice_workers=int(tile_dense_retirement_slice_workers),
                            sparse_retire_dir=temp_dir,
                        )
                        tile_cleanup_futures[fut] = (str(model_name), str(view.name), str(tile_job.config_id), str(tile_job.tile_id))
                finally:
                    close_prediction_volume_ref(prediction_ref, keep_temp=bool(keep_temp_artifacts))
                    _pump_prediction_volume_build_queue()
                continue

            waitables: List[Future] = list(pending_prediction_volume_futures)
            waitables.extend(list(prediction_accumulation_futures.keys()))
            waitables.extend(list(view_processing_futures.keys()))
            waitables.extend(list(tile_cleanup_futures.keys()))
            waitables.extend(list(tile_parent_gate_futures.keys()))
            waitables.extend(list(tile_bridge_gate_futures.keys()))
            waitables.extend(list(tile_consolidation_futures.keys()))
            waitables.extend(list(tile_parent_finalization_futures.keys()))
            if not waitables:
                _drain_parent_mask_ready_events()
                _flush_ready_postprocessed_tiles()
                _flush_ready_residual_tiles()
                _pump_prediction_volume_build_queue()
                if inference_worker_process_active and _process_inference_outstanding():
                    # Backend processes are still inferencing but no CPU-side future is pending;
                    # block briefly on the common result queue so the loop wakes on the next result.
                    _wait_for_one_process_result(timeout=0.5)
                    _check_inference_workers_alive()
                    continue
                scheduler_quiescent = bool(
                    not pending_prediction_build_jobs and
                    not pending_prediction_volume_futures and
                    not prediction_accumulation_futures and
                    not _process_inference_outstanding() and
                    not ready_fullframe and
                    not ready_tile_infer and
                    not tile_parent_gate_futures and
                    not tile_bridge_gate_futures and
                    not tile_cleanup_futures and
                    not tile_consolidation_futures and
                    not tile_parent_finalization_futures and
                    not view_processing_futures
                )
                if scheduler_quiescent:
                    waiting_parent = {
                        key: sorted(waiting)
                        for key, waiting in postprocessed_tiles_waiting_by_parent.items()
                        if waiting
                    }
                    waiting_bridge = {
                        key: sorted(waiting)
                        for key, waiting in residual_tiles_waiting_by_parent.items()
                        if waiting
                    }
                    incomplete_tiles = {
                        key: {
                            'completed': len(tile_completed_by_parent.get(key, set())),
                            'expected': int(expected),
                        }
                        for key, expected in tile_expected_by_parent.items()
                        if len(tile_completed_by_parent.get(key, set())) < int(expected)
                    }
                    if waiting_parent or waiting_bridge or incomplete_tiles:
                        raise RuntimeError(
                            'Tile scheduler became quiescent with unresolved two-stage gate '
                            f'dependencies: waiting_for_parent={waiting_parent}, '
                            f'waiting_for_bridge={waiting_bridge}, '
                            f'incomplete_tiles={incomplete_tiles}'
                        )
                    break
                continue
            _log_scheduler_wait_state()
            if push_drain_active:
                # sleep on the shared wake event — worker results and future
                # completions both set it, so the loop wakes the instant either happens.
                # The heartbeat only bounds the worker-liveness re-check cadence.
                _hook_scheduler_wake(waitables)
                scheduler_wake.wait(timeout=float(scheduler_push_drain_heartbeat_seconds()))
                scheduler_wake.clear()
            else:
                # Poll the inference-worker result queue while CPU-side futures run, so completed
                # inference results are unioned without waiting for a future to finish.
                wait(waitables, timeout=(0.1 if inference_worker_process_active else None), return_when=FIRST_COMPLETED)

    finally:
        if sys.exc_info()[0] is not None:
            # (completion): the wait=True shutdowns below block on render
            # tasks parked in wait_for_volume_ready for still-running streaming
            # producers. Abort the producers FIRST on the error path, or a failure during
            # the decode-overlap window stalls teardown for the remaining multi-hour
            # decode before the exception can even propagate to __main__.
            abort_streaming_producers('scheduler error teardown')
            # (completion): queued eagerly pre-staged prediction sources
            # were never handed to a predict call, so nothing else closes them — their
            # producer threads would otherwise keep VRAM-staged batches and ledger
            # reservations alive (and spin) for the whole teardown.
            for _view, _job, ref in list(ready_fullframe):
                close_prediction_volume_ref(ref, keep_temp=bool(keep_temp_artifacts))
            ready_fullframe.clear()
            for _model_name, _view, _tile_job, ref in list(ready_tile_infer):
                close_prediction_volume_ref(ref, keep_temp=bool(keep_temp_artifacts))
            ready_tile_infer.clear()
        push_drain_stop.set()  # the transport thread exits within one 0.5 s tick
        _set_main_process_gpu_inference_priority_active(False)
        if inference_worker_process_active:
            _shutdown_inference_worker_processes()
        prediction_volume_executor.shutdown(wait=True)
        prediction_join_executor.shutdown(wait=True)
        prediction_result_executor.shutdown(wait=True)
        parent_postprocess_executor.shutdown(wait=True)
        tile_dense_retirement_executor.shutdown(wait=True)
        tile_postprocess_executor.shutdown(wait=True)
        # Error teardown can bypass the normal tile-result lifecycle. Every worker and
        # tile-postprocess future is quiescent now, so release any parent-owned memfd/disk
        # mappings that never reached sparse retirement without racing an active consumer.
        for tile_task_id in list(
            set(gpu_worker_tile_dense_result_reservations)
            | set(gpu_worker_tile_dense_result_memfd_reservations)
            | set(gpu_worker_tile_dense_result_reserved_at)
            | set(gpu_worker_tile_dense_result_workspaces)
        ):
            _release_tile_dense_result_task_id(
                int(tile_task_id), reason='scheduler teardown', refill=False,
            )
        if not bool(keep_temp_artifacts) and gpu_worker_result_dir.exists():
            # All worker processes, D2H publications, cleanup futures, and gate futures are
            # quiescent here. Remove any orphaned result pathname left by an exception or an
            # interrupted memfd fallback transaction instead of carrying it into final output.
            release_memfd_owners_under(gpu_worker_result_dir)
            shutil.rmtree(gpu_worker_result_dir, ignore_errors=True)
            if gpu_worker_result_dir.exists():
                print(
                    f'Warning: final inference-worker result scratch sweep could not remove '
                    f'{gpu_worker_result_dir}'
                )
        if prediction_render_executor is not None:
            try:
                prediction_render_executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                prediction_render_executor.shutdown(wait=True)
        set_interpolation_process_executor(None, 0)
        if interpolation_process_executor is not None:
            try:
                interpolation_process_executor.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                interpolation_process_executor.shutdown(wait=True)

        _restore_parent_post_inference_affinity()

        _set_main_process_gpu_pending_inference(False)
        _set_main_process_gpu_stage_wake_callback(None)
        _reset_main_process_gpu_stage_coordinator()

    _drain_completed_prediction_volume_futures()
    _drain_completed_prediction_accumulation_futures()
    _drain_completed_background_futures()

    post_inference_tail_started = (
        float(gpu_inference_drained_at)
        if gpu_inference_drained_at is not None
        else time.perf_counter()
    )
    runtime_telemetry().gauge('pipeline.phase', 'post_inference_final_assembly')
    active_layer_sink = nrrd_layer_sink()
    if active_layer_sink is not None:
        nrrd_done_at_boundary, nrrd_total_at_boundary = active_layer_sink.progress_counts()
    else:
        nrrd_done_at_boundary, nrrd_total_at_boundary = 0, 0
    print('\n=== Scheduler postprocessing drained; entering final assembly/output tail ===')
    print(
        f'Inference tasks completed={int(gpu_worker_results_collected)}/{int(gpu_worker_total_tasks)}; '
        f'queued NRRD writes completed={int(nrrd_done_at_boundary)}/{int(nrrd_total_at_boundary)}. '
        'Subsequent phase banners and telemetry identify every remaining tail stage.'
    )

    # The scheduler's finally block above has joined every GPU/OpenVINO inference worker and
    # interpolation executor. Only now may CPU-only tail stages reclaim the CPU reservations
    # held for live inference subprocesses. Keep
    # Separate budgets preserve the configured per-stage sizes when expansion is disabled.
    tail_slice_workers = int(
        tail_worker_budget if tail_budget_expansion_active else slice_postprocess_workers
    )
    tail_output_workers = int(
        tail_worker_budget if tail_budget_expansion_active else output_workers
    )
    tail_tile_slice_workers = int(
        tail_worker_budget if tail_budget_expansion_active else tile_slice_postprocess_workers
    )
    tail_output_frame_workers = int(
        max(
            1,
            _env_int(
                'YOLO_TTA_OUTPUT_FRAME_WORKERS',
                max(1, min(_cpu_count(), int(tail_output_workers))),
            ),
        )
        if tail_budget_expansion_active
        else output_frame_workers
    )
    if tail_budget_expansion_active and int(output_manager.max_workers) != int(tail_output_workers):
        # BackgroundOutputManager owns a fixed-size executor. Settle any earlier submissions
        # before replacing it; in the current schedule none are submitted before this boundary,
        # but waiting here keeps the transition correct if an earlier output is added later.
        output_manager.wait()
        output_manager = BackgroundOutputManager(max_workers=int(tail_output_workers))
        _run_resources().track_output_manager(output_manager)
    print(
        'v13.3.10 G7 post-inference CPU expansion: '
        f'slice workers={int(tail_slice_workers)}, output workers={int(tail_output_workers)}, '
        f'output frame workers={int(tail_output_frame_workers)}.'
    )
    if tail_budget_expansion_active:
        spec_notes.append(
            'v13.3.10 G7 tail CPU expansion: after all GPU/OpenVINO inference workers and interpolation '
            f'executors joined, strictly post-inference stages used worker_budget={int(tail_worker_budget)}; '
            'YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND=0 restores inference-phase tail sizing.'
        )
    else:
        spec_notes.append(
            'v13.3.10 G7 tail CPU expansion was disabled by '
            'YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND=0; legacy per-stage tail sizing was retained.'
        )

    if preprocess_streaming_active:
        if isinstance(volume_rgb, LazyProcessingCube) and not volume_rgb.materialized:
            # this barrier exists to settle active producers. An untouched
            # lazy cube has no producer and final source-geometry output reads the decoded
            # volume, so wait only for decode and preserve the no-cube fast path.
            print(
                'Ensuring streaming decode completed; deferred host cube remained unused '
                '(v13.3.17 C10 fast path).'
            )
            wait_for_volume_ready(input_volume_rgb)
        else:
            print('Ensuring streaming preprocessing producers have completed before final output/backprojection stages.')
            wait_for_volume_ready(volume_rgb)

    for cache_name, cache_mm in list(view_frame_caches.items()):
        close_memmap_array(cache_mm)
        cache_path = view_frame_cache_paths.get(cache_name)
        if not keep_temp_artifacts and cache_path is not None:
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass
    view_frame_caches.clear()
    view_frame_cache_paths.clear()

    if not bool(keep_temp_artifacts):
        swept_mkvs = purge_remaining_temporary_mkvs(temp_dir, keep_temp=False)
        if int(swept_mkvs) > 0:
            print(f'Final legacy temporary MKV sweep removed {int(swept_mkvs)} leftover file(s).')
        spec_notes.append(
            'No prediction MKVs are produced by the v12 in-memory path. A best-effort final sweep still removes '
            f'{int(swept_mkvs)} legacy/interrupted scratch MKV file(s) when YOLO_TTA_KEEP_TEMP is disabled.'
        )
    else:
        spec_notes.append('YOLO_TTA_KEEP_TEMP retained scratch artifacts; the v12.2.0 in-memory inference path itself does not create prediction MKVs.')

    # radial/tilted results are backprojected DIRECTLY into the
    # original source geometry (single resample) instead of the working geometry.
    source_output_shape_tyx = (int(input_T), int(input_H), int(input_W))
    final_backprojection_jobs: List[ViewBackprojectionQueueJob] = []
    fused_projected_layer_refs: List[NrrdLayerRef] = []  #
    for view in views:
        for model_name, _ in yolo_models:
            d1_base_ref = d1_layer_ref_by_parent.get((str(model_name), str(view.name)))
            d1_dense_orthogonal_additions_present = bool(
                str(view.name) in view_volumes_by_model.get(str(model_name), {})
            )
            if d1_base_ref is not None and d1_dense_orthogonal_additions_present:
                # Orthogonal dense D1 continuation contains additions only and takes the early
                # dense-view branch below. Contribute the already-source-space base exactly once.
                # Radial/Tilted D1 views instead reach the component-ref branch, which already
                # includes both this base and every projected addition layer.
                fused_projected_layer_refs.append(d1_base_ref)
            if view.name in view_volumes_by_model[model_name]:
                continue
            view_projected_layer_refs = [
                ref for ref in nrrd_layer_refs
                if str(getattr(ref, 'view_name', '')) == str(view.name)
            ]
            if view_projected_layer_refs and all(len(tuple(ref.shape)) == 3 for ref in view_projected_layer_refs):
                print(
                    f'Final {model_name}/{view.name}: contributing '
                    f'{len(view_projected_layer_refs)} orthogonal component layer(s) directly '
                    f'to the terminal output-geometry union (sparse-retirement path).'
                )
                # assemble_current_view_union_volume has an exact per-layer fallback when
                # G5 is disabled or working geometry already equals output geometry.  Feeding
                # refs directly in every case avoids recreating and retaining one dense volume
                # per retired view immediately before the final union.
                fused_projected_layer_refs.extend(view_projected_layer_refs)
                continue

            if (view.family != 'radial' and not is_tilted_view(view)):
                continue
            if view.family == 'radial':
                native_source = radial_native_output_by_model[model_name].get(view.name)
            else:
                native_source = tilted_native_output_by_model[model_name].get(view.name)
            if native_source is None:
                native_source = native_view_support_by_model[model_name].get(view.name)
            if native_source is None:
                continue
            final_backprojection_jobs.append(ViewBackprojectionQueueJob(
                model_name=str(model_name),
                view=view,
                native_source=native_source,
                out_path=temp_dir / 'view_volumes' / str(model_name) / f'{view.name}.u8.dat',
                desc=f'Backprojecting final {model_name}/{view.name}',
                min_radius=0.0,
                workers=1,
                out_shape_tyx=source_output_shape_tyx,
            ))

    if final_backprojection_jobs:
        # Run one set at a time with the full CPU fallback budget. Transverse Radial may use
        # GPU backprojection; all upright bases can use orientation-aware sink-only projection.
        per_backproject_workers = max(1, int(tail_slice_workers))
        final_backprojection_jobs = [
            ViewBackprojectionQueueJob(
                model_name=job.model_name,
                view=job.view,
                native_source=job.native_source,
                out_path=job.out_path,
                desc=job.desc,
                min_radius=job.min_radius,
                workers=int(per_backproject_workers),
                out_shape_tyx=job.out_shape_tyx,
            )
            for job in final_backprojection_jobs
        ]
        print(
            f'Final radial/tilted backprojection queue: tasks={len(final_backprojection_jobs)}, '
            f'max_active=1 CPU-only, per-set CPU workers={int(per_backproject_workers)}'
        )
        backproject_queue = HybridBackprojectionQueue(
            cpu_workers=int(per_backproject_workers),
        )
        for model_name_done, view_name_done, projected_volume in backproject_queue.run(final_backprojection_jobs):
            view_volumes_by_model[model_name_done][view_name_done] = projected_volume

    output_manager.reap_completed()

    # v16.4.0 keeps every TTA angle independent through cleanup, interpolation, tiling,
    # and NRRD emission. Collapse dense angle variants exactly once here, immediately before
    # the physical-view/global union. This restores the physical Cartesian keys required by
    # the terminal axis-aware assembler and retires redundant per-angle dense workspaces.
    retired_tta_volume_ids: set[int] = set()
    view_volumes_by_model = collapse_tta_variant_volumes_to_physical_views(
        view_volumes_by_model,
        views,
        workers=int(tail_slice_workers),
        retired_volume_ids=retired_tta_volume_ids,
    )
    release_unretained_volume_maps(
        (
            native_view_support_by_model,
            radial_native_output_by_model,
            tilted_native_output_by_model,
        ),
        view_volumes_by_model,
        already_retired_ids=retired_tta_volume_ids,
    )
    if bool(component_ref_dense_retirement_active):
        spec_notes.append(
            'v17.0.5 component-ref TTA retirement: every --angle variant remained logically '
            'independent through cleanup, interpolation, per-tile parent/bridge gating, and '
            'component-layer output. Non-tiled dense variants retired after an immutable terminal '
            'representation was materialized; tiled variants retired after consolidation. Requested '
            'NRRD component layers are reused when available; otherwise one private pathname-backed '
            'row-wise packed raw-bbox final-view layer is created and is not submitted as output. These refs contribute '
            'directly to the terminal union without rebuilding one dense volume per angle. '
            'YOLO_TTA_KEEP_TEMP=1 preserves dense diagnostics.'
        )
    else:
        spec_notes.append(
            'v16.4.0 TTA-angle finalization: every --angle variant remained independent through '
            'cleanup, interpolation, per-tile parent/bridge gating, and component-layer output; '
            'dense variants were OR-collapsed only at physical-view finalization immediately before '
            'the global view union. Component-ref dense retirement is disabled because '
            'YOLO_TTA_KEEP_TEMP=1 preserves dense diagnostics.'
        )

    print('\n=== Building final single-model view union after physical-view TTA collapse ===')
    # the union is assembled at ORIGINAL SOURCE dimensions.
    # Cartesian working stacks are restored working->source with one resample while merging;
    # radial/tilted volumes arrive already backprojected to source geometry. Void fill,
    # Gaussian smoothing (sigma in SOURCE voxels) and postprocessing keep_objects run at source dimensions.
    final_union_mm = assemble_final_union_after_view_union(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        out_path=temp_dir / 'final_union_volume.u8.dat',
        temp_dir=temp_dir,
        out_shape_tyx=source_output_shape_tyx,
        enable_3d_void_fill=bool(args.enable_3d_void_fill),
        keep_temp=bool(keep_temp_artifacts),
        prefer_memory=True,
        workers=tail_slice_workers,
        projected_layer_refs=fused_projected_layer_refs,
    )

    if int(args.centerline_filter_passes) > 0:
        discard_binary_volume_slice_metadata(final_union_mm)
    centerline_filter_stats: Dict[str, object] = apply_v14_centerline_filter_inplace(
        final_union_mm,
        model_name=str(model_name), temp_dir=temp_dir,
        passes=int(args.centerline_filter_passes),
        backend='embedded',
        radius_factor=float(args.centerline_radius_factor),
        temporal_context=int(args.centerline_temporal_context),
        automatic_removal_enabled=bool(args.centerline_auto_remove),
        surface_max_dim=int(args.centerline_surface_max_dim),
        surface_points=int(args.centerline_surface_points),
        timeout_seconds=float(args.centerline_timeout),
        workers=int(tail_slice_workers), keep_temp=bool(keep_temp_artifacts),
        nrrd_layer_refs=nrrd_layer_refs,
    )
    if bool(centerline_filter_stats.get('enabled', False)):
        pass_summaries = [
            (
                f"pass {int(record.get('pass_index', 0))}: backend={record.get('backend', 'unknown')}, "
                f"backend_allows_removal={bool(record.get('backend_automatic_removal_allowed', False))}, "
                f"anomaly_fraction={float(record.get('section_anomaly_fraction', 0.0)):.3f}, "
                f"reliability_guard={bool(record.get('section_reliability_guard_triggered', False))}, "
                f"events={int(record.get('longitudinal_events', 0))}, "
                f"removed_components={int(record.get('removed_components', 0))}, "
                f"removed_voxels={int(record.get('removed_voxels', 0))}, "
                f"watershed_voxels={int(record.get('watershed_voxels', 0))}, "
                f"marker_mode={record.get('marker_mode', 'unknown')}"
            )
            for record in centerline_filter_stats.get('passes', [])
        ]
        spec_notes.append(
            'Centerline post-union filter: pass 0 preserves the untouched union; '
            'the embedded backend uses exact 3D EDT on the block-max raster plus three-axis '
            'medial-ridge tracking; '
            f'X={float(args.centerline_radius_factor):g}; anomaly duration is uncapped; '
            f'automatic component removal requested={bool(args.centerline_auto_remove)}; '
            'protected or otherwise unsafe 2D components are marker-only; '
            f"stop={centerline_filter_stats.get('stop_reason', 'unknown')}. "
            + ('; '.join(pass_summaries) if pass_summaries else 'No filter pass completed.')
        )

    if bool(nrrd_layers_needed and gaussian_smoothing_enabled):
        pre_smoothing_ref = materialize_nrrd_global_layer(
            final_union_mm,
            model_name=str(model_name),
            source='global',
            mask_kind='union',
            pass_index=0,
            stage='pre_smoothing',
            description='Global union after all active view/tile layers, optional 3D void fill, and centerline filtering; input to Gaussian smoothing.',
            temp_dir=temp_dir,
            workers=int(tail_slice_workers),
        )
        if pre_smoothing_ref is not None:
            nrrd_layer_refs.append(pre_smoothing_ref)

    gaussian_smoothing_stats: Optional[Dict[str, object]] = None
    if bool(gaussian_smoothing_enabled):
        print('\n=== Applying Gaussian smoothing ===')
        discard_binary_volume_slice_metadata(final_union_mm)
        gaussian_smoothing_stats = apply_gaussian_smoothing_inplace(
            final_union_mm,
            sigma=float(gaussian_smoothing_sigma),
            passes=int(gaussian_smoothing_passes),
            temp_dir=temp_dir,
            keep_temp=bool(keep_temp_artifacts),
            prefer_memory=True,
            workers=tail_slice_workers,
            nrrd_layers=nrrd_layer_refs if bool(nrrd_layers_needed) else None,
            nrrd_model_name=str(model_name),
        )

    keep_objects_stats: Optional[Dict[str, int | float]] = None
    if int(args.keep_objects) > 0:
        print(f'\n=== Keeping largest {int(args.keep_objects)} final object(s) ===')
        keep_objects_stats = apply_keep_largest_objects_inplace(
            final_union_mm,
            int(args.keep_objects),
            temp_dir=temp_dir,
            keep_temp=bool(keep_temp_artifacts),
            prefer_memory=True,
            workers=tail_slice_workers,
        )

    final_output_mask_mm = final_union_mm
    output_volume_rgb = input_volume_rgb
    output_T, output_H, output_W = int(input_T), int(input_H), int(input_W)
    # the final union is already assembled at source dimensions,
    # so this tail restore is an identity no-op kept purely as a shape safety net.
    if tuple(int(v) for v in final_union_mm.shape) != (output_T, output_H, output_W):
        print('\n=== Restoring final mask to original input geometry for default outputs ===')
        final_output_mask_mm = restore_mask_volume_to_original_shape(
            final_union_mm,
            (output_T, output_H, output_W),
            temp_dir / 'final_union_original_geometry.u8.dat',
            workers=int(tail_slice_workers),
            prefer_memory=True,
        )
    if bool(nrrd_layers_needed):
        # Materialize the final single-layer NRRD here; the sink writes it in the background
        # while the remaining default outputs are produced. This checkpoint is not a
        # recomposition layer.
        final_output_ref = materialize_nrrd_global_layer(
            final_output_mask_mm,
            model_name=str(model_name),
            source='global',
            mask_kind='union',
            pass_index=0,
            stage='final_output_after_all_postprocessing',
            description='Final binary output after view/tile union, optional smoothing, optional keep_objects, and geometry restoration. Not intended for recomposition.',
            temp_dir=temp_dir,
            workers=int(tail_slice_workers),
            # nothing mutates the final volume after this point (overlay/video
            # writers only read it), and it stays open until after layer_sink.wait — the
            # sink streams it directly instead of a store encode + read-back.
            volume_is_immutable=True,
            keep_temp=bool(keep_temp_artifacts),
        )
        if final_output_ref is not None:
            nrrd_layer_refs.append(final_output_ref)

    final_output_volume_for_low_quality = output_volume_rgb
    final_paths: Dict[str, Path] = {}


    native_final_outputs_requested = bool(
        save_high_quality_enabled or save_binary_enabled or save_labels_enabled
    )
    if native_final_outputs_requested:
        print('\n=== Scheduling selected native-resolution outputs in background ===')
        final_output_paths, final_futures = collect_pipeline_output_futures(
            output_manager.executor,
            volume_rgb=output_volume_rgb,
            mask_u8=final_output_mask_mm,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_high_quality=bool(save_high_quality_enabled),
            save_binary_pattern_value='__DEFAULT__' if save_binary_enabled else None,
            save_labels_pattern_value='__DEFAULT__' if save_labels_enabled else None,
            tag=None,
            frame_workers=tail_output_frame_workers,
            show_progress=False,
            nrrd_temp_dir=temp_dir,
        )
        final_paths.update(final_output_paths)
        output_manager.submit(BackgroundOutputSubmission(
            label='selected native-resolution outputs',
            result_paths=final_output_paths,
            futures=final_futures,
            resources=[],
        ))

    if bool(low_quality_requested):
        print('\n=== Scheduling low-quality isotropic outputs in background ===')
        low_quality_paths, low_quality_futures = collect_low_quality_output_futures(
            output_manager.executor,
            volume_gray=final_output_volume_for_low_quality,
            mask_u8=final_output_mask_mm,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            downbin_specs=low_quality_downbin_specs,
            temp_dir=temp_dir,
            workers=tail_output_workers,
            show_progress=False,
        )
        final_paths.update(low_quality_paths)
        output_manager.submit(BackgroundOutputSubmission(
            label='low-quality outputs',
            result_paths=low_quality_paths,
            futures=low_quality_futures,
            resources=[],
        ))

    voxel_volume = None
    if bool(save_voxel_volume_enabled):
        voxel_counts = np.zeros((int(final_output_mask_mm.shape[0]),), dtype=np.int64)

        def _count_voxels(z: int) -> None:
            voxel_counts[int(z)] = np.int64(np.count_nonzero(np.asarray(final_output_mask_mm[int(z)])))

        parallel_for_indices(
            int(final_output_mask_mm.shape[0]),
            _count_voxels,
            max_workers=choose_slice_parallel_workers(int(tail_slice_workers), int(final_output_mask_mm.shape[0])),
            desc='Counting voxel_volume',
        )
        voxel_volume = int(np.sum(voxel_counts, dtype=np.int64))

    runtime_telemetry().gauge('pipeline.phase', 'post_output_wait')
    print('\n=== Waiting for background video/label output tasks ===')
    output_manager.wait()

    nrrd_manifest_path: Optional[Path] = None
    nrrd_layer_files_written = 0
    nrrd_low_quality_layer_files_written = 0
    nrrd_centerline_audit_files_written = 0
    nrrd_low_quality_centerline_audit_files_written = 0
    layer_sink = nrrd_layer_sink()
    if layer_sink is not None:
        runtime_telemetry().gauge('pipeline.phase', 'post_nrrd_wait')
        print('\n=== Finishing single-layer NRRD writes ===')
        layer_sink.wait()
        nrrd_layer_files_written = int(layer_sink.layer_count())
        nrrd_low_quality_layer_files_written = int(layer_sink.low_quality_layer_count())
        nrrd_centerline_audit_files_written = int(layer_sink.centerline_audit_layer_count())
        nrrd_low_quality_centerline_audit_files_written = int(
            layer_sink.low_quality_centerline_audit_layer_count()
        )
        nrrd_manifest_path = layer_sink.write_manifest()
        layer_sink.shutdown()
        set_nrrd_layer_sink(None)
        final_paths['nrrd_dir'] = nrrd_dir
        if nrrd_manifest_path is not None:
            final_paths['nrrd_manifest'] = nrrd_manifest_path

    if bool(save_images_enabled):
        print('\n=== Saving active-view image sequences ===')
        for view in views:
            image_dir = write_view_images(
                volume_rgb=volume_rgb,
                view=view,
                out_dir=out_dir,
                stem=input_path.stem,
                channel_format=channel_format,
                workers=tail_output_frame_workers,
                show_progress=False,
            )
            final_paths[f'{view.name}_images_dir'] = image_dir

    if bool(nrrd_layers_needed):
        legacy_nrrd_count = max(
            0, int(nrrd_layer_files_written) - int(nrrd_centerline_audit_files_written),
        )
        spec_notes.append(
            f'NRRD decomposition (v13.2.0): {int(legacy_nrrd_count)} legacy component/checkpoint NRRD file(s) written to '
            f'{nrrd_dir} as one uint8 binary mask per component layer (X,Y,t source geometry), named '
            f'{OUTPUT_NRRD_PREFIX}{{Filestem}}_{{ViewToken|Global}}_{{layer}}.seg.nrrd with the model name dropped, tagged as a 3D Slicer '
            'segmentation (v13.2.3: segment named after the file, deterministic per-layer color). Each layer is created during '
            'the intermediate pipeline steps (e.g. the Transverse layer compresses while Tiled Transverse is still '
            'inferencing) and, when Gaussian smoothing is enabled, Global_union_presmoothing is written while smoothing runs; a single '
            f'{variant_nrrd_stem(input_path.stem)}_nrrd_manifest.json sidecar lists every written layer. The previous mega 4D '
            'decomposed NRRD (one file with a trailing list axis) has been removed.'
        )
        if int(nrrd_low_quality_layer_files_written) > 0:
            legacy_lq_nrrd_count = max(
                0,
                int(nrrd_low_quality_layer_files_written)
                - int(nrrd_low_quality_centerline_audit_files_written),
            )
            spec_notes.append(
                f'Low-quality NRRD decomposition (v13.2.1, bug #2): {int(legacy_lq_nrrd_count)} legacy '
                'downbinned single-layer NRRD file(s) written under low_quality/<token>/nrrd/, mirroring the '
                'full-quality component layers per --save low_quality[:LOW_QUALITY_DOWNBIN] spec and written on the same '
                f'view-completion schedule. Each downbin has its own {OUTPUT_NRRD_PREFIX}{{Filestem}}_nrrd_manifest.json with layer '
                'suffixes matching the full-quality nrrd/ folder.'
            )
    if bool(centerline_audit_nrrd_needed):
        spec_notes.append(
            f'v14.0.1 centerline audit NRRDs: pass 0 and, for each completed detection pass, '
            f'sparse removed-component/watershed-candidate layers plus a result checkpoint were written under '
            f'{nrrd_dir}. Audit-only mode uses the same full reference raster as every other NRRD '
            f'and does not enable the legacy per-view decomposition. The manifest identifies explicit select/subtract/marker '
            f'roles for {int(nrrd_centerline_audit_files_written)} full-quality centerline audit file(s), '
            f'{int(nrrd_low_quality_centerline_audit_files_written)} matching low-quality audit mirror file(s), and '
            f'{int(nrrd_layer_files_written)} total full-quality layer file(s) from this run. Removed-component '
            'downbins use diagnostic_only and watershed-candidate downbins use none; neither participates in '
            'low-quality recomposition because max-pool downbinning does not commute with subtraction.'
        )

    summary_path: Optional[Path] = None
    if bool(save_summary_enabled):
        summary_path = write_summary_file(
            out_dir / f'{input_path.stem}_Summary.txt',
            command=shlex.join([str(x) for x in sys.argv]),
            input_path=input_path,
            out_dir=out_dir,
            scratch_dir=temp_dir,
            source_shape_x_y_t=(input_W, input_H, input_T),
            volume_shape=(T, H, W),
            fps=fps,
            model_paths=model_paths,
            view_names=[
                (
                    f'{v.name} ({int(v.num_slices)} frames; centers {int(v.tilt_frame_start)}..{int(v.tilt_frame_stop)})'
                    if is_tilted_view(v)
                    else f'{v.name} ({int(v.num_slices)} frames)'
                )
                for v in views
            ],
            view_prediction_stats=view_prediction_stats,
            interpolation_stats=interpolation_stats,
            view_prediction_labels=view_prediction_labels,
            enable_3d_void_fill=bool(args.enable_3d_void_fill),
            gaussian_smoothing_stats=gaussian_smoothing_stats,
            keep_objects_stats=keep_objects_stats,
            voxel_volume=voxel_volume,
            final_paths=final_paths,
            augmentation_workers=augmentation_workers,
            slice_postprocess_workers=slice_postprocess_workers,
            interpolation_workers=interpolation_workers,
            output_workers=tail_output_workers,
        )


    if final_output_mask_mm is not final_union_mm:
        close_memmap_array(final_output_mask_mm)
    close_memmap_array(final_union_mm)
    for model_support in native_view_support_by_model.values():
        for mm in model_support.values():
            close_memmap_array(mm)
        model_support.clear()
    for model_views in radial_native_output_by_model.values():
        for mm in model_views.values():
            close_memmap_array(mm)
        model_views.clear()
    for model_views in tilted_native_output_by_model.values():
        for mm in model_views.values():
            close_memmap_array(mm)
        model_views.clear()
    for model_views in view_volumes_by_model.values():
        for mm in model_views.values():
            close_memmap_array(mm)
        model_views.clear()
    for model_support in parent_mask_support_by_model.values():
        for mm in model_support.values():
            close_raw_store_or_memmap_volume(mm, keep_temp=bool(keep_temp_artifacts))
        model_support.clear()
    for model_support in parent_bridge_support_by_model.values():
        for mm in model_support.values():
            close_raw_store_or_memmap_volume(mm, keep_temp=bool(keep_temp_artifacts))
        model_support.clear()
    for mm in tile_accumulator_by_set.values():
        archive_or_delete_binary_volume_storage(
            mm,
            keep_temp=bool(keep_temp_artifacts),
            workers=int(tail_tile_slice_workers),
            desc='remaining consolidated tile accumulator',
        )
    tile_accumulator_by_set.clear()
    for mm in tile_parent_mask_accumulator_by_set.values():
        archive_or_delete_binary_volume_storage(
            mm,
            keep_temp=bool(keep_temp_artifacts),
            workers=int(tail_tile_slice_workers),
            desc='remaining parent-mask tile category accumulator',
        )
    tile_parent_mask_accumulator_by_set.clear()
    for mm in tile_parent_bridge_accumulator_by_set.values():
        archive_or_delete_binary_volume_storage(
            mm,
            keep_temp=bool(keep_temp_artifacts),
            workers=int(tail_tile_slice_workers),
            desc='remaining parent-bridge tile category accumulator',
        )
    tile_parent_bridge_accumulator_by_set.clear()
    for mm in baseline_union_by_model_view.values():
        close_memmap_array(mm)
    for mm in baseline_confmap_by_model_view.values():
        close_memmap_array(mm)
    for _, yolo in yolo_models:
        if yolo is not None:
            unload_yolo_model(yolo)
    if volume_rgb is not input_volume_rgb:
        close_memmap_array(volume_rgb)
    close_memmap_array(input_volume_rgb)
    trim_cuda_memory()
    gc.collect()

    if not keep_temp_artifacts:
        released_memfd_files = release_memfd_owners_under(temp_dir)
        if int(released_memfd_files) > 0:
            print(f'Released {int(released_memfd_files)} memfd-backed scratch payload(s).')
        try:
            for child in list(temp_dir.iterdir()):
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if temp_dir != out_dir / 'temp':
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_link = out_dir / 'temp'
                if temp_link.is_symlink():
                    temp_link.unlink(missing_ok=True)
        except Exception:
            pass

    post_inference_tail_seconds = float(time.perf_counter() - post_inference_tail_started)
    runtime_telemetry().gauge('post_inference_tail.seconds', post_inference_tail_seconds)
    runtime_telemetry().gauge('pipeline.phase', 'complete')
    print(f'\nPost-inference final assembly/output tail: {post_inference_tail_seconds:.1f}s.')
    print('\nDone.')
    print(f'Output dir: {out_dir}')
    print(f'Scratch dir: {temp_dir}')
    if 'overlay' in final_paths:
        print(f"Final overlay: {final_paths['overlay']}")
    if summary_path is not None:
        print(f'Summary: {summary_path}')


# Late imports keep callable-only dependency cycles import-safe.
from ._latebind import bind_late_symbols as _bind_late_symbols

_bind_late_symbols(
    __name__,
    globals(),
    {
        "assembly": (
            "_delete_tile_result_storage",
            "apply_gaussian_smoothing_inplace",
            "defer_open_tile_result_store",
            "finalize_consolidated_tile_volume_for_parent",
            "finalize_parent_without_tile_contribution_for_sparse_retirement",
            "gate_tile_residual_against_parent_bridge",
            "gate_tile_result_against_parent_mask",
            "materialize_nrrd_global_layer",
            "postprocess_tile_volume_after_inference",
            "prepare_view_volume_after_fullframe",
            "set_final_source_output_shape",
            "spill_waiting_tile_result_to_raw_store",
        ),
        "backprojection": (
            "HybridBackprojectionQueue",
            "ViewBackprojectionQueueJob",
            "_MAIN_PROCESS_GPU_STAGE_COORDINATOR",
            "_configure_main_process_gpu_stage_workers",
            "_main_process_gpu_stage_begin_inference",
            "_main_process_gpu_stage_can_dispatch_inference",
            "_main_process_gpu_stage_finish_inference",
            "_radial_view_nominal_spacing_deg",
            "_reset_main_process_gpu_stage_coordinator",
            "_set_main_process_gpu_inference_priority_active",
            "_set_main_process_gpu_pending_inference",
            "_set_main_process_gpu_stage_wake_callback",
            "fused_angle_variant_radial_component_layer_enabled",
            "main_process_gpu_stage_inference_priority_enabled",
        ),
        "config": (
            "GIB",
            "NRRD_SPACE",
            "OUTPUT_NRRD_PREFIX",
            "RADIAL_TEXTURE_VARIANT_LABEL",
            "SCRIPT_VERSION",
            "build_argparser",
            "quantize_display",
            "resolve_auto_positive_int",
            "resolve_backend_batches",
            "resolve_backend_devices",
            "resolve_backend_models",
            "resolve_backend_precisions",
            "resolve_cartesian_views",
            "resolve_channel_format",
            "resolve_postprocessing_options",
            "resolve_quantize",
            "resolve_radial_view_requests",
            "resolve_save_request",
            "resolve_tilted_view_groups",
            "resolve_tta_angles",
            "tilted_group_base_views",
            "variant_nrrd_stem",
        ),
        "cuda_backend": (
            "_fused_preflight_family",
            "fused_renderer_preflight_enabled",
            "gpu_cube_resize_enabled",
            "open_existing_gray_memmap",
            "union_conf_volume_into_volume_inplace",
        ),
        "cuda_d1": (
            "_memmap_backing_path",
            "_nrrd_layer_key",
            "archive_or_delete_binary_volume_storage",
            "close_raw_store_or_memmap_volume",
            "d1_owner_pipeline_enabled",
            "raw_bbox_nrrd_layers_enabled",
            "tile_dense_worker_result_limit_bytes",
            "tile_dense_worker_result_limit_tasks",
            "tile_dense_worker_result_warn_seconds",
            "tile_intermediate_accumulator_reserve_bytes",
            "tile_intermediate_accumulators_prefer_memory",
        ),
        "finalization": (
            "apply_keep_largest_objects_inplace",
            "apply_v14_centerline_filter_inplace",
            "assemble_final_union_after_view_union",
            "collapse_tta_variant_volumes_to_physical_views",
            "release_unretained_volume_maps",
            "scheduler_push_drain_enabled",
            "scheduler_push_drain_heartbeat_seconds",
        ),
        "geometry": (
            "AugJob",
            "DenseTileJob",
            "PredictionVolumeRef",
            "StreamingYoloVolumeSource",
            "ViewInfo",
            "_prediction_ref_has_gpu_input_staging",
            "build_aug_job_for_variant",
            "build_dense_tile_jobs_for_aug",
            "build_view_frame_cache",
            "close_prediction_volume_ref",
            "expand_views_into_tta_variants",
            "get_view_infos",
            "gpu_input_staging_ahead_sources",
            "gpu_input_staging_enabled",
            "is_radial_view",
            "is_tilted_radial_view",
            "is_tilted_view",
            "iter_aug_jobs_round_robin",
            "make_dense_tile_channel_renderer",
            "make_fullframe_channel_renderer",
            "materialize_dense_tile_prediction_volume_for_job",
            "materialize_fullframe_prediction_volume_for_job",
            "maybe_eager_stage_prediction_ref_on_gpu",
            "orthogonal_views_only",
            "output_to_view_processing_affine",
            "physical_view_name",
            "pretty_view_name",
            "queued_streaming_source_cpu_warmup_slots",
            "radial_base_view_name",
            "radial_target_base_view",
            "radial_target_diameter",
            "resolve_prediction_render_workers",
            "resolve_prediction_source_queue_slots",
            "resolve_tile_configs",
            "shared_streaming_render_pool_enabled",
            "should_cache_view_frames",
            "streaming_prediction_source_autostart_enabled",
            "streaming_prediction_source_prefetch_frames",
            "streaming_prediction_source_workers",
            "streaming_prediction_sources_enabled",
            "view_output_token",
            "view_processing_min_radius",
            "view_processing_volume_shape",
            "write_aug_job_meta",
            "write_dense_tile_job_meta",
        ),
        "inference": (
            "PredictConfig",
            "PredictionAccumulationHandle",
            "async_predict_join_workers",
            "async_predict_postprocess_enabled",
            "cleanup_backend",
            "cpu_mask_postprocess_pending_limit",
            "gpu_device_hole_fill_enabled",
            "gpu_device_union_enabled",
            "gpu_worker_chunk_hole_fill_enabled",
            "offload_between_jobs_enabled",
            "offload_yolo_from_gpu",
            "predict_in_memory_volume_and_accumulate",
            "predict_in_memory_volume_and_submit_accumulation",
            "resolve_gaussian_smoothing_settings",
            "set_angle_variant_gpu_fastpath",
            "set_inference_batch_size",
            "set_retina_mask_processor",
            "trim_cuda_memory",
            "unload_yolo_model",
        ),
        "interpolation": (
            "DeferredTilePostprocessResult",
            "NrrdLayerRef",
            "PreparedViewResult",
            "TilePostprocessResult",
            "TilePostprocessTask",
            "_ByteAdmissionPool",
            "_DirectUnionBackingLease",
            "_view_uses_interpolation",
            "interpolation_compiled_kernels_status",
            "materialize_raw_bbox_mask_store_workspace",
        ),
        "media": (
            "LazyProcessingCube",
            "_cube_t_axis_resize_backend",
            "_path_is_relative_to",
            "abort_streaming_producers",
            "compute_cube_resize_shape",
            "decode_video_to_memmap_gray8",
            "decode_video_to_memmap_gray8_streaming",
            "ffprobe_info",
            "processing_volume_mode",
            "purge_remaining_temporary_mkvs",
            "reset_streaming_state_for_new_run",
            "resize_volume_to_processing_cube_gray8",
            "resize_volume_to_processing_cube_gray8_streaming",
            "resolve_radial_azimuth_angles",
            "restore_mask_volume_to_original_shape",
            "should_resize_to_processing_cube",
            "streaming_preprocess_enabled",
            "wait_for_volume_ready",
            "wait_for_streaming_producers",
        ),
        "outputs": (
            "BackgroundOutputManager",
            "BackgroundOutputSubmission",
            "NrrdLayerSink",
            "collect_low_quality_output_futures",
            "collect_pipeline_output_futures",
            "ffv1_segment_count",
            "nrrd_layer_output_suffix",
            "nrrd_layer_sink",
            "nrrd_layer_sink_workers",
            "resolve_low_quality_downbin_specs",
            "set_nrrd_layer_sink",
            "shutdown_nrrd_gzip_executors",
            "write_summary_file",
            "write_view_images",
        ),
        "runtime": (
            "CpuInferenceInstancePlan",
            "HYBRID_DEFERRED_RESULT_MODE",
            "INTERPOLATION_PROCESS_WORKER_DEFAULT_CAP",
            "_GpuWorkerAuxInterpolationPool",
            "_attach_memfd_transfers_to_task",
            "_filesystem_free_bytes",
            "_memfd_owner_key_from_array",
            "_mount_fstype_for_path",
            "_sanitize_filesystem_token",
            "_sched_setaffinity_all_threads",
            "allocate_workspace_array",
            "array_nbytes",
            "choose_scratch_dir",
            "choose_slice_parallel_workers",
            "close_memmap_array",
            "close_memmap_array_without_flush",
            "configure_interpolation_pass_admission",
            "copy_workspace_array",
            "cpu_inference_supports_view",
            "cpu_inference_task_priority",
            "cpu_worker_default_seconds_per_frame",
            "cpu_worker_initial_lease_slices",
            "cpu_worker_max_lease_slices",
            "cpu_worker_min_lease_slices",
            "cpu_worker_target_lease_seconds",
            "create_interpolation_process_executor",
            "default_worker_budget",
            "expose_scratch_in_output",
            "flush_array",
            "gpu_feeder_reserved_physical_cores",
            "gpu_worker_aux_interpolation_enabled",
            "gpu_worker_aux_interpolation_pool",
            "gpu_worker_cpu_share",
            "gpu_worker_default_seconds_per_frame",
            "gpu_worker_direct_union_enabled",
            "gpu_worker_fullframe_task_ranges",
            "gpu_worker_initial_lease_slices",
            "gpu_worker_max_lease_slices",
            "gpu_worker_min_lease_slices",
            "gpu_worker_tail_split_point",
            "gpu_worker_target_lease_seconds",
            "gpu_worker_task_cost_key",
            "hybrid_cpu_affinity_overlap_enabled",
            "hybrid_cpu_reserved_view_count",
            "hybrid_gpu_stealback_enabled",
            "hybrid_gpu_stealback_eta_ratio",
            "hybrid_gpu_stealback_max_fraction",
            "hybrid_gpu_stealback_min_cpu_samples",
            "hybrid_gpu_stealback_min_lead_seconds",
            "initialize_runtime_observability",
            "interpolation_process_backend_enabled",
            "interpolation_process_cv2_threads",
            "interpolation_process_fallback_enabled",
            "interpolation_process_start_method",
            "main_process_worker_budget",
            "memfd_workspace_enabled",
            "parallel_for_indices",
            "path_is_memory_backed",
            "plan_gpu_feeder_core_reservations",
            "plan_gpu_worker_affinity",
            "plan_openvino_cpu_instances",
            "preflight_multiprocessing_payload",
            "raw_store_memfd_enabled",
            "register_unique_run_scratch_cleanup",
            "release_memfd_owners_under",
            "resolve_parent_interpolation_worker_allocation",
            "resolve_parent_postprocess_worker_allocation",
            "resolve_worker_count",
            "reset_runtime_state_for_new_run",
            "runtime_telemetry",
            "scratch_dir_is_memory_backed",
            "set_gpu_worker_aux_interpolation_pool",
            "set_interpolation_process_executor",
            "shutdown_parallel_pool_cache",
            "tail_worker_budget_expansion_enabled",
            "workspace_anon_cap_bytes",
        ),
        "topology": (
            "configure_gpu_slice_labeling_devices",
            "discard_binary_volume_slice_metadata",
        ),
        "workers": (
            "_cpu_inference_worker_main",
            "_gpu_inference_worker_main",
            "_pin_cuda_visible_device_token",
        ),
        "workspace": (
            "_cpu_count",
            "_env_flag",
            "_env_float",
            "_env_int",
            "available_anon_work_bytes",
            "radial_source_mode",
            "v1613_d1_backprojection_overlap_enabled",
            "v1613_fast_bundle_requested",
        ),
    },
)
