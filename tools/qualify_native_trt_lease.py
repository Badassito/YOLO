"""Compare bounded, complete native-mask TensorRT leases on a real video block.

Each lease calls production predict_source_and_accumulate: render, input precision
conversion, TensorRT, compaction, mask postprocessing, device union and native-mask
host publication. It stops before whole-volume backprojection, view hole fill,
NRRD, final union and topology. This is not an H100 pipeline-walltime predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np

from tools.capture_native_proto_outputs import chosen_views, digest_file, load_video


def array_digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def serializable_stats(stats):
    result = {}
    for name, value in stats.items():
        if name == 'slice_meta':
            result[name] = ({key: {'shape': list(array.shape), 'dtype': str(array.dtype),
                                   'sha256': array_digest(array)}
                             for key, array in value.items()} if value else None)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        else:
            result[name] = str(type(value).__name__)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=3072)
    parser.add_argument('--start-frame', type=int, default=900)
    parser.add_argument('--source-frames', type=int, default=24)
    parser.add_argument('--lease-frames', type=int, default=8)
    parser.add_argument('--conf', type=float, default=.5)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    args = parser.parse_args()
    args.input = args.input.resolve(strict=True)
    args.engine = args.engine.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if (args.output_dir.is_relative_to(ROOT) or not 1 <= args.source_frames <= 64 or
            not 1 <= args.lease_frames <= 32 or args.start_frame < 0 or
            args.imgsz <= 0 or args.imgsz % 64 or args.heat_seconds < 0 or
            not 0 <= args.conf <= 1):
        parser.error('Use task Scratch, bounded frame counts, valid confidence and raster size')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_dir = args.output_dir / 'ultralytics-config'
    config_dir.mkdir()
    os.environ.update(YOLO_TTA_FAST_GEOMETRY='1', YOLO_TTA_NATIVE_TRT_RING='0',
                      YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(config_dir))

    import torch
    import tensorrt as trt
    from XTA import backprojection, cuda_backend, geometry, inference
    from XTA.config import resolve_channel_format
    from XTA.media import compute_cube_resize_shape
    from tests.test_cylindrical_cuda import resident_engine

    torch.set_num_threads(2)
    started = time.perf_counter()
    volume = load_video(args.input, args.source_frames, args.start_frame)
    decode_seconds = time.perf_counter() - started
    logical_shape = tuple(map(int, compute_cube_resize_shape(*volume.shape)))
    if logical_shape[1:] != volume.shape[1:]:
        raise ValueError('Use a source block requiring only logical T expansion')
    choices = list(chosen_views(logical_shape, args.imgsz))
    selected = [next(pair for pair in choices if pair[0].family == 'radial'
                     and pair[0].radial_base_view == 'sagittal'),
                next(pair for pair in choices if pair[0].family == 'spherical'
                     and pair[0].spherical_face == 4)]
    cfg = inference.PredictConfig(imgsz=args.imgsz, conf=args.conf, device='cuda:0',
                                  quantize='fp16', batch=1, input_channels=1, channel_token='gray')
    inference.set_retina_mask_processor('gpu')
    inference.set_angle_variant_gpu_fastpath(0., 0.)
    started = time.perf_counter()
    renderer = resident_engine(volume, 'cuda:0', logical_t=logical_shape[0])
    renderer._stream = torch.cuda.Stream(device=renderer.device)
    renderer._stream.wait_stream(torch.cuda.current_stream())
    renderer._stream.synchronize()
    upload_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = inference.load_ultralytics_model(str(args.engine), task='segment')
    inference.require_channel_aware_yolo_preprocess_patch('gray')
    predictor = inference._ensure_predictor_for_direct_predict(model, cfg)
    if predictor is None:
        raise RuntimeError('Actual direct TensorRT predictor could not be initialized')
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started
    backend = predictor.model
    trt_engine = backprojection._trt_engine_from_autobackend(backend)
    if trt_engine is None:
        raise RuntimeError('Actual TensorRT engine is required')
    names, input_name, _, _ = backprojection._trt_binding_layout_for_backend(backend, trt_engine)
    input_dtype = backprojection._torch_dtype_for_trt_binding(backend, trt_engine, input_name, torch)
    if input_dtype != torch.float32:
        raise RuntimeError('This qualification expects the exported FP32 binding interface')
    binding_shapes = {name: list(trt_engine.get_tensor_shape(name)) for name in names}
    if binding_shapes[input_name] != [1, 1, args.imgsz, args.imgsz]:
        raise RuntimeError(f'Wrong engine input shape: {binding_shapes[input_name]}')

    report = {'scope': __doc__, 'source': str(args.input), 'engine': str(args.engine),
              'source_sha256': digest_file(args.input), 'engine_sha256': digest_file(args.engine),
              'source_start_frame': args.start_frame, 'source_shape': list(volume.shape),
              'logical_shape': logical_shape, 'torch': torch.__version__, 'tensorrt': trt.__version__,
              'device': torch.cuda.get_device_name(), 'binding_shapes': binding_shapes,
              'binding_input_dtype': str(input_dtype), 'render_normalization_dtype': 'torch.float16',
              'shared_setup_seconds': {'decode': decode_seconds, 'source_upload': upload_seconds,
                                       'model_and_backend_initialization': setup_seconds},
              'timing_scope': 'Host walltime includes source creation, host-target allocation, actual '
                              'prediction API, ring preparation/cache reuse, CUDA synchronization, '
                              'device-union retirement/host publication, and source close. Shared '
                              'decode/upload/model initialization and artifact hashing/writes are separate. '
                              'Mode-transition context cleanup is recorded separately, not charged to '
                              'the next mode. CUDA graph warmup work is included in preparation time.',
              'planned_source_frames_per_mode': 2 * 2 * args.lease_frames,
              'source_file_hashes': {name: digest_file(ROOT / name) for name in
                                    ('XTA/inference.py', 'XTA/backprojection.py', 'XTA/cuda_backend.py',
                                     'tools/qualify_native_trt_lease.py')},
              'runs': [], 'transitions': [], 'all_masks_exact': None, 'all_counts_exact': None}
    output = args.output_dir / 'qualification.json'
    save_json(output, report)
    if args.heat_seconds:
        from tools.benchmark_radial_setup import heatsoak
        report['heat_seconds_actual'] = heatsoak(args.heat_seconds, 0)
    baselines = {}
    previous_mode = None
    channel_format = resolve_channel_format('gray')
    for physical, middle_frame in selected:
        view = geometry.expand_views_into_tta_variants((physical,), (0.,))[0]
        offset = min(max(0, middle_frame - args.lease_frames // 2),
                     physical.num_slices - args.lease_frames)
        if offset < 0:
            raise ValueError('View has fewer frames than the requested fixed lease')
        job = geometry.build_aug_job_for_variant(view, args.imgsz, args.output_dir)
        native_h, native_w = geometry.view_processing_plane_shape(view, args.imgsz)
        matrix = geometry.output_to_view_processing_affine(view, job.aff.M_out_to_src, args.imgsz)
        for mode in ('generic', 'native', 'native', 'generic'):
            if mode != previous_mode:
                started = time.perf_counter()
                retired = backprojection._release_resident_trt_pipeline_cache()
                torch.cuda.synchronize()
                report['transitions'].append({'after_mode': previous_mode, 'before_mode': mode,
                                               'retired_context_groups': retired,
                                               'seconds': time.perf_counter() - started})
            previous_mode = mode
            os.environ['YOLO_TTA_NATIVE_TRT_RING'] = str(int(mode == 'native'))
            label = f'{len(report["runs"]):02d}-{physical.family}-{mode}'
            print(f'BEGIN COMPLETE LEASE {label}: {physical.name} [{offset},{offset + args.lease_frames})', flush=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            target = np.zeros((args.lease_frames, native_h, native_w), dtype=np.uint8)
            source = cuda_backend.GpuRenderedYoloSource(
                renderer, view, job, slice_offset=offset, num_frames=args.lease_frames,
                batch_size=1, out_size=args.imgsz, fp16=True, name=label,
                channel_format=channel_format)
            stats = inference.predict_source_and_accumulate(
                model, source, source_label=label, num_frames=args.lease_frames,
                out_size=args.imgsz, cfg=cfg, view_union_mm=target, view_confmap_mm=None,
                M_out_to_native=matrix, native_h=native_h, native_w=native_w,
                postprocess_workers=2, streaming_cleanup_enabled=False,
                device_hole_fill=False, defer_device_union_flush=False,
                require_device_union=True)
            future = stats.pop('_device_union_flush_future', None)
            if future is not None:
                stats.update(future.result())
            source.close()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            consumed = int(source._direct_count if mode == 'native' else source.count)
            native_active = bool(getattr(source, '_native_trt_data_consumed', False))
            if consumed != args.lease_frames or native_active != (mode == 'native'):
                raise AssertionError(f'{label}: source frame count/route mismatch ({consumed}, {native_active})')
            if int(stats.get('device_hole_filled_frames', 0)):
                raise AssertionError('This native-mask lease must not add morphology')
            if mode == 'generic' and source._direct_count:
                raise AssertionError('Generic mode unexpectedly consumed ring slots')
            mask_sha = array_digest(target)
            counts = (int(stats['prediction_count']), int(stats['frames_with_predictions']))
            baseline = baselines.setdefault(physical.family, (mask_sha, counts))
            row = {'mode': mode, 'label': label, 'view': physical.name, 'family': physical.family,
                   'slice_offset': offset, 'source_frames_consumed': consumed,
                   'native_ring_active': native_active, 'lease_seconds': elapsed,
                   'mask_shape': list(target.shape), 'foreground_voxels': int(np.count_nonzero(target)),
                   'mask_sha256': mask_sha, 'exact_mask_match': mask_sha == baseline[0],
                   'exact_count_match': counts == baseline[1], 'stats': serializable_stats(stats),
                   'sampler_actual': getattr(renderer, '_spherical_sampler_mode', None),
                   'cuda_memory_allocated': torch.cuda.memory_allocated(),
                   'cuda_memory_reserved': torch.cuda.memory_reserved()}
            # Preserve compact, lossless publication evidence outside the timed interval.
            np.packbits(target.reshape(-1), bitorder='little').tofile(args.output_dir / f'{label}.mask.packbits')
            report['runs'].append(row)
            save_json(output, report)
            print(json.dumps(row), flush=True)
            del target, source
    started = time.perf_counter()
    retired = backprojection._release_resident_trt_pipeline_cache()
    torch.cuda.synchronize()
    report['terminal_context_retirement'] = {'context_groups': retired,
                                             'seconds': time.perf_counter() - started}
    report['all_masks_exact'] = all(row['exact_mask_match'] for row in report['runs'])
    report['all_counts_exact'] = all(row['exact_count_match'] for row in report['runs'])
    report['actual_source_frames_per_mode'] = {
        mode: sum(row['source_frames_consumed'] for row in report['runs'] if row['mode'] == mode)
        for mode in ('generic', 'native')}
    report['median_lease_seconds'] = {
        family: {mode: statistics.median(row['lease_seconds'] for row in report['runs']
                                         if row['family'] == family and row['mode'] == mode)
                 for mode in ('generic', 'native')}
        for family in ('radial', 'spherical')}
    save_json(output, report)
    if not report['all_masks_exact'] or not report['all_counts_exact']:
        raise AssertionError('Native ring changed a mask or prediction count in the identical lease')
    print(f'Complete lease qualification passed: {output}', flush=True)


if __name__ == '__main__':
    main()
