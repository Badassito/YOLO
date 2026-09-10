"""Capture eight actual TensorRT head/prototype pairs from native view renders.

Uses an explicitly supplied local engine and a bounded real-video volume. The
production Radial/Spherical renderer supplies FP16-normalized grayscale inputs,
then the binding receives their exact FP32 cast, matching --quantize gpu:fp16.
This writes diagnostic captures only and does not modify any engine or XTA code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_video(path, max_frames, start_frame=0):
    import cv2

    reader = cv2.VideoCapture(str(path))
    frames = []
    try:
        if start_frame and not reader.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame)):
            raise ValueError('Could not seek to the requested source frame')
        while len(frames) < max_frames:
            ok, image = reader.read()
            if not ok:
                break
            # Fixtures are gray8 videos; OpenCV may expose repeated BGR channels.
            if image.ndim == 3:
                if not (np.array_equal(image[..., 0], image[..., 1]) and
                        np.array_equal(image[..., 1], image[..., 2])):
                    raise ValueError('Expected an actual grayscale source video')
                image = image[..., 0]
            if image.dtype != np.uint8 or max(image.shape) > 4096:
                raise ValueError('Use a bounded gray8 fixture with axes no larger than 4096')
            frames.append(image.copy())
    finally:
        reader.release()
    if not frames:
        raise ValueError('No decoded source frames')
    return np.stack(frames)


def read_plan(path):
    data = path.read_bytes()
    length = int.from_bytes(data[:4], 'little', signed=True)
    if 0 < length < min(16 * 2**20, len(data) - 4):
        try:
            metadata = json.loads(data[4:4 + length])
        except (ValueError, UnicodeError):
            pass
        else:
            return data[4 + length:], metadata
    return data, {}


def chosen_views(shape, size):
    from XTA.cylindrical_geometry import build_radial_view_infos
    from XTA.spherical_geometry import build_spherical_view_infos

    radial = build_radial_view_infos(*shape, targets=('transverse', 'sagittal', 'coronal'),
                                    min_radius=1., patch_size=size, tilted_views=())
    for base, fractions in [('transverse', (.25, .75)), ('sagittal', (.5,)), ('coronal', (.5,))]:
        view = next(view for view in radial if view.radial_base_view == base)
        for fraction in fractions:
            yield view, min(view.num_slices - 1, int(fraction * view.num_slices))
    spherical = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=1.,
                                          patch_size=size, tilted_views=())
    for face in (0, 1, 4, 5):
        view = next(view for view in spherical if view.spherical_face == face)
        yield view, view.num_slices // 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-source-frames', type=int, default=24)
    parser.add_argument('--start-frame', type=int, default=0)
    args = parser.parse_args()
    args.input = args.input.resolve(strict=True)
    args.engine = args.engine.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.is_relative_to(ROOT) or not 1 <= args.max_source_frames <= 64 or args.start_frame < 0:
        parser.error('Use task Scratch and a source-frame bound of 1 through 64')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_TTA_FAST_GEOMETRY'] = '1'
    volume = load_video(args.input, args.max_source_frames, args.start_frame)

    import torch
    import tensorrt as trt
    from XTA import geometry
    from XTA.config import resolve_channel_format
    from XTA.media import compute_cube_resize_shape
    from tests.test_cylindrical_cuda import resident_engine

    torch.set_num_threads(2)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    plan, metadata = read_plan(args.engine)
    trt_engine = runtime.deserialize_cuda_engine(plan)
    if trt_engine is None:
        raise RuntimeError('Could not load the explicitly supplied local engine')
    context = trt_engine.create_execution_context()
    names = [trt_engine.get_tensor_name(i) for i in range(trt_engine.num_io_tensors)]
    inputs = [name for name in names if trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    if len(inputs) != 1:
        raise ValueError('Expected one static input tensor')
    buffers = {}
    for name in names:
        shape = tuple(trt_engine.get_tensor_shape(name))
        if min(shape) <= 0 or trt_engine.get_tensor_dtype(name) != trt.float32:
            raise ValueError(f'{name}: expected static FP32 binding, got {shape}')
        buffers[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
        context.set_tensor_address(name, buffers[name].data_ptr())
    network_input = buffers[inputs[0]]
    if network_input.shape[:2] != (1, 1) or network_input.shape[2] != network_input.shape[3]:
        raise ValueError('Expected static batch-one grayscale square input')
    size = int(network_input.shape[2])
    output_names = [name for name in names if name not in inputs]
    head_name = next(name for name in output_names if buffers[name].ndim == 3)
    proto_name = next(name for name in output_names if buffers[name].ndim == 4)
    logical_shape = tuple(map(int, compute_cube_resize_shape(*volume.shape)))
    if logical_shape[1:] != volume.shape[1:]:
        raise ValueError('Fixture must require at most logical T expansion')
    renderer = resident_engine(volume, 'cuda:0', logical_t=logical_shape[0])
    channel_format = resolve_channel_format('gray')
    receipt = {'source': str(args.input), 'source_sha256': digest_file(args.input),
               'source_start_frame': args.start_frame,
               'source_shape': list(volume.shape), 'logical_shape': logical_shape,
               'engine': str(args.engine), 'engine_sha256': hashlib.sha256(args.engine.read_bytes()).hexdigest(),
               'engine_metadata': metadata, 'tensorrt': trt.__version__, 'torch': torch.__version__,
               'preprocessing': 'production native renderer, gray8 -> normalized FP16 -> FP32 binding',
               'renderer_fixture': 'tests.test_cylindrical_cuda.resident_engine with real decoded gray8 volume',
               'requested_confidence': .5, 'forward_passes': 0, 'captures': []}
    stream = torch.cuda.current_stream()
    for ordinal, (physical, frame) in enumerate(chosen_views(logical_shape, size)):
        view = geometry.expand_views_into_tta_variants((physical,), (0.,))[0]
        job = geometry.build_aug_job_for_variant(view, size, args.output_dir)
        rendered, ready = renderer.render_fullframe_batch(view, job, (frame,), size,
                                                         True, channel_format=channel_format)
        stream.wait_event(ready)
        network_input.copy_(rendered.to(torch.float32))
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execute_async_v3 failed')
        stream.synchronize()
        receipt['forward_passes'] += 1
        head = buffers[head_name][0].cpu().numpy().copy()
        proto = buffers[proto_name][0].cpu().numpy().copy()
        filename = f'{ordinal:02d}-{physical.name}-frame{frame}.npz'
        np.savez(args.output_dir / filename, head=head, proto=proto,
                 input_hw=np.array((size, size), np.int32), conf=np.array(.5, np.float32),
                 source_view=np.array(physical.name), source_family=np.array(physical.family),
                 source_frame=np.array(frame, np.int32))
        scores = head[4]
        row = {'path': filename, 'view': physical.name, 'family': physical.family, 'frame': frame,
               'head_shape': list(head.shape), 'proto_shape': list(proto.shape),
               'retained_at_0_5': int((scores >= .5).sum()),
               'retained_at_0_00001_diagnostic': int((scores >= .00001).sum()),
               'max_confidence': float(scores.max()),
               'input_nonzero_pixels': int(torch.count_nonzero(rendered).item()),
               'sampler_actual': getattr(renderer, '_spherical_sampler_mode', None),
               'sha256': hashlib.sha256((args.output_dir / filename).read_bytes()).hexdigest()}
        receipt['captures'].append(row)
        (args.output_dir / 'capture.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(row), flush=True)
    if receipt['forward_passes'] != 8:
        raise AssertionError('Capture helper must issue exactly eight forward passes')
    stream.synchronize()
    print(f'Saved eight native captures in {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
