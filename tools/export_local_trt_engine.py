"""Export a separate static TensorRT validation engine into task Scratch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--workspace-gib', type=float, default=1.)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.is_relative_to(root) or args.imgsz <= 0 or args.imgsz % 64 or args.workspace_gib <= 0:
        parser.error('Use task Scratch, a positive multiple-of-64 size, and positive workspace')
    output.mkdir(parents=True, exist_ok=True)
    engine_path = output / f'local-4090-{args.imgsz}-fp16.engine'
    onnx_path = engine_path.with_suffix('.onnx')
    if engine_path.exists() or onnx_path.exists():
        parser.error('Refusing to replace an existing engine or ONNX export')
    config = output / 'ultralytics-config'
    config.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config)
    os.environ['YOLO_AUTOINSTALL'] = 'false'

    import torch
    import tensorrt as trt
    import onnx
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, Detect

    torch.set_num_threads(2)
    model = YOLO(str(args.model.resolve())).model.eval().float().cuda()
    model = model.fuse(verbose=False, imgsz=(args.imgsz, args.imgsz))
    channels = next(m.in_channels for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, Detect):
            if getattr(module, 'end2end', False):
                raise ValueError('Validation exporter requires a non-end-to-end segmentation head')
            module.export, module.dynamic, module.format = True, False, 'onnx'
            module.shape = None
        elif isinstance(module, C2f):
            module.forward = module.forward_split
    sample = torch.zeros((1, channels, args.imgsz, args.imgsz), device='cuda')
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(sample)
        if (not isinstance(outputs, (tuple, list)) or len(outputs) != 2
                or outputs[0].ndim != 3 or outputs[1].ndim != 4):
            raise ValueError('Expected an exported segmentation head and prototype pair')
        shapes = [tuple(value.shape) for value in outputs]
        torch.onnx.export(model, sample, str(onnx_path), input_names=['images'],
                          output_names=['output0', 'output1'], opset_version=17,
                          do_constant_folding=True, dynamo=False)
    export_seconds = time.perf_counter() - started
    onnx.checker.check_model(str(onnx_path))
    names = dict(model.names)
    stride = int(model.stride.max())
    del outputs, model, sample
    torch.cuda.empty_cache()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 2**30))
    configuration.set_flag(trt.BuilderFlag.FP16)
    # Match the production engine's FP32 output interface. The builder may use
    # FP16 internally; the numerical comparison always uses this same engine.
    for index in range(network.num_outputs):
        network.get_output(index).dtype = trt.float32
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, configuration)
    if plan is None:
        raise RuntimeError('TensorRT failed to build the local validation engine')
    build_seconds = time.perf_counter() - started
    metadata = {'task': 'segment', 'batch': 1, 'imgsz': [args.imgsz, args.imgsz],
                'stride': stride, 'names': names, 'channels': channels,
                'description': 'Local 4090 validation copy; not the H100 reference engine'}
    encoded = json.dumps(metadata).encode('utf-8')
    engine_path.write_bytes(len(encoded).to_bytes(4, 'little', signed=True) + encoded + bytes(plan))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(plan))
    bindings = [{'name': engine.get_tensor_name(i),
                 'shape': list(engine.get_tensor_shape(engine.get_tensor_name(i))),
                 'dtype': str(engine.get_tensor_dtype(engine.get_tensor_name(i)))}
                for i in range(engine.num_io_tensors)]
    result = {'engine': str(engine_path), 'onnx': str(onnx_path), 'bindings': bindings,
              'source_model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'engine_sha256': hashlib.sha256(engine_path.read_bytes()).hexdigest(),
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'tensorrt': trt.__version__, 'onnx_version': onnx.__version__,
              'export_seconds': export_seconds, 'build_seconds': build_seconds,
              'workspace_gib': args.workspace_gib, 'expected_outputs': shapes}
    engine_path.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
