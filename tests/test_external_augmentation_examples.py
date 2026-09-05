from __future__ import annotations

import ast
import copy
import importlib
import io
import math
import os
import re
import sys
import tokenize
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "XTA" / "examples" / "external_augmentations"
GPU_PROFILES = (
    ("light", EXAMPLES / "GPU_light.py"),
    ("baseline", EXAMPLES / "GPU_baseline.py"),
    ("heavy", EXAMPLES / "GPU_heavy.py"),
    ("superheavy", EXAMPLES / "GPU_superheavy.py"),
)
CPU_PROFILES = tuple(
    (profile, EXAMPLES / f"CPU_{profile}.py")
    for profile, _path in GPU_PROFILES
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    raise AssertionError(f"{path}: missing {class_name}.{method_name}")


def _isolated_sample_function(
    path: Path,
    *,
    class_name: str,
) -> object:
    function = copy.deepcopy(
        _class_method(path, class_name, "_sample_parameters")
    )
    function.name = "sample_parameters"
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.Import(names=[ast.alias(name="random")]),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["sample_parameters"]


def _elastic_amplitude(path: Path, *, class_name: str) -> float:
    function = _class_method(path, class_name, "_elastic_displacement")
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        raise AssertionError(f"{path}: expected one elastic-displacement return")
    multipliers = [
        node
        for node in ast.walk(returns[0].value)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, (int, float))
    ]
    if len(multipliers) != 1:
        raise AssertionError(f"{path}: elastic amplitude is not an explicit multiplier")
    return float(multipliers[0].right.value)  # type: ignore[union-attr]


def _multiplicative_noise_range(path: Path, *, class_name: str) -> tuple[float, float]:
    function = _class_method(path, class_name, "_apply_intensity_noise")
    method_name = "uniform_" if class_name == "GPUAugmentation" else "uniform"
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
        and len(node.args) >= 2
        and all(
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, (int, float))
            for argument in node.args[:2]
        )
    ]
    if len(calls) != 1:
        raise AssertionError(
            f"{path}: expected one explicit multiplicative-noise {method_name} range"
        )
    return tuple(float(argument.value) for argument in calls[0].args[:2])  # type: ignore[return-value,union-attr]


def _selectors(sample: dict[str, object]) -> tuple[object, ...]:
    return (
        int(sample["d4"]),
        bool(float(sample["scale"]) < 1.0),
        bool(sample["elastic"]),
        bool(float(sample["brightness"]) != 1.0),
        bool(float(sample["blur_sigma"]) != 0.0),
        int(sample["noise_family"]),
        bool(float(sample["salt_pepper_amount"]) != 0.0),
    )


def _magnitudes(sample: dict[str, object], *, height: int, width: int) -> tuple[float, ...]:
    noise_strength = float(sample["noise_strength"])
    noise_magnitude = (
        abs(noise_strength - 1.0)
        if int(sample["noise_family"]) == 2
        else noise_strength
    )
    return (
        abs(float(sample["rotation"])),
        abs(float(sample["scale"]) - 1.0),
        abs(float(sample["translate_x"])) / float(width),
        abs(float(sample["translate_y"])) / float(height),
        abs(float(sample["shear_x"])),
        abs(float(sample["shear_y"])),
        abs(float(sample["brightness"]) - 1.0),
        float(sample["blur_sigma"]),
        noise_magnitude,
        float(sample["salt_pepper_amount"]),
    )


class ExternalAugmentationExampleTests(unittest.TestCase):
    def test_gpu_files_have_one_supported_export_and_no_versioned_definition_names(self) -> None:
        supported_exports = {
            "custom_transforms",
            "augmentation",
            "build_augmentation",
            "build_gpu_augmentation",
        }
        version_token = re.compile(r"\bv\d+(?:\.\d+)*\b|Augments[_A-Za-z]*V?\d+", re.IGNORECASE)
        versioned_name = re.compile(r"(?:^|_)v\d+(?:_|$)|V\d+")

        for profile, path in GPU_PROFILES:
            with self.subTest(profile=profile):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                top_level_names = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
                self.assertEqual(top_level_names & supported_exports, {"build_gpu_augmentation"})
                self.assertIn("GPUAugmentation", top_level_names)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.assertIsNone(versioned_name.search(node.name), node.name)
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        docstring = ast.get_docstring(node, clean=False) or ""
                        self.assertIsNone(version_token.search(docstring), docstring)
                comments = (
                    token.string
                    for token in tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type == tokenize.COMMENT
                )
                for comment in comments:
                    self.assertIsNone(version_token.search(comment), comment)

    def test_gpu_profiles_preserve_branch_selection_and_increase_magnitudes(self) -> None:
        height, width = 257, 383
        samplers = [
            _isolated_sample_function(path, class_name="GPUAugmentation")
            for _profile, path in GPU_PROFILES
        ]
        self.assertEqual(
            [
                _elastic_amplitude(path, class_name="GPUAugmentation")
                for _profile, path in GPU_PROFILES
            ],
            [15.0, 20.0, 27.5, 35.0],
        )
        self.assertEqual(
            [
                _multiplicative_noise_range(path, class_name="GPUAugmentation")
                for _profile, path in GPU_PROFILES
            ],
            [(0.65, 1.35), (0.5, 1.5), (0.35, 1.65), (0.15, 1.85)],
        )
        self.assertEqual(
            _multiplicative_noise_range(
                EXAMPLES / "CPU_baseline.py",
                class_name="CPUAugmentation",
            ),
            (0.5, 1.5),
        )

        for seed in range(2048):
            samples = [
                sampler(seed, height, width)  # type: ignore[operator]
                for sampler in samplers
            ]
            self.assertTrue(all(_selectors(sample) == _selectors(samples[0]) for sample in samples[1:]))
            magnitudes = [
                _magnitudes(sample, height=height, width=width)
                for sample in samples
            ]
            for field_values in zip(*magnitudes):
                self.assertTrue(
                    all(left <= right + 1e-12 for left, right in zip(field_values, field_values[1:])),
                    (seed, field_values),
                )

    def test_cpu_profiles_match_their_gpu_parameter_graphs(self) -> None:
        self.assertEqual(
            [
                _elastic_amplitude(path, class_name="CPUAugmentation")
                for _profile, path in CPU_PROFILES
            ],
            [15.0, 20.0, 27.5, 35.0],
        )
        self.assertEqual(
            [
                _multiplicative_noise_range(path, class_name="CPUAugmentation")
                for _profile, path in CPU_PROFILES
            ],
            [(0.65, 1.35), (0.5, 1.5), (0.35, 1.65), (0.15, 1.85)],
        )

        for (profile, gpu_path), (_cpu_profile, cpu_path) in zip(
            GPU_PROFILES,
            CPU_PROFILES,
        ):
            gpu_sample = _isolated_sample_function(
                gpu_path,
                class_name="GPUAugmentation",
            )
            cpu_sample = _isolated_sample_function(
                cpu_path,
                class_name="CPUAugmentation",
            )

            for seed in range(512):
                with self.subTest(profile=profile, seed=seed):
                    self.assertEqual(
                        cpu_sample(seed, 257, 383),  # type: ignore[operator]
                        gpu_sample(seed, 257, 383),  # type: ignore[operator]
                    )

    def test_cpu_profiles_are_seeded_deterministic_and_preserve_empty_masks(self) -> None:
        loaded_cv2 = sys.modules.get("cv2")
        if loaded_cv2 is not None and type(loaded_cv2).__name__ == "_StubModule":
            self.skipTest("CPU example execution requires real OpenCV, not import stubs")
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated host
            if exc.name != "cv2":
                raise
            self.skipTest(f"CPU example dependencies are unavailable: {exc}")

        height, width = 64, 80
        image = np.arange(height * width, dtype=np.uint16).reshape(height, width)
        image = np.asarray(image % 256, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for profile, path in CPU_PROFILES:
            with self.subTest(profile=profile):
                module = importlib.import_module(
                    f"XTA.examples.external_augmentations.{path.stem}"
                )
                policy = module.build_augmentation()
                policy.set_random_seed(12345)
                first = policy(image=image, mask=mask)
                policy.set_random_seed(12345)
                second = policy(image=image, mask=mask)

                np.testing.assert_array_equal(first["image"], second["image"])
                np.testing.assert_array_equal(first["mask"], second["mask"])
                self.assertEqual(first["image"].shape, image.shape)
                self.assertEqual(first["image"].dtype, np.uint8)
                self.assertEqual(first["mask"].dtype, np.uint8)
                self.assertEqual(int(np.count_nonzero(first["mask"])), 0)

    @unittest.skipUnless(
        os.environ.get("XTA_RUN_EXTERNAL_AUGMENTATION_CUDA", "").strip() == "1",
        "set XTA_RUN_EXTERNAL_AUGMENTATION_CUDA=1 on a CUDA PyTorch host",
    )
    def test_gpu_profiles_execute_deterministically_on_cuda(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - hardware gate
            self.fail(f"CUDA example gate was enabled without PyTorch: {exc}")
        self.assertTrue(torch.cuda.is_available(), "CUDA example gate requires torch.cuda")

        height, width = 96, 112
        image = np.arange(height * width, dtype=np.uint16).reshape(height, width)
        image = np.asarray(image % 256, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[24:72, 28:84] = 1
        seeds = (None, 7, 410, 586)

        outputs: list[object] = []
        with mock.patch.dict(os.environ, {"PTA_GPU_TORCH_COMPILE": "0"}):
            for profile, path in GPU_PROFILES:
                with self.subTest(profile=profile):
                    module = importlib.import_module(
                        f"XTA.examples.external_augmentations.{path.stem}"
                    )
                    policy = module.build_gpu_augmentation(
                        device="cuda:0",
                        batch_size=len(seeds),
                    )
                    first_images, first_masks = policy.apply_batch(
                        image=image,
                        mask=mask,
                        seeds=seeds,
                        output_size=(height, width),
                    )
                    second_images, second_masks = policy.apply_batch(
                        image=image,
                        mask=mask,
                        seeds=seeds,
                        output_size=(height, width),
                    )
                    torch.cuda.synchronize()
                    self.assertEqual(
                        tuple(first_images.shape),
                        (len(seeds), 1, height, width),
                    )
                    self.assertEqual(
                        tuple(first_masks.shape),
                        (len(seeds), height, width),
                    )
                    self.assertEqual(first_images.dtype, torch.uint8)
                    self.assertEqual(first_masks.dtype, torch.uint8)
                    self.assertTrue(torch.equal(first_images, second_images))
                    self.assertTrue(torch.equal(first_masks, second_masks))
                    self.assertLessEqual(int(first_masks.max().item()), 1)
                    outputs.append(first_images.detach().cpu())

        self.assertTrue(
            all(not torch.equal(left, right) for left, right in zip(outputs, outputs[1:])),
            "adjacent magnitude profiles unexpectedly produced identical seeded batches",
        )


class GPUAugmentationGaussianMathTests(unittest.TestCase):
    """Check GPU policy math on CPU without constructing a CUDA policy."""

    @classmethod
    def setUpClass(cls) -> None:
        loaded_torch = sys.modules.get("torch")
        if loaded_torch is not None and type(loaded_torch).__name__ == "_StubModule":
            raise unittest.SkipTest("Gaussian execution requires real PyTorch, not import stubs")
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise unittest.SkipTest("Gaussian execution requires PyTorch") from exc

        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.modules = [
            (
                profile,
                importlib.import_module(f"XTA.examples.external_augmentations.{path.stem}"),
            )
            for profile, path in GPU_PROFILES
        ]

    def _square_reference(self, sample: object, sigma: float) -> object:
        # Build the isotropic 2D density independently of the policy's 1D kernel.
        radius = max(1, int(math.ceil(3.0 * max(0.05, sigma))))
        coords = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(
            -(coords[:, None] ** 2 + coords[None, :] ** 2)
            / (2.0 * max(0.05, sigma) ** 2)
        )
        kernel /= kernel.sum()
        channels = int(sample.shape[1])
        weight = self.torch.from_numpy(kernel.astype(np.float32))
        weight = weight.view(1, 1, *weight.shape).repeat(channels, 1, 1, 1)
        pad_mode = "reflect" if min(sample.shape[-2:]) > radius else "replicate"
        padded = self.torch.nn.functional.pad(
            sample, (radius, radius, radius, radius), mode=pad_mode
        )
        return self.torch.nn.functional.conv2d(padded, weight, groups=channels)

    def test_blur_matches_square_gaussian_for_channels_and_boundary_modes(self) -> None:
        cases = (
            ((2, 3, 29, 31), 5.0),
            ((1, 1, 9, 17), 1.25),
            ((1, 3, 7, 31), 5.0),
            ((1, 2, 31, 7), 5.0),
            ((1, 3, 1, 2), 8.0),
            ((1, 1, 49, 53), 8.0),
            ((1, 3, 15, 17), 5.0),
            ((1, 3, 3, 4), 0.1),
        )
        generator = self.torch.Generator(device="cpu").manual_seed(1729)
        for shape, sigma in cases:
            sample = self.torch.rand(shape, generator=generator)
            expected = self._square_reference(sample, sigma)
            for profile, module in self.modules:
                with self.subTest(profile=profile, shape=shape, sigma=sigma):
                    actual = module._blur_sample(sample, sigma)
                    self.assertEqual(tuple(actual.shape), shape)
                    self.torch.testing.assert_close(
                        actual, expected, rtol=4e-6, atol=4e-6
                    )

    def test_disabled_blur_preserves_the_input_tensor(self) -> None:
        sample = self.torch.arange(12, dtype=self.torch.float32).reshape(1, 1, 3, 4)
        for profile, module in self.modules:
            for sigma in (0.0, 0.05):
                with self.subTest(profile=profile, sigma=sigma):
                    self.assertIs(module._blur_sample(sample, sigma), sample)

    def test_elastic_field_preserves_seed_and_profile_amplitude(self) -> None:
        seed = 31891
        for (profile, module), amplitude in zip(
            self.modules, (15.0, 20.0, 27.5, 35.0)
        ):
            # Only the pure tensor helper is executed; no CUDA constructor,
            # device probe, pinned staging, or optional compilation is needed.
            policy = module.GPUAugmentation.__new__(module.GPUAugmentation)
            policy.device = self.torch.device("cpu")
            for height, width in ((27, 31), (7, 31), (31, 7), (1, 2)):
                with self.subTest(profile=profile, shape=(height, width)):
                    generator = self.torch.Generator(device="cpu")
                    generator.manual_seed(module._subseed(seed, 101))
                    noise = self.torch.rand(
                        (1, 2, height, width),
                        generator=generator,
                        dtype=self.torch.float32,
                    ) * 2.0 - 1.0
                    expected = self._square_reference(noise, 5.0)[0] * amplitude
                    actual = policy._elastic_displacement(seed, height, width)
                    self.torch.testing.assert_close(
                        actual, expected, rtol=4e-6, atol=2e-5
                    )
                    self.assertTrue(
                        self.torch.equal(
                            actual, policy._elastic_displacement(seed, height, width)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
