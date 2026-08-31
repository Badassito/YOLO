"""Heavy GPU augmentation policy for XTA PTA offline generation.

This file is intentionally the complete augmentation policy.  PTA imports only
``build_gpu_augmentation`` and treats the returned object's ``apply_batch``
methods as the single-file GPU contract. This profile preserves the supplied
policy's transform choices while increasing their sampled magnitudes.

The port is distribution-compatible rather than pixel-identical:

* D4 and affine are composed into one destination-to-source grid.
* Optional elastic displacement is added to that grid, so image and mask each
  incur one geometric resampling pass.
* Brightness and the selected noise family are applied on CUDA tensors.
* Gaussian blur is executed only for selected samples.
* ``None`` seeds identify unaugmented originals; integer seeds are deterministic
  across GPU count and batch size.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PTA_GPU_POLICY_API = 2
PTA_GPU_RUNTIME = "torch-cuda-fused-grid-v1"
AUGMENTATION_PROFILE = "heavy"


def _subseed(seed: int, salt: int) -> int:
    value = (int(seed) ^ (int(salt) * 0x9E3779B97F4A7C15)) & ((1 << 63) - 1)
    return value or 1


def _gaussian_kernel_2d(sigma: float, *, device: torch.device) -> torch.Tensor:
    sigma_f = max(0.05, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma_f)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma_f * sigma_f))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def _blur_sample(sample: torch.Tensor, sigma: float) -> torch.Tensor:
    if float(sigma) <= 0.05:
        return sample
    channels = int(sample.shape[1])
    kernel = _gaussian_kernel_2d(float(sigma), device=sample.device)
    radius = int(kernel.shape[0]) // 2
    weight = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
    pad_mode = "reflect" if min(int(sample.shape[-2]), int(sample.shape[-1])) > radius else "replicate"
    padded = F.pad(sample, (radius, radius, radius, radius), mode=pad_mode)
    return F.conv2d(padded, weight, groups=channels)


def _fused_pointwise(
    images: torch.Tensor,
    brightness: torch.Tensor,
    multiplier: torch.Tensor,
    additive: torch.Tensor,
    shot_selected: torch.Tensor,
    shot_values: torch.Tensor,
    pepper: torch.Tensor,
    salt: torch.Tensor,
) -> torch.Tensor:
    regular = images * brightness * multiplier + additive
    noisy = torch.where(shot_selected, shot_values, regular)
    noisy = torch.where(pepper, torch.zeros_like(noisy), noisy)
    noisy = torch.where(salt, torch.ones_like(noisy), noisy)
    return torch.clamp(noisy, 0.0, 1.0)


class GPUAugmentation:
    """Torch-CUDA implementation of the baseline probability graph."""

    supports_cuda_sources = True

    def __init__(self, *, device: str, batch_size: int = 32) -> None:
        self.device = torch.device(str(device))
        if self.device.type != "cuda":
            raise ValueError(f"GPUAugmentation requires a CUDA device, got {self.device}")
        self.batch_size = max(1, int(batch_size))
        self._pixel_grid_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._pointwise_kernel = _fused_pointwise
        self._pointwise_compiled = False
        if os.environ.get("PTA_GPU_TORCH_COMPILE", "1").strip().lower() not in {"0", "false", "no"}:
            compile_fn = getattr(torch, "compile", None)
            if callable(compile_fn):
                try:
                    self._pointwise_kernel = compile_fn(
                        _fused_pointwise,
                        fullgraph=True,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                    self._pointwise_compiled = True
                except Exception:
                    self._pointwise_kernel = _fused_pointwise
                    self._pointwise_compiled = False

    def _pixel_grid(self, height: int, width: int) -> torch.Tensor:
        key = (int(height), int(width))
        cached = self._pixel_grid_cache.get(key)
        if cached is not None:
            return cached
        ys, xs = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        grid = torch.stack((xs, ys, ones), dim=-1)
        self._pixel_grid_cache[key] = grid
        return grid

    @staticmethod
    def _sample_parameters(seed: int, height: int, width: int) -> Dict[str, object]:
        rng = random.Random(int(seed))
        d4 = int(rng.randrange(8))
        rotation = float(rng.uniform(-55.0, 55.0))
        if rng.random() < 0.5:
            scale = float(rng.uniform(0.4, 1.0))
        else:
            scale = float(rng.uniform(1.0, 2.5))
        translate_x = float(rng.uniform(-0.125, 0.125) * width)
        translate_y = float(rng.uniform(-0.125, 0.125) * height)
        shear_x = float(rng.uniform(-37.5, 37.5))
        shear_y = float(rng.uniform(-37.5, 37.5))
        elastic = bool(rng.random() < 0.30)
        brightness = float(rng.uniform(0.75, 1.25)) if rng.random() < 0.50 else 1.0
        blur_sigma = float(rng.uniform(0.0, 6.5)) if rng.random() < 0.25 else 0.0
        noise_family = int(rng.randrange(3))
        noise_strength = (
            float(rng.uniform(0.0, 0.65))
            if noise_family == 0
            else float(rng.uniform(0.0, 0.065))
            if noise_family == 1
            else float(rng.uniform(0.35, 1.65))
        )
        salt_pepper_amount = float(rng.uniform(0.0, 0.065)) if rng.random() < 0.25 else 0.0
        return {
            "d4": d4,
            "rotation": rotation,
            "scale": scale,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "shear_x": shear_x,
            "shear_y": shear_y,
            "elastic": elastic,
            "brightness": brightness,
            "blur_sigma": blur_sigma,
            "noise_family": noise_family,
            "noise_strength": noise_strength,
            "salt_pepper_amount": salt_pepper_amount,
        }

    @staticmethod
    def _forward_matrix(params: Dict[str, object], height: int, width: int) -> List[List[float]]:
        d4 = int(params["d4"])
        quarter_angle = math.radians(90.0 * float(d4 % 4))
        qc, qs = math.cos(quarter_angle), math.sin(quarter_angle)
        reflect = -1.0 if d4 >= 4 else 1.0
        d00, d01 = qc * reflect, -qs
        d10, d11 = qs * reflect, qc

        sx = math.tan(math.radians(float(params["shear_x"])))
        sy = math.tan(math.radians(float(params["shear_y"])))
        scale = float(params["scale"])
        # Shear @ (uniform scale * D4).
        a00 = scale * (d00 + sx * d10)
        a01 = scale * (d01 + sx * d11)
        a10 = scale * (sy * d00 + d10)
        a11 = scale * (sy * d01 + d11)

        angle = math.radians(float(params["rotation"]))
        c, s = math.cos(angle), math.sin(angle)
        l00, l01 = c * a00 - s * a10, c * a01 - s * a11
        l10, l11 = s * a00 + c * a10, s * a01 + c * a11

        center_x = (float(width) - 1.0) * 0.5
        center_y = (float(height) - 1.0) * 0.5
        tx = center_x + float(params["translate_x"]) - (l00 * center_x + l01 * center_y)
        ty = center_y + float(params["translate_y"]) - (l10 * center_x + l11 * center_y)
        return [[l00, l01, tx], [l10, l11, ty], [0.0, 0.0, 1.0]]

    def _elastic_displacement(self, seed: int, height: int, width: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(_subseed(seed, 101))
        noise = torch.rand(
            (1, 2, height, width),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        ) * 2.0 - 1.0
        kernel = _gaussian_kernel_2d(5.0, device=self.device)
        radius = int(kernel.shape[0]) // 2
        weight = kernel.view(1, 1, *kernel.shape).repeat(2, 1, 1, 1)
        pad_mode = "reflect" if min(int(height), int(width)) > radius else "replicate"
        padded = F.pad(noise, (radius, radius, radius, radius), mode=pad_mode)
        return F.conv2d(padded, weight, groups=2)[0] * 27.5

    def _apply_intensity_noise(
        self,
        images: torch.Tensor,
        seeds: Sequence[Optional[int]],
        params_by_sample: Sequence[Optional[Dict[str, object]]],
    ) -> torch.Tensor:
        output = images.clone()
        count = int(output.shape[0])
        brightness = torch.ones((count, 1, 1, 1), device=self.device, dtype=output.dtype)
        multiplier = torch.ones_like(output)
        additive = torch.zeros_like(output)
        shot_selected = torch.zeros((count, 1, 1, 1), device=self.device, dtype=torch.bool)
        shot_values = torch.zeros_like(output)
        pepper = torch.zeros_like(output, dtype=torch.bool)
        salt = torch.zeros_like(output, dtype=torch.bool)
        for index, (seed, params) in enumerate(zip(seeds, params_by_sample)):
            if seed is None or params is None:
                continue
            sample = _blur_sample(output[index:index + 1], float(params["blur_sigma"]))
            output[index:index + 1] = sample
            brightness[index] = float(params["brightness"])
            generator = torch.Generator(device=self.device)
            generator.manual_seed(_subseed(int(seed), 211))
            family = int(params["noise_family"])
            strength = float(params["noise_strength"])
            if family == 0 and strength > 0.0:
                additive[index:index + 1] = torch.randn(
                    sample.shape,
                    device=self.device,
                    dtype=sample.dtype,
                    generator=generator,
                ) * strength
            elif family == 1 and strength > 1e-6:
                shot_selected[index] = True
                shot_input = torch.clamp(sample * float(params["brightness"]), 0.0, 1.0)
                shot_values[index:index + 1] = torch.poisson(
                    shot_input / strength,
                    generator=generator,
                ) * strength
            elif family == 2:
                multiplier[index:index + 1] = torch.empty_like(sample).uniform_(
                    0.35,
                    1.65,
                    generator=generator,
                )

            amount = float(params["salt_pepper_amount"])
            if amount > 0.0:
                chooser = torch.rand(
                    sample.shape,
                    device=self.device,
                    dtype=sample.dtype,
                    generator=generator,
                )
                pepper[index:index + 1] = chooser < (amount * 0.5)
                salt[index:index + 1] = chooser > (1.0 - amount * 0.5)
        arguments = (
            output,
            brightness,
            multiplier,
            additive,
            shot_selected,
            shot_values,
            pepper,
            salt,
        )
        try:
            return self._pointwise_kernel(*arguments)
        except Exception:
            if not self._pointwise_compiled:
                raise
            self._pointwise_kernel = _fused_pointwise
            self._pointwise_compiled = False
            return _fused_pointwise(*arguments)

    @torch.inference_mode()
    def apply_batch_many(
        self,
        *,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        seeds: Sequence[Sequence[Optional[int]]],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upload several source ROIs once and emit one flat CUDA batch.

        Results are source-major and preserve the order of each nested seed
        sequence. A ``None`` seed identifies an unaugmented original.
        """
        if not images or len(images) != len(masks) or len(images) != len(seeds):
            raise ValueError(
                "apply_batch_many requires equally sized nonempty images, masks, and seeds"
            )
        flat_seeds = [seed for source_seeds in seeds for seed in source_seeds]
        if not flat_seeds:
            raise ValueError("apply_batch_many requires at least one output seed/original marker")
        if len(flat_seeds) > self.batch_size:
            raise ValueError(
                f"batch length {len(flat_seeds)} exceeds configured batch_size={self.batch_size}"
            )
        out_h, out_w = int(output_size[0]), int(output_size[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"invalid output_size={output_size}")

        cuda_image_inputs = [
            bool(torch.is_tensor(image) and bool(getattr(image, "is_cuda", False)))
            for image in images
        ]
        if any(cuda_image_inputs) and not all(cuda_image_inputs):
            raise ValueError("apply_batch_many cannot mix NumPy and CUDA source images")
        image_arrays = (
            []
            if all(cuda_image_inputs)
            else [np.ascontiguousarray(np.asarray(image), dtype=np.uint8) for image in images]
        )
        mask_arrays = [
            np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8))
            for mask in masks
        ]
        image_shape = tuple(images[0].shape) if all(cuda_image_inputs) else tuple(image_arrays[0].shape)
        if any(tuple(image.shape) != image_shape for image in images):
            raise ValueError("apply_batch_many source images must share one shape")
        if any(mask.ndim != 2 or tuple(mask.shape) != image_shape[:2] for mask in mask_arrays):
            raise ValueError(
                f"apply_batch_many image/mask shape mismatch for source shape={image_shape}"
            )
        if all(cuda_image_inputs):
            for image in images:
                if image.dtype != torch.uint8 or image.device != self.device:
                    raise ValueError(
                        "CUDA policy source images must be uint8 tensors on "
                        f"{self.device}; got dtype={image.dtype}, device={image.device}"
                    )
            if len(image_shape) == 2:
                image_tensor = torch.stack(
                    [image.unsqueeze(0) for image in images], dim=0
                ).contiguous()
            elif len(image_shape) == 3 and int(image_shape[2]) >= 1:
                image_tensor = torch.stack(
                    [image.permute(2, 0, 1) for image in images], dim=0
                ).contiguous()
            else:
                raise ValueError(
                    f"GPU policy supports HxW gray or HxWxC images with C>=1, got {image_shape}"
                )
        elif len(image_shape) == 2:
            image_nchw = np.stack(image_arrays, axis=0)[:, None, :, :]
            image_host = torch.from_numpy(np.ascontiguousarray(image_nchw)).pin_memory()
            image_tensor = image_host.to(self.device, non_blocking=True)
        elif len(image_shape) == 3 and int(image_shape[2]) >= 1:
            image_nchw = np.transpose(np.stack(image_arrays, axis=0), (0, 3, 1, 2))
            image_host = torch.from_numpy(np.ascontiguousarray(image_nchw)).pin_memory()
            image_tensor = image_host.to(self.device, non_blocking=True)
        else:
            raise ValueError(
                f"GPU policy supports HxW gray or HxWxC images with C>=1, got {image_shape}"
            )
        mask_nhw = np.stack(mask_arrays, axis=0)
        mask_host = torch.from_numpy(np.ascontiguousarray(mask_nhw)).pin_memory()
        mask_tensor = mask_host.to(self.device, non_blocking=True).unsqueeze(1)

        image_float = image_tensor.to(torch.float32) / 255.0
        mask_float = mask_tensor.to(torch.float32)
        if tuple(image_float.shape[-2:]) != (out_h, out_w):
            image_float = F.interpolate(image_float, size=(out_h, out_w), mode="bilinear", align_corners=False)
            mask_float = F.interpolate(mask_float, size=(out_h, out_w), mode="nearest")

        source_indices = torch.as_tensor(
            [
                source_index
                for source_index, source_seeds in enumerate(seeds)
                for _seed in source_seeds
            ],
            device=self.device,
            dtype=torch.int64,
        )
        image_batch = image_float.index_select(0, source_indices)
        mask_batch = mask_float.index_select(0, source_indices)
        if all(seed is None for seed in flat_seeds):
            # ``augmentation_ratio=1`` emits originals only. The rendered
            # sources have already received any required output resize above,
            # so building an identity inverse matrix/grid and sampling both
            # tensors again is pure overhead.
            images_u8 = torch.clamp(
                torch.round(image_batch * 255.0), 0.0, 255.0
            ).to(torch.uint8)
            masks_u8 = (mask_batch[:, 0] >= 0.5).to(torch.uint8)
            return images_u8.contiguous(), masks_u8.contiguous()
        params_by_sample: List[Optional[Dict[str, object]]] = []
        forward_matrices: List[List[List[float]]] = []
        for seed in flat_seeds:
            if seed is None:
                params_by_sample.append(None)
                forward_matrices.append([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            else:
                params = self._sample_parameters(int(seed), out_h, out_w)
                params_by_sample.append(params)
                forward_matrices.append(self._forward_matrix(params, out_h, out_w))

        forward = torch.tensor(forward_matrices, device=self.device, dtype=torch.float32)
        inverse = torch.linalg.inv(forward)
        pixel_grid = self._pixel_grid(out_h, out_w)
        source = torch.einsum("nij,hwj->nhwi", inverse, pixel_grid)
        for index, (seed, params) in enumerate(zip(flat_seeds, params_by_sample)):
            if seed is not None and params is not None and bool(params["elastic"]):
                displacement = self._elastic_displacement(int(seed), out_h, out_w)
                source[index, :, :, 0] += displacement[0]
                source[index, :, :, 1] += displacement[1]

        if out_w > 1:
            grid_x = source[:, :, :, 0] * (2.0 / float(out_w - 1)) - 1.0
        else:
            grid_x = torch.zeros_like(source[:, :, :, 0])
        if out_h > 1:
            grid_y = source[:, :, :, 1] * (2.0 / float(out_h - 1)) - 1.0
        else:
            grid_y = torch.zeros_like(source[:, :, :, 1])
        sampling_grid = torch.stack((grid_x, grid_y), dim=-1)

        warped_images = F.grid_sample(
            image_batch,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_masks = F.grid_sample(
            mask_batch,
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_images = self._apply_intensity_noise(warped_images, flat_seeds, params_by_sample)
        images_u8 = torch.clamp(torch.round(warped_images * 255.0), 0.0, 255.0).to(torch.uint8)
        masks_u8 = (warped_masks[:, 0] >= 0.5).to(torch.uint8)
        return images_u8.contiguous(), masks_u8.contiguous()

    @torch.inference_mode()
    def apply_batch(
        self,
        *,
        image: np.ndarray,
        mask: np.ndarray,
        seeds: Sequence[Optional[int]],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compatibility wrapper for one-source GPU policy callers."""

        return self.apply_batch_many(
            images=(image,),
            masks=(mask,),
            seeds=(tuple(seeds),),
            output_size=output_size,
        )


def build_gpu_augmentation(*, device: str, batch_size: int = 32) -> GPUAugmentation:
    """XTA PTA single-file GPU policy factory."""
    return GPUAugmentation(device=device, batch_size=batch_size)
