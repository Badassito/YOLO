"""CPU companion to the light XTA PTA GPU augmentation policy.

The policy preserves the light transform graph, parameter ranges, affine
composition, image/mask interpolation choices, and integer-seed contract. It is
distribution-compatible rather than pixel-identical: OpenCV and NumPy do not
share CUDA's resampling implementation or random-number streams.

XTA's CPU external-policy loader calls set_random_seed before each image/mask
pair. The returned object intentionally implements that small callable contract
without constructing an Albumentations transform tree.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping

import cv2
import numpy as np


AUGMENTATION_PROFILE = "light"


def _subseed(seed: int, salt: int) -> int:
    value = (int(seed) ^ (int(salt) * 0x9E3779B97F4A7C15)) & ((1 << 63) - 1)
    return value or 1


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma_f = max(0.05, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma_f)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2.0 * sigma_f * sigma_f))
    return np.ascontiguousarray(kernel / np.sum(kernel), dtype=np.float32)


def _blur_sample(sample: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.05:
        return sample
    kernel = _gaussian_kernel_1d(float(sigma))
    radius = int(kernel.size) // 2
    border = (
        cv2.BORDER_REFLECT_101
        if min(int(sample.shape[0]), int(sample.shape[1])) > radius
        else cv2.BORDER_REPLICATE
    )
    blurred = cv2.sepFilter2D(
        sample,
        ddepth=-1,
        kernelX=kernel,
        kernelY=kernel,
        borderType=border,
    )
    if sample.ndim == 3 and blurred.ndim == 2:
        blurred = blurred[:, :, None]
    return np.ascontiguousarray(blurred, dtype=np.float32)


class CPUAugmentation:
    """Seedable OpenCV/NumPy implementation of the light probability graph."""

    def __init__(self) -> None:
        self._seed = 1
        self._mask_interpolation = cv2.INTER_NEAREST

    def set_random_seed(self, seed: int) -> None:
        self._seed = int(seed)

    def set_mask_interpolation(self, interpolation: int) -> None:
        if int(interpolation) != int(cv2.INTER_NEAREST):
            raise ValueError("CPUAugmentation requires nearest-neighbor mask interpolation")
        self._mask_interpolation = int(interpolation)

    @staticmethod
    def _sample_parameters(seed: int, height: int, width: int) -> Dict[str, object]:
        rng = random.Random(int(seed))
        d4 = int(rng.randrange(8))
        rotation = float(rng.uniform(-35.0, 35.0))
        if rng.random() < 0.5:
            scale = float(rng.uniform(2.0 / 3.0, 1.0))
        else:
            scale = float(rng.uniform(1.0, 1.5))
        translate_x = float(rng.uniform(-0.075, 0.075) * width)
        translate_y = float(rng.uniform(-0.075, 0.075) * height)
        shear_x = float(rng.uniform(-22.5, 22.5))
        shear_y = float(rng.uniform(-22.5, 22.5))
        elastic = bool(rng.random() < 0.30)
        brightness = float(rng.uniform(0.85, 1.15)) if rng.random() < 0.50 else 1.0
        blur_sigma = float(rng.uniform(0.0, 3.5)) if rng.random() < 0.25 else 0.0
        noise_family = int(rng.randrange(3))
        noise_strength = (
            float(rng.uniform(0.0, 0.35))
            if noise_family == 0
            else float(rng.uniform(0.0, 0.035))
            if noise_family == 1
            else float(rng.uniform(0.65, 1.35))
        )
        salt_pepper_amount = (
            float(rng.uniform(0.0, 0.035)) if rng.random() < 0.25 else 0.0
        )
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
    def _forward_matrix(
        params: Mapping[str, object],
        height: int,
        width: int,
    ) -> List[List[float]]:
        d4 = int(params["d4"])
        quarter_angle = math.radians(90.0 * float(d4 % 4))
        qc, qs = math.cos(quarter_angle), math.sin(quarter_angle)
        reflect = -1.0 if d4 >= 4 else 1.0
        d00, d01 = qc * reflect, -qs
        d10, d11 = qs * reflect, qc

        sx = math.tan(math.radians(float(params["shear_x"])))
        sy = math.tan(math.radians(float(params["shear_y"])))
        scale = float(params["scale"])
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
        tx = (
            center_x
            + float(params["translate_x"])
            - (l00 * center_x + l01 * center_y)
        )
        ty = (
            center_y
            + float(params["translate_y"])
            - (l10 * center_x + l11 * center_y)
        )
        return [[l00, l01, tx], [l10, l11, ty], [0.0, 0.0, 1.0]]

    @staticmethod
    def _elastic_displacement(seed: int, height: int, width: int) -> np.ndarray:
        generator = np.random.default_rng(_subseed(seed, 101))
        noise = generator.uniform(-1.0, 1.0, size=(height, width, 2)).astype(
            np.float32
        )
        kernel = _gaussian_kernel_1d(5.0)
        radius = int(kernel.size) // 2
        border = (
            cv2.BORDER_REFLECT_101
            if min(int(height), int(width)) > radius
            else cv2.BORDER_REPLICATE
        )
        displacement = cv2.sepFilter2D(
            noise,
            ddepth=-1,
            kernelX=kernel,
            kernelY=kernel,
            borderType=border,
        )
        return np.ascontiguousarray(displacement * 15.0, dtype=np.float32)

    @staticmethod
    def _apply_intensity_noise(
        image: np.ndarray,
        *,
        seed: int,
        params: Mapping[str, object],
    ) -> np.ndarray:
        sample = _blur_sample(image, float(params["blur_sigma"]))
        generator = np.random.default_rng(_subseed(seed, 211))
        family = int(params["noise_family"])
        strength = float(params["noise_strength"])
        brightness = float(params["brightness"])

        if family == 0 and strength > 0.0:
            additive = generator.standard_normal(sample.shape).astype(np.float32)
            output = sample * brightness + additive * strength
        elif family == 1 and strength > 1e-6:
            shot_input = np.clip(sample * brightness, 0.0, 1.0)
            output = generator.poisson(shot_input / strength).astype(np.float32)
            output *= strength
        elif family == 2:
            multiplier = generator.uniform(0.65, 1.35, size=sample.shape).astype(
                np.float32
            )
            output = sample * brightness * multiplier
        else:
            output = sample * brightness

        amount = float(params["salt_pepper_amount"])
        if amount > 0.0:
            chooser = generator.random(sample.shape).astype(np.float32)
            output = np.where(chooser < amount * 0.5, 0.0, output)
            output = np.where(chooser > 1.0 - amount * 0.5, 1.0, output)
        return np.ascontiguousarray(np.clip(output, 0.0, 1.0), dtype=np.float32)

    def __call__(self, *, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask)
        if image_array.dtype != np.uint8 or image_array.ndim not in (2, 3):
            raise ValueError(
                "CPUAugmentation expects a uint8 HxW or HxWxC image; "
                f"got shape={image_array.shape}, dtype={image_array.dtype}"
            )
        if (
            mask_array.ndim != 2
            or tuple(mask_array.shape) != tuple(image_array.shape[:2])
        ):
            raise ValueError(
                "CPUAugmentation image/mask shape mismatch: "
                f"image={image_array.shape}, mask={mask_array.shape}"
            )

        height, width = (int(value) for value in image_array.shape[:2])
        params = self._sample_parameters(self._seed, height, width)
        forward = np.asarray(
            self._forward_matrix(params, height, width),
            dtype=np.float64,
        )
        inverse = np.linalg.inv(forward)
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        source_x = (
            inverse[0, 0] * grid_x
            + inverse[0, 1] * grid_y
            + inverse[0, 2]
        ).astype(np.float32)
        source_y = (
            inverse[1, 0] * grid_x
            + inverse[1, 1] * grid_y
            + inverse[1, 2]
        ).astype(np.float32)
        if bool(params["elastic"]):
            displacement = self._elastic_displacement(self._seed, height, width)
            source_x += displacement[:, :, 0]
            source_y += displacement[:, :, 1]

        image_float = np.ascontiguousarray(
            image_array.astype(np.float32) / 255.0
        )
        warped_image = cv2.remap(
            image_float,
            source_x,
            source_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if image_array.ndim == 3 and warped_image.ndim == 2:
            warped_image = warped_image[:, :, None]
        warped_mask = cv2.remap(
            np.ascontiguousarray((mask_array > 0).astype(np.uint8)),
            source_x,
            source_y,
            interpolation=self._mask_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        augmented = self._apply_intensity_noise(
            np.ascontiguousarray(warped_image, dtype=np.float32),
            seed=self._seed,
            params=params,
        )
        image_u8 = np.clip(np.rint(augmented * 255.0), 0.0, 255.0).astype(
            np.uint8
        )
        mask_u8 = np.ascontiguousarray((warped_mask >= 0.5).astype(np.uint8))
        return {
            "image": np.ascontiguousarray(image_u8),
            "mask": mask_u8,
        }


def build_augmentation() -> CPUAugmentation:
    """Build a seedable light policy for XTA's CPU augmentation backend."""
    return CPUAugmentation()
