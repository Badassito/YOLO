from __future__ import annotations

import ctypes
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from XTA import nvtiff_backend


def _integer(value: object) -> int:
    return int(getattr(value, "value", value))


class _FakeFunction:
    def __init__(self, owner: "_FakeNvTiffLibrary", name: str) -> None:
        self.owner = owner
        self.name = name
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self, *arguments: object) -> int:
        self.owner.calls.append((self.name, arguments))
        status = int(self.owner.failures.get(self.name, 0))
        if status != 0:
            return status

        if self.name == "nvtiffGetProperty":
            property_type = _integer(arguments[0])
            output = ctypes.cast(arguments[1], ctypes.POINTER(ctypes.c_int))
            output[0] = int(self.owner.version[property_type])
        elif self.name == "nvtiffEncoderCreate":
            output = ctypes.cast(arguments[0], ctypes.POINTER(ctypes.c_void_p))
            output[0] = ctypes.c_void_p(self.owner.encoder_handle)
        elif self.name == "nvtiffEncodeParamsCreate":
            output = ctypes.cast(arguments[0], ctypes.POINTER(ctypes.c_void_p))
            output[0] = ctypes.c_void_p(self.owner.params_handle)
        elif self.name == "nvtiffEncodeParamsSetImageInfo":
            pointer = ctypes.cast(
                arguments[1], ctypes.POINTER(nvtiff_backend._NvTiffImageInfo)
            )
            self.owner.image_info = nvtiff_backend._NvTiffImageInfo.from_buffer_copy(
                ctypes.string_at(pointer, ctypes.sizeof(nvtiff_backend._NvTiffImageInfo))
            )
        elif self.name == "nvtiffEncodeParamsSetTiffVariant":
            self.owner.tiff_variants.append(_integer(arguments[1]))
        elif self.name == "nvtiffEncodeParamsSetInputs":
            count = _integer(arguments[2])
            pointer_array = ctypes.cast(
                arguments[1],
                ctypes.POINTER(nvtiff_backend._Uint8DevicePointer),
            )
            self.owner.input_pointers = tuple(
                int(ctypes.cast(pointer_array[index], ctypes.c_void_p).value or 0)
                for index in range(count)
            )
        elif self.name in {"nvtiffEncode", "nvtiffEncodeFinalize"}:
            params_array = ctypes.cast(
                arguments[1], ctypes.POINTER(ctypes.c_void_p)
            )
            self.owner.params_array_values.append(int(params_array[0] or 0))
        elif self.name == "nvtiffWriteTiffFile":
            params_array = ctypes.cast(
                arguments[1], ctypes.POINTER(ctypes.c_void_p)
            )
            self.owner.params_array_values.append(int(params_array[0] or 0))
            raw_path = arguments[3]
            assert isinstance(raw_path, bytes)
            output_path = Path(os.fsdecode(raw_path))
            self.owner.native_output_paths.append(output_path)
            output_path.write_bytes(self.owner.output_payload)
        return 0


class _FakeNvTiffLibrary:
    FUNCTION_NAMES = tuple(nvtiff_backend._FUNCTION_SIGNATURES)

    def __init__(
        self,
        *,
        version: tuple[int, int, int] = (0, 8, 0),
        failures: dict[str, int] | None = None,
        output_payload: bytes = b"II*\x00fake-nvtiff",
    ) -> None:
        self.version = tuple(version)
        self.failures = dict(failures or {})
        self.output_payload = bytes(output_payload)
        self.encoder_handle = 0xE001
        self.params_handle = 0xA001
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.image_info: nvtiff_backend._NvTiffImageInfo | None = None
        self.input_pointers: tuple[int, ...] = ()
        self.params_array_values: list[int] = []
        self.native_output_paths: list[Path] = []
        self.tiff_variants: list[int] = []
        for name in self.FUNCTION_NAMES:
            setattr(self, name, _FakeFunction(self, name))

    def clear_calls(self) -> None:
        self.calls.clear()
        self.params_array_values.clear()


class _FakeCudaPages:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (3, 4, 5),
        dtype: object = "torch.uint8",
        is_cuda: bool = True,
        contiguous: bool = True,
        strides: tuple[int, ...] = (20, 5, 1),
        pointer: int = 0x1000,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._contiguous = contiguous
        self._strides = strides
        self._pointer = pointer

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self) -> tuple[int, ...]:
        return self._strides

    def data_ptr(self) -> int:
        return self._pointer


class NvTiffAbiTests(unittest.TestCase):
    def _backend(
        self,
        library: _FakeNvTiffLibrary | None = None,
    ) -> tuple[nvtiff_backend.NvTiffBackend, _FakeNvTiffLibrary]:
        fake = library or _FakeNvTiffLibrary()
        backend = nvtiff_backend.NvTiffBackend(
            0, "<fake-libnvtiff>", _library=fake
        )
        fake.clear_calls()
        return backend, fake

    def test_ctypes_signatures_match_public_encode_abi(self) -> None:
        _, fake = self._backend()

        self.assertEqual(
            fake.nvtiffGetProperty.argtypes,
            [ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
        )
        self.assertEqual(
            fake.nvtiffEncodeParamsSetImageInfo.argtypes,
            [
                ctypes.c_void_p,
                ctypes.POINTER(nvtiff_backend._NvTiffImageInfo),
            ],
        )
        self.assertEqual(
            fake.nvtiffEncodeParamsSetTiffVariant.argtypes,
            [ctypes.c_void_p, ctypes.c_int],
        )
        self.assertEqual(
            fake.nvtiffEncodeParamsSetInputs.argtypes,
            [
                ctypes.c_void_p,
                ctypes.POINTER(nvtiff_backend._Uint8DevicePointer),
                ctypes.c_uint32,
            ],
        )
        self.assertEqual(
            fake.nvtiffWriteTiffFile.argtypes,
            [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.c_void_p,
            ],
        )

    def test_image_info_layout_matches_nvtiff_08_header(self) -> None:
        image_info = nvtiff_backend._NvTiffImageInfo
        self.assertEqual(ctypes.sizeof(image_info), 124)
        self.assertEqual(
            {
                name: getattr(image_info, name).offset
                for name, _field_type in image_info._fields_
            },
            {
                "image_type": 0,
                "image_width": 4,
                "image_height": 8,
                "compression": 12,
                "photometric_int": 16,
                "planar_config": 20,
                "samples_per_pixel": 24,
                "bits_per_pixel": 26,
                "bits_per_sample": 28,
                "sample_format": 60,
            },
        )

    def test_success_sets_lossless_multipage_fields_and_preserves_page_order(self) -> None:
        backend, fake = self._backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "custom.tif"
            result = backend.write_multipage_lzw_from_device_pointers(
                output,
                (0x1000, 0x2000, 0x3000),
                height=7,
                width=11,
                cuda_stream=0xCAFE,
            )

            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes(), b"II*\x00fake-nvtiff")
        backend.close()

        self.assertEqual(
            [name for name, _arguments in fake.calls],
            [
                "nvtiffEncoderCreate",
                "nvtiffEncodeParamsCreate",
                "nvtiffEncodeParamsSetImageInfo",
                "nvtiffEncodeParamsSetInputs",
                "nvtiffEncode",
                "nvtiffEncodeFinalize",
                "nvtiffWriteTiffFile",
                "nvtiffEncodeParamsDestroy",
                "nvtiffEncoderDestroy",
            ],
        )
        self.assertEqual(fake.input_pointers, (0x1000, 0x2000, 0x3000))
        self.assertEqual(
            fake.params_array_values,
            [fake.params_handle, fake.params_handle, fake.params_handle],
        )
        self.assertIsNotNone(fake.image_info)
        assert fake.image_info is not None
        self.assertEqual(fake.image_info.image_type, 0x2)
        self.assertEqual(fake.image_info.image_width, 11)
        self.assertEqual(fake.image_info.image_height, 7)
        self.assertEqual(fake.image_info.compression, 5)
        self.assertEqual(fake.image_info.photometric_int, 1)
        self.assertEqual(fake.image_info.planar_config, 1)
        self.assertEqual(fake.image_info.samples_per_pixel, 1)
        self.assertEqual(fake.image_info.bits_per_pixel, 8)
        self.assertEqual(fake.image_info.bits_per_sample[0], 8)
        self.assertEqual(fake.image_info.sample_format[0], 1)
        self.assertTrue(all(value == 0 for value in fake.image_info.bits_per_sample[1:]))
        self.assertTrue(all(value == 0 for value in fake.image_info.sample_format[1:]))

        create_stream = fake.calls[0][1][3]
        destroy_params_stream = fake.calls[-2][1][1]
        destroy_encoder_stream = fake.calls[-1][1][1]
        self.assertEqual(_integer(create_stream), 0xCAFE)
        self.assertEqual(_integer(destroy_params_stream), 0xCAFE)
        self.assertEqual(_integer(destroy_encoder_stream), 0xCAFE)

    def test_empty_native_success_is_rejected_without_replacing_final(self) -> None:
        backend, _fake = self._backend(
            _FakeNvTiffLibrary(output_payload=b""),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "custom.tif"
            output.write_bytes(b"OLD")
            with self.assertRaisesRegex(
                nvtiff_backend.NvTiffError,
                "empty or invalid staged TIFF",
            ):
                backend.write_multipage_lzw_from_device_pointers(
                    output,
                    (0x1000,),
                    height=7,
                    width=11,
                    cuda_stream=0xCAFE,
                )
            self.assertEqual(output.read_bytes(), b"OLD")
            self.assertEqual(list(Path(temp_dir).glob(".*.nvtiff.tmp")), [])
        backend.close()

    def test_native_handles_are_cleaned_after_every_encode_stage_failure(self) -> None:
        stages = (
            "nvtiffEncoderCreate",
            "nvtiffEncodeParamsCreate",
            "nvtiffEncodeParamsSetImageInfo",
            "nvtiffEncodeParamsSetInputs",
            "nvtiffEncode",
            "nvtiffEncodeFinalize",
            "nvtiffWriteTiffFile",
        )
        for stage in stages:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp_dir:
                backend, fake = self._backend(
                    _FakeNvTiffLibrary(failures={stage: 6})
                )
                output = Path(temp_dir) / "failed.tiff"
                with self.assertRaises(nvtiff_backend.NvTiffCallError) as caught:
                    backend.write_multipage_lzw_from_device_pointers(
                        output,
                        (0x1000, 0x2000),
                        height=4,
                        width=5,
                        cuda_stream=0x123,
                    )
                self.assertEqual(caught.exception.operation, stage)
                self.assertFalse(output.exists())
                self.assertFalse(any(Path(temp_dir).iterdir()))

                call_names = [name for name, _arguments in fake.calls]
                if stage == "nvtiffEncoderCreate":
                    self.assertNotIn("nvtiffEncoderDestroy", call_names)
                    self.assertNotIn("nvtiffEncodeParamsDestroy", call_names)
                elif stage == "nvtiffEncodeParamsCreate":
                    self.assertEqual(call_names[-1], "nvtiffEncoderDestroy")
                    self.assertNotIn("nvtiffEncodeParamsDestroy", call_names)
                else:
                    self.assertEqual(
                        call_names[-2:],
                        ["nvtiffEncodeParamsDestroy", "nvtiffEncoderDestroy"],
                    )

    def test_encoder_destroy_still_runs_when_params_destroy_fails(self) -> None:
        backend, fake = self._backend(
            _FakeNvTiffLibrary(failures={"nvtiffEncodeParamsDestroy": 8})
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "failed.tif"
            with self.assertRaisesRegex(nvtiff_backend.NvTiffError, "cleanup failed"):
                backend.write_multipage_lzw_from_device_pointers(
                    output,
                    (0x1000,),
                    height=4,
                    width=5,
                    cuda_stream=0,
                )
            self.assertFalse(output.exists())
            self.assertFalse(any(Path(temp_dir).iterdir()))

        self.assertEqual(
            [name for name, _arguments in fake.calls][-2:],
            ["nvtiffEncodeParamsDestroy", "nvtiffEncoderDestroy"],
        )

    def test_cleanup_failure_is_attached_without_masking_primary_error(self) -> None:
        fake = _FakeNvTiffLibrary(
            failures={
                "nvtiffEncode": 6,
                "nvtiffEncodeParamsDestroy": 8,
            }
        )
        backend, fake = self._backend(fake)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(nvtiff_backend.NvTiffCallError) as caught:
                backend.write_multipage_lzw_from_device_pointers(
                    Path(temp_dir) / "failed.tif",
                    (0x1000,),
                    height=4,
                    width=5,
                    cuda_stream=0,
                )
        self.assertEqual(caught.exception.operation, "nvtiffEncode")
        notes = getattr(caught.exception, "__notes__", ())
        self.assertTrue(any("cleanup also failed" in note for note in notes))
        self.assertEqual(fake.calls[-1][0], "nvtiffEncoderDestroy")

    def test_persistent_encoder_is_reused_and_close_is_idempotent(self) -> None:
        backend, fake = self._backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            for index in range(2):
                backend.write_multipage_lzw_from_device_pointers(
                    Path(temp_dir) / f"channels-{index}.tif",
                    (0x1000, 0x2000),
                    height=4,
                    width=5,
                    cuda_stream=0x555,
                )
        backend.close()
        backend.close()

        call_names = [name for name, _arguments in fake.calls]
        self.assertEqual(call_names.count("nvtiffEncoderCreate"), 1)
        self.assertEqual(call_names.count("nvtiffEncode"), 2)
        self.assertEqual(call_names.count("nvtiffEncodeParamsCreate"), 2)
        self.assertEqual(call_names.count("nvtiffEncodeParamsDestroy"), 2)
        self.assertEqual(call_names.count("nvtiffEncoderDestroy"), 1)

    def test_poisoned_encoder_is_recreated_after_failure(self) -> None:
        backend, fake = self._backend()
        fake.failures["nvtiffEncode"] = 6
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(nvtiff_backend.NvTiffCallError):
                backend.write_multipage_lzw_from_device_pointers(
                    Path(temp_dir) / "failed.tif",
                    (0x1000,),
                    height=4,
                    width=5,
                    cuda_stream=0x555,
                )
            del fake.failures["nvtiffEncode"]
            backend.write_multipage_lzw_from_device_pointers(
                Path(temp_dir) / "recovered.tif",
                (0x1000,),
                height=4,
                width=5,
                cuda_stream=0x555,
            )
        backend.close()

        call_names = [name for name, _arguments in fake.calls]
        self.assertEqual(call_names.count("nvtiffEncoderCreate"), 2)
        self.assertEqual(call_names.count("nvtiffEncoderDestroy"), 2)

    def test_backend_rejects_stream_change_and_use_after_close(self) -> None:
        backend, _fake = self._backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            backend.write_multipage_lzw_from_device_pointers(
                Path(temp_dir) / "first.tif",
                (0x1000,),
                height=4,
                width=5,
                cuda_stream=0x111,
            )
            with self.assertRaisesRegex(ValueError, "one backend per stream"):
                backend.write_multipage_lzw_from_device_pointers(
                    Path(temp_dir) / "wrong-stream.tif",
                    (0x1000,),
                    height=4,
                    width=5,
                    cuda_stream=0x222,
                )
            backend.close()
            with self.assertRaisesRegex(nvtiff_backend.NvTiffError, "closed"):
                backend.write_multipage_lzw_from_device_pointers(
                    Path(temp_dir) / "closed.tif",
                    (0x1000,),
                    height=4,
                    width=5,
                    cuda_stream=0x111,
                )

    def test_large_raw_stack_selects_bigtiff(self) -> None:
        backend, fake = self._backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            backend.write_multipage_lzw_from_device_pointers(
                Path(temp_dir) / "large.tif",
                (0x1000,),
                height=65536,
                width=65536,
                cuda_stream=0,
            )
        backend.close()
        self.assertEqual(fake.tiff_variants, [1])

    def test_bigtiff_selection_failure_destroys_params_and_encoder(self) -> None:
        backend, fake = self._backend(
            _FakeNvTiffLibrary(
                failures={"nvtiffEncodeParamsSetTiffVariant": 2}
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "large.tif"
            with self.assertRaises(nvtiff_backend.NvTiffCallError) as caught:
                backend.write_multipage_lzw_from_device_pointers(
                    output,
                    (0x1000,),
                    height=65536,
                    width=65536,
                    cuda_stream=0,
                )
            self.assertFalse(output.exists())
        self.assertEqual(
            caught.exception.operation,
            "nvtiffEncodeParamsSetTiffVariant",
        )
        self.assertEqual(
            [name for name, _arguments in fake.calls][-2:],
            ["nvtiffEncodeParamsDestroy", "nvtiffEncoderDestroy"],
        )


class NvTiffInputTests(unittest.TestCase):
    def test_tensor_adapter_derives_tightly_packed_page_pointers(self) -> None:
        fake = _FakeNvTiffLibrary()
        backend = nvtiff_backend.NvTiffBackend(0, "<fake>", _library=fake)
        fake.clear_calls()
        pages = _FakeCudaPages()
        with tempfile.TemporaryDirectory() as temp_dir:
            backend.write_multipage_lzw(
                Path(temp_dir) / "channels.tiff",
                pages,
                cuda_stream=0x777,
            )
        backend.close()
        self.assertEqual(fake.input_pointers, (0x1000, 0x1014, 0x1028))

    def test_tensor_adapter_accepts_cuda_array_interface(self) -> None:
        class InterfacePages:
            shape = (2, 3, 4)
            dtype = None
            __cuda_array_interface__ = {
                "shape": shape,
                "typestr": "|u1",
                "strides": None,
                "data": (0x5000, False),
                "version": 3,
                "stream": 0xBEEF,
            }

        pointers, height, width = nvtiff_backend._cuda_pages_device_pointers(
            InterfacePages()
        )
        self.assertEqual((pointers, height, width), ((0x5000, 0x500C), 3, 4))

    def test_tensor_adapter_uses_cai_v3_producer_stream(self) -> None:
        class InterfacePages:
            shape = (2, 3, 4)
            dtype = None
            __cuda_array_interface__ = {
                "shape": shape,
                "typestr": "|u1",
                "strides": None,
                "data": (0x5000, False),
                "version": 3,
                "stream": 0xBEEF,
            }

        fake = _FakeNvTiffLibrary()
        backend = nvtiff_backend.NvTiffBackend(0, "<fake>", _library=fake)
        fake.clear_calls()
        with tempfile.TemporaryDirectory() as temp_dir:
            backend.write_multipage_lzw(
                Path(temp_dir) / "cai.tif",
                InterfacePages(),
            )
        backend.close()
        create_call = next(
            arguments
            for name, arguments in fake.calls
            if name == "nvtiffEncoderCreate"
        )
        self.assertEqual(_integer(create_call[3]), 0xBEEF)

    def test_tensor_writer_rejects_unsynchronized_cai_stream_override(self) -> None:
        class InterfacePages:
            shape = (2, 3, 4)
            dtype = None
            __cuda_array_interface__ = {
                "shape": shape,
                "typestr": "|u1",
                "strides": None,
                "data": (0x5000, False),
                "version": 3,
                "stream": 0xBEEF,
            }

        backend = nvtiff_backend.NvTiffBackend(
            0, "<fake>", _library=_FakeNvTiffLibrary()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "does not match.*producer stream"):
                backend.write_multipage_lzw(
                    Path(temp_dir) / "race.tif",
                    InterfacePages(),
                    cuda_stream=0xCAFE,
                )
        backend.close()

    def test_tensor_writer_rejects_ambiguous_cai_stream_zero(self) -> None:
        class InterfacePages:
            shape = (2, 3, 4)
            dtype = None
            __cuda_array_interface__ = {
                "shape": shape,
                "typestr": "|u1",
                "strides": None,
                "data": (0x5000, False),
                "version": 3,
                "stream": 0,
            }

        backend = nvtiff_backend.NvTiffBackend(
            0, "<fake>", _library=_FakeNvTiffLibrary()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "forbids producer stream 0"):
                backend.write_multipage_lzw(
                    Path(temp_dir) / "ambiguous.tif",
                    InterfacePages(),
                )
        backend.close()

    def test_tensor_adapter_rejects_wrong_memory_contracts(self) -> None:
        cases = (
            (_FakeCudaPages(shape=(4, 5)), ValueError, "shape"),
            (_FakeCudaPages(dtype="torch.float32"), TypeError, "uint8"),
            (_FakeCudaPages(is_cuda=False), ValueError, "CUDA"),
            (_FakeCudaPages(contiguous=False), ValueError, "contiguous"),
            (_FakeCudaPages(strides=(21, 5, 1)), ValueError, "strides"),
            (_FakeCudaPages(pointer=0), ValueError, "positive"),
        )
        for pages, exception_type, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(exception_type, pattern):
                    nvtiff_backend._cuda_pages_device_pointers(pages)

    def test_tensor_writer_requires_safe_stream_and_matching_device(self) -> None:
        backend = nvtiff_backend.NvTiffBackend(
            0, "<fake>", _library=_FakeNvTiffLibrary()
        )
        pages = _FakeCudaPages()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "channels.tif"
            with self.assertRaisesRegex(ValueError, "cuda_stream is required"):
                backend.write_multipage_lzw(output, pages)

            class Device:
                type = "cuda"
                index = 1

            pages.device = Device()  # type: ignore[attr-defined]
            with self.assertRaisesRegex(ValueError, "device 1.*device 0"):
                backend.write_multipage_lzw(
                    output,
                    pages,
                    cuda_stream=0,
                )
        backend.close()

    def test_raw_pointer_writer_validates_before_native_calls(self) -> None:
        backend = nvtiff_backend.NvTiffBackend(
            0, "<fake>", _library=_FakeNvTiffLibrary()
        )
        invalid = (
            ((), {"height": 1, "width": 1, "cuda_stream": 0}, ValueError),
            ((0,), {"height": 1, "width": 1, "cuda_stream": 0}, ValueError),
            ((1,), {"height": 0, "width": 1, "cuda_stream": 0}, ValueError),
            ((1,), {"height": 1, "width": 0, "cuda_stream": 0}, ValueError),
            ((1,), {"height": 1, "width": 1, "cuda_stream": -1}, ValueError),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for pointers, keywords, exception_type in invalid:
                with self.subTest(pointers=pointers, keywords=keywords):
                    with self.assertRaises(exception_type):
                        backend.write_multipage_lzw_from_device_pointers(
                            Path(temp_dir) / "invalid.tif",
                            pointers,
                            **keywords,
                        )
            with self.assertRaisesRegex(ValueError, r"\.tif"):
                backend.write_multipage_lzw_from_device_pointers(
                    Path(temp_dir) / "invalid.png",
                    (1,),
                    height=1,
                    width=1,
                    cuda_stream=0,
                )


class NvTiffCapabilityTests(unittest.TestCase):
    def test_version_floor_rejects_pre_08_library(self) -> None:
        with self.assertRaisesRegex(
            nvtiff_backend.NvTiffUnavailableError,
            r"0\.7\.9.*0\.8\.0",
        ):
            nvtiff_backend.NvTiffBackend(
                0, "<old>", _library=_FakeNvTiffLibrary(version=(0, 7, 9))
            )

    def test_missing_required_symbol_has_controlled_diagnostic(self) -> None:
        fake = _FakeNvTiffLibrary()
        del fake.nvtiffEncodeFinalize
        with self.assertRaisesRegex(
            nvtiff_backend.NvTiffUnavailableError,
            r"missing required 0\.8 symbols: nvtiffEncodeFinalize",
        ):
            nvtiff_backend.NvTiffBackend(0, "<incomplete>", _library=fake)

    def test_probe_is_non_throwing_and_actionable_when_load_fails(self) -> None:
        with (
            mock.patch.object(
                nvtiff_backend, "_library_candidates", return_value=("missing-nvtiff",)
            ),
            mock.patch.object(
                nvtiff_backend,
                "_load_cdll",
                side_effect=OSError("loader rejected library"),
            ),
            mock.patch("builtins.print") as print_mock,
        ):
            capability = nvtiff_backend.probe_nvtiff()

        self.assertFalse(capability.available)
        self.assertIsNone(capability.version)
        self.assertIn("nvTIFF >= 0.8.0 is unavailable", capability.diagnostic)
        self.assertIn("nvidia-nvtiff-cu12", capability.diagnostic)
        self.assertEqual(len(capability.attempts), 1)
        print_mock.assert_not_called()

    def test_probe_reports_loaded_version_and_path(self) -> None:
        fake = _FakeNvTiffLibrary(version=(0, 8, 2))
        with (
            mock.patch.object(
                nvtiff_backend, "_library_candidates", return_value=("fake-nvtiff",)
            ),
            mock.patch.object(nvtiff_backend, "_load_cdll", return_value=fake),
        ):
            capability = nvtiff_backend.probe_nvtiff()

        self.assertTrue(capability.available)
        self.assertEqual(capability.version, (0, 8, 2))
        self.assertEqual(capability.library_path, "fake-nvtiff")
        self.assertIn("nvTIFF 0.8.2 available", capability.diagnostic)

    def test_explicit_environment_path_suppresses_implicit_search(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {nvtiff_backend.NVTIFF_LIBRARY_ENV: "/opt/custom/libnvtiff.so.0"},
            ),
            mock.patch.object(
                nvtiff_backend,
                "_wheel_library_candidates",
                side_effect=AssertionError("implicit search must not run"),
            ),
        ):
            candidates = nvtiff_backend._library_candidates()
        self.assertEqual(candidates, ("/opt/custom/libnvtiff.so.0",))

    def test_mixed_cuda_major_wheels_require_explicit_selection(self) -> None:
        def fake_candidates(distribution_names: object) -> list[str]:
            names = tuple(distribution_names)  # type: ignore[arg-type]
            return [f"/{names[0]}/libnvtiff.so.0"]

        with mock.patch.object(
            nvtiff_backend,
            "_wheel_candidates_for_distributions",
            side_effect=fake_candidates,
        ):
            with self.assertRaisesRegex(
                nvtiff_backend.NvTiffUnavailableError,
                "multiple CUDA-major.*pass cuda_major",
            ):
                tuple(nvtiff_backend._wheel_library_candidates())

    def test_cuda_major_limits_wheel_distribution_search(self) -> None:
        with mock.patch.object(
            nvtiff_backend,
            "_wheel_candidates_for_distributions",
            return_value=["/cu12/libnvtiff.so.0"],
        ) as candidates_mock:
            result = tuple(nvtiff_backend._wheel_library_candidates(12))

        self.assertEqual(result, ("/cu12/libnvtiff.so.0",))
        candidates_mock.assert_called_once_with(
            ("nvidia-nvtiff-cu12", "nvidia-nvtiff-tegra-cu12")
        )


@unittest.skipUnless(
    os.environ.get("XTA_RUN_NVTIFF_INTEGRATION", "").strip() == "1",
    "set XTA_RUN_NVTIFF_INTEGRATION=1 on an nvTIFF/CUDA host",
)
class NvTiffNativeIntegrationTests(unittest.TestCase):
    def test_cuda_roundtrip_preserves_ordered_gray8_ifds(self) -> None:
        import cv2
        import numpy as np
        import tifffile
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA PyTorch is unavailable")
        cuda_version = str(torch.version.cuda or "")
        if not cuda_version:
            self.skipTest("PyTorch does not report a CUDA runtime version")
        cuda_major = int(cuda_version.partition(".")[0])
        device_id = int(torch.cuda.current_device())
        capability = nvtiff_backend.probe_nvtiff(cuda_major=cuda_major)
        if not capability.available:
            self.skipTest(capability.diagnostic)

        values = torch.arange(3 * 17 * 19, device="cuda", dtype=torch.int64)
        pages = values.remainder(251).to(torch.uint8).reshape(3, 17, 19).contiguous()
        cuda_stream = int(torch.cuda.current_stream(device_id).cuda_stream)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "roundtrip.tif"
            with nvtiff_backend.NvTiffBackend(
                device_id,
                cuda_major=cuda_major,
            ) as backend:
                backend.write_multipage_lzw(
                    output,
                    pages,
                    cuda_stream=cuda_stream,
                )

            loaded, decoded_pages = cv2.imreadmulti(
                str(output), flags=cv2.IMREAD_UNCHANGED
            )
            self.assertTrue(loaded)
            self.assertEqual(len(decoded_pages), 3)
            expected = pages.cpu().numpy()
            for index, decoded in enumerate(decoded_pages):
                self.assertEqual(decoded.dtype, np.uint8)
                self.assertEqual(decoded.shape, (17, 19))
                np.testing.assert_array_equal(decoded, expected[index])
            with tifffile.TiffFile(output) as tiff:
                self.assertEqual(len(tiff.pages), 3)
                for page in tiff.pages:
                    self.assertEqual(int(page.compression), 5)
                    self.assertEqual(int(page.photometric), 1)
                    self.assertEqual(page.samplesperpixel, 1)
                    self.assertEqual(page.bitspersample, 8)


if __name__ == "__main__":
    unittest.main()
