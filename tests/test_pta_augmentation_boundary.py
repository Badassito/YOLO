from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest

from tools.smoke_import import install_stubs


def _module_is_available(name: str) -> bool:
    loaded = sys.modules.get(name)
    if loaded is not None:
        return getattr(loaded, "__spec__", None) is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


if not all(_module_is_available(name) for name in ("cv2", "scipy", "tqdm")):
    install_stubs()

from XTA import pta
from XTA import pta_augmentation


class PtaAugmentationBoundaryTests(unittest.TestCase):
    EXPORTED_NAMES = (
        "AugmentationDefinition",
        "LoadedAugmentation",
        "LoadedGpuAugmentation",
        "OfflineAugmentation",
        "_augmented_image_to_uint8",
        "_augmented_mask_to_binary",
        "_load_external_python_module",
        "apply_augmentation_pair",
        "assert_augmentation_definition_unchanged",
        "assert_augmentation_did_not_synthesize_mask",
        "inspect_augmentation_definition",
        "load_augmentation_definition",
        "load_gpu_augmentation_definition",
        "load_offline_augmentation_definition",
        "validate_seedable_augmentation_pipeline",
    )

    def test_pta_reexports_the_augmentation_owner_objects(self) -> None:
        self.assertEqual(pta_augmentation.__all__, self.EXPORTED_NAMES)
        for name in self.EXPORTED_NAMES:
            with self.subTest(name=name):
                owned = getattr(pta_augmentation, name)
                self.assertIs(getattr(pta, name), owned)
                if inspect.isfunction(owned) or inspect.isclass(owned):
                    self.assertEqual(owned.__module__, "XTA.pta_augmentation")


if __name__ == "__main__":
    unittest.main()
