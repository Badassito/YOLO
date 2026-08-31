from __future__ import annotations

import unittest
from pathlib import Path

import XTA
from XTA import cli, config


ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "18.0.2"
CURRENT_LAUNCHER = "GPT-5.6-Sol-Ultra_v18.0.2_SLURM.py"


def _toml_section(source: str, name: str) -> str:
    marker = f"[{name}]"
    start = source.index(marker) + len(marker)
    remainder = source[start:]
    next_section = remainder.find("\n[")
    return remainder if next_section < 0 else remainder[:next_section]


class PackageMetadataTests(unittest.TestCase):
    def test_runtime_version_constants_are_aligned(self) -> None:
        self.assertEqual(XTA.__version__, CURRENT_VERSION)
        self.assertEqual(config.SCRIPT_VERSION, CURRENT_VERSION)
        self.assertEqual(config.SCRIPT_VERSION_COMPACT, "1802")
        self.assertEqual(config.SCRIPT_BASENAME, CURRENT_LAUNCHER)
        self.assertEqual(cli.SCRIPT_VERSION, CURRENT_VERSION)
        self.assertEqual(cli.SCRIPT_BASENAME, CURRENT_LAUNCHER)

    def test_project_metadata_uses_mode_dispatcher(self) -> None:
        source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        project = _toml_section(source, "project")
        scripts = _toml_section(source, "project.scripts")
        package_data = _toml_section(source, "tool.setuptools.package-data")
        data_files = _toml_section(source, "tool.setuptools.data-files")

        self.assertIn('name = "xta"', project)
        self.assertIn(f'version = "{CURRENT_VERSION}"', project)
        self.assertNotIn("readme =", project)
        self.assertIn('xta = "XTA.cli:run"', scripts)
        self.assertIn(
            '"XTA.examples.external_augmentations" = ["README.md"]',
            package_data,
        )
        self.assertIn(f'"{CURRENT_LAUNCHER}"', data_files)
        self.assertNotIn('GPT-5.6-Sol-Ultra_v18.0.0_SLURM.py', data_files)
        self.assertNotIn('"README.md"', data_files)

    def test_source_distribution_has_one_versioned_launcher(self) -> None:
        manifest_lines = {
            line.strip()
            for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        }
        self.assertIn(f"include {CURRENT_LAUNCHER}", manifest_lines)
        self.assertNotIn("include GPT-5.6-Sol-Ultra_v18.0.0_SLURM.py", manifest_lines)
        self.assertNotIn("include README.md", manifest_lines)
        self.assertIn("include XTA/_package_inventory.json", manifest_lines)
        self.assertIn("recursive-include XTA/examples *.py *.md", manifest_lines)
        self.assertIn("recursive-include tools *.py", manifest_lines)
        self.assertTrue((ROOT / CURRENT_LAUNCHER).is_file())
        self.assertFalse((ROOT / "GPT-5.6-Sol-Ultra_v18.0.0_SLURM.py").exists())
        self.assertFalse((ROOT / "GPT-5.6-Sol-Ultra_v18.0.1_SLURM.py").exists())

    def test_legacy_distribution_identity_is_absent_from_text_sources(self) -> None:
        forbidden = (
            "volume" + "-tta",
            "volume" + "_tta",
            "VOLUME" + "_TTA",
            "Volume" + " TTA",
        )
        text_suffixes = {".c", ".h", ".in", ".json", ".md", ".py", ".toml"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(path=path.relative_to(ROOT), token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
