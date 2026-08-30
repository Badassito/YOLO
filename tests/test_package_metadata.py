from __future__ import annotations

import unittest
from pathlib import Path

import XTA
from XTA import cli, config


ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "18.0.1"
CURRENT_LAUNCHER = "GPT-5.6-Sol-Ultra_v18.0.1_SLURM.py"


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
        self.assertEqual(config.SCRIPT_VERSION_COMPACT, "1801")
        self.assertEqual(config.SCRIPT_BASENAME, CURRENT_LAUNCHER)
        self.assertEqual(cli.SCRIPT_VERSION, CURRENT_VERSION)
        self.assertEqual(cli.SCRIPT_BASENAME, CURRENT_LAUNCHER)

    def test_project_metadata_uses_mode_dispatcher(self) -> None:
        source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        project = _toml_section(source, "project")
        scripts = _toml_section(source, "project.scripts")
        data_files = _toml_section(source, "tool.setuptools.data-files")

        self.assertIn(f'version = "{CURRENT_VERSION}"', project)
        self.assertNotIn("readme =", project)
        self.assertIn('volume-tta = "XTA.cli:run"', scripts)
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
        self.assertIn("recursive-include tools *.py", manifest_lines)
        self.assertTrue((ROOT / CURRENT_LAUNCHER).is_file())
        self.assertFalse((ROOT / "GPT-5.6-Sol-Ultra_v18.0.0_SLURM.py").exists())


if __name__ == "__main__":
    unittest.main()
