from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
import tempfile
import unittest

from XTA.lta_execution import _artifact_ownership_path, _unlink_consumed_temp_artifacts


class LtaArtifactContainmentTests(unittest.TestCase):
    def test_equivalent_extended_dos_and_unc_paths_remain_inside_their_root(self):
        long_tail = "\\".join("component-" + str(index) + "-" + "x" * 48 for index in range(6))
        for root, candidate in (
            (r"C:\Scratch\job", "\\\\?\\C:\\Scratch\\job\\" + long_tail + "\\union.pack"),
            (r"\\?\C:\Scratch\job", "C:\\Scratch\\job\\" + long_tail + "\\union.pack"),
            (r"\\server\share\job", "\\\\?\\UNC\\server\\share\\job\\" + long_tail + "\\union.pack"),
            (r"\\?\UNC\server\share\job", "\\\\server\\share\\job\\" + long_tail + "\\union.pack"),
        ):
            with self.subTest(root=root):
                normalized_root = _artifact_ownership_path(PureWindowsPath(root))
                normalized_candidate = _artifact_ownership_path(PureWindowsPath(candidate))
                self.assertEqual(normalized_candidate.relative_to(normalized_root).name, "union.pack")
                self.assertGreater(len(candidate), 260)

    def test_outside_sibling_and_unknown_namespaces_fail_closed(self):
        for root, candidate in (
            (r"C:\Scratch\job", r"\\?\C:\Scratch\job-other\union.pack"),
            (r"C:\Scratch\job", r"\\?\D:\Scratch\job\union.pack"),
            (r"\\server\share\job", r"\\?\UNC\server\share\job-other\union.pack"),
            (r"\\server\share\job", r"\\?\UNC\server\other-share\job\union.pack"),
            (r"\\server\share\job", r"\\?\UNC\other-server\share\job\union.pack"),
            (r"C:\Scratch\job", r"\\?\GLOBALROOT\Device\HarddiskVolume1\Scratch\job\union.pack"),
            (r"C:\Scratch\job", r"\\.\C:\Scratch\job\union.pack"),
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    _artifact_ownership_path(PureWindowsPath(candidate)).relative_to(
                        _artifact_ownership_path(PureWindowsPath(root)),
                    )

    @unittest.skipUnless(os.name == "nt", "requires Windows extended-length filesystem paths")
    def test_real_long_path_is_unlinked_and_equivalent_spellings_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root
            for index in range(6):
                parent = parent / (f"generation-{index}-" + "x" * 45)
            parent.mkdir(parents=True)
            artifact = parent / "union.pack"
            artifact.write_bytes(b"verified artifact")
            self.assertGreater(len(str(artifact)), 260)
            extended = Path("\\\\?\\" + str(artifact))
            _unlink_consumed_temp_artifacts((artifact, extended), temp_root=root)
            self.assertFalse(artifact.exists())
            self.assertTrue(parent.is_dir())

    def test_escape_is_rejected_before_any_file_is_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / "job"
            sibling = base / "job-other"
            root.mkdir()
            sibling.mkdir()
            inside = root / "inside.pack"
            outside = sibling / "outside.pack"
            inside.write_bytes(b"inside")
            outside.write_bytes(b"outside")
            escaped = root / ".." / sibling.name / outside.name
            if os.name == "nt":
                escaped = Path("\\\\?\\" + str(escaped))
            with self.assertRaisesRegex(RuntimeError, "outside temp root"):
                _unlink_consumed_temp_artifacts((inside, escaped), temp_root=root)
            self.assertEqual(inside.read_bytes(), b"inside")
            self.assertEqual(outside.read_bytes(), b"outside")


if __name__ == "__main__":
    unittest.main()
