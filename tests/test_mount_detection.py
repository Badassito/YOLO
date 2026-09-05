"""Effective filesystem detection for stacked SLURM job mounts."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.smoke_import import install_stubs

install_stubs()
from XTA import runtime


MOUNTS = """353 1 8:1 / / rw - ext4 /dev/root rw
477 353 0:37 / /tmp rw - tmpfs tmpfs rw,size=193206204k
480 477 0:38 / /tmp/covered-child rw - ramfs ramfs rw
692 477 9:127 /job_tmp/142020/.142020/_tmp /tmp rw - ext4 /dev/md127 rw
"""


def read_proc(path, *args, **kwargs):
    if path.name == 'mountinfo':
        return MOUNTS
    if path.parent.name == 'fdinfo':
        return 'pos:\t0\nflags:\t012000000\nmnt_id:\t692\nino:\t1234\n'
    raise AssertionError(f'Unexpected proc read: {path}')


class MountDetectionTests(unittest.TestCase):
    def test_kernel_id_selects_overmount_even_below_a_hidden_child(self):
        with (
            mock.patch.object(os, 'O_PATH', 0o10000000, create=True),
            mock.patch.object(os, 'open', return_value=71) as opened,
            mock.patch.object(os, 'close') as closed,
            mock.patch.object(Path, 'read_text', autospec=True, side_effect=read_proc),
        ):
            self.assertEqual(runtime._mount_fstype_for_path(Path('/tmp/covered-child/work')), 'ext4')
        self.assertTrue(opened.call_args.args[1] & 0o10000000)
        closed.assert_called_once_with(71)

    def test_missing_path_uses_closest_existing_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / 'new' / 'child'

            def open_existing(path, flags):
                if Path(path) != root:
                    raise FileNotFoundError(path)
                return 72

            with (
                mock.patch.object(os, 'O_PATH', 0o10000000, create=True),
                mock.patch.object(os, 'open', side_effect=open_existing) as opened,
                mock.patch.object(os, 'close') as closed,
                mock.patch.object(Path, 'read_text', autospec=True, side_effect=read_proc),
            ):
                self.assertEqual(runtime._mount_fstype_for_path(target), 'ext4')
            self.assertEqual([Path(call.args[0]) for call in opened.call_args_list], [target, target.parent, root])
            closed.assert_called_once_with(72)

    def test_unknown_or_unavailable_metadata_closes_descriptor(self):
        for metadata in ('mnt_id: 999\n', 'mnt_id: invalid\n', 'pos: 0\n'):
            def read(path, *args, **kwargs):
                return MOUNTS if path.name == 'mountinfo' else metadata

            with (
                self.subTest(metadata=metadata),
                mock.patch.object(os, 'O_PATH', 0o10000000, create=True),
                mock.patch.object(os, 'open', return_value=73),
                mock.patch.object(os, 'close') as closed,
                mock.patch.object(Path, 'read_text', autospec=True, side_effect=read),
            ):
                self.assertIsNone(runtime._mount_fstype_for_path(Path('/tmp')))
            closed.assert_called_once_with(73)
        with (
            mock.patch.object(os, 'O_PATH', 0o10000000, create=True),
            mock.patch.object(os, 'open', return_value=74),
            mock.patch.object(os, 'close') as closed,
            mock.patch.object(Path, 'read_text', side_effect=PermissionError('proc denied')),
        ):
            self.assertIsNone(runtime._mount_fstype_for_path(Path('/tmp')))
        closed.assert_called_once_with(74)

    def test_permission_failure_does_not_guess_parent_filesystem(self):
        with (
            mock.patch.object(os, 'O_PATH', 0o10000000, create=True),
            mock.patch.object(os, 'open', side_effect=PermissionError('denied')) as opened,
            mock.patch.object(os, 'close') as closed,
        ):
            self.assertIsNone(runtime._mount_fstype_for_path(Path('/tmp')))
        self.assertEqual(opened.call_count, 1)
        closed.assert_not_called()


if __name__ == '__main__':
    unittest.main()
