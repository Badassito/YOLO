from __future__ import annotations

import ast
import copy
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tools.verify_package_inventory import (
    INTENTIONALLY_CHANGED_BINDINGS,
    LOCAL_IMPORT_SEAM_MARKER,
    MANIFEST,
    REVIEWED_V20_ADDED_DEFINITIONS,
    REVIEWED_V20_ADDED_STATEMENTS,
    azimuthal_rename_replacements,
    digest,
    main as verify_inventory,
    reviewed_local_import_seams,
    stable_ast_dump,
)


def inspect_seams(source: str):
    source = textwrap.dedent(source)
    return reviewed_local_import_seams("sample", source, ast.parse(source))


class PackageInventoryTests(unittest.TestCase):
    def test_numba_compile_policy_is_pinned_outside_function_bodies(self) -> None:
        key = ('cylindrical_projection', 'numba_compile_policy')
        _expected_hash, reason = REVIEWED_V20_ADDED_STATEMENTS[key]
        with mock.patch.dict(REVIEWED_V20_ADDED_STATEMENTS, {key: ('0' * 64, reason)}):
            with self.assertRaisesRegex(RuntimeError, 'reviewed added statement changed or is missing: cylindrical_projection.numba_compile_policy'):
                verify_inventory()

    def test_scalar_cuda_kernel_has_an_independent_review_pin(self) -> None:
        key = ('cuda_backend', '_radial_native_kernels')
        _expected_hash, reason = REVIEWED_V20_ADDED_DEFINITIONS[key]
        with mock.patch.dict(REVIEWED_V20_ADDED_DEFINITIONS, {key: ('0' * 64, reason)}):
            with self.assertRaisesRegex(RuntimeError, 'reviewed added definition changed or is missing: cuda_backend._radial_native_kernels'):
                verify_inventory()

    def test_azimuthal_rename_pins_exact_ast_without_accepting_shell_names_or_changed_math(self) -> None:
        old = ast.parse('def radial_sample(value):\n    return value + 1\n').body[0]
        renamed = ast.parse('def azimuthal_sample(value):\n    return value + 1\n').body[0]
        altered = ast.parse('def azimuthal_sample(value):\n    return value + 2\n').body[0]
        baseline_hash = digest(old)
        record = {
            'module': 'sample', 'baseline_sha256': baseline_hash,
            'renamed_sha256': digest(renamed), 'baseline_name': 'radial_sample',
            'current_name': 'azimuthal_sample',
        }
        manifest = {
            'statements': [{'module': 'sample', 'sha256': baseline_hash, 'name': 'radial_sample'}],
            'v20_azimuthal_rename': {'statements': [record]},
        }
        replacement = azimuthal_rename_replacements(manifest)['sample', baseline_hash]
        self.assertEqual(replacement, digest(renamed))
        self.assertNotEqual(replacement, digest(old), 'A new shell function reusing the old name is a different declaration')
        self.assertNotEqual(replacement, digest(altered), 'The rename map must not normalize arithmetic changes')
        for updates, message in (
            ({'baseline_sha256': '0' * 64}, 'absent immutable'),
            ({'current_name': 'radial_sample'}, 'identity mismatch'),
            ({'renamed_sha256': 'not-a-digest'}, 'invalid reviewed'),
        ):
            broken = copy.deepcopy(manifest)
            broken['v20_azimuthal_rename']['statements'][0].update(updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(RuntimeError, message):
                azimuthal_rename_replacements(broken)
        duplicated = copy.deepcopy(manifest)
        duplicated['v20_azimuthal_rename']['statements'].append(record)
        with self.assertRaisesRegex(RuntimeError, 'duplicate Azimuthal rename'):
            azimuthal_rename_replacements(duplicated)

    def test_original_inventory_cannot_be_silently_rebaselined(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        manifest['statements'][0]['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'inventory.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with mock.patch('tools.verify_package_inventory.MANIFEST', path):
                with self.assertRaisesRegex(RuntimeError, 'immutable inventory digest mismatch'):
                    verify_inventory()

    def test_ast_dump_keeps_empty_fields_across_python_versions(self) -> None:
        function = ast.parse("def callback():\n    pass\n").body[0]

        dumped = stable_ast_dump(function)

        self.assertIn("decorator_list=[]", dumped)

    def test_marker_move_changes_seam_digest_even_when_definition_ast_is_unchanged(self) -> None:
        first_import_reviewed = inspect_seams(
            f"""
            def callback(flag):
                {LOCAL_IMPORT_SEAM_MARKER}
                from .first import run_first
                # The other callback remains local for unrelated reasons.
                from .second import run_second
                return run_first() if flag else run_second()
            """
        )
        second_import_reviewed = inspect_seams(
            f"""
            def callback(flag):
                # The first callback remains local for unrelated reasons.
                from .first import run_first
                {LOCAL_IMPORT_SEAM_MARKER}
                from .second import run_second
                return run_first() if flag else run_second()
            """
        )

        first_definition, first_seam = first_import_reviewed[("sample", "callback")]
        second_definition, second_seam = second_import_reviewed[("sample", "callback")]
        self.assertEqual(first_definition, second_definition)
        self.assertNotEqual(first_seam, second_seam)

    def test_marker_inside_method_pins_the_enclosing_top_level_class(self) -> None:
        reviewed = inspect_seams(
            f"""
            class CallbackOwner:
                def callback(self):
                    {LOCAL_IMPORT_SEAM_MARKER}
                    from .dependency import run
                    return run()
            """
        )
        self.assertEqual(set(reviewed), {("sample", "CallbackOwner")})

    def test_marker_must_immediately_precede_a_relative_function_local_import(self) -> None:
        invalid_sources = (
            f"""
            def callback():
                {LOCAL_IMPORT_SEAM_MARKER}

                from .dependency import run
            """,
            f"""
            def callback():
                {LOCAL_IMPORT_SEAM_MARKER}
                from dependency import run
            """,
            f"""
            {LOCAL_IMPORT_SEAM_MARKER}
            from .dependency import run
            """,
        )
        for source in invalid_sources:
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                inspect_seams(source)

    def test_changed_binding_review_must_reference_the_immutable_inventory(self) -> None:
        unknown = ("config", "0" * 64)
        with mock.patch.dict(
            INTENTIONALLY_CHANGED_BINDINGS,
            {unknown: "SAVE_OPTION_TOKENS"},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "changed-binding entries are absent from the immutable inventory",
            ):
                verify_inventory()


if __name__ == "__main__":
    unittest.main()
