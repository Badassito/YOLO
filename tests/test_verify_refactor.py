from __future__ import annotations

import ast
import textwrap
import unittest

from tools.verify_refactor import (
    LOCAL_IMPORT_SEAM_MARKER,
    reviewed_local_import_seams,
)


def inspect_seams(source: str):
    source = textwrap.dedent(source)
    return reviewed_local_import_seams("sample", source, ast.parse(source))


class RefactorVerifierTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
