from __future__ import annotations

import unittest

from XTA.config import (
    SAVE_OPTION_TOKENS,
    build_argparser,
    resolve_save_request,
)


class TtaSaveOverlayAliasTests(unittest.TestCase):
    def test_overlay_is_an_accepted_documented_save_value(self) -> None:
        self.assertIn('overlay', SAVE_OPTION_TOKENS)
        help_text = build_argparser().format_help()
        self.assertIn('overlay', help_text)
        self.assertIn('alias', help_text)

    def test_overlay_canonicalizes_to_existing_high_quality_sink(self) -> None:
        request = resolve_save_request(['overlay'])
        self.assertEqual(request.options, ('high_quality',))
        self.assertEqual(request.low_quality_downbins, ())

    def test_alias_and_canonical_name_schedule_only_one_sink(self) -> None:
        for values in (
            ['overlay', 'high_quality'],
            ['high_quality', 'overlay'],
            ['overlay,high_quality,overlay'],
        ):
            with self.subTest(values=values):
                request = resolve_save_request(values)
                self.assertEqual(request.options, ('high_quality',))
                self.assertEqual(request.options.count('high_quality'), 1)
                self.assertNotIn('overlay', request.options)

    def test_overlay_terminates_embedded_low_quality_downbin_collection(self) -> None:
        request = resolve_save_request(
            ['low_quality:0.5,1024,overlay,summary']
        )
        self.assertEqual(
            request.options,
            ('low_quality', 'high_quality', 'summary'),
        )
        self.assertEqual(request.low_quality_downbins, ('0.5', '1024'))


if __name__ == '__main__':
    unittest.main()
