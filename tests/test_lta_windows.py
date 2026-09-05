from __future__ import annotations

import unittest

from XTA.lta_sam import LTA_SESSION_FRAMES
from XTA.lta_windows import (
    AnchorDomain,
    WindowPlan,
    owned_frame_range,
    plan_anchor_domains,
    plan_domain_windows,
    plan_directional_windows,
)


class LtaWindowPlanningTests(unittest.TestCase):
    def test_nearest_anchor_domains_cover_once_at_deterministic_midpoints(self) -> None:
        domains = plan_anchor_domains(
            (20, 10, 20),
            frame_start=0,
            frame_stop=31,
        )

        self.assertEqual(
            domains,
            (
                AnchorDomain(anchor_frame=10, frame_start=0, frame_stop=16),
                AnchorDomain(anchor_frame=20, frame_start=16, frame_stop=31),
            ),
        )
        owned = [
            frame
            for domain in domains
            for frame in range(domain.frame_start, domain.frame_stop)
        ]
        self.assertEqual(owned, list(range(31)))

    def test_short_domain_is_one_authoritative_window(self) -> None:
        domain = AnchorDomain(anchor_frame=12, frame_start=3, frame_stop=30)

        windows = plan_domain_windows(domain)

        self.assertEqual(
            windows,
            (
                WindowPlan(
                    branch="center",
                    ordinal=0,
                    frame_start=3,
                    frame_stop=30,
                    prompt_frame=12,
                    direction="both",
                    seed_kind="authoritative",
                ),
            ),
        )
        self.assertEqual(owned_frame_range(windows[0]), (3, 30))

    def test_long_domain_uses_fixed_overlapping_directional_dogfood_chains(self) -> None:
        domain = AnchorDomain(anchor_frame=50, frame_start=0, frame_stop=100)

        windows = plan_domain_windows(domain)
        center = windows[0]
        backward = tuple(item for item in windows if item.branch == "backward")
        forward = tuple(item for item in windows if item.branch == "forward")

        self.assertEqual((center.frame_start, center.frame_stop), (36, 66))
        self.assertEqual(
            [(item.frame_start, item.frame_stop, item.prompt_frame) for item in backward],
            [(7, 37, 36), (0, 8, 7)],
        )
        self.assertEqual(
            [(item.frame_start, item.frame_stop, item.prompt_frame) for item in forward],
            [(65, 95, 65), (94, 100, 94)],
        )
        self.assertTrue(
            all(1 <= item.frame_count <= LTA_SESSION_FRAMES for item in windows)
        )
        self.assertTrue(all(item.seed_kind == "dogfood" for item in (*backward, *forward)))

        # Every chained session shares exactly its prompt boundary with the
        # preceding session in that temporal direction.
        self.assertEqual(backward[0].frame_stop - 1, center.frame_start)
        self.assertEqual(backward[1].frame_stop - 1, backward[0].frame_start)
        self.assertEqual(forward[0].frame_start, center.frame_stop - 1)
        self.assertEqual(forward[1].frame_start, forward[0].frame_stop - 1)

        owned = [
            frame
            for window in windows
            for frame in range(*owned_frame_range(window))
        ]
        self.assertEqual(sorted(owned), list(range(100)))
        self.assertEqual(len(owned), len(set(owned)))

    def test_center_window_shifts_inside_domain_near_anchor_edge(self) -> None:
        windows = plan_domain_windows(
            AnchorDomain(anchor_frame=2, frame_start=0, frame_stop=70)
        )

        self.assertEqual(
            (windows[0].frame_start, windows[0].frame_stop, windows[0].prompt_frame),
            (0, LTA_SESSION_FRAMES, 2),
        )
        self.assertFalse(any(item.branch == "backward" for item in windows))
        self.assertTrue(any(item.branch == "forward" for item in windows))

    def test_spatial_relay_chains_own_forward_and_backward_ranges_once(self) -> None:
        forward = plan_directional_windows(
            frame_start=0,
            frame_stop=75,
            prompt_frame=12,
            direction="forward",
        )
        backward = plan_directional_windows(
            frame_start=0,
            frame_stop=75,
            prompt_frame=62,
            direction="backward",
        )

        self.assertEqual(forward[0].seed_kind, "spatial_relay")
        self.assertEqual(backward[0].seed_kind, "spatial_relay")
        self.assertTrue(all(item.frame_count <= LTA_SESSION_FRAMES for item in (*forward, *backward)))
        self.assertEqual(
            sorted(
                frame
                for window in forward
                for frame in range(*owned_frame_range(window))
            ),
            list(range(12, 75)),
        )
        self.assertEqual(
            sorted(
                frame
                for window in backward
                for frame in range(*owned_frame_range(window))
            ),
            list(range(0, 63)),
        )

    def test_invalid_domains_and_window_contracts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside"):
            plan_anchor_domains((31,), frame_start=0, frame_stop=31)
        with self.assertRaisesRegex(ValueError, "inside"):
            AnchorDomain(anchor_frame=8, frame_start=0, frame_stop=8)
        with self.assertRaisesRegex(ValueError, "backward dogfood"):
            WindowPlan(
                branch="backward",
                ordinal=1,
                frame_start=0,
                frame_stop=10,
                prompt_frame=8,
                direction="backward",
                seed_kind="dogfood",
            )
        with self.assertRaisesRegex(ValueError, "at most"):
            WindowPlan(
                branch="center",
                ordinal=0,
                frame_start=0,
                frame_stop=LTA_SESSION_FRAMES + 1,
                prompt_frame=5,
                direction="both",
                seed_kind="authoritative",
            )


if __name__ == "__main__":
    unittest.main()
