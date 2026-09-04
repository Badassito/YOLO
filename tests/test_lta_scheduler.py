from __future__ import annotations

import unittest

from XTA.lta_scheduler import (
    LtaBackprojectionClaim,
    LtaSessionWork,
    LtaViewAffinityScheduler,
    LtaViewKey,
    assign_view_owners,
)


def _work(
    work_id: str,
    view_name: str,
    plan_order: int,
    *,
    cost: float = 1.0,
    session_index: int | None = None,
    tail_eligible: bool = True,
    relay_generation: int = 0,
) -> LtaSessionWork:
    session = plan_order if session_index is None else session_index
    return LtaSessionWork(
        work_id=work_id,
        view=LtaViewKey("volume", view_name),
        runtime_view_id=f"{view_name}::runtime",
        session_index=session,
        frame_start=session * 10,
        frame_stop=session * 10 + 10,
        plan_order=plan_order,
        estimated_cost=cost,
        projection_key=f"projection::{view_name}",
        tail_eligible=tail_eligible,
        relay_generation=relay_generation,
    )


class LtaViewAffinitySchedulerTests(unittest.TestCase):
    def test_whole_view_ownership_is_balanced_and_deterministic(self) -> None:
        work = (
            _work("alpha-0", "alpha", 0, cost=4.0, session_index=0),
            _work("beta-0", "beta", 2, cost=9.0, session_index=0),
            _work("alpha-1", "alpha", 1, cost=6.0, session_index=1),
            _work("gamma-0", "gamma", 3, cost=8.0, session_index=0),
        )

        expected = {
            "alpha": (7, 10.0, 2),
            "beta": (3, 9.0, 1),
            "gamma": (3, 8.0, 1),
        }
        for candidate in (work, tuple(reversed(work)), (work[2], work[0], work[3], work[1])):
            assignments = assign_view_owners(candidate, (7, 3))
            self.assertEqual(
                {
                    item.view.physical_view_id: (
                        item.owner_device_id,
                        item.estimated_cost,
                        item.work_count,
                    )
                    for item in assignments
                },
                expected,
            )

    def test_single_device_owns_every_view_and_executes_without_tail_assist(self) -> None:
        work = (
            _work("b", "beta", 1),
            _work("a", "alpha", 0),
        )
        scheduler = LtaViewAffinityScheduler(work, (5,))

        self.assertEqual(
            {assignment.owner_device_id for assignment in scheduler.assignments},
            {5},
        )
        for assignment in scheduler.assignments:
            scheduler.mark_projection_ready(assignment.view, device_id=5)

        claims = []
        while (claim := scheduler.claim(5)) is not None:
            claims.append(claim)
            scheduler.complete(claim, claim.work.work_id.upper())

        self.assertEqual([claim.work.work_id for claim in claims], ["a", "b"])
        self.assertTrue(all(not claim.tail_assist for claim in claims))
        self.assertFalse(scheduler.done)
        self.assertEqual(
            [item.work_id for item, _result in scheduler.drain_committable()],
            ["a", "b"],
        )
        alpha_claim = scheduler.claim_backprojection(5)
        assert alpha_claim is not None
        self.assertEqual(alpha_claim.view, LtaViewKey("volume", "alpha"))
        scheduler.complete_backprojection(alpha_claim)
        beta_claim = scheduler.claim_backprojection(5)
        assert beta_claim is not None
        self.assertEqual(beta_claim.view, LtaViewKey("volume", "beta"))
        scheduler.complete_backprojection(beta_claim)
        self.assertTrue(scheduler.done)

    def test_owner_work_is_claimed_before_any_eligible_foreign_tail(self) -> None:
        work = (
            _work("alpha-head", "alpha", 0, cost=5.0, session_index=0),
            _work("alpha-tail", "alpha", 1, cost=5.0, session_index=1),
            _work("beta-own", "beta", 2, cost=9.0, session_index=0),
        )
        scheduler = LtaViewAffinityScheduler(work, (0, 1))
        alpha = LtaViewKey("volume", "alpha")
        beta = LtaViewKey("volume", "beta")
        self.assertEqual(scheduler.owner_for_view(alpha), 0)
        self.assertEqual(scheduler.owner_for_view(beta), 1)
        scheduler.mark_projection_ready(alpha, device_id=0)
        scheduler.mark_projection_ready(beta, device_id=1)

        claim = scheduler.claim(1)

        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(claim.work.work_id, "beta-own")
        self.assertFalse(claim.tail_assist)
        self.assertEqual(claim.owner_device_id, 1)

    def test_tail_assist_waits_for_an_empty_own_queue_and_ready_projection(self) -> None:
        work = (
            _work("alpha-head", "alpha", 0, cost=5.0, session_index=0),
            _work("alpha-tail", "alpha", 1, cost=5.0, session_index=1),
            _work("beta-own", "beta", 2, cost=9.0, session_index=0),
        )
        alpha = LtaViewKey("volume", "alpha")
        beta = LtaViewKey("volume", "beta")
        scheduler = LtaViewAffinityScheduler(work, (0, 1))
        scheduler.mark_projection_ready(alpha, device_id=0)

        # Device 1 is not allowed to escape its blocked owner queue by stealing.
        self.assertIsNone(scheduler.claim(1))

        scheduler.mark_projection_ready(beta, device_id=1)
        own = scheduler.claim(1)
        self.assertIsNotNone(own)
        assert own is not None
        self.assertEqual(own.work.work_id, "beta-own")
        scheduler.complete(own, "beta-result")

        assisted = scheduler.claim(1)
        self.assertIsNotNone(assisted)
        assert assisted is not None
        self.assertEqual(assisted.work.work_id, "alpha-tail")
        self.assertTrue(assisted.tail_assist)
        self.assertEqual(assisted.owner_device_id, 0)
        self.assertEqual(assisted.execution_device_id, 1)

        unready = LtaViewAffinityScheduler(
            (_work("only", "only-view", 0),),
            (0, 1),
        )
        self.assertIsNone(unready.claim(1))
        unready.mark_projection_ready(LtaViewKey("volume", "only-view"), device_id=0)
        self.assertTrue(unready.claim(1).tail_assist)  # type: ignore[union-attr]

    def test_claim_is_an_atomic_session_lease(self) -> None:
        scheduler = LtaViewAffinityScheduler(
            (
                _work("first", "alpha", 0, session_index=0),
                _work("second", "alpha", 1, session_index=1),
            ),
            (0, 1),
        )
        view = LtaViewKey("volume", "alpha")
        scheduler.mark_projection_ready(view, device_id=0)
        owner_claim = scheduler.claim(0)
        helper_claim = scheduler.claim(1)
        self.assertIsNotNone(owner_claim)
        self.assertIsNotNone(helper_claim)
        assert owner_claim is not None and helper_claim is not None
        self.assertNotEqual(owner_claim.work.work_id, helper_claim.work.work_id)

        with self.assertRaisesRegex(RuntimeError, "already owns an active"):
            scheduler.claim(0)
        with self.assertRaisesRegex(RuntimeError, "already owns an active"):
            scheduler.claim(1)

        scheduler.complete(helper_claim, "second-result")
        with self.assertRaisesRegex(ValueError, "not the active lease"):
            scheduler.complete(helper_claim, "duplicate-result")
        scheduler.complete(owner_claim, "first-result")
        self.assertEqual(
            set(scheduler.snapshot().completed_work_ids),
            {"first", "second"},
        )

    def test_retry_lease_rejects_the_structurally_equal_stale_claim(self) -> None:
        scheduler = LtaViewAffinityScheduler(
            (_work("retry-me", "alpha", 0),),
            (0,),
        )
        scheduler.mark_projection_ready(LtaViewKey("volume", "alpha"), device_id=0)
        stale = scheduler.claim(0)
        assert stale is not None
        scheduler.fail(stale, retry=True)
        current = scheduler.claim(0)
        assert current is not None

        # Retrying the same work creates an equal-valued claim, but it is a new
        # lease.  A late completion from the failed attempt must not settle it.
        self.assertEqual(current, stale)
        self.assertIsNot(current, stale)
        with self.assertRaisesRegex(ValueError, "not the active lease"):
            scheduler.complete(stale, "stale-result")
        scheduler.complete(current, "current-result")
        self.assertEqual(scheduler.snapshot().completed_work_ids, ("retry-me",))

    def test_completed_results_commit_strictly_in_plan_order(self) -> None:
        scheduler = LtaViewAffinityScheduler(
            (
                _work("third", "alpha", 2, session_index=2),
                _work("first", "alpha", 0, session_index=0),
                _work("second", "alpha", 1, session_index=1),
            ),
            (0, 1),
        )
        view = LtaViewKey("volume", "alpha")
        scheduler.mark_projection_ready(view, device_id=0)
        first = scheduler.claim(0)
        third = scheduler.claim(1)
        assert first is not None and third is not None
        self.assertEqual((first.work.work_id, third.work.work_id), ("first", "third"))

        scheduler.complete(third, "result-3")
        self.assertEqual(scheduler.drain_committable(), ())
        scheduler.complete(first, "result-1")
        self.assertEqual(
            [(item.work_id, result) for item, result in scheduler.drain_committable()],
            [("first", "result-1")],
        )
        second = scheduler.claim(0)
        assert second is not None
        scheduler.complete(second, "result-2")
        self.assertEqual(
            [(item.work_id, result) for item, result in scheduler.drain_committable()],
            [("second", "result-2"), ("third", "result-3")],
        )
        self.assertEqual(scheduler.drain_committable(), ())

    def test_failed_commit_claim_replays_without_advancing_frontier(self) -> None:
        scheduler = LtaViewAffinityScheduler(
            (_work("first", "alpha", 0),),
            (0,),
        )
        scheduler.mark_projection_ready(LtaViewKey("volume", "alpha"), device_id=0)
        work_claim = scheduler.claim(0)
        assert work_claim is not None
        scheduler.complete(work_claim, "result")

        stale = scheduler.claim_committable()
        assert stale is not None
        self.assertEqual(stale.entries[0][0].work_id, "first")
        with self.assertRaisesRegex(RuntimeError, "commit batch is already active"):
            scheduler.claim_committable()
        scheduler.fail_commit(stale)
        current = scheduler.claim_committable()
        assert current is not None
        self.assertIsNot(current, stale)
        with self.assertRaisesRegex(ValueError, "not the active"):
            scheduler.complete_commit(stale)
        scheduler.complete_commit(current)
        self.assertEqual(scheduler.snapshot().active_commit_work_ids, ())

    def test_backprojection_is_owner_only_and_exactly_once_after_view_settles(self) -> None:
        scheduler = LtaViewAffinityScheduler(
            (
                _work("head", "alpha", 0, session_index=0),
                _work("tail", "alpha", 1, session_index=1),
            ),
            (0, 1),
        )
        view = LtaViewKey("volume", "alpha")
        scheduler.mark_projection_ready(view, device_id=0)
        owner_claim = scheduler.claim(0)
        helper_claim = scheduler.claim(1)
        assert owner_claim is not None and helper_claim is not None
        scheduler.complete(helper_claim, "tail-result")

        with self.assertRaisesRegex(RuntimeError, "SAM session is active"):
            scheduler.claim_backprojection(0)
        self.assertIsNone(scheduler.claim_backprojection(1))
        scheduler.complete(owner_claim, "head-result")
        self.assertIsNone(scheduler.claim_backprojection(0))
        scheduler.drain_committable()
        self.assertIsNone(scheduler.claim_backprojection(1))
        first_backprojection = scheduler.claim_backprojection(0)
        assert first_backprojection is not None
        self.assertEqual(first_backprojection.view, view)
        with self.assertRaisesRegex(RuntimeError, "active LTA backprojection"):
            scheduler.claim_backprojection(0)
        with self.assertRaisesRegex(RuntimeError, "active LTA backprojection"):
            scheduler.claim(0)
        with self.assertRaisesRegex(ValueError, "not the active"):
            scheduler.complete_backprojection(
                LtaBackprojectionClaim(view=view, owner_device_id=1)
            )

        scheduler.fail_backprojection(first_backprojection)
        retry_backprojection = scheduler.claim_backprojection(0)
        assert retry_backprojection is not None
        self.assertEqual(retry_backprojection.view, view)
        with self.assertRaisesRegex(ValueError, "not the active"):
            scheduler.complete_backprojection(first_backprojection)
        scheduler.complete_backprojection(retry_backprojection)
        self.assertIsNone(scheduler.claim_backprojection(0))
        with self.assertRaisesRegex(ValueError, "not the active"):
            scheduler.complete_backprojection(retry_backprojection)
        self.assertEqual(scheduler.snapshot().backprojected_views, (view,))
        self.assertTrue(scheduler.done)

    def test_later_view_cannot_backproject_ahead_of_the_commit_frontier(self) -> None:
        scheduler = LtaViewAffinityScheduler(
            (
                _work("alpha", "alpha", 0),
                _work("beta", "beta", 1),
            ),
            (0, 1),
        )
        alpha = LtaViewKey("volume", "alpha")
        beta = LtaViewKey("volume", "beta")
        scheduler.mark_projection_ready(alpha, device_id=scheduler.owner_for_view(alpha))
        scheduler.mark_projection_ready(beta, device_id=scheduler.owner_for_view(beta))
        alpha_claim = scheduler.claim(scheduler.owner_for_view(alpha))
        beta_claim = scheduler.claim(scheduler.owner_for_view(beta))
        assert alpha_claim is not None and beta_claim is not None
        scheduler.complete(beta_claim, "beta")
        self.assertEqual(scheduler.drain_committable(), ())
        self.assertIsNone(scheduler.claim_backprojection(scheduler.owner_for_view(beta)))
        scheduler.complete(alpha_claim, "alpha")
        scheduler.drain_committable()
        backprojection = scheduler.claim_backprojection(scheduler.owner_for_view(beta))
        assert backprojection is not None
        self.assertEqual(backprojection.view, beta)

    def test_device_ids_and_costs_are_strict(self) -> None:
        work = (_work("alpha", "alpha", 0),)
        with self.assertRaisesRegex(ValueError, "unique"):
            LtaViewAffinityScheduler(work, (0, 0))
        with self.assertRaises(TypeError):
            LtaViewAffinityScheduler(work, (False,))
        with self.assertRaises(TypeError):
            LtaViewAffinityScheduler(work, (1.5,))
        with self.assertRaisesRegex(ValueError, "finite"):
            _work("infinite", "alpha", 1, cost=float("inf"))
        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            LtaViewAffinityScheduler(
                (_work("orphan-relay", "alpha", 0, relay_generation=1),),
                (0,),
            )

    def test_relay_generation_waits_for_committed_prior_work(self) -> None:
        scheduler = LtaViewAffinityScheduler(
            (
                _work("authoritative", "alpha", 0, relay_generation=0),
                _work("relay", "alpha", 1, relay_generation=1),
            ),
            (0, 1),
        )
        view = LtaViewKey("volume", "alpha")
        scheduler.mark_projection_ready(view, device_id=0)
        authoritative = scheduler.claim(0)
        assert authoritative is not None
        self.assertEqual(authoritative.work.work_id, "authoritative")
        self.assertIsNone(scheduler.claim(1))
        scheduler.complete(authoritative, "seed")
        self.assertIsNone(scheduler.claim(1))
        scheduler.drain_committable()
        relay = scheduler.claim(1)
        assert relay is not None
        self.assertEqual(relay.work.work_id, "relay")
        self.assertTrue(relay.tail_assist)


if __name__ == "__main__":
    unittest.main()
