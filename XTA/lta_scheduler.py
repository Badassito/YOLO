"""Device-independent view-affinity scheduler for LTA tracking work.

One device owns each physical view and therefore its projection and final
backprojection.  Idle devices may steal only unopened, atomic session/window
items after their own queues drain.  A stolen item consumes the owner's
immutable rendered-view cache; it never starts another projection or migrates
a live SAM session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import operator
from typing import Deque, Iterable, Mapping, Sequence


def _nonnegative_index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(resolved)


@dataclass(frozen=True, order=True)
class LtaViewKey:
    volume_id: str
    physical_view_id: str

    def __post_init__(self) -> None:
        for name in ("volume_id", "physical_view_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)

    @property
    def token(self) -> str:
        return f"{self.volume_id}::{self.physical_view_id}"


@dataclass(frozen=True)
class LtaSessionWork:
    """One unopened session/window that can execute atomically on one GPU."""

    work_id: str
    view: LtaViewKey
    runtime_view_id: str
    session_index: int
    frame_start: int
    frame_stop: int
    plan_order: int
    estimated_cost: float
    projection_key: str
    tile_index: int | None = None
    tile_config_id: str | None = None
    tail_eligible: bool = True
    relay_generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.view, LtaViewKey):
            raise TypeError("view must be an LtaViewKey")
        for name in ("work_id", "runtime_view_id", "projection_key"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        for name in ("session_index", "frame_start", "plan_order", "relay_generation"):
            value = _nonnegative_index(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        stop = _nonnegative_index(self.frame_stop, name="frame_stop")
        if stop <= self.frame_start:
            raise ValueError("frame_stop must be greater than frame_start")
        object.__setattr__(self, "frame_stop", stop)
        cost = float(self.estimated_cost)
        if not math.isfinite(cost) or not cost > 0.0:
            raise ValueError("estimated_cost must be finite and positive")
        object.__setattr__(self, "estimated_cost", cost)
        if self.tile_index is not None:
            tile_index = _nonnegative_index(self.tile_index, name="tile_index")
            object.__setattr__(self, "tile_index", tile_index)
        if self.tile_config_id is not None:
            tile_config_id = str(self.tile_config_id).strip()
            if not tile_config_id:
                raise ValueError("tile_config_id must not be empty when supplied")
            object.__setattr__(self, "tile_config_id", tile_config_id)
        if (self.tile_index is None) != (self.tile_config_id is None):
            raise ValueError("tile_index and tile_config_id must be supplied together")
        object.__setattr__(self, "tail_eligible", bool(self.tail_eligible))

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start


@dataclass(frozen=True)
class LtaWorkClaim:
    work: LtaSessionWork
    owner_device_id: int
    execution_device_id: int
    tail_assist: bool


@dataclass(frozen=True)
class LtaBackprojectionClaim:
    """Identity-bearing owner lease for one settled physical view."""

    view: LtaViewKey
    owner_device_id: int


@dataclass(frozen=True, eq=False)
class LtaCommitClaim:
    """Identity-bearing ordered result batch awaiting publication acknowledgement."""

    entries: tuple[tuple[LtaSessionWork, object], ...]


@dataclass(frozen=True)
class LtaViewAssignment:
    view: LtaViewKey
    owner_device_id: int
    estimated_cost: float
    work_count: int


@dataclass(frozen=True)
class LtaScheduleSnapshot:
    assignments: tuple[LtaViewAssignment, ...]
    pending_work_ids: tuple[str, ...]
    active_claims: tuple[LtaWorkClaim, ...]
    active_backprojection_claims: tuple[LtaBackprojectionClaim, ...]
    active_commit_work_ids: tuple[str, ...]
    completed_work_ids: tuple[str, ...]
    projection_ready_views: tuple[LtaViewKey, ...]
    backprojected_views: tuple[LtaViewKey, ...]


def assign_view_owners(
    work: Sequence[LtaSessionWork],
    device_ids: Sequence[int],
) -> tuple[LtaViewAssignment, ...]:
    """Balance whole physical views with deterministic longest-processing-time assignment."""

    devices = tuple(
        _nonnegative_index(value, name="device_id") for value in device_ids
    )
    if not devices:
        raise ValueError("LTA scheduling requires at least one device")
    if len(devices) != len(set(devices)):
        raise ValueError("device ids must be unique")
    grouped: dict[LtaViewKey, list[LtaSessionWork]] = {}
    for item in work:
        if not isinstance(item, LtaSessionWork):
            raise TypeError("work must contain only LtaSessionWork instances")
        grouped.setdefault(item.view, []).append(item)
    loads = {device: 0.0 for device in devices}
    assignments: list[LtaViewAssignment] = []
    ordered_views = sorted(
        grouped,
        key=lambda view: (-sum(item.estimated_cost for item in grouped[view]), view),
    )
    for view in ordered_views:
        owner = min(devices, key=lambda device: (loads[device], devices.index(device)))
        cost = float(sum(item.estimated_cost for item in grouped[view]))
        assignments.append(
            LtaViewAssignment(
                view=view,
                owner_device_id=owner,
                estimated_cost=cost,
                work_count=len(grouped[view]),
            )
        )
        loads[owner] += cost
    return tuple(sorted(assignments, key=lambda item: item.view))


class LtaViewAffinityScheduler:
    """Exactly-once state machine for owner-first work and bounded tail assistance.

    All state transitions are intentionally owned by one coordinator thread;
    GPU workers exchange claims and results with that coordinator through
    process-safe queues rather than calling this object concurrently.
    """

    def __init__(
        self,
        work: Iterable[LtaSessionWork],
        device_ids: Sequence[int],
        *,
        allow_tail_assist: bool = True,
    ) -> None:
        self.device_ids = tuple(
            _nonnegative_index(value, name="device_id") for value in device_ids
        )
        if not self.device_ids:
            raise ValueError("device_ids must contain CUDA indexes")
        if len(self.device_ids) != len(set(self.device_ids)):
            raise ValueError("device_ids must be unique")
        self.work = tuple(sorted(tuple(work), key=lambda item: item.plan_order))
        if not all(isinstance(item, LtaSessionWork) for item in self.work):
            raise TypeError("work must contain only LtaSessionWork instances")
        ids = [item.work_id for item in self.work]
        orders = [item.plan_order for item in self.work]
        if len(ids) != len(set(ids)):
            raise ValueError("work ids must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("plan_order values must be unique")
        projection_keys: dict[LtaViewKey, str] = {}
        generations_by_view: dict[LtaViewKey, set[int]] = {}
        for item in self.work:
            previous = projection_keys.setdefault(item.view, item.projection_key)
            if previous != item.projection_key:
                raise ValueError("all work in one physical view must share a projection key")
            generations_by_view.setdefault(item.view, set()).add(item.relay_generation)
        for view, generations in generations_by_view.items():
            expected = set(range(max(generations) + 1))
            if generations != expected:
                raise ValueError(
                    f"relay generations for {view.token} must be contiguous from zero"
                )
        self.assignments = assign_view_owners(self.work, self.device_ids)
        self._owner_by_view = {
            assignment.view: assignment.owner_device_id for assignment in self.assignments
        }
        self._queues: dict[int, Deque[LtaSessionWork]] = {
            device: deque(
                item
                for item in self.work
                if self._owner_by_view[item.view] == device
            )
            for device in self.device_ids
        }
        self._allow_tail_assist = bool(allow_tail_assist)
        self._projection_ready: set[LtaViewKey] = set()
        self._active_by_device: dict[int, LtaWorkClaim] = {}
        self._active_ids: set[str] = set()
        self._completed: dict[str, object] = {}
        self._committed_ids: set[str] = set()
        self._active_commit_claim: LtaCommitClaim | None = None
        self._backprojection_claimed: set[LtaViewKey] = set()
        self._backprojection_by_device: dict[int, LtaBackprojectionClaim] = {}
        self._backprojected: set[LtaViewKey] = set()

    def owner_for_view(self, view: LtaViewKey) -> int:
        try:
            return self._owner_by_view[view]
        except KeyError as exc:
            raise ValueError(f"unknown LTA view {view}") from exc

    def mark_projection_ready(self, view: LtaViewKey, *, device_id: int) -> None:
        """Publish the one immutable rendered-view cache from its affinity owner."""

        owner = self.owner_for_view(view)
        device = _nonnegative_index(device_id, name="device_id")
        if device != owner:
            raise ValueError("only a view owner may publish its projection cache")
        self._projection_ready.add(view)

    def _generation_ready(self, item: LtaSessionWork) -> bool:
        """Keep relay-derived work behind committed prior generations."""

        if item.relay_generation == 0:
            return True
        prerequisites = {
            candidate.work_id
            for candidate in self.work
            if candidate.view == item.view
            and candidate.relay_generation < item.relay_generation
        }
        return bool(prerequisites) and prerequisites.issubset(self._committed_ids)

    def _pop_owner_work(self, device_id: int) -> LtaSessionWork | None:
        queue = self._queues[device_id]
        for _ in range(len(queue)):
            item = queue[0]
            if item.view in self._projection_ready and self._generation_ready(item):
                return queue.popleft()
            queue.rotate(-1)
        return None

    def _steal_tail_work(self, device_id: int) -> tuple[int, LtaSessionWork] | None:
        if not self._allow_tail_assist:
            return None
        # A projection-blocked owner queue is still assigned work, not spare
        # capacity.  Tail help starts only once this device's own queue has
        # genuinely drained; otherwise a device can abandon its affinity view
        # while its owner is still preparing the shared render cache.
        if self._queues[device_id]:
            return None
        candidates: list[tuple[float, LtaViewKey, int, int, LtaSessionWork]] = []
        for owner, queue in self._queues.items():
            if owner == device_id:
                continue
            remaining_cost = sum(item.estimated_cost for item in queue)
            for reverse_index, item in enumerate(reversed(queue)):
                if (
                    item.tail_eligible
                    and item.view in self._projection_ready
                    and self._generation_ready(item)
                ):
                    candidates.append(
                        (remaining_cost, item.view, owner, reverse_index, item)
                    )
                    break
        if not candidates:
            return None
        _cost, _view, owner, reverse_index, item = min(
            candidates,
            key=lambda value: (-value[0], value[1].token, value[2]),
        )
        queue = self._queues[owner]
        position = len(queue) - 1 - reverse_index
        del queue[position]
        return owner, item

    def claim(self, device_id: int) -> LtaWorkClaim | None:
        """Claim owner work first, then one unopened tail item from another owner."""

        device = _nonnegative_index(device_id, name="device_id")
        if device not in self._queues:
            raise ValueError(f"unknown LTA device {device}")
        if device in self._backprojection_by_device:
            raise RuntimeError(
                f"device {device} already owns an active LTA backprojection"
            )
        if device in self._active_by_device:
            raise RuntimeError(f"device {device} already owns an active LTA session")
        item = self._pop_owner_work(device)
        owner = device
        if item is None:
            stolen = self._steal_tail_work(device)
            if stolen is None:
                return None
            owner, item = stolen
        if item.work_id in self._active_ids or item.work_id in self._completed:
            raise RuntimeError(f"work {item.work_id} was scheduled more than once")
        claim = LtaWorkClaim(
            work=item,
            owner_device_id=owner,
            execution_device_id=device,
            tail_assist=owner != device,
        )
        self._active_by_device[device] = claim
        self._active_ids.add(item.work_id)
        return claim

    def complete(self, claim: LtaWorkClaim, result: object) -> None:
        """Settle one complete session; partial/live-session migration is unsupported."""

        active = self._active_by_device.get(int(claim.execution_device_id))
        # Claim values can repeat after a retry.  Identity distinguishes the
        # current lease from a structurally equal late completion (the ABA
        # case) from the failed attempt.
        if active is not claim:
            raise ValueError("claim is not the active lease for its execution device")
        self._active_by_device.pop(int(claim.execution_device_id))
        self._active_ids.remove(claim.work.work_id)
        self._completed[claim.work.work_id] = result

    def fail(self, claim: LtaWorkClaim, *, retry: bool = False) -> None:
        """Release a failed atomic session and optionally put it back on its owner queue."""

        active = self._active_by_device.get(int(claim.execution_device_id))
        if active is not claim:
            raise ValueError("claim is not the active lease for its execution device")
        self._active_by_device.pop(int(claim.execution_device_id))
        self._active_ids.remove(claim.work.work_id)
        if retry:
            self._queues[claim.owner_device_id].appendleft(claim.work)

    def claim_committable(self) -> LtaCommitClaim | None:
        """Lease the next deterministic result prefix without acknowledging it."""

        if self._active_commit_claim is not None:
            raise RuntimeError("an LTA commit batch is already active")
        ready: list[tuple[LtaSessionWork, object]] = []
        for item in self.work:
            if item.work_id in self._committed_ids:
                continue
            if item.work_id not in self._completed:
                break
            ready.append((item, self._completed[item.work_id]))
        if not ready:
            return None
        claim = LtaCommitClaim(entries=tuple(ready))
        self._active_commit_claim = claim
        return claim

    def complete_commit(self, claim: LtaCommitClaim) -> None:
        """Acknowledge publication of a leased result prefix."""

        if self._active_commit_claim is not claim:
            raise ValueError("claim is not the active LTA commit batch")
        self._committed_ids.update(item.work_id for item, _result in claim.entries)
        self._active_commit_claim = None

    def fail_commit(self, claim: LtaCommitClaim) -> None:
        """Release a failed publication batch without advancing the commit frontier."""

        if self._active_commit_claim is not claim:
            raise ValueError("claim is not the active LTA commit batch")
        self._active_commit_claim = None

    def drain_committable(self) -> tuple[tuple[LtaSessionWork, object], ...]:
        """Convenience path that leases and immediately acknowledges one prefix."""

        claim = self.claim_committable()
        if claim is None:
            return ()
        entries = claim.entries
        self.complete_commit(claim)
        return entries

    def claim_backprojection(self, device_id: int) -> LtaBackprojectionClaim | None:
        """Return one settled view to its original owner for exactly-once backprojection."""

        device = _nonnegative_index(device_id, name="device_id")
        if device not in self.device_ids:
            raise ValueError(f"unknown LTA device {device}")
        if device in self._active_by_device:
            raise RuntimeError(
                f"device {device} cannot backproject while a SAM session is active"
            )
        if device in self._backprojection_by_device:
            raise RuntimeError(
                f"device {device} already owns an active LTA backprojection"
            )
        for view in sorted(self._owner_by_view):
            if self._owner_by_view[view] != device:
                continue
            if view in self._backprojection_claimed or view in self._backprojected:
                continue
            view_ids = {item.work_id for item in self.work if item.view == view}
            if view_ids and view_ids.issubset(self._committed_ids):
                claim = LtaBackprojectionClaim(view=view, owner_device_id=device)
                self._backprojection_claimed.add(view)
                self._backprojection_by_device[device] = claim
                return claim
        return None

    def complete_backprojection(self, claim: LtaBackprojectionClaim) -> None:
        if not isinstance(claim, LtaBackprojectionClaim):
            raise TypeError("claim must be an LtaBackprojectionClaim")
        active = self._backprojection_by_device.get(claim.owner_device_id)
        if active is not claim:
            raise ValueError("claim is not the active backprojection lease for its owner")
        self._backprojection_claimed.remove(claim.view)
        self._backprojection_by_device.pop(claim.owner_device_id)
        self._backprojected.add(claim.view)

    def fail_backprojection(self, claim: LtaBackprojectionClaim) -> None:
        """Release a failed owner-only backprojection so it may be retried."""

        if not isinstance(claim, LtaBackprojectionClaim):
            raise TypeError("claim must be an LtaBackprojectionClaim")
        active = self._backprojection_by_device.get(claim.owner_device_id)
        if active is not claim:
            raise ValueError("claim is not the active backprojection lease for its owner")
        self._backprojection_by_device.pop(claim.owner_device_id)
        self._backprojection_claimed.remove(claim.view)

    @property
    def done(self) -> bool:
        return (
            len(self._completed) == len(self.work)
            and len(self._committed_ids) == len(self.work)
            and self._active_commit_claim is None
            and not self._active_by_device
            and not self._backprojection_by_device
            and len(self._backprojected) == len(self._owner_by_view)
        )

    def snapshot(self) -> LtaScheduleSnapshot:
        pending = tuple(item.work_id for queue in self._queues.values() for item in queue)
        return LtaScheduleSnapshot(
            assignments=self.assignments,
            pending_work_ids=tuple(sorted(pending)),
            active_claims=tuple(
                self._active_by_device[key] for key in sorted(self._active_by_device)
            ),
            active_backprojection_claims=tuple(
                self._backprojection_by_device[key]
                for key in sorted(self._backprojection_by_device)
            ),
            active_commit_work_ids=(
                ()
                if self._active_commit_claim is None
                else tuple(
                    item.work_id for item, _result in self._active_commit_claim.entries
                )
            ),
            completed_work_ids=tuple(sorted(self._completed)),
            projection_ready_views=tuple(sorted(self._projection_ready)),
            backprojected_views=tuple(sorted(self._backprojected)),
        )


def assignment_manifest(
    assignments: Sequence[LtaViewAssignment],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "volume_id": item.view.volume_id,
            "physical_view_id": item.view.physical_view_id,
            "owner_device_id": item.owner_device_id,
            "estimated_cost": item.estimated_cost,
            "work_count": item.work_count,
        }
        for item in assignments
    )


__all__ = (
    "LtaScheduleSnapshot",
    "LtaBackprojectionClaim",
    "LtaCommitClaim",
    "LtaSessionWork",
    "LtaViewAffinityScheduler",
    "LtaViewAssignment",
    "LtaViewKey",
    "LtaWorkClaim",
    "assign_view_owners",
    "assignment_manifest",
)
