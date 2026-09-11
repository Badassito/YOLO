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
from itertools import islice
from typing import Deque, Iterable, Mapping, Sequence


# A concrete tile graph should pass ``tile_count - 1`` to the scheduler.  The
# conservative default remains finite so a malformed coordinator cannot admit
# an unbounded relay wave before concrete topology is connected.
DEFAULT_MAX_RELAY_GENERATION = 1024


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


@dataclass(frozen=True, order=True)
class LtaSpatialRelayKey:
    """Idempotency key for one accumulated relay-mask revision.

    Source tile and relay generation are deliberately absent so equivalent
    arrivals can be merged. The frame distinguishes legitimate later re-entry;
    the revision digest lets complementary masks discovered through a longer
    route advance the same event monotonically while an exact ping-pong is
    suppressed.
    """

    view: LtaViewKey
    runtime_view_id: str
    tile_config_id: str
    lineage_id: str
    destination_tile_index: int
    frame_index: int
    temporal_direction: str
    mask_revision_sha256: str = "0" * 64

    def __post_init__(self) -> None:
        if not isinstance(self.view, LtaViewKey):
            raise TypeError("view must be an LtaViewKey")
        for name in ("runtime_view_id", "tile_config_id", "lineage_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "destination_tile_index",
            _nonnegative_index(
                self.destination_tile_index,
                name="destination_tile_index",
            ),
        )
        object.__setattr__(
            self,
            "frame_index",
            _nonnegative_index(self.frame_index, name="frame_index"),
        )
        direction = str(self.temporal_direction).strip().lower()
        if direction not in {"forward", "backward"}:
            raise ValueError("temporal_direction must be 'forward' or 'backward'")
        object.__setattr__(self, "temporal_direction", direction)
        revision = str(self.mask_revision_sha256).strip().lower()
        if len(revision) != 64 or any(value not in "0123456789abcdef" for value in revision):
            raise ValueError("mask_revision_sha256 must be a SHA256 hexadecimal digest")
        object.__setattr__(self, "mask_revision_sha256", revision)


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
    dependency_work_ids: tuple[str, ...] = ()

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
        dependencies = tuple(str(value).strip() for value in self.dependency_work_ids)
        if any(not value for value in dependencies) or len(set(dependencies)) != len(dependencies):
            raise ValueError("work dependencies must be unique nonempty work ids")
        if self.work_id in dependencies:
            raise ValueError("work cannot depend on itself")
        object.__setattr__(self, "dependency_work_ids", dependencies)

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
    sealed_generations: tuple[tuple[LtaViewKey, int], ...] = ()
    sealed_views: tuple[LtaViewKey, ...] = ()
    settled_spatial_relay_keys: tuple[LtaSpatialRelayKey, ...] = ()


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
    """Exactly-once state machine for owner-first work and unopened helper work.

    All state transitions are intentionally owned by one coordinator thread;
    GPU workers exchange claims and results with that coordinator through
    process-safe queues rather than calling this object concurrently.
    Helpers default to suffix work. Production opts into head order to keep
    the earliest commit dependencies running while later dense unions reduce.
    """

    def __init__(
        self,
        work: Iterable[LtaSessionWork],
        device_ids: Sequence[int],
        *,
        allow_tail_assist: bool = True,
        helper_queue_order: str = "tail",
        max_relay_generation: int = DEFAULT_MAX_RELAY_GENERATION,
    ) -> None:
        self.device_ids = tuple(
            _nonnegative_index(value, name="device_id") for value in device_ids
        )
        if not self.device_ids:
            raise ValueError("device_ids must contain CUDA indexes")
        if len(self.device_ids) != len(set(self.device_ids)):
            raise ValueError("device_ids must be unique")
        self.helper_queue_order = str(helper_queue_order).strip().lower()
        if self.helper_queue_order not in {"head", "tail"}:
            raise ValueError("helper_queue_order must be 'head' or 'tail'")
        self.max_relay_generation = _nonnegative_index(
            max_relay_generation,
            name="max_relay_generation",
        )
        self.work = tuple(sorted(tuple(work), key=lambda item: item.plan_order))
        if not all(isinstance(item, LtaSessionWork) for item in self.work):
            raise TypeError("work must contain only LtaSessionWork instances")
        ids = [item.work_id for item in self.work]
        orders = [item.plan_order for item in self.work]
        if len(ids) != len(set(ids)):
            raise ValueError("work ids must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("plan_order values must be unique")
        self._validate_dependencies(self.work, self.work)
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
            if max(generations) > self.max_relay_generation:
                raise ValueError(
                    f"relay generation {max(generations)} exceeds configured maximum "
                    f"{self.max_relay_generation} for {view.token}"
                )
        self.assignments = assign_view_owners(self.work, self.device_ids)
        self._owner_by_view = {
            assignment.view: assignment.owner_device_id for assignment in self.assignments
        }
        self._projection_key_by_view = dict(projection_keys)
        self._work_ids = set(ids)
        self._plan_orders = set(orders)
        self._generations_by_view = {
            view: set(generations)
            for view, generations in generations_by_view.items()
        }
        # Constructor-supplied generations are immutable complete batches and
        # are therefore sealed immediately. Dynamically registered generations
        # stay open until ``seal_generation`` is called.
        self._sealed_generations: set[tuple[LtaViewKey, int]] = {
            (view, generation)
            for view, generations in self._generations_by_view.items()
            for generation in generations
        }
        self._sealed_views: set[LtaViewKey] = set()
        self._settled_spatial_relay_keys: set[LtaSpatialRelayKey] = set()
        self._queues: dict[int, Deque[LtaSessionWork]] = {
            device: deque(
                item
                for item in self.work
                if self._owner_by_view[item.view] == device and not item.dependency_work_ids
            )
            for device in self.device_ids
        }
        self._allow_tail_assist = bool(allow_tail_assist)
        self._projection_ready: set[LtaViewKey] = set()
        self._active_by_device: dict[int, LtaWorkClaim] = {}
        self._active_ids: set[str] = set()
        self._completed: dict[str, object] = {}
        self._dependency_children: dict[str, list[LtaSessionWork]] = {}
        self._waiting_dependencies: dict[str, set[str]] = {}
        self._generation_uncommitted: dict[tuple[LtaViewKey, int], int] = {}
        self._work_by_id = {item.work_id: item for item in self.work}
        self._register_dependency_state(self.work)
        self._commit_cursor = 0
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

    @staticmethod
    def _validate_dependencies(items: Sequence[LtaSessionWork], available: Sequence[LtaSessionWork]) -> None:
        by_id = {item.work_id: item for item in available}
        for item in items:
            for dependency_id in item.dependency_work_ids:
                dependency = by_id.get(dependency_id)
                if dependency is None:
                    raise ValueError(f"unknown LTA work dependency {dependency_id!r}")
                if (dependency.view != item.view or dependency.plan_order >= item.plan_order
                        or dependency.relay_generation > item.relay_generation):
                    raise ValueError("work dependencies must precede their child in the same view")

    def _require_view(self, view: LtaViewKey) -> LtaViewKey:
        if not isinstance(view, LtaViewKey):
            raise TypeError("view must be an LtaViewKey")
        if view not in self._owner_by_view:
            raise ValueError(f"unknown LTA view {view}")
        return view

    def _register_dependency_state(self, items: Sequence[LtaSessionWork]) -> None:
        for item in items:
            key = item.view, item.relay_generation
            self._generation_uncommitted[key] = self._generation_uncommitted.get(key, 0) + 1
            remaining = set(item.dependency_work_ids) - self._completed.keys()
            if remaining:
                self._waiting_dependencies[item.work_id] = remaining
                for dependency in remaining:
                    self._dependency_children.setdefault(dependency, []).append(item)

    def _activate_dependents(self, work_id: str) -> None:
        for child in self._dependency_children.pop(work_id, ()):
            remaining = self._waiting_dependencies[child.work_id]
            remaining.remove(work_id)
            if remaining:
                continue
            del self._waiting_dependencies[child.work_id]
            owner = self._owner_by_view[child.view]
            queue = self._queues[owner]
            queue.append(child)
            self._queues[owner] = deque(sorted(queue, key=lambda item: item.plan_order))

    def _generation_work_ids(self, view: LtaViewKey, generation: int) -> set[str]:
        return {
            item.work_id
            for item in self.work
            if item.view == view and item.relay_generation == generation
        }

    def _generation_is_settled(self, view: LtaViewKey, generation: int) -> bool:
        key = (view, int(generation))
        if key not in self._sealed_generations:
            return False
        return self._generation_uncommitted.get(key, 0) == 0

    def register_generation(
        self,
        view: LtaViewKey,
        generation: int,
        work: Iterable[LtaSessionWork],
    ) -> tuple[LtaSessionWork, ...]:
        """Register one dynamic relay generation without changing view affinity.

        The first call for a generation may occur only after the preceding
        generation is sealed and committed. Multiple calls may append to the
        same open generation; none of its work is claimable until
        :meth:`seal_generation` closes that batch.
        """

        resolved_view = self._require_view(view)
        resolved_generation = _nonnegative_index(
            generation,
            name="generation",
        )
        if resolved_generation > self.max_relay_generation:
            raise ValueError(
                f"relay generation {resolved_generation} exceeds configured maximum "
                f"{self.max_relay_generation}"
            )
        if resolved_view in self._sealed_views:
            raise RuntimeError(
                f"cannot register late work after view {resolved_view.token} was sealed"
            )
        if (
            resolved_view in self._backprojection_claimed
            or resolved_view in self._backprojected
        ):
            raise RuntimeError("cannot register work after backprojection admission")
        if (resolved_view, resolved_generation) in self._sealed_generations:
            raise RuntimeError(
                f"relay generation {resolved_generation} is already sealed for "
                f"{resolved_view.token}"
            )

        registered = self._generations_by_view[resolved_view]
        highest = max(registered)
        if resolved_generation not in registered:
            if resolved_generation != highest + 1:
                raise ValueError(
                    f"relay generations for {resolved_view.token} must be registered "
                    f"contiguously; expected {highest + 1}, got {resolved_generation}"
                )
            if not self._generation_is_settled(resolved_view, highest):
                raise RuntimeError(
                    f"relay generation {highest} for {resolved_view.token} is not settled"
                )

        items = tuple(work)
        if not all(isinstance(item, LtaSessionWork) for item in items):
            raise TypeError("work must contain only LtaSessionWork instances")
        expected_projection = self._projection_key_by_view[resolved_view]
        for item in items:
            if item.view != resolved_view:
                raise ValueError("registered work belongs to a different physical view")
            if item.relay_generation != resolved_generation:
                raise ValueError("registered work has the wrong relay_generation")
            if item.projection_key != expected_projection:
                raise ValueError(
                    "dynamic work must consume the view's immutable projection cache"
                )

        new_ids = [item.work_id for item in items]
        new_orders = [item.plan_order for item in items]
        if len(new_ids) != len(set(new_ids)) or any(
            work_id in self._work_ids for work_id in new_ids
        ):
            raise ValueError("dynamic work ids must be globally unique")
        if len(new_orders) != len(set(new_orders)) or any(
            order in self._plan_orders for order in new_orders
        ):
            raise ValueError("dynamic plan_order values must be globally unique")
        current_max_order = max(self._plan_orders, default=-1)
        if new_orders and min(new_orders) <= current_max_order:
            raise ValueError(
                "dynamic plan_order values must follow all previously registered work"
            )
        self._validate_dependencies(items, (*self.work, *items))

        # All validation precedes mutation so a rejected generation cannot
        # partially alter queue or commit order.
        registered.add(resolved_generation)
        self.work = tuple(sorted((*self.work, *items), key=lambda item: item.plan_order))
        self._work_ids.update(new_ids)
        self._work_by_id.update({item.work_id: item for item in items})
        self._plan_orders.update(new_orders)
        self._register_dependency_state(items)
        owner = self.owner_for_view(resolved_view)
        self._queues[owner].extend(
            item for item in sorted(items, key=lambda item: item.plan_order)
            if item.work_id not in self._waiting_dependencies
        )
        return items

    def seal_generation(self, view: LtaViewKey, generation: int) -> None:
        """Declare that no more work will be added to a registered generation."""

        resolved_view = self._require_view(view)
        resolved_generation = _nonnegative_index(generation, name="generation")
        if resolved_generation not in self._generations_by_view[resolved_view]:
            raise ValueError(
                f"relay generation {resolved_generation} is not registered for "
                f"{resolved_view.token}"
            )
        self._sealed_generations.add((resolved_view, resolved_generation))

    def generation_settled(self, view: LtaViewKey, generation: int) -> bool:
        """Return whether a closed generation has been committed completely."""

        resolved_view = self._require_view(view)
        resolved_generation = _nonnegative_index(generation, name="generation")
        if resolved_generation not in self._generations_by_view[resolved_view]:
            raise ValueError(
                f"relay generation {resolved_generation} is not registered for "
                f"{resolved_view.token}"
            )
        return self._generation_is_settled(resolved_view, resolved_generation)

    def admit_spatial_relay(self, key: LtaSpatialRelayKey) -> bool:
        """Admit one generation-independent accumulated mask revision once."""

        if not isinstance(key, LtaSpatialRelayKey):
            raise TypeError("key must be an LtaSpatialRelayKey")
        self._require_view(key.view)
        if key.view in self._sealed_views:
            raise RuntimeError(
                f"cannot admit a relay after view {key.view.token} was sealed"
            )
        if key in self._settled_spatial_relay_keys:
            return False
        self._settled_spatial_relay_keys.add(key)
        return True

    def seal_view(self, view: LtaViewKey) -> None:
        """Seal a view after its final relay generation reaches fixed point."""

        resolved_view = self._require_view(view)
        if resolved_view in self._sealed_views:
            return
        unsettled = [
            generation
            for generation in sorted(self._generations_by_view[resolved_view])
            if not self._generation_is_settled(resolved_view, generation)
        ]
        if unsettled:
            raise RuntimeError(
                f"cannot seal view {resolved_view.token}; unsettled relay generations: "
                f"{unsettled}"
            )
        self._sealed_views.add(resolved_view)

    def mark_projection_ready(self, view: LtaViewKey, *, device_id: int) -> None:
        """Publish the one immutable rendered-view cache from its affinity owner."""

        owner = self.owner_for_view(view)
        device = _nonnegative_index(device_id, name="device_id")
        if device != owner:
            raise ValueError("only a view owner may publish its projection cache")
        self._projection_ready.add(view)

    def _generation_ready(self, item: LtaSessionWork) -> bool:
        """Keep relay-derived work behind committed prior generations."""

        generation = int(item.relay_generation)
        if any(work_id not in self._completed for work_id in item.dependency_work_ids):
            return False
        if (item.view, generation) not in self._sealed_generations:
            return False
        if generation == 0:
            return True
        return all(
            self._generation_is_settled(item.view, prior)
            for prior in range(generation)
        )

    def _pop_owner_work(self, device_id: int) -> LtaSessionWork | None:
        queue = self._queues[device_id]
        for position, item in enumerate(queue):
            if item.view in self._projection_ready and self._generation_ready(item):
                del queue[position]
                return item
        return None

    def _steal_helper_work(self, device_id: int) -> tuple[int, LtaSessionWork] | None:
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
            positions = (
                range(len(queue) - 1, -1, -1)
                if self.helper_queue_order == "tail"
                else range(len(queue))
            )
            for position in positions:
                item = queue[position]
                if (
                    item.tail_eligible
                    and item.view in self._projection_ready
                    and self._generation_ready(item)
                ):
                    candidates.append(
                        (remaining_cost, item.view, owner, position, item)
                    )
                    break
        if not candidates:
            return None
        _cost, _view, owner, position, item = min(
            candidates,
            key=lambda value: (-value[0], value[1].token, value[2]),
        )
        queue = self._queues[owner]
        del queue[position]
        return owner, item

    def claim(self, device_id: int) -> LtaWorkClaim | None:
        """Claim owner work first, then one unopened item in the helper order."""

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
            stolen = self._steal_helper_work(device)
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
        self._activate_dependents(claim.work.work_id)

    def fail(self, claim: LtaWorkClaim, *, retry: bool = False) -> None:
        """Release a failed atomic session and optionally put it back on its owner queue."""

        active = self._active_by_device.get(int(claim.execution_device_id))
        if active is not claim:
            raise ValueError("claim is not the active lease for its execution device")
        self._active_by_device.pop(int(claim.execution_device_id))
        self._active_ids.remove(claim.work.work_id)
        if retry:
            self._queues[claim.owner_device_id].appendleft(claim.work)

    def skip_pending(self, work_id: str, result: object) -> None:
        """Settle an unopened continuation after its verified boundary is empty."""

        if work_id in self._active_ids or work_id in self._completed:
            raise ValueError("only pending work may be skipped")
        for queue in self._queues.values():
            for position, item in enumerate(queue):
                if item.work_id != work_id:
                    continue
                if not self._generation_ready(item):
                    raise ValueError("skipped work must have completed dependencies")
                del queue[position]
                self._completed[work_id] = result
                self._activate_dependents(work_id)
                return
        raise ValueError(f"unknown pending LTA work {work_id!r}")

    def claim_committable(self) -> LtaCommitClaim | None:
        """Lease the next deterministic result prefix without acknowledging it."""

        if self._active_commit_claim is not None:
            raise RuntimeError("an LTA commit batch is already active")
        ready: list[tuple[LtaSessionWork, object]] = []
        for item in islice(self.work, self._commit_cursor, None):
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
        self._commit_cursor += len(claim.entries)
        for item, _result in claim.entries:
            self._generation_uncommitted[(item.view, item.relay_generation)] -= 1
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
            if view not in self._sealed_views:
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
            and len(self._sealed_views) == len(self._owner_by_view)
            and len(self._backprojected) == len(self._owner_by_view)
        )

    def queue_counts(self) -> dict[str, int]:
        """Return compact scheduling state without enumerating blocked work ids."""

        ready = sum(
            item.view in self._projection_ready and self._generation_ready(item)
            for queue in self._queues.values() for item in queue
        )
        return {
            "ready": int(ready),
            "blocked": len(self._waiting_dependencies),
            "active": len(self._active_ids),
            "completed_awaiting_commit": len(self._completed) - len(self._committed_ids),
        }

    def snapshot(self) -> LtaScheduleSnapshot:
        pending = (
            *(item.work_id for queue in self._queues.values() for item in queue),
            *self._waiting_dependencies,
        )
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
            sealed_generations=tuple(sorted(self._sealed_generations)),
            sealed_views=tuple(sorted(self._sealed_views)),
            settled_spatial_relay_keys=tuple(
                sorted(self._settled_spatial_relay_keys)
            ),
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
    "DEFAULT_MAX_RELAY_GENERATION",
    "LtaScheduleSnapshot",
    "LtaBackprojectionClaim",
    "LtaCommitClaim",
    "LtaSessionWork",
    "LtaSpatialRelayKey",
    "LtaViewAffinityScheduler",
    "LtaViewAssignment",
    "LtaViewKey",
    "LtaWorkClaim",
    "assign_view_owners",
    "assignment_manifest",
)
