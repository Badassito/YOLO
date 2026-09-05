"""Deterministic temporal windows for authoritative-mask LTA propagation.

Each positive anchor owns its nearest-anchor temporal domain.  A domain of at
most :data:`XTA.lta_sam.LTA_SESSION_FRAMES` frames is one authoritative SAM
session.  Longer domains start with one anchor-containing session and extend
outward through directional sessions that overlap their predecessor by exactly
one dogfood frame.

The repeated frame is context, not output ownership: a backward dogfood window
does not own its final frame and a forward dogfood window does not own its
first.  Consequently every frame in an anchor domain is owned exactly once
even though boundary predictions are reused as the next injected mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Iterable

from .lta_sam import LTA_SESSION_FRAMES


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@dataclass(frozen=True, order=True)
class AnchorDomain:
    """Half-open temporal range assigned to its nearest positive anchor."""

    anchor_frame: int
    frame_start: int
    frame_stop: int

    def __post_init__(self) -> None:
        for name in ("anchor_frame", "frame_start", "frame_stop"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name),
            )
        if self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_stop <= self.frame_start:
            raise ValueError("frame_stop must be greater than frame_start")
        if not self.frame_start <= self.anchor_frame < self.frame_stop:
            raise ValueError("anchor_frame must lie inside its domain")

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start


@dataclass(frozen=True, order=True)
class WindowPlan:
    """One fixed-size authoritative or directional dogfood SAM session."""

    branch: str
    ordinal: int
    frame_start: int
    frame_stop: int
    prompt_frame: int
    direction: str
    seed_kind: str

    def __post_init__(self) -> None:
        branch = str(self.branch).strip().lower()
        direction = str(self.direction).strip().lower()
        seed_kind = str(self.seed_kind).strip().lower()
        if branch not in {"center", "backward", "forward"}:
            raise ValueError(f"unknown window branch: {self.branch!r}")
        if direction not in {"both", "backward", "forward"}:
            raise ValueError(f"unknown propagation direction: {self.direction!r}")
        if seed_kind not in {"authoritative", "spatial_relay", "dogfood"}:
            raise ValueError(f"unknown seed kind: {self.seed_kind!r}")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "seed_kind", seed_kind)

        for name in ("ordinal", "frame_start", "frame_stop", "prompt_frame"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name),
            )
        if self.ordinal < 0 or self.frame_start < 0:
            raise ValueError("window ordinal and frame_start must be non-negative")
        if self.frame_stop <= self.frame_start:
            raise ValueError("frame_stop must be greater than frame_start")
        if self.frame_count > LTA_SESSION_FRAMES:
            raise ValueError(
                f"one LTA window may contain at most {LTA_SESSION_FRAMES} frames"
            )
        if not self.frame_start <= self.prompt_frame < self.frame_stop:
            raise ValueError("prompt_frame must lie inside its window")

        if seed_kind == "authoritative":
            if branch != "center" or direction != "both" or self.ordinal != 0:
                raise ValueError(
                    "an authoritative window must be center ordinal 0 in both directions"
                )
        elif seed_kind == "spatial_relay":
            if self.ordinal != 0 or branch not in {"backward", "forward"}:
                raise ValueError(
                    "a spatial-relay window must be directional ordinal 0"
                )
            if branch != direction:
                raise ValueError("a spatial-relay branch must match its direction")
            if direction == "backward" and self.prompt_frame != self.frame_stop - 1:
                raise ValueError("a backward spatial relay must prompt its final frame")
            if direction == "forward" and self.prompt_frame != self.frame_start:
                raise ValueError("a forward spatial relay must prompt its first frame")
        elif self.ordinal < 1 or self.frame_count < 2:
            raise ValueError(
                "a dogfood window needs a positive ordinal and at least two frames"
            )
        elif branch == "backward":
            if direction != "backward" or self.prompt_frame != self.frame_stop - 1:
                raise ValueError(
                    "a backward dogfood window must be prompted at its final frame"
                )
        elif branch == "forward":
            if direction != "forward" or self.prompt_frame != self.frame_start:
                raise ValueError(
                    "a forward dogfood window must be prompted at its first frame"
                )
        else:
            raise ValueError("a dogfood window must be backward or forward")

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start


def plan_anchor_domains(
    anchor_frames: Iterable[int],
    *,
    frame_start: int,
    frame_stop: int,
) -> tuple[AnchorDomain, ...]:
    """Partition a selected interval into deterministic nearest-anchor domains."""

    start = _integer(frame_start, name="frame_start")
    stop = _integer(frame_stop, name="frame_stop")
    if start < 0 or stop <= start:
        raise ValueError("frame range must be nonempty and non-negative")

    frames_list: list[int] = []
    for raw in anchor_frames:
        frames_list.append(_integer(raw, name="anchor_frame"))
    frames = tuple(sorted(set(frames_list)))
    if not frames:
        return ()
    if frames[0] < start or frames[-1] >= stop:
        raise ValueError("all anchors must lie inside the selected frame range")

    domains = tuple(
        AnchorDomain(
            anchor_frame=anchor,
            frame_start=(
                start
                if index == 0
                else (frames[index - 1] + anchor) // 2 + 1
            ),
            frame_stop=(
                stop
                if index + 1 == len(frames)
                else (anchor + frames[index + 1]) // 2 + 1
            ),
        )
        for index, anchor in enumerate(frames)
    )
    if domains[0].frame_start != start or domains[-1].frame_stop != stop:
        raise RuntimeError("nearest-anchor domains did not cover the selected range")
    if any(left.frame_stop != right.frame_start for left, right in zip(domains, domains[1:])):
        raise RuntimeError("nearest-anchor domains contain a gap or overlap")
    return domains


def owned_frame_range(window: WindowPlan) -> tuple[int, int]:
    """Return the half-open output range after removing dogfood context."""

    if not isinstance(window, WindowPlan):
        raise TypeError("window must be a WindowPlan")
    if window.seed_kind in {"authoritative", "spatial_relay"}:
        return window.frame_start, window.frame_stop
    if window.direction == "backward":
        return window.frame_start, window.frame_stop - 1
    if window.direction == "forward":
        return window.frame_start + 1, window.frame_stop
    raise RuntimeError("a dogfood window must be directional")


def plan_domain_windows(domain: AnchorDomain) -> tuple[WindowPlan, ...]:
    """Plan a fixed 30-frame center-out dogfood chain for one anchor domain."""

    if not isinstance(domain, AnchorDomain):
        raise TypeError("domain must be an AnchorDomain")
    if domain.frame_count <= LTA_SESSION_FRAMES:
        return (
            WindowPlan(
                branch="center",
                ordinal=0,
                frame_start=domain.frame_start,
                frame_stop=domain.frame_stop,
                prompt_frame=domain.anchor_frame,
                direction="both",
                seed_kind="authoritative",
            ),
        )

    # A 30-frame center has fourteen frames available to the left of its
    # anchor before the right side receives the remaining fifteen.  Near a
    # domain edge the complete fixed window shifts inward without moving the
    # authoritative prompt.
    left_budget = (LTA_SESSION_FRAMES - 1) // 2
    center_start = max(domain.frame_start, domain.anchor_frame - left_budget)
    center_stop = min(domain.frame_stop, center_start + LTA_SESSION_FRAMES)
    if center_stop - center_start < LTA_SESSION_FRAMES:
        center_start = max(domain.frame_start, center_stop - LTA_SESSION_FRAMES)
    center = WindowPlan(
        branch="center",
        ordinal=0,
        frame_start=center_start,
        frame_stop=center_stop,
        prompt_frame=domain.anchor_frame,
        direction="both",
        seed_kind="authoritative",
    )

    backward: list[WindowPlan] = []
    boundary = center.frame_start
    ordinal = 1
    while boundary > domain.frame_start:
        start = max(
            domain.frame_start,
            boundary - (LTA_SESSION_FRAMES - 1),
        )
        backward.append(
            WindowPlan(
                branch="backward",
                ordinal=ordinal,
                frame_start=start,
                frame_stop=boundary + 1,
                prompt_frame=boundary,
                direction="backward",
                seed_kind="dogfood",
            )
        )
        boundary = start
        ordinal += 1

    forward: list[WindowPlan] = []
    boundary = center.frame_stop - 1
    ordinal = 1
    while boundary + 1 < domain.frame_stop:
        stop = min(domain.frame_stop, boundary + LTA_SESSION_FRAMES)
        forward.append(
            WindowPlan(
                branch="forward",
                ordinal=ordinal,
                frame_start=boundary,
                frame_stop=stop,
                prompt_frame=boundary,
                direction="forward",
                seed_kind="dogfood",
            )
        )
        boundary = stop - 1
        ordinal += 1

    windows = (center, *backward, *forward)
    owned = [
        frame
        for window in windows
        for frame in range(*owned_frame_range(window))
    ]
    expected = list(range(domain.frame_start, domain.frame_stop))
    if sorted(owned) != expected or len(owned) != len(set(owned)):
        raise RuntimeError("dogfood window planning did not assign each domain frame once")
    return windows


def plan_directional_windows(
    *,
    frame_start: int,
    frame_stop: int,
    prompt_frame: int,
    direction: str,
) -> tuple[WindowPlan, ...]:
    """Plan a relay-seeded directional chain to one selected-range edge."""

    start = _integer(frame_start, name="frame_start")
    stop = _integer(frame_stop, name="frame_stop")
    prompt = _integer(prompt_frame, name="prompt_frame")
    resolved_direction = str(direction).strip().lower()
    if start < 0 or stop <= start or not start <= prompt < stop:
        raise ValueError("relay prompt and range are inconsistent")
    if resolved_direction not in {"forward", "backward"}:
        raise ValueError("relay direction must be forward or backward")

    windows: list[WindowPlan] = []
    if resolved_direction == "forward":
        boundary = prompt
        ordinal = 0
        while True:
            window_stop = min(stop, boundary + LTA_SESSION_FRAMES)
            windows.append(
                WindowPlan(
                    branch="forward",
                    ordinal=ordinal,
                    frame_start=boundary,
                    frame_stop=window_stop,
                    prompt_frame=boundary,
                    direction="forward",
                    seed_kind="spatial_relay" if ordinal == 0 else "dogfood",
                )
            )
            if window_stop >= stop:
                break
            boundary = window_stop - 1
            ordinal += 1
    else:
        boundary = prompt
        ordinal = 0
        while True:
            window_start = max(start, boundary - (LTA_SESSION_FRAMES - 1))
            windows.append(
                WindowPlan(
                    branch="backward",
                    ordinal=ordinal,
                    frame_start=window_start,
                    frame_stop=boundary + 1,
                    prompt_frame=boundary,
                    direction="backward",
                    seed_kind="spatial_relay" if ordinal == 0 else "dogfood",
                )
            )
            if window_start <= start:
                break
            boundary = window_start
            ordinal += 1

    owned = [frame for window in windows for frame in range(*owned_frame_range(window))]
    expected = (
        list(range(prompt, stop))
        if resolved_direction == "forward"
        else list(range(start, prompt + 1))
    )
    if sorted(owned) != expected or len(owned) != len(set(owned)):
        raise RuntimeError("directional relay windows do not own their range exactly once")
    return tuple(windows)


__all__ = (
    "AnchorDomain",
    "WindowPlan",
    "owned_frame_range",
    "plan_anchor_domains",
    "plan_domain_windows",
    "plan_directional_windows",
)
