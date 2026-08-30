"""Mode-owned channel expansion over the existing canonical TTA grammar."""

from __future__ import annotations

from typing import Tuple

from XTA.config import ChannelFormat, resolve_channel_format

from .contracts import ChannelLayout, ChannelVariant, PipelineMode


def resolve_channel_layout(value: str | ChannelFormat | None) -> ChannelLayout:
    """Resolve exactly one value using the existing TTA ``ChannelFormat`` parser."""

    if isinstance(value, (list, tuple, set, frozenset, dict)):
        raise ValueError("--channel_format accepts exactly one value in the v18 interface")
    resolved = resolve_channel_format(value)
    return ChannelLayout(
        token=resolved.token,
        kind=resolved.kind,
        channel_count=resolved.channel_count,
        stride=resolved.stride,
        offsets=tuple(resolved.offsets),
    )


def expand_channel_variants(
    mode: PipelineMode | str,
    layout: ChannelLayout,
) -> Tuple[ChannelVariant, ...]:
    """Apply the v18 mode contract to one canonical channel layout.

    TTA supplies only the canonical ascending order.  PTA adds the reversed
    channel order only when reversal changes the actual channel sequence.
    """

    resolved_mode = PipelineMode.coerce(mode)
    if not isinstance(layout, ChannelLayout):
        raise TypeError("layout must be a ChannelLayout")
    ascending = ChannelVariant(layout=layout, direction="ascending")
    if resolved_mode is PipelineMode.TTA:
        return (ascending,)
    reversed_offsets = tuple(reversed(layout.offsets))
    if reversed_offsets == layout.offsets:
        return (ascending,)
    return (
        ascending,
        ChannelVariant(layout=layout, direction="reversed"),
    )


def resolve_channel_variants(
    mode: PipelineMode | str,
    value: str | ChannelFormat | None,
) -> Tuple[ChannelVariant, ...]:
    """Resolve one public value and perform mode-owned direction expansion."""

    return expand_channel_variants(mode, resolve_channel_layout(value))


__all__ = (
    "expand_channel_variants",
    "resolve_channel_layout",
    "resolve_channel_variants",
)
