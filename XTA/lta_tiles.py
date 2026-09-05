"""Concrete LTA tile geometry and eight-neighbor topology.

The command-line tile grammar lives in :mod:`XTA.unification.tiles`; this
module owns positioned, source-space tiles used by tracking and publication.
It has no accelerator imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TilePlan:
    """One native-resolution square tile in a row-major source grid."""

    left: int
    top: int
    size: int
    source_width: int
    source_height: int
    row: int | None = None
    column: int | None = None

    def __post_init__(self) -> None:
        values = {
            "left": self.left,
            "top": self.top,
            "size": self.size,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }
        for name, raw in values.items():
            if isinstance(raw, bool):
                raise TypeError(f"{name} must be an integer")
            value = int(raw)
            if name in {"size", "source_width", "source_height"} and value < 1:
                raise ValueError(f"{name} must be positive")
            if name in {"left", "top"} and value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.size > self.source_width or self.size > self.source_height:
            raise ValueError("tile size must fit inside the source dimensions")
        if self.left + self.size > self.source_width:
            raise ValueError("tile extends beyond the source width")
        if self.top + self.size > self.source_height:
            raise ValueError("tile extends beyond the source height")
        for name in ("row", "column"):
            raw = getattr(self, name)
            if raw is not None:
                if isinstance(raw, bool) or int(raw) < 0:
                    raise ValueError(f"{name} must be a non-negative integer or None")
                object.__setattr__(self, name, int(raw))

    @property
    def width(self) -> int:
        return self.size

    @property
    def height(self) -> int:
        return self.size

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.size, self.top + self.size


@dataclass(frozen=True, order=True)
class TileNeighbor:
    """One directed Moore-neighborhood edge between positioned tiles."""

    source_index: int
    destination_index: int
    row_delta: int
    column_delta: int
    overlap_xyxy: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if int(self.source_index) == int(self.destination_index):
            raise ValueError("a tile cannot be its own neighbor")
        if self.row_delta not in {-1, 0, 1} or self.column_delta not in {-1, 0, 1}:
            raise ValueError("neighbor deltas must be in {-1,0,1}")
        if self.row_delta == 0 and self.column_delta == 0:
            raise ValueError("a neighbor edge needs a nonzero direction")
        x0, y0, x1, y1 = (int(value) for value in self.overlap_xyxy)
        if x1 < x0 or y1 < y0:
            raise ValueError("neighbor tiles cannot have a spatial gap")
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "destination_index", int(self.destination_index))
        object.__setattr__(self, "overlap_xyxy", (x0, y0, x1, y1))

    @property
    def direction(self) -> str:
        names = {
            (-1, -1): "northwest",
            (-1, 0): "north",
            (-1, 1): "northeast",
            (0, -1): "west",
            (0, 1): "east",
            (1, -1): "southwest",
            (1, 0): "south",
            (1, 1): "southeast",
        }
        return names[(self.row_delta, self.column_delta)]


def plan_axis_starts(extent: int, tile_size: int, stride: int) -> tuple[int, ...]:
    """Cover one axis, pinning the final tile to the far source edge."""

    extent = int(extent)
    tile_size = int(tile_size)
    stride = int(stride)
    if extent < 1 or tile_size < 1 or stride < 1:
        raise ValueError("extent, tile size, and stride must be positive")
    if tile_size > extent:
        raise ValueError(f"tile size {tile_size} exceeds source extent {extent}")
    if stride > tile_size:
        raise ValueError("tile stride must not exceed tile size (gaps are forbidden)")
    last = extent - tile_size
    starts = list(range(0, last + 1, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return tuple(sorted(set(int(value) for value in starts)))


def plan_tile_grid(
    *,
    source_width: int,
    source_height: int,
    tile_size: int = 1008,
    tile_stride: int = 756,
) -> tuple[TilePlan, ...]:
    """Return a row-major, edge-pinned overlapping native tile grid."""

    xs = plan_axis_starts(source_width, tile_size, tile_stride)
    ys = plan_axis_starts(source_height, tile_size, tile_stride)
    return tuple(
        TilePlan(
            left=x,
            top=y,
            size=int(tile_size),
            source_width=int(source_width),
            source_height=int(source_height),
            row=row,
            column=column,
        )
        for row, y in enumerate(ys)
        for column, x in enumerate(xs)
    )


def plan_object_tile(
    polygon: Any,
    *,
    source_width: int,
    source_height: int,
    size: int = 1008,
) -> TilePlan:
    """Place one native tile around a normalized polygon bounding box."""

    source_width = int(source_width)
    source_height = int(source_height)
    size = int(size)
    if source_width <= 0 or source_height <= 0 or size <= 0:
        raise ValueError("source dimensions and tile size must be positive")
    if source_width < size or source_height < size:
        raise ValueError(
            f"source dimensions {source_width}x{source_height} cannot contain "
            f"a {size}x{size} native tile"
        )
    x0, y0, x1, y1 = (float(value) for value in polygon.box_xyxy)
    pixel_box = (
        x0 * source_width,
        y0 * source_height,
        x1 * source_width,
        y1 * source_height,
    )
    if pixel_box[2] - pixel_box[0] > size or pixel_box[3] - pixel_box[1] > size:
        raise ValueError(f"selected polygon bbox does not fit a {size}x{size} source tile")
    center_x = (pixel_box[0] + pixel_box[2]) * 0.5
    center_y = (pixel_box[1] + pixel_box[3]) * 0.5
    left = min(max(0, int(round(center_x - size * 0.5))), source_width - size)
    top = min(max(0, int(round(center_y - size * 0.5))), source_height - size)
    plan = TilePlan(left, top, size, source_width, source_height)
    if (
        plan.left > pixel_box[0]
        or plan.top > pixel_box[1]
        or plan.left + plan.size < pixel_box[2]
        or plan.top + plan.size < pixel_box[3]
    ):
        raise RuntimeError("object-centered tile does not contain the complete polygon bbox")
    return plan


def transform_polygon_to_tile(polygon: Any, plan: TilePlan) -> Any:
    """Transform a normalized source polygon to normalized tile coordinates."""

    points = tuple(
        (
            (float(x) * plan.source_width - plan.left) / plan.size,
            (float(y) * plan.source_height - plan.top) / plan.size,
        )
        for x, y in polygon.points
    )
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    return SimpleNamespace(
        points=points,
        box_xyxy=(min(xs), min(ys), max(xs), max(ys)),
        row_index=getattr(polygon, "row_index", -1),
    )


def select_polygon_row(polygons: Sequence[Any], row_index: int) -> Any:
    matches = [
        polygon
        for polygon in polygons
        if int(getattr(polygon, "row_index", -1)) == int(row_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"selected polygon row {row_index} is not unique")
    return matches[0]


def rasterize_polygons(polygons: Sequence[Any], size: int = 1008):
    """Rasterize normalized polygon points into one immutable-size binary tile."""

    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("1", (int(size), int(size)), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(
            [(float(x) * int(size), float(y) * int(size)) for x, y in polygon.points],
            fill=1,
        )
    return np.asarray(image, dtype=bool)


def rasterize_polygons_to_shape(
    polygons: Sequence[Any],
    *,
    height: int,
    width: int,
):
    """Rasterize normalized polygons directly in a rectangular source frame."""

    import numpy as np
    from PIL import Image, ImageDraw

    resolved_height = int(height)
    resolved_width = int(width)
    if resolved_height < 1 or resolved_width < 1:
        raise ValueError("raster dimensions must be positive")
    image = Image.new("1", (resolved_width, resolved_height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(
            [
                (float(x) * resolved_width, float(y) * resolved_height)
                for x, y in polygon.points
            ],
            fill=1,
        )
    return np.asarray(image, dtype=bool)


def polygon_intersects_tile(polygon: Any, tile: TilePlan) -> bool:
    x0, y0, x1, y1 = (float(value) for value in polygon.box_xyxy)
    source_x0 = x0 * tile.source_width
    source_y0 = y0 * tile.source_height
    source_x1 = x1 * tile.source_width
    source_y1 = y1 * tile.source_height
    left, top, right, bottom = tile.xyxy
    return bool(
        source_x1 > left
        and source_x0 < right
        and source_y1 > top
        and source_y0 < bottom
    )


def masks_for_tile(polygons: Iterable[Any], tile: TilePlan) -> tuple[Any, ...]:
    """Rasterize each nonempty source polygon portion visible in a tile."""

    masks = []
    for polygon in polygons:
        if not polygon_intersects_tile(polygon, tile):
            continue
        local = transform_polygon_to_tile(polygon, tile)
        mask = rasterize_polygons((local,), size=tile.size)
        if bool(mask.any()):
            masks.append(mask)
    return tuple(masks)


def eight_neighbor_graph(tiles: Sequence[TilePlan]) -> Mapping[int, tuple[TileNeighbor, ...]]:
    """Build a symmetric eight-neighbor graph from positioned grid geometry.

    Row/column coordinates are inferred from unique source starts when a caller
    supplies legacy ``TilePlan`` objects without explicit grid coordinates.
    Actual overlap rectangles are computed from geometry, which handles the
    irregular final stride introduced by edge pinning.
    """

    plans = tuple(tiles)
    if not plans:
        return {}
    source_shapes = {(item.source_width, item.source_height) for item in plans}
    if len(source_shapes) != 1:
        raise ValueError("all tiles in a neighbor graph must share source dimensions")
    if len({item.xyxy for item in plans}) != len(plans):
        raise ValueError("a neighbor graph cannot contain duplicate tile rectangles")
    xs = {value: index for index, value in enumerate(sorted({item.left for item in plans}))}
    ys = {value: index for index, value in enumerate(sorted({item.top for item in plans}))}
    coordinates: dict[int, tuple[int, int]] = {}
    for index, item in enumerate(plans):
        inferred = (ys[item.top], xs[item.left])
        explicit = (item.row, item.column)
        if explicit != (None, None) and explicit != inferred:
            raise ValueError(
                f"tile {index} row/column {explicit} disagrees with geometry {inferred}"
            )
        coordinates[index] = inferred
    by_coordinate = {coordinate: index for index, coordinate in coordinates.items()}
    if len(by_coordinate) != len(plans):
        raise ValueError("tile grid coordinates must be unique")
    graph: dict[int, list[TileNeighbor]] = {index: [] for index in range(len(plans))}
    for source_index, (row, column) in coordinates.items():
        source = plans[source_index]
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                destination_index = by_coordinate.get((row + row_delta, column + column_delta))
                if destination_index is None:
                    continue
                destination = plans[destination_index]
                sx0, sy0, sx1, sy1 = source.xyxy
                dx0, dy0, dx1, dy1 = destination.xyxy
                overlap = max(sx0, dx0), max(sy0, dy0), min(sx1, dx1), min(sy1, dy1)
                if overlap[2] < overlap[0] or overlap[3] < overlap[1]:
                    raise ValueError(
                        f"neighbor tiles {source_index}/{destination_index} have a gap"
                    )
                graph[source_index].append(
                    TileNeighbor(
                        source_index=source_index,
                        destination_index=destination_index,
                        row_delta=row_delta,
                        column_delta=column_delta,
                        overlap_xyxy=overlap,
                    )
                )
    return {
        index: tuple(sorted(neighbors, key=lambda edge: edge.destination_index))
        for index, neighbors in graph.items()
    }


__all__ = (
    "TileNeighbor",
    "TilePlan",
    "eight_neighbor_graph",
    "masks_for_tile",
    "plan_axis_starts",
    "plan_object_tile",
    "plan_tile_grid",
    "polygon_intersects_tile",
    "rasterize_polygons",
    "rasterize_polygons_to_shape",
    "select_polygon_row",
    "transform_polygon_to_tile",
)
