from dataclasses import dataclass
from typing import Literal, TypeAlias

Dim: TypeAlias = Literal["X", "Y"]
ScaleGroup: TypeAlias = Literal[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
Element: TypeAlias = Literal[1, 2, 3, 4, 5, 6]
NormRegion: TypeAlias = Literal["white", "black"]
FitResult: TypeAlias = object
MtfResult: TypeAlias = object

@dataclass
class BarSection:
    group: int
    element: int
    dim: Dim

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Roi:
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point

@dataclass
class Anchor:
    roi: Roi

    @property
    def centroid(self) -> Point:
        roi_points = (self.roi.top_left, self.roi.top_right, self.roi.bottom_left, self.roi.bottom_right)
        area_twice = 0.0
        centroid_x = 0.0
        centroid_y = 0.0

        for index, current in enumerate(roi_points):
            following = roi_points[(index + 1) % len(roi_points)]
            cross = (current.x * following.y) - (following.x * current.y)
            area_twice += cross
            centroid_x += (current.x + following.x) * cross
            centroid_y += (current.y + following.y) * cross

        if area_twice == 0.0:
            count = float(len(roi_points))
            return Point(
                x=sum(vertex.x for vertex in roi_points) / count,
                y=sum(vertex.y for vertex in roi_points) / count,
            )

        factor = 1.0 / (3.0 * area_twice)
        return Point(x=centroid_x * factor, y=centroid_y * factor)

@dataclass
class RoiConfig:
    anchor: Anchor
    scale_groups: list[ScaleGroup]
    bar_rois: dict[BarSection, Roi]
    norm_rois: dict[NormRegion, Roi]

@dataclass
class Profile:
    raw_values: list[float]
    norm_values: list[float]
