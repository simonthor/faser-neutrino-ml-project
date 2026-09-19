"""Geometry definitions and fiducial volume checks for FASERnu.

This module provides the geometric definitions of the fiducial volume zones
and functions to check if vertices fall within these zones.

The fiducial volume is defined by 6 polygonal zones (the Takubo polygons)
in the x-y plane, combined with z-position cuts based on plate numbers.
"""

import os

import numba as nb
import numpy as np
from numpy.typing import NDArray

# Whether numba kernels should run in parallel. Controlled by the ``RUN_PARALLEL``
# environment variable: parallel processing is on by default and can be
# disabled by setting ``RUN_PARALLEL=0`` before running the program.
RUN_PARALLEL_ENABLED = os.environ.get("RUN_PARALLEL", "1") != "0"

# Z positions of key detector plates (in cm)
# These define the z-extent of the fiducial volume
FIRST_PLATE = 7
LAST_PLATE = 626

# Zone polygons in detector coordinates (from target mass studies)
# Reference: https://indico.cern.ch/event/1627580/contributions/6856640/attachments/3196514/5689588/TargetMassMoriond2026.pdf
# Original coordinates are in mm, converted to cm by dividing by 10
_POLYGONS_MM = [
    [[22.4, 21.8218], [127.0, 8.956], [127.0, 98.55], [22.4, 98.55]],
    [[127.0, 14.2], [236.0, 14.2], [236.0, 98.55], [127.0, 98.55]],
    [[20.8, 98.55], [127.0, 98.55], [127.0, 190.4], [20.8, 190.4]],
    [[127.0, 98.55], [240.0, 98.55], [240.0, 190.4], [127.0, 190.4]],
    [[28.0, 190.4], [127.0, 190.4], [127.0, 280.1], [28.0, 280.1]],
    [[127.0, 190.4], [236.0, 190.4], [236.0, 275.9], [127.0, 275.9]],
]

# Convert to numpy arrays in cm
ZONE_EDGES: tuple[NDArray[np.float64], ...] = tuple(
    np.array([[x / 10.0, y / 10.0] for x, y in poly]) for poly in _POLYGONS_MM
)

# Coordinate offset to convert from detector to rescaled coordinates
# Vertex positions from ROOT files need this offset added after dividing by 10000
COORDINATE_OFFSET = np.array([12.5, 15.0, 0.0])


@nb.njit(cache=True)
def point_in_polygon(x: float, y: float, polygon: NDArray[np.float64]) -> bool:
    """Check if a point (x, y) is inside a polygon using ray casting.

    Parameters
    ----------
    x : float
        X coordinate of the point.
    y : float
        Y coordinate of the point.
    polygon : NDArray
        Array of shape (N, 2) defining the polygon vertices.

    Returns
    -------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@nb.njit(parallel=RUN_PARALLEL_ENABLED, cache=True)
def vertices_in_zones(
    vertices: NDArray[np.float64],
    zone_edges: tuple[NDArray[np.float64], ...],
) -> NDArray[np.bool_]:
    """Check which vertices are inside any of the fiducial zones.

    Parameters
    ----------
    vertices : NDArray
        Array of shape (N, 2) with x, y coordinates of vertices (in cm).
    zone_edges : tuple of NDArray
        Tuple of polygon arrays defining the zone boundaries.

    Returns
    -------
    NDArray[np.bool_]
        Boolean array of length N, True if vertex is inside any zone.
    """
    n_vertices = vertices.shape[0]
    in_zone = np.zeros(n_vertices, dtype=np.bool_)

    for i in nb.prange(n_vertices):  # ty: ignore[not-iterable]
        x, y = vertices[i, 0], vertices[i, 1]
        for poly_points in zone_edges:
            if point_in_polygon(x, y, poly_points):
                in_zone[i] = True
                break

    return in_zone
