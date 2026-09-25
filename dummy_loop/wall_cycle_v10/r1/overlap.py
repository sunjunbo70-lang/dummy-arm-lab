"""Conservative planar overlap building block; not an adopted material model.
Coordinates and areas are SI. A blade cell is a square in the wall plane.
Pitch projection and constitutive fluxes belong to the caller, not this module.
"""
import math
import numpy as np
from numba import njit

GEOMETRY_VERSION = 'v10r1.planar_polygon_overlap.v1'

@njit(cache=True)
def _clip(poly, count, axis, bound, lower):
    out = np.empty((12, 2), dtype=np.float64)
    n = 0
    for k in range(count):
        a = poly[(k - 1) % count]
        b = poly[k]
        ina = a[axis] >= bound if lower else a[axis] <= bound
        inb = b[axis] >= bound if lower else b[axis] <= bound
        if ina != inb:
            t = (bound - a[axis]) / (b[axis] - a[axis])
            out[n] = a + t * (b - a)
            n += 1
        if inb:
            out[n] = b
            n += 1
    return out, n

@njit(cache=True)
def _area_in_box(poly, x0, y0, cell):
    p, n = _clip(poly, 4, 0, x0, True)
    p, n = _clip(p, n, 0, x0 + cell, False)
    p, n = _clip(p, n, 1, y0, True)
    p, n = _clip(p, n, 1, y0 + cell, False)
    if n < 3:
        return 0.0
    # Translate to avoid cancellation for small cells far from the origin.
    area = 0.0
    for k in range(n):
        a = p[k] - p[0]
        b = p[(k + 1) % n] - p[0]
        area += a[0] * b[1] - a[1] * b[0]
    return abs(area) * 0.5

@njit(cache=True)
def planar_overlap(center, angle, blade_shape, blade_cell, wall_shape, wall_cell, origin):
    """Return sparse blade indices, wall indices, m^2 areas, outside m^2.

    Row axis = (cos(angle), sin(angle)); column axis is its left normal.
    Both grids are cell-centered, with origin the wall lower-left corner.
    The caller must transform its own blade convention explicitly.
    """
    nr, nc = blade_shape
    ny, nx = wall_shape
    # A rotated square bounding box has width at most sqrt(2)*blade_cell.
    bound = int(math.ceil(math.sqrt(2.0) * blade_cell / wall_cell)) + 2
    cap = nr * nc * bound * bound
    bi = np.empty(cap, dtype=np.int64)
    wi = np.empty(cap, dtype=np.int64)
    areas = np.empty(cap, dtype=np.float64)
    outside = np.empty(nr * nc, dtype=np.float64)
    e = np.array([math.cos(angle), math.sin(angle)])
    f = np.array([-math.sin(angle), math.cos(angle)])
    count = 0
    for r in range(nr):
        for c in range(nc):
            mid = center + (r + .5 - nr / 2.) * blade_cell * e + (c + .5 - nc / 2.) * blade_cell * f
            poly = np.empty((4, 2), dtype=np.float64)
            poly[0] = mid - .5 * blade_cell * e - .5 * blade_cell * f
            poly[1] = mid + .5 * blade_cell * e - .5 * blade_cell * f
            poly[2] = mid + .5 * blade_cell * e + .5 * blade_cell * f
            poly[3] = mid - .5 * blade_cell * e + .5 * blade_cell * f
            j0 = max(0, int(math.floor((np.min(poly[:, 0]) - origin[0]) / wall_cell)))
            j1 = min(nx - 1, int(math.floor((np.max(poly[:, 0]) - origin[0]) / wall_cell)))
            i0 = max(0, int(math.floor((np.min(poly[:, 1]) - origin[1]) / wall_cell)))
            i1 = min(ny - 1, int(math.floor((np.max(poly[:, 1]) - origin[1]) / wall_cell)))
            total = 0.
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    area = _area_in_box(poly, origin[0] + j * wall_cell, origin[1] + i * wall_cell, wall_cell)
                    if area > 0.:
                        bi[count], wi[count], areas[count] = r * nc + c, i * nx + j, area
                        count += 1
                        total += area
            outside[r * nc + c] = max(0., blade_cell * blade_cell - total)
    return bi[:count], wi[:count], areas[:count], outside


def deposit(blade_volumes, mapping, blade_cell, wall_shape):
    """Distribute the supplied volume once; return wall volume and lost volume.
    Does not mutate inputs or add/absorb any material automatically.
    """
    bi, wi, area, outside = mapping
    volumes = np.asarray(blade_volumes).reshape(-1)
    wall = np.bincount(wi, weights=volumes[bi] * area / blade_cell**2,
                       minlength=int(np.prod(wall_shape))).reshape(wall_shape)
    return wall, float(np.dot(volumes, outside / blade_cell**2))


def gather(wall_volumes, mapping, wall_cell, blade_shape):
    """Transfer covered fractions, leaving uncovered volume on the wall."""
    bi, wi, area, _ = mapping
    volumes = np.asarray(wall_volumes).reshape(-1)
    transferred = volumes[wi] * area / wall_cell**2
    blade = np.bincount(bi, weights=transferred, minlength=int(np.prod(blade_shape)))
    taken = np.bincount(wi, weights=transferred, minlength=volumes.size)
    return blade.reshape(blade_shape), (volumes - taken).reshape(wall_volumes.shape)
