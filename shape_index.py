"""Module that computes indexes for shapely (2D) and polydata (3D) shapes"""

import math
from shapely.geometry import Point

def circularity(shape):
    """Returns circularity 2D for a given polygon"""

    return 4 * math.pi * shape.area / math.pow(shape.length, 2)

def hemisphericality(mesh):
    """Returns hemisphericality for a given volume.
    
    Hemisphericality can be perceived as a similar metric
    to circularity in 2D. But in 3D no building is expected
    to be spherical, but can be relatively hemispherical
    (i.e. starting with a big footpring and narrowing towards
    the roof).
    """

    return 3 * math.sqrt(2) * math.sqrt(math.pi) * mesh.volume / math.pow(mesh.area, 3/2)

def convexity_2d(shape):
    """Returns the convexity in 2D"""

    return shape.area / shape.convex_hull.area

def fractality_2d(shape):
    """Returns the fractality in 2D for a given polygon"""

    return 1 - math.log(shape.area) / (2 * math.log(shape.length))

def fractality_3d(mesh):
    """Returns the fractality in 3D for a given volume"""

    # TODO: Check this formula
    return 1 - math.log(mesh.volume) / (2 * math.log(mesh.area))

def squareness(shape):
    """Returns the squareness in 2D for a given polygon"""

    return 4 * math.sqrt(shape.area) / shape.length

def cubeness(mesh):
    """Returns the cubeness in 3D for a given volume"""

    return 6 * math.pow(mesh.volume, 2/3) / mesh.area

def get_box_dimensions(box):
    """Given a box (as shapely polygon) returns its dimensions as a tuple
    (small, large)
    """

    obb_pts = list(box.boundary.coords)

    S = Point(obb_pts[1]).distance(Point(obb_pts[0]))
    L = Point(obb_pts[2]).distance(Point(obb_pts[1]))

    if S > L:
        L, S = S, L

    return S, L

def elongation(S, L):
    """Returns the elongation for the given dimensions"""

    if S > L:
        return 1 - L / S

    return 1 - S / L

def equivalent_rectangular_index(shape, obb_2d=None):
    """Returns the equivalent rectangular index"""

    if obb_2d is None:
        obb_2d = shape.minimum_rotated_rectangle

    k = math.sqrt(shape.area / obb_2d.area)

    return k * obb_2d.length / shape.length

def equivalent_prism_index(mesh, obb):
    """Returns the equivalent rectangular prism index"""

    k = math.pow(mesh.volume / obb.volume, 2/3)

    # evrp: equal volume rectangular prism
    A_evrp = k * obb.area

    return A_evrp / mesh.area
