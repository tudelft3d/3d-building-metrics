import math

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