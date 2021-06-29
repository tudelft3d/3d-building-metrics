import numpy as np
from shapely.geometry import Polygon, MultiPolygon

def get_surface_boundaries(geom):
    """Returns the boundaries for all surfaces"""

    if geom["type"] == "MultiSurface":
        return geom["boundaries"]
    elif geom["type"] == "Solid":
        return geom["boundaries"][0]
    else:
        raise Exception("Geometry not supported")

def to_shapely(geom, vertices, ground_only=True):
    """Returns a shapely geometry of the footprint from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    if ground_only and "semantics" in geom:
        semantics = geom["semantics"]
        if geom["type"] == "MultiSurface":
            values = semantics["values"]
        else:
            values = semantics["values"][0]
        
        ground_idxs = [semantics["surfaces"][i]["type"] == "GroundSurface" for i in values]

        boundaries = np.array(boundaries)[ground_idxs]
    
    shape = MultiPolygon([Polygon([vertices[v] for v in boundary[0]]) for boundary in boundaries])

    shape = shape.buffer(0)
    
    return shape