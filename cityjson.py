"""Module with functions for manipulating CityJSON data"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from helpers.geometry import project_2d, surface_normal, triangulate, triangulate_polygon

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

def to_polydata(geom, vertices):
    """Returns the polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    f = [[len(r[0])] + r[0] for r in [f for f in boundaries]]
    faces = np.hstack(f)

    mesh = pv.PolyData(vertices, faces, n_faces=len(boundaries))

    if "semantics" in geom:        
        semantics = geom["semantics"]
        if geom["type"] == "MultiSurface":
            values = semantics["values"]
        else:
            values = semantics["values"][0]
        
        mesh.cell_arrays["semantics"] = [semantics["surfaces"][i]["type"] for i in values]
    
    return mesh

def to_triangulated_polydata(geom, vertices):
    """Returns the polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    final_mesh = pv.PolyData()
    
    if "semantics" in geom:        
        semantics = geom["semantics"]
        if geom["type"] == "MultiSurface":
            values = semantics["values"]
        else:
            values = semantics["values"][0]
        
        semantic_types = [semantics["surfaces"][i]["type"] for i in values]

    for fid, face in enumerate(boundaries):
        points, triangles = triangulate_polygon(face, vertices)

        if len(triangles) == 0:
            continue

        new_mesh = pv.PolyData(points, triangles, n_faces=t_count)
        if "semantics" in geom:
            new_mesh["semantics"] = [semantic_types[fid] for _ in np.arange(t_count)]

        final_mesh = final_mesh + new_mesh
    
    final_mesh.clean()

    return final_mesh