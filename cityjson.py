"""Module with functions for manipulating CityJSON data"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from helpers.geometry import project_2d, surface_normal, triangulate, triangulate_polygon
import pyvista as pv

def get_surface_boundaries(geom):
    """Returns the boundaries for all surfaces"""

    if geom["type"] == "MultiSurface" or geom["type"] == "CompositeSurface":
        return geom["boundaries"]
    elif geom["type"] == "Solid":
        return geom["boundaries"][0]
    else:
        raise Exception("Geometry not supported")

def get_points(geom, verts):
    """Return the points of the geometry"""

    boundaries = get_surface_boundaries(geom)

    f = [v for ring in boundaries for v in ring[0]]
    points = [verts[i] for i in f]

    return points

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

        boundaries = np.array(boundaries, dtype=object)[ground_idxs]
    
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
        
        mesh.cell_data["semantics"] = [semantics["surfaces"][i]["type"] for i in values]
    
    return mesh

def to_triangulated_polydata(geom, vertices, clean=True):
    """Returns the polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)
    
    if "semantics" in geom:        
        semantics = geom["semantics"]
        if geom["type"] == "MultiSurface":
            values = semantics["values"]
        else:
            values = semantics["values"][0]
        
        semantic_types = [semantics["surfaces"][i]["type"] for i in values]

    points = []
    triangles = []
    semantics = []
    triangle_count = 0
    for fid, face in enumerate(boundaries):
        try:
            new_points, new_triangles = triangulate_polygon(face, vertices, len(points))
        except:
            continue

        points.extend(new_points)
        triangles.extend(new_triangles)
        t_count = int(len(new_triangles) / 4)

        triangle_count += t_count

        if "semantics" in geom:
            semantics.extend([semantic_types[fid] for _ in np.arange(t_count)])
    
    mesh = pv.PolyData(points, triangles, n_faces=triangle_count)

    if "semantics" in geom:
        mesh["semantics"] = semantics
    
    if clean:
        mesh = mesh.clean()

    return mesh

def get_bbox(geom, verts):
    pts = np.array(get_points(geom, verts))

    return np.hstack([[np.min(pts[:, i]), np.max(pts[:, i])] for i in range(np.shape(pts)[1])])