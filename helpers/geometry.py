"""Module with functions for 3D geometrical operations"""

import numpy as np
import mapbox_earcut as earcut
import pyvista as pv
from shapely.geometry import Polygon, MultiPolygon

def surface_normal(poly):
    n = [0.0, 0.0, 0.0]

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly)]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    if all([c == 0 for c in n]):
        raise ValueError("No normal. Possible colinear points!")

    normalised = [i/np.linalg.norm(n) for i in n]

    return normalised

def axes_of_normal(normal):
    """Returns an x-axis and y-axis on a plane of the given normal"""
    if normal[2] > 0.001 or normal[2] < -0.001:
        x_axis = [1, 0, -normal[0]/normal[2]];
    elif normal[1] > 0.001 or normal[1] < -0.001:
        x_axis = [1, -normal[0]/normal[1], 0];
    else:
        x_axis = [-normal[1] / normal[0], 1, 0];
    
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)

    return x_axis, y_axis

def project_2d(points, normal, origin=None):
    if origin is None:
        origin = points[0]

    x_axis, y_axis = axes_of_normal(normal)
     
    return [[np.dot(p - origin, x_axis), np.dot(p - origin, y_axis)] for p in points]

def triangulate(mesh):
    """Triangulates a mesh in the proper way"""
    
    final_mesh = pv.PolyData()
    n_cells = mesh.n_cells
    for i in np.arange(n_cells):
        if not mesh.cell_type(i) in [5, 6, 7, 9, 10]:
            continue

        pts = mesh.cell_points(i)
        p = project_2d(pts, mesh.face_normals[i])
        result = earcut.triangulate_float32(p, [len(p)])

        t_count = len(result.reshape(-1,3))
        triangles = np.hstack([[3] + list(t) for t in result.reshape(-1,3)])
        
        new_mesh = pv.PolyData(pts, triangles, n_faces=t_count)
        for k in mesh.cell_data:
            new_mesh[k] = [mesh.cell_data[k][i] for _ in np.arange(t_count)]
        
        final_mesh = final_mesh + new_mesh
    
    return final_mesh

def triangulate_polygon(face, vertices, offset = 0):
    """Returns the points and triangles for a given CityJSON polygon"""

    points = vertices[np.hstack(face)]
    normal = surface_normal(points)
    holes = [0]
    for ring in face:
        holes.append(len(ring) + holes[-1])
    holes = holes[1:]

    points_2d = project_2d(points, normal)

    result = earcut.triangulate_float32(points_2d, holes)

    result += offset

    t_count = len(result.reshape(-1,3))
    if t_count == 0:
        return points,  []
    triangles = np.hstack([[3] + list(t) for t in result.reshape(-1,3)])

    return points, triangles

def plane_params(normal, origin, rounding=2):
    """Returns the params (a, b, c, d) of the plane equation for the given
    normal and origin point.
    """
    a, b, c = np.round_(normal, 3)
    x0, y0, z0 = origin
    
    d = -(a * x0 + b * y0 + c * z0)
    
    if rounding >= 0:
        d = round(d, rounding)
    
    return np.array([a, b, c, d])

def project_mesh(mesh, normal, origin):
    """Project the faces of a mesh to the given plane"""

    p = []
    for i in range(mesh.n_cells):
        pts = mesh.cell_points(i)
        
        pts_2d = project_2d(pts, normal, origin)
        
        p.append(Polygon(pts_2d))
    
    return MultiPolygon(p).buffer(0)

def to_3d(points, normal, origin):
    """Returns the 3d coordinates of a 2d points from a given plane"""

    xa, ya = axes_of_normal(normal)
    
    mat = np.array([xa, ya])
    pts = np.array(points)
    
    return np.dot(pts, mat) + origin
