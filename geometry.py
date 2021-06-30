"""Module to manipulate geometry of pyvista meshes"""

import numpy as np
from pymeshfix import MeshFix
import pyvista as pv

def get_points_of_type(mesh, surface_type):
    """Returns the points that belong to the given surface type"""

    if not "semantics" in mesh.cell_arrays:
        return []
    
    idxs = [s == surface_type for s in mesh.cell_arrays["semantics"]]

    points = np.array([mesh.cell_points(i) for i in range(mesh.number_of_cells)])

    return np.vstack(points[idxs])

def extrude(shape, min, max):
    """Create a pyvista mesh from a polygon"""

    points = np.array([[p[0], p[1], min] for p in shape.boundary.coords])
    mesh = pv.PolyData(points).delaunay_2d()

    # Transform to 0, 0, 0 to avoid precision issues
    pts = mesh.points
    t = np.mean(pts, axis=0)
    mesh.points = mesh.points - t
    
    mesh = mesh.extrude([0.0, 0.0, max - min])
    
    # Transform back to origina coords
    # mesh.points = mesh.points + t

    m = MeshFix(mesh.clean().triangulate())
    m.repair()
    mesh = m.mesh

    return mesh
