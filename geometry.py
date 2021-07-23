"""Module to manipulate geometry of pyvista meshes"""

import numpy as np
from pymeshfix import MeshFix
import pyvista as pv

def get_points_of_type(mesh, surface_type):
    """Returns the points that belong to the given surface type"""

    if not "semantics" in mesh.cell_arrays:
        return []
    
    idxs = [s == surface_type for s in mesh.cell_arrays["semantics"]]

    points = np.array([mesh.cell_points(i) for i in range(mesh.number_of_cells)], dtype=object)

    if all([i == False for i in idxs]):
        return []

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

def oriented_bounding_box(dataset, fix=True):
    """Return the oriented bounding box of the PolyData (only works for vertical
    objects)
    """
    
    obb_2d = MinimumBoundingBox([(p[0], p[1]) for p in dataset.clean().points])

    ground_z = np.min(dataset.clean().points[:, 2])
    height = np.max(dataset.clean().points[:, 2]) - ground_z
    box = np.array([[p[0], p[1], ground_z] for p in list(obb_2d.corner_points)])

    t = np.mean(box, axis=0)
    obb = pv.PolyData(box).delaunay_2d()
    obb.points = obb.points - t
    obb = obb.extrude([0.0, 0.0, height])
    obb.points = obb.points + t

    if fix:
        m = MeshFix(obb.clean().triangulate())
        m.repair()
        obb = m.mesh

    return obb

def area_by_surface(mesh, tri_mesh=None):
    """Compute the area per semantic surface"""

    area = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    point_count = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    surface_count = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    # Compute the triangulated surfaces to fix issues with areas
    if tri_mesh is None:
        tri_mesh = mesh.triangulate()

    if "semantics" in mesh.cell_arrays:
        # Compute area per surface type
        sized = tri_mesh.compute_cell_sizes()
        surface_areas = sized.cell_arrays["Area"]

        points_per_cell = np.array([mesh.cell_n_points(i) for i in range(mesh.number_of_cells)])

        for surface_type in area:
            triangle_idxs = [s == surface_type for s in tri_mesh.cell_arrays["semantics"]]
            area[surface_type] = sum(surface_areas[triangle_idxs])

            face_idxs = [s == surface_type for s in mesh.cell_arrays["semantics"]]

            point_count[surface_type] = sum(points_per_cell[face_idxs])
            surface_count[surface_type] = sum(face_idxs)
    
    return area, point_count, surface_count