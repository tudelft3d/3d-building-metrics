import numpy as np
import pymesh
import pyvista as pv

def to_pymesh(mesh):
    """Returns a pymesh from a pyvista PolyData"""
    v = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:]

    return pymesh.form_mesh(v, f)

def to_pyvista(mesh):
    """Return a PolyData from a pymesh"""
    v = mesh.vertices
    f = mesh.faces
    
    f = np.hstack([[len(f)] + list(f) for f in mesh.faces])
    
    return pv.PolyData(v, f, len(mesh.faces))

def intersect(mesh1, mesh2, engine="igl"):
    """Returns the intersection of two meshes (in pymesh format)"""

    return pymesh.boolean(mesh1, mesh2, operation="intersection", engine=engine)

def symmetric_difference(mesh1, mesh2, engine="igl"):
    """Returns the symmetric difference of two volumes (in pymesh format)"""

    return pymesh.boolean(mesh1, mesh2, operation="symmetric_difference", engine=engine)

def difference(mesh1, mesh2, engine="igl"):
    """Returns the difference between two volumes (in pymesh format)"""

    return pymesh.boolean(mesh1, mesh2, operation="difference", engine=engine)