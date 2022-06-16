"""Module to manipulate geometry of pyvista meshes"""

import numpy as np
import pyvista as pv
from helpers.geometry import plane_params, project_mesh, to_3d
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering

def get_points_of_type(mesh, surface_type):
    """Returns the points that belong to the given surface type"""

    if not "semantics" in mesh.cell_data:
        return []
    
    idxs = [s == surface_type for s in mesh.cell_data["semantics"]]

    points = np.array([mesh.cell_points(i) for i in range(mesh.number_of_cells)], dtype=object)

    if all([i == False for i in idxs]):
        return []

    return np.vstack(points[idxs])

def move_to_origin(mesh):
    """Moves the object to the origin"""
    pts = mesh.points
    t = np.min(pts, axis=0)
    mesh.points = mesh.points - t

    return mesh, t

def extrude(shape, min, max):
    """Create a pyvista mesh from a polygon"""

    points = np.array([[p[0], p[1], min] for p in shape.boundary.coords])
    mesh = pv.PolyData(points).delaunay_2d()

    if min == max:
        return mesh

    # Transform to 0, 0, 0 to avoid precision issues
    pts = mesh.points
    t = np.mean(pts, axis=0)
    mesh.points = mesh.points - t
    
    mesh = mesh.extrude([0.0, 0.0, max - min], capping=True)
    
    # Transform back to origina coords
    # mesh.points = mesh.points + t

    mesh = mesh.clean().triangulate()

    return mesh

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

    if "semantics" in mesh.cell_data:
        # Compute area per surface type
        sized = tri_mesh.compute_cell_sizes()
        surface_areas = sized.cell_data["Area"]

        points_per_cell = np.array([mesh.cell_n_points(i) for i in range(mesh.number_of_cells)])

        for surface_type in area:
            triangle_idxs = [s == surface_type for s in tri_mesh.cell_data["semantics"]]
            area[surface_type] = sum(surface_areas[triangle_idxs])

            face_idxs = [s == surface_type for s in mesh.cell_data["semantics"]]

            point_count[surface_type] = sum(points_per_cell[face_idxs])
            surface_count[surface_type] = sum(face_idxs)
    
    return area, point_count, surface_count

def face_planes(mesh):
    """Return the params of all planes in a given mesh"""

    return [plane_params(mesh.face_normals[i], mesh.cell_points(i)[0])
            for i in range(mesh.n_cells)]

def cluster_meshes(meshes, threshold=0.1):
    """Clusters the faces of the given meshes"""
    
    n_meshes = len(meshes)
    
    # Compute the "absolute" plane params for every face of the two meshes
    planes = [face_planes(mesh) for mesh in meshes]
    mesh_ids = [[m for _ in range(meshes[m].n_cells)] for m in range(n_meshes)]
    
    # Find the common planes between the two faces
    all_planes = np.concatenate(planes)
    all_labels, n_clusters = cluster_faces(all_planes, threshold)
    areas = []
    
    labels = np.array_split(all_labels, [meshes[m].n_cells for m in range(n_meshes - 1)])
    
    return labels, n_clusters

def cluster_faces(data, threshold=0.1):
    """Clusters the given planes"""
    ndata = np.array(data)
    
    dm1 = distance_matrix(ndata, ndata)
    dm2 = distance_matrix(ndata, -ndata)

    dist_mat = np.minimum(dm1, dm2)

    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=threshold,
                                         affinity='precomputed',
                                         linkage='average').fit(dist_mat)
    
    return clustering.labels_, clustering.n_clusters_

def intersect_surfaces(meshes):
    """Return the intersection between the surfaces of multiple meshes"""

    def get_area_from_ring(areas, area, geom, normal, origin, subtract=False):
        pts = to_3d(geom.coords, normal, origin)
        common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
        if subtract:
            common_mesh["area"] = [-area]
        else:
            common_mesh["area"] = [area]
        areas.append(common_mesh)

    def get_area_from_polygon(areas, geom, normal, origin):
        # polygon with holes:
        if geom.boundary.type == 'MultiLineString':
            get_area_from_ring(areas, geom.area, geom.boundary[0], normal, origin)
            for sgeom in geom.boundary[1:]:
                get_area_from_ring(areas, 0, sgeom, normal, origin, subtract=True)
        # polygon without holes:
        elif geom.boundary.type == 'LineString':
            get_area_from_ring(areas, geom.area, geom.boundary, normal, origin)
    
    n_meshes = len(meshes)
    
    areas = []
    
    labels, n_clusters = cluster_meshes(meshes)
    
    for plane in range(n_clusters):
        # For every common plane, extract the faces that belong to it
        idxs = [[i for i, p in enumerate(labels[m]) if p == plane] for m in range(n_meshes)]
                
        if any([len(idx) == 0 for idx in idxs]):
            continue
        
        msurfaces = [mesh.extract_cells(idxs[i]).extract_surface() for i, mesh in enumerate(meshes)]
                
        # Set the normal and origin point for a plane to project the faces
        origin = msurfaces[0].clean().points[0]
        normal = msurfaces[0].face_normals[0]
        
        # Create the two 2D polygons by projecting the faces
        polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]
        
        # Intersect the 2D polygons
        inter = polys[0]
        for i in range(1, len(polys)):
            inter = inter.intersection(polys[i])
        
        if inter.area > 0.001:
            if inter.type == "MultiPolygon" or inter.type == "GeometryCollection":
                for geom in inter.geoms:
                    if geom.type != "Polygon":
                        continue
                    get_area_from_polygon(areas, geom, normal, origin)
            elif inter.type == "Polygon":
                get_area_from_polygon(areas, inter, normal, origin)
    
    return areas
