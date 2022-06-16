"""Module that computes indexes for shapely (2D) and polydata (3D) shapes"""

import math
from shapely.geometry import Point, MultiPoint, Polygon
from helpers.geometry import surface_normal
try:
    from helpers.mesh import to_pymesh, to_pyvista, intersect
    pymesh_exists = True
except:
    print("WARNING: pymesh not found! Exchange index calculation will be omitted...")
    pymesh_exists = False
import miniball
import numpy as np
import pyvista as pv
# import pymesh

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
    return 1 - math.log(mesh.volume) / (3/2 * math.log(mesh.area))

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

def create_grid_2d(shape, density):
    """Return the grid for a given polygon"""
    
    x_min, y_min, x_max, y_max = shape.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    x, y = np.meshgrid(x, y)
    
    x = np.hstack(x)
    y = np.hstack(y)
    
    return [(x[i], y[i]) for i in range(len(x))]

def create_grid_3d(mesh, density, check_surface=False):
    """Returns the grid for a given mesh"""
    voxel = pv.voxelize(mesh, density=density, check_surface=check_surface)
    
    return voxel.cell_centers().points

def to_3d(points, normal, origin):
    """Translate local 2D coordinates to 3D"""
    
    x_axis, y_axis = axes_of_normal(normal)
  
    return (np.repeat([origin], len(points), axis=0)
        + np.matmul(points, [x_axis, y_axis]))    

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

def project_2d(points, normal):
    origin = points[0]

    x_axis, y_axis = axes_of_normal(normal)
     
    return [[np.dot(p - origin, x_axis), np.dot(p - origin, y_axis)] for p in points]

def create_surface_grid(mesh, density=1):
    """Create a 2-dimensional grid along the surface of a 3D mesh"""
    
    result = []
    
    sized = mesh.compute_cell_sizes()
    
    for i in range(mesh.n_cells):
        if not mesh.cell_type(i) in [5, 6, 7, 9, 10]:
            continue
        
        pts = mesh.cell_points(i)
        
        try:
            normal = surface_normal(pts)
        except:
            continue
        
        pts_2d = project_2d(pts, normal)
        poly_2d = Polygon(pts_2d)

        if not poly_2d.is_valid:
            continue
        
        grid = create_grid_2d(poly_2d, density)
        grid = MultiPoint(grid).intersection(poly_2d)
        
        if grid.is_empty:
            continue
        elif grid.geom_type == "Point":
            grid = np.array(grid.coords)
        else:
            grid = np.array([list(p.coords[0]) for p in grid.geoms])
        
        # TODO: Randomise the origin
        result.extend(list(to_3d(grid, normal, pts[0])))
            
    return result

def distance(x, y):
    """Returns the euclidean distance between two points"""
    
    return math.sqrt(sum([math.pow(x[c] - y[c], 2) for c in range(len(x))]))

def cohesion_2d(shape, grid=None, density=1):
    """Returns the cohesion index in 2D for a given polygon"""
    
    if grid is None:
        grid = create_grid_2d(shape, density)
    
    if isinstance(grid, list):
        grid = MultiPoint(grid).intersection(shape)
    
    d = 0
    for pi in grid.geoms:
        for pj in grid.geoms:
            if pi == pj:
                continue
            
            d += pi.distance(pj)

    n = len(grid.geoms)
    return 0.9054 * math.sqrt(shape.area / math.pi) / (1 / (n * (n - 1)) * d)

def cohesion_3d(mesh, grid=None, density=1, check_surface=False):
    """Returns the cohesion index in 3D for a given mesh"""
    
    if grid is None:
        grid = create_grid_3d(density=density, check_surface=check_surface)
    
    d = 0
    for pi in grid:
        for pj in grid:
            d += distance(pi, pj)
    
    
    n = len(grid)
    return 36 / 35 * math.pow(3 * mesh.volume / (4 * math.pi), 1/3) / (1 / (n * (n - 1)) * d)

def proximity_2d(shape, density=1, grid=None):
    """Returns the proximity index in 2D for a given polygon"""
    
    if grid is None:
        grid = create_grid_2d(shape, density)
    
    if isinstance(grid, list):
        grid = MultiPoint(grid).intersection(shape)

        if grid.is_empty:
            return -1

        if grid.geom_type == "Point":
            grid = MultiPoint([grid])
    
    centroid = shape.centroid
    
    return 2 / 3 * math.sqrt(shape.area / math.pi) / np.mean([centroid.distance(p) for p in grid])

def proximity_3d(mesh, grid=None, density=1, check_surface=False):
    """Returns the cohesion index in 3D for a given mesh"""
    
    if grid is None:
        grid = create_grid_3d(mesh, density=density, check_surface=check_surface)

    centroid = np.mean(grid, axis=0)
    
    # TODO: Verify the formula here
    r = math.pow(3 * mesh.volume / (4 * math.pi), 1/3)

    return (3 * r / 4) / np.mean([distance(centroid, p) for p in grid])

def equal_volume_radius(volume):
    """Returns the radius of the equal volume sphere"""
    
    return math.pow(3 * volume / (4 * math.pi), 1/3)

def equal_volume_sphere(mesh, position=(0, 0, 0)):
    """Returns the sphere that has the same volume as the given mesh"""
    
    r = math.pow(3 * mesh.volume / (4 * math.pi), 1/3)
    
    return pv.Sphere(radius=r, center=position)

def exchange_2d(shape):
    """Returns the exchange index in 2D for a given polygon"""
    
    r = math.sqrt(shape.area / math.pi)
    
    eac = shape.centroid.buffer(r)
    
    return shape.intersection(eac).area / shape.area

def exchange_3d(mesh, evs=None, density=0.25, engine="igl"):
    """Returns the exhange index in 3D for a given mesh
    
    mesh: The pyvista mesh to evaluate
    evs: The equal volume sphere (if provided speeds up the calculation)
    density: If no evs is provided, it is used to create a grid to compute the center of mass
    enginge: The engine for the boolean operations
    """

    if not pymesh_exists:
        return -1
    
    if evs is None:
        voxel = pv.voxelize(mesh, density=density, check_surface=False)
        grid = voxel.cell_centers().points

        if len(grid) == 0:
            centroid = mesh.center
        else:
            centroid = np.mean(grid, axis=0)

        evs = equal_volume_sphere(mesh, centroid)
    
    if mesh.n_open_edges > 0:
        return -1

    pm_mesh = to_pymesh(mesh)
    pm_evs = to_pymesh(evs)

    try:
        inter = intersect(pm_mesh, pm_evs, engine)
    except:
        return -1

    return inter.volume / mesh.volume

def spin_2d(shape, grid=None, density=1):
    if grid is None:
        grid = create_grid_2d(shape, density)
    
    if isinstance(grid, list):
        grid = MultiPoint(grid).intersection(shape)

        if grid.is_empty:
            return -1

        if grid.geom_type == "Point":
            grid = MultiPoint([grid])
    
    centroid = shape.centroid
    
    return 0.5 * (shape.area / math.pi) / np.mean([math.pow(centroid.distance(p), 2) for p in grid])

def spin_3d(mesh, grid=None, density=1, check_surface=False):
    """Returns the cohesion index in 3D for a given mesh"""
    
    if grid is None:
        voxel = pv.voxelize(mesh, density=density, check_surface=check_surface)
        grid = voxel.cell_centers().points
    
    centroid = np.mean(grid, axis=0)
    
    r = math.pow(3 * mesh.volume / (4 * math.pi), 1/3)
    # TODO: Calculate the actual formula here
    return 3 / 5 * math.pow(r, 2) / np.mean([math.pow(distance(centroid, p), 2) for p in grid])

def perimeter_index(shape):
    return 2 * math.sqrt(math.pi * shape.area) / shape.length

def circumference_index_3d(mesh):
    return 4 * math.pi * math.pow(3 * mesh.volume / (4 * math.pi), 2 / 3) / mesh.area
    
def depth_2d(shape, grid=None, density=1):
    if grid is None:
        grid = create_grid_2d(shape, density)
    
    if isinstance(grid, list):
        grid = MultiPoint(grid).intersection(shape)

        if grid.is_empty:
            return -1

        if grid.geom_type == "Point":
            grid = MultiPoint([grid])
        
    return 3 * np.mean([p.distance(shape.boundary) for p in grid]) / math.sqrt(shape.area / math.pi)

def depth_3d(mesh, grid=None, density=1, check_surface=False):
    """Returns the depth index in 3D for a given mesh"""
    
    if grid is None:
        voxel = pv.voxelize(mesh, density=density, check_surface=check_surface)
        grid = voxel.cell_centers()
        
    dist = grid.compute_implicit_distance(mesh)
    
    r = math.pow(3 * mesh.volume / (4 * math.pi), 1/3)
    return 4 * np.mean(np.absolute(dist["implicit_distance"])) / r

from polylabel import polylabel

def largest_inscribed_circle(shape):
    """Returns the largest inscribed circle of a polygon in 2D"""

    centre, r = polylabel([list([list(c)[:2] for c in shape.boundary.coords])], with_distance=True)  # ([0.5, 0.5], 0.5)

    lic = Point(centre).buffer(r)
    
    return lic

def largest_inscribed_sphere(mesh, grid=None, density=1, check_surface=False):
    """Returns the largest inscribed sphere of a mesh in 3D"""
    
    if grid is None:
        voxel = pv.voxelize(mesh, density=density, check_surface=check_surface)
        grid = voxel.cell_centers()
    
    if not isinstance(grid, pv.PolyData):
        grid = pv.PolyData(grid)
        
    dist = grid.compute_implicit_distance(mesh)

    if grid.n_points == 0:
        return pv.Sphere(center=(0, 0, 0), radius=(mesh.bounds[2] - mesh.bounds[0]) / 2)
    
    # The largest inscribed circle's radius is the largest (internal) distance,
    # hence the lowest value (as internal distance is negative)
    lis_radius = np.min(dist["implicit_distance"])
    lis_center = dist.points[np.where(dist["implicit_distance"] == lis_radius)][0]
    
    return pv.Sphere(center=lis_center, radius=abs(lis_radius))

def girth_2d(shape):
    """Return the girth index in 2D for a given polygon"""
    
    lic = largest_inscribed_circle(shape)

    if lic.is_empty:
        return -1
    
    # Compute the radius as half the bounding box width
    r = (lic.bounds[2] - lic.bounds[0]) / 2
    
    return r / math.sqrt(shape.area / math.pi)

def girth_3d(mesh, grid=None, density=1, check_surface=False):
    """Return the girth index in 3D for a given mesh"""
    
    lis = largest_inscribed_sphere(mesh,
                                   grid=grid,
                                   density=density,
                                   check_surface=check_surface)
    
    r = (lis.bounds[1] - lis.bounds[0]) / 2
    r_evs = math.pow(3 * mesh.volume / (4 * math.pi), 1/3)
    
    return r / r_evs

def range_2d(shape):
    """Returns the range index in 2D for a given polygon"""
    
    from helpers.smallestenclosingcircle import make_circle

    x, y, r = make_circle([c[:2] for c in  shape.boundary.coords])
    
    return math.sqrt(shape.area / math.pi) / r

def get_bounding_ball_radius(points):
    """Returns the bounding ball for a set of points"""

    try:
        _, r2 = miniball.get_bounding_ball(points)
    except:
        return -1
    
    return r2

def range_3d(mesh):
    """Returns the range index in 3D for a given mesh"""
    
    r2 = -1

    pts = mesh.clean().points
    t = np.mean(pts, axis=0)
    pts = pts - t

    count = 0

    while r2 < 0:
        r2 = get_bounding_ball_radius(pts)
        count += 1

        if count > 10:
            return -1
    
    r_scc = math.sqrt(r2)
    
    return math.pow(3 * mesh.volume / (4 * math.pi), 1/3) / r_scc

def dispersion_2d(shape, density=0.2):
    """Returns the dispersion index in 2d for a given polygon"""
    
    c = shape.centroid
    b = shape.boundary
    
    r = math.sqrt(shape.area / math.pi)
    
    r_dev = 0
    r_ibp = 0
    for l in np.arange(0, b.length, density):
        p = b.interpolate(l)
        
        r_dev += abs(p.distance(c) - r)
        r_ibp += p.distance(c)
    
    return 1 - (r_dev / r_ibp)

def dispersion_3d(mesh, grid, density=0.5):
    """Returns the dispersion index in 3d for a given mesh"""
    
    centroid = np.mean(grid, axis=0)
    
    s_grid = create_surface_grid(mesh, density)
    
    r = equal_volume_radius(mesh.volume)
    
    r_dev = 0
    r_ibp = 0
    for p in s_grid:
        d_i = distance(centroid, p)
        r_dev += abs(d_i - r)
        r_ibp += d_i

    return 1 - (r_dev / r_ibp)

def roughness_index_2d(shape, density=0.2):
    c = shape.centroid
    b = shape.boundary

    if b.length < 1:
        return -1
        
    r_ibp = 0
    for l in np.arange(0, b.length, density):
        p = b.interpolate(l)
        
        r_ibp += p.distance(c)
    
    m_r = r_ibp / math.floor(b.length / density)
    
    return 42.62 * math.pow(m_r, 2) / (shape.area + math.pow(shape.length, 2))

def roughness_index_3d(mesh, grid, density=0.5):
    centroid = np.mean(grid, axis=0)
        
    s_grid = create_surface_grid(mesh, density)

    if len(s_grid) == 0:
        return -1
        
    r_ibp = 0
    for p in s_grid:
        d_i = distance(centroid, p)
        r_ibp += d_i
    
    m_r = r_ibp / len(s_grid)
    
    return 48.735 * math.pow(m_r, 3) / (mesh.volume + math.pow(mesh.area, 3/2))
