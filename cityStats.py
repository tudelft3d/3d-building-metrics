import click
import json
import numpy as np
import pyvista as pv
import scipy.spatial as ss
from pymeshfix import MeshFix
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import MultiPolygon, Polygon

def get_bearings(values, num_bins, weights):
    """Divides the values depending on the bins"""

    n = num_bins * 2

    bins = np.arange(n + 1) * 360 / n

    count, bin_edges = np.histogram(values, bins=bins, weights=weights)

    # move last bin to front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    bin_counts = count[::2] + count[1::2]

    # because we merged the bins, their edges are now only every other one
    bin_edges = bin_edges[range(0, len(bin_edges), 2)]

    return bin_counts, bin_edges

def get_wall_bearings(dataset, num_bins):
    """Returns the bearings of the azimuth angle of the normals for vertical
    surfaces of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_arrays:
        wall_idxs = [s == "WallSurface" for s in dataset.cell_arrays["semantics"]]
    else:
        wall_idxs = [n[2] == 0 for n in normals]

    normals = normals[wall_idxs]

    azimuth = [get_point_azimuth(n) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_arrays["Area"][wall_idxs]

    return get_bearings(azimuth, num_bins, surface_areas)

def get_roof_bearings(dataset, num_bins):
    """Returns the bearings of the (vertical surfaces) of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_arrays:
        roof_idxs = [s == "RoofSurface" for s in dataset.cell_arrays["semantics"]]
    else:
        roof_idxs = [n[2] > 0 for n in normals]

    normals = normals[roof_idxs]

    xz_angle = [get_azimuth(n[0], n[2]) for n in normals]
    yz_angle = [get_azimuth(n[1], n[2]) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_arrays["Area"][roof_idxs]

    xz_counts, bin_edges = get_bearings(xz_angle, num_bins, surface_areas)
    yz_counts, bin_edges = get_bearings(yz_angle, num_bins, surface_areas)

    return xz_counts, yz_counts, bin_edges

def plot_orientations(
    bin_counts,
    bin_edges,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None
):
    if title_font is None:
        title_font = {"family": "DejaVu Sans", "size": 12, "weight": "bold"}

    width = 2 * np.pi / num_bins

    positions = np.radians(bin_edges[:-1])

    radius = bin_counts / bin_counts.sum()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=radius.max())

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, radius.max(), 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=xticklabels)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.bar(
        positions,
        height=radius,
        width=width,
        align="center",
        bottom=0,
        zorder=2
    )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)

    plt.show()

def get_surface_plot(
    dataset,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None
):
    """Prints a plot for the surface normals of a polyData"""
    
    bin_counts, bin_edges = get_wall_bearings(dataset, num_bins)

    plot_orientations(bin_counts, bin_edges)
    

def get_azimuth(dx, dy):
    """Returns the azimuth angle for the given coordinates"""
    
    return (math.atan2(dx, dy) * 180 / np.pi) % 360

def get_point_azimuth(p):
    """Returns the azimuth angle of the given point"""

    return get_azimuth(p[0], p[1])

def get_point_zenith(p):
    """Return the zenith angle of the given 3d point"""

    z = [0.0, 0.0, 1.0]

    cosine_angle = np.dot(p, z) / (np.linalg.norm(p) * np.linalg.norm(z))
    angle = np.arccos(cosine_angle)

    return (angle * 180 / np.pi) % 360

def get_stats(values, percentile = 90, percentage = 75):
    """
    Returns the stats (mean, median, max, min, range etc.) for a set of values.
    
    Author: Anna Labetski
    """
    hDic = {'Mean': np.mean(values), 'Median': np.median(values),
    'Max': max(values), 'Min': min(values), 'Range': (max(values) - min(values)),
    'Std': np.std(values)}
    m = max([values.count(a) for a in values])
    if percentile:
        hDic['Percentile'] = np.percentile(values, percentile)
    if percentage:
        hDic['Percentage'] = (percentage/100.0) * hDic['Range'] + hDic['Min']
    if m>1:
        hDic['ModeStatus'] = 'Y'
        modeCount = [x for x in values if values.count(x) == m][0]
        hDic['Mode'] = modeCount
    else:
        hDic['ModeStatus'] = 'N'
        hDic['Mode'] = np.mean(values)
    return hDic

def add_value(dict, key, value):
    """Does dict[key] = dict[key] + value"""

    if key in dict:
        dict[key] = dict[key] + value
    else:
        area[key] = value

def get_area_by_surface(mesh, tri_mesh=None):
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

def get_points_of_type(mesh, surface_type):
    """Returns the points that belong to the given surface type"""

    if not "semantics":
        return 0
    
    idxs = [s == surface_type for s in mesh.cell_arrays["semantics"]]

    points = np.array([mesh.cell_points(i) for i in range(mesh.number_of_cells)])

    return np.vstack(points[idxs])

def get_surface_boundaries(geom):
    """Returns the boundaries for all surfaces"""

    if geom["type"] == "MultiSurface":
        return geom["boundaries"]
    elif geom["type"] == "Solid":
        return geom["boundaries"][0]
    else:
        raise Exception("Geometry not supported")

def get_points(geom, verts):
    """Return the points of the geometry"""

    boundaries = get_surface_boundaries(geom)

    # Compute the convex hull volume
    f = [v for ring in boundaries for v in ring[0]]
    points = [verts[i] for i in f]

    return points

def get_convexhull_volume(points):
    """Returns the volume of the convex hull"""

    try:
        return ss.ConvexHull(points).volume
    except:
        return 0

def get_boundingbox_volume(points):
    """Returns the volume of the bounding box"""
    
    minx = min(p[0] for p in points)
    maxx = max(p[0] for p in points)
    miny = min(p[1] for p in points)
    maxy = max(p[1] for p in points)
    minz = min(p[2] for p in points)
    maxz = max(p[2] for p in points)

    return (maxx - minx) * (maxy - miny) * (maxz - minz)

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

def to_shapely(geom, vertices):
    """Returns a shapely geometry of the footprint from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    if "semantics" in geom:
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
        

def get_errors_from_report(report, objid, cm):
    """Return the report for the feature of the given obj"""

    if not "features" in report:
        return []
    
    fid = objid

    obj = cm["CityObjects"][objid]
    primidx = 0

    if "parents" in obj:
        parid = obj["parents"][0]

        primidx = cm["CityObjects"][parid]["children"].index(objid)
        fid = parid

    for f in report["features"]:
        if f["id"] == fid:
            if "errors" in f["primitives"][primidx]:
                return list(map(lambda e: e["code"], f["primitives"][primidx]["errors"]))
            else:
                return []

    return []

def validate_report(report, cm):
    """Returns true if the report is actually for this file"""

    return True

# Assume semantic surfaces
@click.command()
@click.argument("input", type=click.File("rb"))
@click.option('-o', '--output', type=click.File("wb"))
@click.option('-v', '--val3dity-report', type=click.File("rb"))
@click.option('-f', '--filter')
@click.option('-r', '--repair', flag_value=True)
@click.option('-p', '--plot-buildings', flag_value=True)
def main(input, output, val3dity_report, filter, repair, plot_buildings):
    cm = json.load(input)

    if "transform" in cm:
        s = cm["transform"]["scale"]
        t = cm["transform"]["translate"]
        verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]]
    else:
        verts = cm["vertices"]
    
    if val3dity_report is None:
        report = {}
    else:
        report = json.load(val3dity_report)

        if not validate_report(report, cm):
            print("This doesn't seem like the right report for this file.")
            return

    # mesh points
    vertices = np.array(verts)

    epointsListSemantics = {}

    stats = {}

    total_xy = np.zeros(36)
    total_xz = np.zeros(36)
    total_yz = np.zeros(36)

    for obj in tqdm(cm["CityObjects"]):
        building = cm["CityObjects"][obj]

        if not filter is None and filter != obj:
            continue

        # TODO: Add options for all skip conditions below

        # Skip if type is not Building or Building part
        if not building["type"] in ["Building", "BuildingPart"]:
            continue

        # Skip if no geometry
        if not "geometry" in building or len(building["geometry"]) == 0:
            continue

        geom = building["geometry"][0]
        
        mesh = to_polydata(geom, vertices)

        tri_mesh = mesh.triangulate()

        if plot_buildings:
            print(f"Plotting {obj}")
            tri_mesh.plot()

        # get_surface_plot(dataset, title=obj)

        bin_count, bin_edges = get_wall_bearings(mesh, 36)

        xzc, yzc, be = get_roof_bearings(mesh, 36)
        # plot_orientations(xzc, be, title=f"XZ orientation [{obj}]")
        # plot_orientations(yzc, be, title=f"YZ orientation [{obj}]")

        total_xy = total_xy + bin_count
        total_xz = total_xz + xzc
        total_yz = total_yz + yzc

        if repair:
            mfix = MeshFix(mesh)
            mfix.repair()

            fixed = mfix.mesh
        else:
            fixed = tri_mesh

        # holes = mfix.extract_holes()

        # plotter = pv.Plotter()
        # plotter.add_mesh(dataset, color=True)
        # plotter.add_mesh(holes, color='r', line_width=5)
        # plotter.enable_eye_dome_lighting() # helps depth perception
        # _ = plotter.show()

        points = get_points(geom, vertices)

        bb_volume = get_boundingbox_volume(points)

        ch_volume = get_convexhull_volume(points)

        area, point_count, surface_count = get_area_by_surface(mesh)

        roof_points = get_points_of_type(mesh, "RoofSurface")
        ground_points = get_points_of_type(mesh, "GroundSurface")

        if len(roof_points) == 0:
            height_stats = get_stats([0])
            ground_z = 0
        else:
            height_stats = get_stats([v[2] for v in roof_points])
            ground_z = min([v[2] for v in ground_points])
        
        shape = to_shapely(geom, vertices)

        errors = get_errors_from_report(report, obj, cm)

        stats[obj] = [
            building["type"],
            len(points),
            len(get_surface_boundaries(geom)),
            fixed.volume,
            ch_volume,
            bb_volume,
            shape.length,
            mesh.area,
            area["GroundSurface"],
            area["WallSurface"],
            area["RoofSurface"],
            point_count["GroundSurface"],
            point_count["WallSurface"],
            point_count["RoofSurface"],
            surface_count["GroundSurface"],
            surface_count["WallSurface"],
            surface_count["RoofSurface"],
            height_stats["Max"],
            height_stats["Min"],
            height_stats["Range"],
            height_stats["Mean"],
            height_stats["Median"],
            height_stats["Std"],
            height_stats["Mode"] if height_stats["ModeStatus"] == "Y" else "NA",
            ground_z,
            bin_count,
            bin_edges,
            errors,
            len(errors) == 0
        ]
    
    plot_orientations(total_xy, bin_edges, title="Orientation plot")
    plot_orientations(total_xz, bin_edges, title="XZ plot")
    plot_orientations(total_yz, bin_edges, title="YZ plot")


    columns = [
        "type", # type of the city object
        "point count", # total number of points in the city object
        "surface count", # total number of surfaces in the city object
        "actual volume", # volume of the geometry of city object
        "convex hull volume", # volume of the convex hull of the city object
        "bounding box volume", # volume of the axis-aligned bounding box of the city object
        "footprint perimeter", # perimeter of the footpring of the city object
        "surface area", # total area of all surfaces of the city object
        "ground area", # area of all ground surfaces of the city object
        "wall area", # area of all wall surfaces of the city object
        "roof area", # area of all roof surfaces of the city object
        "ground point count", # number of points in ground surfaces
        "wall point count", # number of point in wall surfaces
        "roof point count", # number of point in roof surfaces
        "ground surface count", # number of ground surfaces
        "wall surface count", # number of wall surfaces
        "roof surface count", # number of roof surfaces
        "max Z", # maximum Z of roof points
        "min Z", # minimum Z of roof points
        "height range", # height range of roof points (ie, max - min)
        "mean Z", # mean Z of roof points
        "median Z", # median Z of roof points
        "std Z", # standard deviation of Z of roof points
        "mode Z", # mode of Z of roof points
        "ground Z", # Z value of the ground points
        "orientation_values", # values of orientation plot of wall surfaces normals
        "orientaiton_edges", # edges of orientation plot of wall surfaces normals
        "errors", # error codes from val3dity for the city object
        "valid" # validity of the city object
    ]

    df = pd.DataFrame.from_dict(stats, orient="index", columns=columns)
    df.index.name = "id"

    if output is None:
        print(df)
    else:
        df.to_csv(output)

if __name__ == "__main__":
    main()

# Run val3dity [X]

# === Per object ===

# Object type [X]

# Volume [X]
# Volume of convex hull [X]
# Volume of BB [X]

# Surface area [X]
# Surface area by semantic surface [X]

# Number of vertices [X]
# Number of surfaces [X]
# Number of vertices by type [X]
# Number of surfaces by type [X]

# Max height [X]
# Min height [X]
# Range (max-min) [X]
# Mean, median, std, mode [X]

# Topology
# Unique vertices
# Unique surfaces
# Connectivity of buildings
# Height difference between adjacent buildings

# Directionality of footprint
# Directionality (?) of surfaces (normals) [X]
# Perimeter of roofprint
# Perimeter of footprint

# Differences and perectages between volumes

# Number of points of footprint
# Shape complexity of footprint
# Shape complexity in 3D
# Spread points and compute the distance to the centroid

# Values for directionality graph of wall surfaces
# Values for zenith angles of roof surfaces

# Shape metrics (for footprint or 3D):
# Spin index (Σd²/n) (normalise by EAC)
# Dispersion (Σd/n) (normalise by EAC)
# Maximum spanning circle (aka girth index) (normalise by radius of EAC)
# Elongation
# Linearity

# === Building level (hierarchy) analysis ===

# Aggregate stats for buildings

# === City level analysis ===

# Density

# Clusters of heights
# Clusters of footprint areas
# Clusters of height range

# Connected components

# Directionality graph for orienation (wall surfaces)
# Zenith graph for roof surfaces (could describe the type of roof)

# Graph of roof slope with respect to the closest road centreline