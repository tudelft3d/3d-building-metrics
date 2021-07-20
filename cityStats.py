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
from helpers.minimumBoundingBox import MinimumBoundingBox
import shape_index as si
import cityjson
import geometry
from concurrent.futures import ProcessPoolExecutor

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

    azimuth = [point_azimuth(n) for n in normals]

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

    xz_angle = [azimuth(n[0], n[2]) for n in normals]
    yz_angle = [azimuth(n[1], n[2]) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_arrays["Area"][roof_idxs]

    xz_counts, bin_edges = get_bearings(xz_angle, num_bins, surface_areas)
    yz_counts, bin_edges = get_bearings(yz_angle, num_bins, surface_areas)

    return xz_counts, yz_counts, bin_edges

def orientation_plot(
    bin_counts,
    bin_edges,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None,
    show=False
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

    if show:
        plt.show()
    
    return plt

def get_surface_plot(
    dataset,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None
):
    """Returns a plot for the surface normals of a polyData"""
    
    bin_counts, bin_edges = get_wall_bearings(dataset, num_bins)

    return orientation_plot(bin_counts, bin_edges)
    

def azimuth(dx, dy):
    """Returns the azimuth angle for the given coordinates"""
    
    return (math.atan2(dx, dy) * 180 / np.pi) % 360

def point_azimuth(p):
    """Returns the azimuth angle of the given point"""

    return azimuth(p[0], p[1])

def point_zenith(p):
    """Return the zenith angle of the given 3d point"""

    z = [0.0, 0.0, 1.0]

    cosine_angle = np.dot(p, z) / (np.linalg.norm(p) * np.linalg.norm(z))
    angle = np.arccos(cosine_angle)

    return (angle * 180 / np.pi) % 360

def compute_stats(values, percentile = 90, percentage = 75):
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

def convexhull_volume(points):
    """Returns the volume of the convex hull"""

    try:
        return ss.ConvexHull(points).volume
    except:
        return 0

def boundingbox_volume(points):
    """Returns the volume of the bounding box"""
    
    minx = min(p[0] for p in points)
    maxx = max(p[0] for p in points)
    miny = min(p[1] for p in points)
    maxy = max(p[1] for p in points)
    minz = min(p[2] for p in points)
    maxz = max(p[2] for p in points)

    return (maxx - minx) * (maxy - miny) * (maxz - minz)

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

    # TODO: Actually validate the report and that it corresponds to this cm
    return True

def process_building(cm,
                     obj,
                     report,
                     val3dity_report,
                     filter,
                     repair,
                     plot_buildings,
                     with_cohesion,
                     density_2d,
                     density_3d,
                     vertices):
    building = cm["CityObjects"][obj]

    if not filter is None and filter != obj:
        return obj, None

    # TODO: Add options for all skip conditions below

    # Skip if type is not Building or Building part
    if not building["type"] in ["Building", "BuildingPart"]:
        return obj, None

    # Skip if no geometry
    if not "geometry" in building or len(building["geometry"]) == 0:
        return obj, None

    geom = building["geometry"][0]
    
    mesh = cityjson.to_polydata(geom, vertices).clean()

    try:
        tri_mesh = cityjson.to_triangulated_polydata(geom, vertices)
    except:
        click.warning(f"{obj} geometry parsing crashed! Omitting...")
        return obj, [building["type"]] + ["NA" for r in range(len(columns) - 1)]

    if plot_buildings:
        print(f"Plotting {obj}")
        tri_mesh.plot()

    # get_surface_plot(dataset, title=obj)

    bin_count, bin_edges = get_wall_bearings(mesh, 36)

    xzc, yzc, be = get_roof_bearings(mesh, 36)
    # plot_orientations(xzc, be, title=f"XZ orientation [{obj}]")
    # plot_orientations(yzc, be, title=f"YZ orientation [{obj}]")

    # total_xy = total_xy + bin_count
    # total_xz = total_xz + xzc
    # total_yz = total_yz + yzc

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

    points = cityjson.get_points(geom, vertices)

    aabb_volume = boundingbox_volume(points)

    ch_volume = convexhull_volume(points)

    area, point_count, surface_count = geometry.area_by_surface(mesh)

    if "semantics" in geom:
        roof_points = geometry.get_points_of_type(mesh, "RoofSurface")
        ground_points = geometry.get_points_of_type(mesh, "GroundSurface")
    else:
        roof_points = []
        ground_points = []

    if len(roof_points) == 0:
        height_stats = compute_stats([0])
        ground_z = 0
    else:
        height_stats = compute_stats([v[2] for v in roof_points])
        ground_z = min([v[2] for v in ground_points])
    
    shape = cityjson.to_shapely(geom, vertices)

    obb_2d = cityjson.to_shapely(geom, vertices, ground_only=False).minimum_rotated_rectangle

    # Compute OBB with shapely
    min_z = np.min(mesh.clean().points[:, 2])
    max_z = np.max(mesh.clean().points[:, 2])
    obb = geometry.extrude(obb_2d, min_z, max_z)

    errors = get_errors_from_report(report, obj, cm)

    # Get the dimensions of the 2D oriented bounding box
    S, L = si.get_box_dimensions(obb_2d)

    voxel = pv.voxelize(mesh, density=density_3d, check_surface=False)
    grid = voxel.cell_centers().points

    return obj, [
        building["type"],
        len(points),
        len(cityjson.get_surface_boundaries(geom)),
        fixed.volume,
        ch_volume,
        obb.volume,
        aabb_volume,
        shape.length,
        S,
        L,
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
        len(errors) == 0,
        si.circularity(shape),
        si.hemisphericality(fixed),
        shape.area / shape.convex_hull.area,
        fixed.volume / ch_volume,
        si.fractality_2d(shape),
        si.fractality_3d(fixed),
        shape.area / shape.minimum_rotated_rectangle.area,
        fixed.volume / obb.volume,
        si.squareness(shape),
        si.cubeness(fixed),
        si.elongation(S, L),
        si.elongation(L, height_stats["Max"]),
        si.elongation(S, height_stats["Max"]),
        shape.area / math.pow(fixed.volume, 2/3),
        si.equivalent_rectangular_index(shape),
        si.equivalent_prism_index(fixed, obb),
        si.proximity_2d(shape, density=density_2d),
        si.proximity_3d(mesh, grid, density=density_3d),
        si.exchange_2d(shape),
        si.exchange_3d(mesh, density=density_3d),
        si.spin_2d(shape, density=density_2d),
        si.spin_3d(mesh, grid, density=density_3d),
        si.perimeter_index(shape),
        si.circumference_index_3d(mesh),
        si.depth_2d(shape, density=density_2d),
        si.depth_3d(mesh, density=density_3d),
        si.girth_2d(shape),
        si.girth_3d(mesh, grid, density=density_3d),
        si.dispersion_2d(shape, density=density_2d),
        si.dispersion_3d(mesh, grid, density=density_3d),
        si.range_2d(shape),
        si.range_3d(mesh),
        si.roughness_index_2d(shape, density=density_2d),
        si.roughness_index_3d(mesh, grid, density_2d)
    ]

# Assume semantic surfaces
@click.command()
@click.argument("input", type=click.File("rb"))
@click.option('-o', '--output', type=click.File("wb"))
@click.option('-v', '--val3dity-report', type=click.File("rb"))
@click.option('-f', '--filter')
@click.option('-r', '--repair', flag_value=True)
@click.option('-p', '--plot-buildings', flag_value=True)
@click.option('-c', '--with-cohesion', flag_value=True)
@click.option('--density-2d', default=1)
@click.option('--density-3d', default=1)
def main(input,
         output,
         val3dity_report,
         filter,
         repair,
         plot_buildings,
         with_cohesion,
         density_2d,
         density_3d):
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

    from concurrent.futures import ProcessPoolExecutor

    num_objs = len(cm["CityObjects"])
    num_cores = 4

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=len(cm["CityObjects"])) as progress:
            futures = []

            for obj in cm["CityObjects"]:
                future = pool.submit(process_building,
                                    cm,
                                    obj,
                                    report,
                                    val3dity_report,
                                    filter,
                                    repair,
                                    plot_buildings,
                                    with_cohesion,
                                    density_2d,
                                    density_3d,
                                    vertices)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            
            results = []
            for future in futures:
                obj, vals = future.result()
                stats[obj] = vals

    # orientation_plot(total_xy, bin_edges, title="Orientation plot")
    # orientation_plot(total_xz, bin_edges, title="XZ plot")
    # orientation_plot(total_yz, bin_edges, title="YZ plot")

    columns = [
        "type", # type of the city object
        "point count", # total number of points in the city object
        "surface count", # total number of surfaces in the city object
        "actual volume", # volume of the geometry of city object
        "convex hull volume", # volume of the convex hull of the city object
        "obb volume", # volume of the oriented bounding box of the city object
        "aabb volume", # volume of the axis-aligned bounding box of the city object
        "footprint perimeter", # perimeter of the footpring of the city object
        "obb width",
        "obb length",
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
        "valid", # validity of the city object
        "circularity (2d)",
        "hemisphericality (3d)",
        "convexity (2d)",
        "convexity (3d)",
        "fractality (2d)",
        "fractality (3d)",
        "rectangularity (2d)",
        "rectangularity (3d)",
        "squareness (2d)",
        "cubeness (3d)",
        "horizontal elongation",
        "min vertical elongation",
        "max vertical elongation",
        "form factor (3D)",
        "equivalent rectangularity index (2d)",
        "equivalent prism index (3d)",
        "proximity index (2d)",
        "proximity index (3d)",
        "exchange index (2d)",
        "exchange index (3d)",
        "spin index (2d)",
        "spin index (3d)",
        "perimeter index (2d)",
        "circumference index (3d)",
        "depth index (2d)",
        "depth index (3d)",
        "girth index (2d)",
        "girth index (3d)",
        "dispersion index (2d)",
        "dispersion index (3d)",
        "range index (2d)",
        "range index (3d)",
        "roughness index (2d)",
        "roughness index (3d)",
    ]

    # print(f"Cohesion | {cohesion_2d(shape):.5f} | {cohesion_3d(mesh, grid):.5f}")
    # print(f"Proximity | {proximity_2d(shape, density=density_2d):.5f} | {proximity_3d(mesh, grid):.5f}")
    # print(f"Exchange | {exchange_2d(shape):.5f} | {exchange_3d(mesh, density=density_3d):.5f}")
    # print(f"Spin | {spin_2d(shape, density=density_2d):.5f} | {spin_3d(mesh, grid):.5f}")
    # print(f"Perimeter/Circumference | {perimeter_index(shape):.5f} | {circumference_index_3d(mesh):.5f} ")
    # print(f"Depth | {depth_2d(shape, density=density_2d):.5f} | {depth_3d(mesh, density=density_3d):.5f} ")
    # print(f"Girth | {girth_2d(shape):.5f} | {girth_3d(mesh, grid):.5f}")
    # print(f"Dispersion | {dispersion_2d(shape, density=density_2d):.5f} | {dispersion_3d(mesh, grid, density=density_3d):0.5f}")
    # print(f"Range | {range_2d(shape):.5f} | {range_3d(mesh):.5f}")
    # print(f"Roughness index | {roughness_index_2d(shape, density_2d)} | {roughness_index_3d(mesh, grid, density_2d)}")

    df = pd.DataFrame.from_dict(stats, orient="index", columns=columns)
    df.index.name = "id"

    if output is None:
        print(df)
    else:
        df.to_csv(output)

if __name__ == "__main__":
    main()
