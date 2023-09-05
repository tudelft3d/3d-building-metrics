import json
import math
from concurrent.futures import ProcessPoolExecutor

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
import pyvista as pv
import rtree.index
import scipy.spatial as ss
from pymeshfix import MeshFix
from tqdm import tqdm

import cityjson
import geometry
import shape_index as si

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

    if "semantics" in dataset.cell_data:
        wall_idxs = [s == "WallSurface" for s in dataset.cell_data["semantics"]]
    else:
        wall_idxs = [n[2] == 0 for n in normals]

    normals = normals[wall_idxs]

    azimuth = [point_azimuth(n) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][wall_idxs]

    return get_bearings(azimuth, num_bins, surface_areas)

def get_roof_bearings(dataset, num_bins):
    """Returns the bearings of the (vertical surfaces) of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_data:
        roof_idxs = [s == "RoofSurface" for s in dataset.cell_data["semantics"]]
    else:
        roof_idxs = [n[2] > 0 for n in normals]

    normals = normals[roof_idxs]

    xz_angle = [azimuth(n[0], n[2]) for n in normals]
    yz_angle = [azimuth(n[1], n[2]) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][roof_idxs]

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

    if not "geometry" in obj or len(obj["geometry"]) == 0:
        return []

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

def tree_generator_function(cm, verts):
    for i, objid in enumerate(cm["CityObjects"]):
        obj = cm["CityObjects"][objid]

        if len(obj["geometry"]) == 0:
            continue

        xmin, xmax, ymin, ymax, zmin, zmax = cityjson.get_bbox(obj["geometry"][0], verts)
        yield (i, (xmin, ymin, zmin, xmax, ymax, zmax), objid)

def get_neighbours(cm, obj, r, verts):
    """Return the neighbours of the given building"""

    building = cm["CityObjects"][obj]

    if len(building["geometry"]) == 0:
        return []
    
    geom = building["geometry"][0]
    xmin, xmax, ymin, ymax, zmin, zmax = cityjson.get_bbox(geom, verts)
    objids = [n.object
            for n in r.intersection((xmin,
                                    ymin,
                                    zmin,
                                    xmax,
                                    ymax,
                                    zmax),
                                    objects=True)
            if n.object != obj]

    if len(objids) == 0:
        objids = [n.object for n in r.nearest((xmin, ymin, zmin, xmax, ymax, zmax), 5, objects=True) if n.object != obj]

    return [cm["CityObjects"][objid]["geometry"][0] for objid in objids]

class StatValuesBuilder:

    def __init__(self, values, indices_list) -> None:
        self.__values = values
        self.__indices_list = indices_list

    def compute_index(self, index_name):
        """Returns True if the given index is supposed to be computed"""

        return self.__indices_list is None or index_name in self.__indices_list
    
    def add_index(self, index_name, index_func):
        """Adds the given index value to the dict"""

        if self.compute_index(index_name):
            self.__values[index_name] = index_func() 
        else:
            self.__values[index_name] = "NC"

def process_building(building,
                     obj,
                     errors,
                     filter,
                     repair,
                     plot_buildings,
                     density_2d,
                     density_3d,
                     vertices,
                     neighbours=[],
                     custom_indices=None):

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
        tri_mesh = cityjson.to_triangulated_polydata(geom, vertices).clean()
    except:
        print(f"{obj} geometry parsing crashed! Omitting...")
        return obj, {"type": building["type"]}

    tri_mesh, t = geometry.move_to_origin(tri_mesh)

    if plot_buildings:
        print(f"Plotting {obj}")
        tri_mesh.plot(show_grid=True)

    # get_surface_plot(dataset, title=obj)

    bin_count, bin_edges = get_wall_bearings(mesh, 36)

    xzc, yzc, be = get_roof_bearings(mesh, 36)
    # plot_orientations(xzc, be, title=f"XZ orientation [{obj}]")
    # plot_orientations(yzc, be, title=f"YZ orientation [{obj}]")

    # total_xy = total_xy + bin_count
    # total_xz = total_xz + xzc
    # total_yz = total_yz + yzc

    if repair:
        mfix = MeshFix(tri_mesh)
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
        if len(ground_points) > 0:
            ground_z = min([v[2] for v in ground_points])
        else:
            ground_z = mesh.bounds[4]
    
    if len(ground_points) > 0:
        shape = cityjson.to_shapely(geom, vertices)
    else:
        shape = cityjson.to_shapely(geom, vertices, ground_only=False)

    obb_2d = cityjson.to_shapely(geom, vertices, ground_only=False).minimum_rotated_rectangle

    # Compute OBB with shapely
    min_z = np.min(mesh.clean().points[:, 2])
    max_z = np.max(mesh.clean().points[:, 2])
    obb = geometry.extrude(obb_2d, min_z, max_z)

    # Get the dimensions of the 2D oriented bounding box
    S, L = si.get_box_dimensions(obb_2d)

    values = {
        "type": building["type"],
        "lod": geom["lod"],
        "point_count": len(points),
        "unique_point_count": fixed.n_points,
        "surface_count": len(cityjson.get_surface_boundaries(geom)),
        "actual_volume": fixed.volume,
        "convex_hull_volume": ch_volume,
        "obb_volume": obb.volume,
        "aabb_volume": aabb_volume,
        "footprint_perimeter": shape.length,
        "obb_width": S,
        "obb_length": L,
        "surface_area": mesh.area,
        "ground_area": area["GroundSurface"],
        "wall_area": area["WallSurface"],
        "roof_area": area["RoofSurface"],
        "ground_point_count": point_count["GroundSurface"],
        "wall_point_count": point_count["WallSurface"],
        "roof_point_count": point_count["RoofSurface"],
        "ground_surface-count": surface_count["GroundSurface"],
        "wall_surface_count": surface_count["WallSurface"],
        "roof_surface_count": surface_count["RoofSurface"],
        "max_Z": height_stats["Max"],
        "min_Z": height_stats["Min"],
        "height_range": height_stats["Range"],
        "mean_Z": height_stats["Mean"],
        "median_Z": height_stats["Median"],
        "std_Z": height_stats["Std"],
        "mode_Z": height_stats["Mode"] if height_stats["ModeStatus"] == "Y" else "NA",
        "ground_Z": ground_z,
        "orientation_values": str(bin_count),
        "orientation_edges": str(bin_edges),
        "errors": str(errors),
        "valid": len(errors) == 0,
        "hole_count": tri_mesh.n_open_edges,
        "geometry": shape
    }

    if custom_indices is None or len(custom_indices) > 0:
        voxel = pv.voxelize(tri_mesh, density=density_3d, check_surface=False)
        grid = voxel.cell_centers().points

        shared_area = 0

        closest_distance = 10000

        if len(neighbours) > 0:
            # Get neighbour meshes
            n_meshes = [cityjson.to_triangulated_polydata(geom, vertices).clean()
                        for geom in neighbours]
            
            for mesh in n_meshes:
                mesh.points -= t
            
            # Compute shared walls
            walls = np.hstack([geometry.intersect_surfaces([fixed, neighbour])
                            for neighbour in n_meshes])
            
            shared_area = sum([wall["area"][0] for wall in walls])

            # Find the closest distance
            for mesh in n_meshes:
                mesh.compute_implicit_distance(fixed, inplace=True)
                            
                closest_distance = min(closest_distance, np.min(mesh["implicit_distance"]))
            
            closest_distance = max(closest_distance, 0)
        else:
            closest_distance = "NA"

        builder = StatValuesBuilder(values, custom_indices)

        builder.add_index("2d_grid_point_count", lambda: len(si.create_grid_2d(shape, density=density_2d)))
        builder.add_index("3d_grid_point_count", lambda: len(grid))

        builder.add_index("circularity_2d", lambda: si.circularity(shape))
        builder.add_index("hemisphericality_3d", lambda: si.hemisphericality(fixed))
        builder.add_index("convexity_2d", lambda: shape.area / shape.convex_hull.area)
        builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume)
        builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume)
        builder.add_index("fractality_2d", lambda: si.fractality_2d(shape))
        builder.add_index("fractality_3d", lambda: si.fractality_3d(fixed))
        builder.add_index("rectangularity_2d", lambda: shape.area / shape.minimum_rotated_rectangle.area)
        builder.add_index("rectangularity_3d", lambda: fixed.volume / obb.volume)
        builder.add_index("squareness_2d", lambda: si.squareness(shape))
        builder.add_index("cubeness_3d", lambda: si.cubeness(fixed))
        builder.add_index("horizontal_elongation", lambda: si.elongation(S, L))
        builder.add_index("min_vertical_elongation", lambda: si.elongation(L, height_stats["Max"]))
        builder.add_index("max_vertical_elongation", lambda: si.elongation(S, height_stats["Max"]))
        builder.add_index("form_factor_3D", lambda: shape.area / math.pow(fixed.volume, 2/3))
        builder.add_index("equivalent_rectangularity_index_2d", lambda: si.equivalent_rectangular_index(shape))
        builder.add_index("equivalent_prism_index_3d", lambda: si.equivalent_prism_index(fixed, obb))
        builder.add_index("proximity_index_2d_", lambda: si.proximity_2d(shape, density=density_2d))
        builder.add_index("proximity_index_3d", lambda: si.proximity_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        builder.add_index("exchange_index_2d", lambda: si.exchange_2d(shape))
        builder.add_index("exchange_index_3d", lambda: si.exchange_3d(tri_mesh, density=density_3d))
        builder.add_index("spin_index_2d", lambda: si.spin_2d(shape, density=density_2d))
        builder.add_index("spin_index_3d", lambda: si.spin_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        builder.add_index("perimeter_index_2d", lambda: si.perimeter_index(shape))
        builder.add_index("circumference_index_3d", lambda: si.circumference_index_3d(tri_mesh))
        builder.add_index("depth_index_2d", lambda: si.depth_2d(shape, density=density_2d))
        builder.add_index("depth_index_3d", lambda: si.depth_3d(tri_mesh, density=density_3d) if len(grid) > 2 else "NA")
        builder.add_index("girth_index_2d", lambda: si.girth_2d(shape))
        builder.add_index("girth_index_3d", lambda: si.girth_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        builder.add_index("dispersion_index_2d", lambda: si.dispersion_2d(shape, density=density_2d))
        builder.add_index("dispersion_index_3d", lambda: si.dispersion_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        builder.add_index("range_index_2d", lambda: si.range_2d(shape))
        builder.add_index("range_index_3d", lambda: si.range_3d(tri_mesh))
        builder.add_index("roughness_index_2d", lambda: si.roughness_index_2d(shape, density=density_2d))
        builder.add_index("roughness_index_3d", lambda: si.roughness_index_3d(tri_mesh, grid, density_2d) if len(grid) > 2 else "NA")
        builder.add_index("shared_walls_area", lambda: shared_area)
        builder.add_index("closest_distance", lambda: closest_distance)

    return obj, values

# Assume semantic surfaces
@click.command()
@click.argument("input", type=click.File("rb"))
@click.option('-o', '--output', type=click.File("wb"))
@click.option('-g', '--gpkg')
@click.option('-v', '--val3dity-report', type=click.File("rb"))
@click.option('-f', '--filter')
@click.option('-r', '--repair', flag_value=True)
@click.option('-p', '--plot-buildings', flag_value=True)
@click.option('--without-indices', flag_value=True)
@click.option('-s', '--single-threaded', flag_value=True)
@click.option('-b', '--break-on-error', flag_value=True)
@click.option('-j', '--jobs', default=1)
@click.option('--density-2d', default=1.0)
@click.option('--density-3d', default=1.0)
def main(input,
         output,
         gpkg,
         val3dity_report,
         filter,
         repair,
         plot_buildings,
         without_indices,
         single_threaded,
         break_on_error,
         jobs,
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

    # Build the index of the city model
    p = rtree.index.Property()
    p.dimension = 3
    r = rtree.index.Index(tree_generator_function(cm, vertices), properties=p)

    if single_threaded or jobs == 1:
        for obj in tqdm(cm["CityObjects"]):
            errors = get_errors_from_report(report, obj, cm)
            
            neighbours = get_neighbours(cm, obj, r, verts)

            indices_list = [] if without_indices else None
            
            try:
                obj, vals = process_building(cm["CityObjects"][obj],
                                obj,
                                errors,
                                filter,
                                repair,
                                plot_buildings,
                                density_2d,
                                density_3d,
                                vertices,
                                neighbours,
                                indices_list)
                if not vals is None:
                    stats[obj] = vals
            except Exception as e:
                print(f"Problem with {obj}")
                if break_on_error:
                    raise e

    else:
        from concurrent.futures import ProcessPoolExecutor

        num_objs = len(cm["CityObjects"])
        num_cores = jobs

        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            with tqdm(total=num_objs) as progress:
                futures = []

                for obj in cm["CityObjects"]:
                    errors = get_errors_from_report(report, obj, cm)

                    neighbours = get_neighbours(cm, obj, r, verts)

                    indices_list = [] if without_indices else None

                    future = pool.submit(process_building,
                                        cm["CityObjects"][obj],
                                        obj,
                                        errors,
                                        filter,
                                        repair,
                                        plot_buildings,
                                        density_2d,
                                        density_3d,
                                        vertices,
                                        neighbours,
                                        indices_list)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                
                results = []
                for future in futures:
                    try:
                        obj, vals = future.result()
                        if not vals is None:
                            stats[obj] = vals
                    except Exception as e:
                        print(f"Problem with {obj}")
                        if break_on_error:
                            raise e

    # orientation_plot(total_xy, bin_edges, title="Orientation plot")
    # orientation_plot(total_xz, bin_edges, title="XZ plot")
    # orientation_plot(total_yz, bin_edges, title="YZ plot")

    click.echo("Building data frame...")

    df = pd.DataFrame.from_dict(stats, orient="index")
    df.index.name = "id"

    if output is None:
        print(df)
    else:
        click.echo("Writing output...")
        df.to_csv(output)
    
    if not gpkg is None:
        gdf = geopandas.GeoDataFrame(df, geometry="geometry")
        gdf.to_file(gpkg, driver="GPKG")

if __name__ == "__main__":
    main()
