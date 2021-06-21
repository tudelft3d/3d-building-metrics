import click
import json
import numpy as np
import pyvista as pv
import scipy.spatial as ss
from pymeshfix import MeshFix
import pandas as pd

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

def get_area_by_surface(dataset, geom, verts):
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

    semantic_points = {"G": [], "R": []}

    if "semantics" in geom:
        # Compute area per surface type
        sized = dataset.compute_cell_sizes()
        surface_areas = sized.cell_arrays["Area"]

        boundaries = get_surface_boundaries(geom)
        
        semantics = geom["semantics"]
        for i in range(len(surface_areas)):
            if geom["type"] == "MultiSurface":
                t = semantics["surfaces"][semantics["values"][i]]["type"]
            elif geom["type"] == "Solid":
                t = semantics["surfaces"][semantics["values"][0][i]]["type"]

            add_value(area, t, surface_areas[i])
            add_value(point_count, t, sized.cell_n_points(i))
            add_value(surface_count, t, 1)

            if t in ["GroundSurface", "RoofSurface"]:
                semantic_points["G" if t == "GroundSurface" else "R"].extend([verts[v] for v in boundaries[i][0]])
    
    return area, point_count, surface_count, semantic_points

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

    return pv.PolyData(vertices, faces)

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
def main(input, output, val3dity_report, filter):
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

    for obj in cm["CityObjects"]:
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
        
        dataset = to_polydata(geom, vertices)

        mfix = MeshFix(dataset)
        # mfix.repair()

        holes = mfix.extract_holes()

        # plotter = pv.Plotter()
        # plotter.add_mesh(dataset, color=True)
        # plotter.add_mesh(holes, color='r', line_width=5)
        # plotter.enable_eye_dome_lighting() # helps depth perception
        # _ = plotter.show()

        points = get_points(geom, vertices)

        bb_volume = get_boundingbox_volume(points)

        fixed = mfix.mesh

        ch_volume = get_convexhull_volume(points)

        area, point_count, surface_count, semantic_points = get_area_by_surface(dataset, geom, vertices)

        roof_points = semantic_points["R"]

        height_stats = get_stats([v[2] for v in roof_points])
        ground_z = min([v[2] for v in semantic_points["G"]])

        errors = get_errors_from_report(report, obj, cm)

        stats[obj] = [
            building["type"],
            len(points),
            len(get_surface_boundaries(geom)),
            fixed.volume,
            ch_volume,
            bb_volume,
            dataset.area,
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
            errors,
            len(errors) == 0
        ]

    columns = [
        "type",
        "point count",
        "surface count",
        "actual volume",
        "convex hull volume",
        "bounding box volume",
        "area",
        "ground area",
        "wall area",
        "roof area",
        "ground point count",
        "wall point count",
        "roof point count",
        "ground surface count",
        "wall surface count",
        "roof surface count",
        "max Z",
        "min Z",
        "height range",
        "mean Z",
        "median Z",
        "std Z",
        "mode Z",
        "ground Z",
        "errors",
        "valid"
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
# Directionality (?) of surfaces (normals)
# Perimeter of footprint

# Differences and perectages between volumes

# Number of points of footprint
# Shape complexity of footprint
# Shape complexity in 3D
# Spread points and compute the distance to the centroid

# Shape metrics (for footprint or 3D):
# Spin index (Σd²/n) (normalise by EAC)
# Dispersion (Σd/n) (normalise by EAC)
# Maximum spanning circle (aka girth index) (normalise by radius of EAC)
# Elongation
# Linearity

# === City level analysis ===

# Density

# Clusters of heights
# Clusters of footprint areas
# Clusters of height range

# Connected components