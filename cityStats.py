import click
import json
import numpy as np
import pyvista as pv
import scipy.spatial as ss
from pymeshfix import MeshFix
import pandas as pd

def get_area_by_surface(dataset, geom, verts):
    """Compute the area per semantic surface"""

    area = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    epointsListSemantics = {"G": [], "R": []}

    if "semantics" in geom:
        # Compute area per surface type
        sized = dataset.compute_cell_sizes()
        surface_areas = sized.cell_arrays["Area"]
        
        semantics = geom["semantics"]
        for i in range(len(surface_areas)):
            if geom["type"] == "MultiSurface":
                t = semantics["surfaces"][semantics["values"][i]]["type"]
            elif geom["type"] == "Solid":
                t = semantics["surfaces"][semantics["values"][0][i]]["type"]

            if t in area:
                area[t] = area[t] + surface_areas[i]
            else:
                area[t] = surface_areas[i]
    
    return area

def get_surface_boundaries(geom):
    """Returns the boundaries for all surfaces"""

    if geom["type"] == "MultiSurface":
        return geom["boundaries"]
    elif geom["type"] == "Solid":
        return geom["boundaries"][0]
    else:
        raise Exception("Geometry not supported")

def get_convexhull_volume(geom, verts):
    """Returns the volume of the convex hull"""

    boundaries = get_surface_boundaries(geom)

    # Compute the convex hull volume
    f = [v for ring in boundaries for v in ring[0]]
    points = [verts[i] for i in f]
    try:
        return ss.ConvexHull(points).volume
    except:
        return 0

def to_polydata(geom, vertices):
    """Returns the polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    f = [[len(r[0])] + r[0] for r in [f for f in boundaries]]
    faces = np.hstack(f) 

    return pv.PolyData(vertices, faces)

def get_feature_from_report(report, obj):
    """Return the report for the feature of the given obj"""

    return {}

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

        fixed = mfix.mesh

        ch_volume = get_convexhull_volume(geom, vertices)

        area = get_area_by_surface(dataset, geom, vertices)

        stats[obj] = [
            building["type"],
            fixed.volume,
            ch_volume,
            dataset.area,
            area["GroundSurface"],
            area["WallSurface"],
            area["RoofSurface"]
        ]

    columns = ["type", "actual volume", "convex hull volume", "area", "ground area", "wall area", "roof area"]

    df = pd.DataFrame.from_dict(stats, orient="index", columns=columns)
    df.index.name = "id"

    df.to_csv(output)

if __name__ == "__main__":
    main()

# Run val3dity

# === Per object ===

# Object type

# Volume
# Volume of convex hull
# Volume of BB

# Surface area
# Surface area by semantic surface

# Number of vertices
# Number of surfaces
# Number of vertices by type
# Number of surfaces by type

# Max height
# Min height
# Range (max-min)
# Mean, median, mode

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