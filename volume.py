import numpy as np
import pyvista as pv
import scipy.spatial as ss
import sys
from pymeshfix import MeshFix

import json

if len(sys.argv) < 2:
    print("Gimme a file, or I'll skip you!")

filename = sys.argv[1]

with open(filename, 'r') as f:
    cm = json.load(f)

if "transform" in cm:
    s = cm["transform"]["scale"]
    t = cm["transform"]["translate"]
    verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
             for v in cm["vertices"]]
else:
    verts = cm["vertices"]

# mesh points
vertices = np.array(verts)

epointsListSemantics = {}

print("id, type, actual volume, convex hull volume, area, ground area, wall area, roof area")
for obj in cm["CityObjects"]:
    building = cm["CityObjects"][obj]

    if len(sys.argv) > 2 and obj != sys.argv[2]:
        continue

    # TODO: Add options for all skip conditions below

    # Skip if type is not Building or Building part
    if not building["type"] in ["Building", "BuildingPart"]:
        continue

    # Skip if no geometry
    if not "geometry" in building or len(building["geometry"]) == 0:
        continue

    geom = building["geometry"][0]

    # Skip if the geometry type is not supported
    if geom["type"] == "MultiSurface":
        boundaries = geom["boundaries"]
    elif geom["type"] == "Solid":
        boundaries = geom["boundaries"][0]
    else:
        continue

    f = [[len(r[0])] + r[0] for r in [f for f in boundaries]]
    faces = np.hstack(f) 

    # Create the pyvista object
    dataset = pv.PolyData(vertices, faces)

    mfix = MeshFix(dataset)
    # mfix.repair()

    holes = mfix.extract_holes()

    # plotter = pv.Plotter()
    # plotter.add_mesh(dataset, color=True)
    # plotter.add_mesh(holes, color='r', line_width=5)
    # plotter.enable_eye_dome_lighting() # helps depth perception
    # _ = plotter.show()

    fixed = mfix.mesh

    # Compute the convex hull volume
    f = [v for ring in boundaries for v in ring[0]]
    points = [verts[i] for i in f]
    try:
        ch_volume = ss.ConvexHull(points).volume
    except:
        ch_volume = 0

    area = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    epointsListSemantics[obj] = {"G": [], "R": []}

    if "semantics" in geom:
        # Compute area per surface type
        sized = dataset.compute_cell_sizes()
        surface_areas = sized.cell_data["Area"]
        
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

            if t == "GroundSurface":
                epointsListSemantics[obj]["G"].append([verts[v] for v in boundaries[i][0]])
            elif t == "RoofSurface":
                epointsListSemantics[obj]["R"].append([verts[v] for v in boundaries[i][0]])

    print(f"{obj}, {building['type']}, {fixed.volume}, {ch_volume}, {dataset.area}, {area['GroundSurface']}, {area['WallSurface']}, {area['RoofSurface']}")