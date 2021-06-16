import numpy as np
import pyvista as pv
import scipy.spatial as ss
import sys

import json

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

print("id, actual volume, convex hull volume, area")
for obj in cm["CityObjects"]:
    building = cm["CityObjects"][obj]

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
    surf = pv.PolyData(vertices, faces)

    # Compute the convex hull volume
    points = [verts[i] for i in np.array(geom["boundaries"]).flatten()]
    ch_volume = ss.ConvexHull(points).volume

    print(f"{obj}, {building['type']}, {surf.volume}, {ch_volume}, {surf.area}")