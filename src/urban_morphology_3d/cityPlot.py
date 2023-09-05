import click
import json
import numpy as np
from urban_morphology_3d.cityjson import to_triangulated_polydata
import pyvista as pv
from tqdm import tqdm

@click.command()
@click.argument("input", type=click.File("rb"))
@click.option("--save", flag_value=True)
def main(input, save):
    cm = json.load(input)

    if "transform" in cm:
        s = cm["transform"]["scale"]
        t = cm["transform"]["translate"]
        verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]]
    else:
        verts = cm["vertices"]

    lods = set([geom["lod"] for obj in cm["CityObjects"]
                            for geom in cm["CityObjects"][obj]["geometry"]])

    if len(lods) > 1:
        lod = click.prompt("Select an LoD:", click.Choice(lods))
    else:
        lod = str(list(lods)[0])

    # mesh points
    vertices = np.array(verts)

    p = pv.Plotter()

    meshes = []

    for obj in tqdm(cm["CityObjects"]):
        co = cm["CityObjects"][obj]

        for geom in co["geometry"]:
            if str(geom["lod"]) == lod:
                mesh = to_triangulated_polydata(geom, vertices)
                meshes.append(mesh)

                p.add_mesh(mesh)

    p.show()

    if save:
        block = pv.MultiBlock(meshes)
        block.save("cm.vtm")

if __name__ == "__main__":
    main()