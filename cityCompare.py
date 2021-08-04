import click
import json
import numpy as np
from tqdm import tqdm
import cityjson
from helpers.mesh import to_pymesh, to_pyvista, intersect
import pyvista as pv

def load_citymodel(file):
    cm = json.load(file)

    if "transform" in cm:
        s = cm["transform"]["scale"]
        t = cm["transform"]["translate"]
        verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]]
    else:
        verts = cm["vertices"]

    # mesh points
    vertices = np.array(verts)

    return cm, vertices

def get_geometry(co, lod):
    """Returns the geometry of the given LoD.
    
    If lod is None then it returns the first one.
    """

    if len(co["geometry"]) == 0:
        return None

    if lod is None:
        return co["geometry"][0]

    for geom in co["geometry"]:
        if str(geom["lod"]) == str(lod):
            return geom

@click.command()
@click.argument("source", type=click.File("rb"))
@click.argument("destination", type=click.File("rb"))
@click.option("--lod_source")
@click.option("--lod_destination")
@click.option("--engine", default="igl")
@click.option("--limit", type=int)
@click.option("--plot", flag_value=True)
def main(source, destination, lod_source, lod_destination, engine, limit, plot):
    cm_source, verts_source = load_citymodel(source)
    cm_dest, verts_dest = load_citymodel(destination)

    i = 0

    for co_id in tqdm(cm_source["CityObjects"]):
        if not co_id in cm_dest["CityObjects"]:
            print(f"WARNING: {co_id} missing from destination file.")
        
        obj_source = cm_source["CityObjects"][co_id]
        obj_dest = cm_dest["CityObjects"][co_id]
        
        geom_source = get_geometry(obj_source, lod_source)
        geom_dest = get_geometry(obj_dest, lod_destination)

        if geom_source is None or geom_dest is None:
            continue
        
        mesh_source = cityjson.to_triangulated_polydata(geom_source, verts_source)
        mesh_dest = cityjson.to_triangulated_polydata(geom_dest, verts_dest)

        pm_source = to_pymesh(mesh_source)
        pm_dest = to_pymesh(mesh_dest)

        try:
            inter = intersect(pm_source, pm_dest, engine)
        except Exception as e:
            print(f"Problem intersecting {co_id}")
            raise e
            continue
        
        print(f"{co_id}: {inter.volume}")
        
        if plot:
            result = to_pyvista(inter)

            p = pv.Plotter()

            p.add_mesh(mesh_source, color="green", opacity=0.1)
            p.add_mesh(mesh_dest, color="red", opacity=0.1)
            
            p.add_mesh(mesh_source.extract_feature_edges(), color="green")
            p.add_mesh(mesh_dest.extract_feature_edges(), color="red")

            p.add_mesh(result)

            p.show()
        
        i += 1

        if not limit is None and i >= limit:
            break

if __name__ == "__main__":
    main()
