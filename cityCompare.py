import click
import json
import numpy as np
from tqdm import tqdm
import cityjson
from helpers.mesh import symmetric_difference, to_pymesh, to_pyvista, intersect
import pyvista as pv
import pandas as pd

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
@click.option("--lod-source", type=str)
@click.option("--lod-destination", type=str)
@click.option("--engine", default="igl")
@click.option("--limit", type=int)
@click.option("--plot", flag_value=True)
@click.option("-o", "--output")
def main(source, destination, lod_source, lod_destination, engine, limit, plot, output):
    cm_source, verts_source = load_citymodel(source)
    cm_dest, verts_dest = load_citymodel(destination)

    i = 0

    result = {}

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

        if mesh_source.n_open_edges > 0 or mesh_dest.n_open_edges > 0:
            click.echo(f"{co_id}: Source or destintation object is not a closed volume...", color="yellow")
            result[co_id] = {
                "source_volume": "NA",
                "destination_volume": "NA",
                "intersection_volume": "NA",
                "symmetric_difference_volume": "NA"
            }
            continue

        pm_source = to_pymesh(mesh_source)
        pm_dest = to_pymesh(mesh_dest)

        try:
            inter = intersect(pm_source, pm_dest, engine)
            sym_dif = symmetric_difference(pm_source, pm_dest, engine)
        except Exception as e:
            print(f"Problem intersecting {co_id}: {str(e)}")
            continue
        
        result[co_id] = {
            "source_volume": mesh_source.volume,
            "destination_volume": mesh_dest.volume,
            "intersection_volume": inter.volume,
            "symmetric_difference_volume": sym_dif.volume
        }
        
        if plot:
            inter_vista = to_pyvista(inter)
            # sym_dif_vista = to_pyvista(sym_dif)

            p = pv.Plotter()

            p.background_color = "white"

            p.add_mesh(mesh_source, color="blue", opacity=0.1)
            p.add_mesh(mesh_dest, color="orange", opacity=0.1)

            p.add_mesh(mesh_source.extract_feature_edges(), color="blue", line_width=3, label=lod_source)
            p.add_mesh(mesh_dest.extract_feature_edges(), color="orange", line_width=3, label=lod_destination)

            p.add_mesh(inter_vista, color="lightgrey", label="Intersection")
            # p.add_mesh(sym_dif_vista, color="black", opacity=0.8, label="Symmetric Difference")

            p.add_legend()

            p.show()
        
        i += 1

        if not limit is None and i >= limit:
            break
    
    df = pd.DataFrame.from_dict(result, orient="index")

    if output is None:
        print(df)
    else:
        df.to_csv(output)

if __name__ == "__main__":
    main()
