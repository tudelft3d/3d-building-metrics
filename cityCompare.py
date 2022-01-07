import click
import json
import numpy as np
from tqdm import tqdm
import cityjson
from helpers.mesh import difference, symmetric_difference, to_pymesh, to_pyvista, intersect
import pyvista as pv
import pandas as pd
import pymesh
from pymeshfix import MeshFix
import os

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

def repair_mesh(mesh):
    mfix = MeshFix(mesh)
    mfix.repair(verbose=False)

    fixed = mfix.mesh
    
    return fixed

def is_valid(mesh):
    return mesh.volume > 0 and mesh.n_open_edges == 0

def compare(co_id,
            obj_source,
            obj_dest,
            lod_source,
            lod_destination,
            verts_source,
            verts_dest,
            repair=False,
            export_path=None,
            plot=False,
            engine="igl"):    
    geom_source = get_geometry(obj_source, lod_source)
    geom_dest = get_geometry(obj_dest, lod_destination)

    if geom_source is None or geom_dest is None:
        raise ValueError("Geometry is missing for source or destination.")
    
    mesh_source = cityjson.to_triangulated_polydata(geom_source, verts_source)
    mesh_dest = cityjson.to_triangulated_polydata(geom_dest, verts_dest)

    if not is_valid(mesh_source) or not is_valid(mesh_dest):
        if repair and not is_valid(mesh_source):
            mesh_source = repair_mesh(mesh_source)
        
        if repair and not is_valid(mesh_dest):
            mesh_dest = repair_mesh(mesh_dest)

        if not is_valid(mesh_source) or not is_valid(mesh_dest):
            raise ValueError("The source or desintation object is not a closed volume.")

    pm_source = to_pymesh(mesh_source)
    pm_dest = to_pymesh(mesh_dest)

    try:
        inter = intersect(pm_source, pm_dest, engine)
        sym_dif = symmetric_difference(pm_source, pm_dest, engine)
        dest_minus_source = difference(pm_dest, pm_source, engine)
    except Exception as e:
        raise ValueError(f"Problem intersecting: {str(e)}")

    if not export_path is None:
        pymesh.save_mesh(export_path, dest_minus_source)
    
    if plot:
        inter_vista = to_pyvista(inter)

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
    
    return co_id, {
        "source_volume": mesh_source.volume,
        "destination_volume": mesh_dest.volume,
        "intersection_volume": inter.volume,
        "symmetric_difference_volume": sym_dif.volume,
        "destination_minus_source": dest_minus_source.volume
    }

@click.command()
@click.argument("source", type=click.File("rb"))
@click.argument("destination", type=click.File("rb"))
@click.option("--lod-source", type=str)
@click.option("--lod-destination", type=str)
@click.option("--engine", default="igl")
@click.option("--limit", type=int)
@click.option("--plot", flag_value=True)
@click.option("-f", "--filter")
@click.option("-o", "--output")
@click.option("-r", "--repair", flag_value=True)
@click.option("-e", "--export-geometry", flag_value=True)
@click.option("-j", "--jobs", default=1)
@click.option("--break_on_error", flag_value=True)
def main(source,
         destination,
         lod_source,
         lod_destination,
         engine,
         limit,
         plot,
         filter,
         output,
         repair,
         export_geometry,
         jobs,
         break_on_error):
    cm_source, verts_source = load_citymodel(source)
    cm_dest, verts_dest = load_citymodel(destination)

    i = 0

    result = {}

    output_path = "output_geom"
    if export_geometry and not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    from concurrent.futures import ProcessPoolExecutor

    num_objs = len(cm_source["CityObjects"])
    num_cores = jobs

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=num_objs if limit is None else limit) as progress:
            futures = []

            for co_id in cm_source["CityObjects"]:
                if not co_id in cm_dest["CityObjects"]:
                    print(f"WARNING: {co_id} missing from destination file.")
                    progress.total -= 1
                    continue
                
                if not filter is None and filter != co_id:
                    progress.total -= 1
                    continue

                future = pool.submit(compare,
                                     co_id,
                                     cm_source["CityObjects"][co_id],
                                     cm_dest["CityObjects"][co_id],
                                     lod_source,
                                     lod_destination,
                                     verts_source,
                                     verts_dest,
                                     repair,
                                     os.path.join(output_path, f"{co_id}.obj"),
                                     plot,
                                     engine)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

                i += 1
                if not limit is None and i >= limit:
                    break
            
            for future in futures:
                try:
                    co_id, vals = future.result()
                    if not vals is None:
                        result[co_id] = vals
                except Exception as e:
                    print(f"Problem with {co_id}: {e}")
                    if break_on_error:
                        raise e
    
    df = pd.DataFrame.from_dict(result, orient="index")

    if output is None:
        print(df)
    else:
        df.to_csv(output)

if __name__ == "__main__":
    main()
