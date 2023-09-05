import click
import json

@click.command()
@click.argument("input", type=click.File("rb"))
@click.option("-l", "--lod")
@click.option("-o", "--output", type=click.File("w"))
def main(input, lod, output):
    cm = json.load(input)

    if "transform" in cm:
        s = cm["transform"]["scale"]
        t = cm["transform"]["translate"]
        verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]]
    else:
        verts = cm["vertices"]

    lods = set([str(geom["lod"]) for obj in cm["CityObjects"]
                            for geom in cm["CityObjects"][obj]["geometry"]])

    if not str(lod) in lods:
        print("LoD not found in the dataset!")
        exit()

    for co_id in cm["CityObjects"]:
        co = cm["CityObjects"][co_id]

        new_geom = []

        for geom in co["geometry"]:
            if str(geom["lod"]) == str(lod):
                new_geom.append(geom)
        
        co["geometry"] = new_geom
    
    json.dump(cm, output)

if __name__ == "__main__":
    main()