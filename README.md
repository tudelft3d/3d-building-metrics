# 3DBM

3D Building Metrics. Elevating geometric analysis for urban morphology, solar potential, CFD etc to the next level 😉

## Installation

You need to install all dependencies first:

```
pip install -r requirements.txt
```

Then take your time and install [pymesh](https://pymesh.readthedocs.io/en/latest/installation.html).

## Wat is het?

A cool script that computes a lot metrics from 3D geometries (mostly intended for buildings).

## Omg, how amazing! Any issues?

Yeah:
- It works with only `MultiSurface` and `Solid` (the latter, only for the first shell)
- It only parses the first geometry
- Expects semantic surfaces

## How?

Running it, saving it, and including a [val3dity](https://github.com/tudelft3d/val3dity) report:

```
python cityStats.py [file_path] -o [output.csv] [-v val3dity_report.json]
```

Default is single-threaded, define the number of threads with:

```
python cityStats.py [file_path] -j [number]
```

Visualising a specific building, which can help with troubleshooting:

```
python cityStats.py [file_path] -p -f [unique_id]
```

Running multiple files in a folder and checking with [val3dity](https://github.com/tudelft3d/val3dity) (make sure you have val3dity installed):

```
for i in *.json; do val3dity $i --report "${i%.json}_v3.json"; python cityStats.py $i -o "${i%.json}.csv" -v "${i%.json}_v3.json"; done
```

## Can I visualise a model?

Tuurlijk! Just:

```
python cityPlot.py [file_path]
```

## Tutorial please!

1) Download or `git clone` this repository.

2) Install all dependencies: `pip install -r requirements.txt`.

3) Download a tile from 3D BAG: `wget --header='Accept-Encoding: gzip' https://data.3dbag.nl/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_5910.json`

4) Run the stats on the data: `python cityStats.py 3dbag_v210908_fd2cee53_5910.json -o 5910.csv`

5) The resutling file `5910.csv` contains all metrics computed for this tile.

You may also run this with a [val3dity](http://geovalidation.bk.tudelft.nl/val3dity/) report. You may download the val3dity report as a json file from the aforementioned website. Assuming the report's filename is `report.json` you can run:

```
python cityStats.py 3dbag_v210908_fd2cee53_5910.json -v report.json -o 5910.csv
```

Then the result will contain more info related to the validation of geometries.
