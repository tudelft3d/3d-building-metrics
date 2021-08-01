# UM3D

Urban Morphology in 3D. Elevating urban morphology to the next level 😉

## Installation

You need to install all dependencies first:

```
pip install -r requirements.txt
```

Then take your time and install [pymesh](https://pymesh.readthedocs.io/en/latest/installation.html).

## Wat is het?

A cool script that computes a lot.

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