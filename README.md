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

Visualising a specific building, which can help with troubleshooting:

```
python cityStats.py [file_path] -p -f [unique_id]
```