# Star Wars

STAtistics for R in WAys to Research StandardCityJSON

*(you can also call it CityJSON geometric stats, but that's boring)*

## Installation

You need to install all dependencies first:

```
pip install -r requirements.txt
```

Then take your time and install [pymesh](https://pymesh.readthedocs.io/en/latest/installation.html)

## Wat is het?

A cool script that computes a lot.

## Omg, how amazing! Any issues?

Yeah:
- It works with only `MultiSurface` and `Solid` (the latter, only for the first shell)
- It only parses the first geometry
- Expects semantic surfaces

## How?

```
python cityStats.py [file_path] -o [output.csv] [-v val3dity_report.json]
```