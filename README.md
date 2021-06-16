# CityJSON geometric stats

A cool script that computes:
- actual volume
- convex hull volume
- total surface area

for every city object in a CityJSON file.

## Omg, how amazing! Any issues?

Yeah:
- It works with only `MultiSurface` and `Solid` (the latter, only for the first shell)
- It only parses the first geometry

## How?

```
python -m venv venv
pip install -r requirements.txt
python volume.py [file_path] >> stats.csv
```