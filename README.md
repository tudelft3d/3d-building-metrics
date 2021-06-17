# Star Wars

STAtistics for R in WAys to Research StandardCityJSON

*(you can also call it CityJSON geometric stats, but that's boring)*

## Wat is het?

A cool script that computes a lot.

## Omg, how amazing! Any issues?

Yeah:
- It works with only `MultiSurface` and `Solid` (the latter, only for the first shell)
- It only parses the first geometry
- Expects semantic surfaces

## How?

```
python -m venv venv
pip install -r requirements.txt
python volume.py [file_path] >> stats.csv
```