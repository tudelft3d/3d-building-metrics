import shape_index
import pyvista as pv
from pytest import approx
from shapely.geometry import Point

def test_hemisphericality():
    hemisphere = pv.Sphere(radius=10).clip()
    index_value = shape_index.hemisphericality(hemisphere)

    assert index_value == approx(1.0, abs=1e-2)

def test_fractality_2d():
    circle = Point(0,0).buffer(10)
    index_value = shape_index.fractality_2d(circle)

    assert index_value == approx(0.3, abs=1e-2)

def test_fractality_3d():
    hemisphere = pv.Sphere(radius=10)
    index_value = shape_index.fractality_3d(hemisphere)

    assert index_value == approx(0.22, abs=1e-2)

def test_cubeness_3d():
    cube = pv.Box().scale(10, inplace=False)
    index_value = shape_index.cubeness(cube)

    assert index_value == approx(1.0, abs=1e-2)