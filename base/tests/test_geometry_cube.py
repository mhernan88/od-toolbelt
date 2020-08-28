import pytest
import numpy as np
from nptyping import NDArray

from geometry import cube, point


def default_arrays() -> NDArray[(2, 2, 2), np.float64]:
    box1 = np.array(((0.48, 0.82), (0.82, 0.48))).astype(np.float64)
    box2 = np.array(((0.14, 0.55), (0.55, 0.14))).astype(np.float64)
    arr = np.zeros((2, 2, 2)).astype(np.float64)
    arr[0, :, :] = box1
    arr[1, :, :] = box2
    return arr


def test_assert_cube1():
    arr = default_arrays()
    cube.assert_cube(arr, True)


def test_assert_cube2():
    box1 = np.array(((0.48, 0.82), (0.82, 0.48), (0.31, 0.42))).astype(np.float64)
    box2 = np.array(((0.14, 0.55), (0.55, 0.14), (0.42, 0.31))).astype(np.float64)
    arr = np.zeros((2, 3, 2)).astype(np.float64)
    arr[0, :, :] = box1
    arr[1, :, :] = box2
    with pytest.raises(cube.AssertCubeError):
        cube.assert_cube(arr, True)


def test_assert_cube3():
    box1 = np.array(((0.48, 0.82, 0.51), (0.82, 0.48, 0.08))).astype(np.float64)
    box2 = np.array(((0.14, 0.55, 0.77), (0.55, 0.14, 0.10))).astype(np.float64)
    arr = np.zeros((2, 2, 3)).astype(np.float64)
    arr[0, :, :] = box1
    arr[1, :, :] = box2
    with pytest.raises(cube.AssertCubeError):
        cube.assert_cube(arr, True)


def test_assert_cube4():
    box1 = np.array(((0.48, 0.82), (0.82, 0.48))).astype(np.float32)
    box2 = np.array(((0.14, 0.55), (0.55, 0.14))).astype(np.float32)
    arr = np.zeros((2, 2, 2)).astype(np.float32)
    arr[0, :, :] = box1
    arr[1, :, :] = box2
    with pytest.raises(cube.AssertCubeError):
        cube.assert_cube(arr, True)


def test_get_one_point1():
    arr = default_arrays()
    pt = cube.get_one_point1(arr, 0)
    point.assert_point(pt, True)
    assert pt[0] == 0.48
    assert pt[1] == 0.82


def test_get_point1():
    arr = default_arrays()
    pts = cube.get_point1(arr)
    for i in np.arange(0, pts.shape[0]):
        point.assert_point(pts[i], True)
    assert pts[0, 0] == 0.48
    assert pts[0, 1] == 0.82


def test_get_point2():
    arr = default_arrays()
    pts = cube.get_point2(arr)
    for i in np.arange(0, pts.shape[0]):
        point.assert_point(pts[i], True)
    assert pts[0, 0] == 0.82
    assert pts[0, 1] == 0.48


def test_get_one_area():
    box1 = np.array(((0.4, 0.4), (0.8, 0.8))).astype(np.float64)
    box2 = np.array(((0.1, 0.1), (0.2, 0.2))).astype(np.float64)
    arr = np.zeros((2, 2, 2)).astype(np.float64)
    arr[0, :, :] = box1
    arr[1, :, :] = box2
    area = cube.get_one_area(arr, 0)
    assert np.round(area, 2) == 0.16


def test_get_area():
    box1 = np.array(((0.4, 0.4), (0.8, 0.8))).astype(np.float64)
    box2 = np.array(((0.1, 0.1), (0.2, 0.2))).astype(np.float64)
    arr = np.zeros((2, 2, 2)).astype(np.float64)
    arr[0, :, :] = box1
    arr[1, :, :] = box2
    areas = cube.get_area(arr)
    assert np.round(areas[0], 2) == 0.16
    assert np.round(areas[1], 2) == 0.01


def test_one_boxes_overlap():
    box1_1 = np.array(((0.4, 0.4), (0.8, 0.8))).astype(np.float64)
    box1_2 = np.array(((0.1, 0.1), (0.2, 0.2))).astype(np.float64)
    arr1 = np.zeros((2, 2, 2)).astype(np.float64)
    arr1[0, :, :] = box1_1
    arr1[1, :, :] = box1_2

    box2_1 = np.array(((0.5, 0.5), (0.9, 0.9))).astype(np.float64)
    box2_2 = np.array(((0.3, 0.3), (0.4, 0.4))).astype(np.float64)
    arr2 = np.zeros((2, 2, 2)).astype(np.float64)
    arr2[0, :, :] = box2_1
    arr2[1, :, :] = box2_2

    assert cube.one_intersection_ok(arr1, arr2, 0)


def test_boxes_overlap():
    box1_1 = np.array(((0.4, 0.4), (0.8, 0.8))).astype(np.float64)
    box1_2 = np.array(((0.1, 0.1), (0.2, 0.2))).astype(np.float64)
    arr1 = np.zeros((2, 2, 2)).astype(np.float64)
    arr1[0, :, :] = box1_1
    arr1[1, :, :] = box1_2

    box2_1 = np.array(((0.5, 0.5), (0.9, 0.9))).astype(np.float64)
    box2_2 = np.array(((0.3, 0.3), (0.4, 0.4))).astype(np.float64)
    arr2 = np.zeros((2, 2, 2)).astype(np.float64)
    arr2[0, :, :] = box2_1
    arr2[1, :, :] = box2_2

    overlaps = cube.intersection_ok(arr1, arr2)
    assert overlaps[0]
    assert not overlaps[1]
