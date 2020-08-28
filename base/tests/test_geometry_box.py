import pytest
import numpy as np

from geometry import box, point


def test_assert_box1():
    arr = np.array(((0.48, 0.82), (0.82, 0.48))).astype(np.float64)
    box.assert_box(arr, True)


def test_assert_box2():
    arr = np.array(((0.48, 0.82), (0.82, 0.48), (0.14, 0.55))).astype(np.float64)
    with pytest.raises(box.AssertBoxError):
        box.assert_box(arr, True)


def test_assert_box3():
    arr = np.array(((0.48, 0.82, 0.14), (0.82, 0.48, 0.55))).astype(np.float64)
    with pytest.raises(box.AssertBoxError):
        box.assert_box(arr, True)


def test_assert_box4():
    arr = np.array(((0.48, 0.82), (0.82, 0.48))).astype(np.float32)
    with pytest.raises(box.AssertBoxError):
        box.assert_box(arr, True)


def test_get_point1_1():
    arr = np.array(((0.48, 0.82), (0.82, 0.48))).astype(np.float64)
    pt = box.get_point1(arr)
    point.assert_point(pt, True)
    assert np.round(pt[0], 2) == 0.48
    assert np.round(pt[1], 2) == 0.82


def test_get_point2_1():
    arr = np.array(((0.48, 0.82), (0.82, 0.48))).astype(np.float64)
    pt = box.get_point2(arr)
    point.assert_point(pt, True)
    assert np.round(pt[0], 2) == 0.82
    assert np.round(pt[1], 2) == 0.48


def test_get_area1():
    arr = np.array(((0.4, 0.4), (0.8, 0.8))).astype(np.float64)
    area = box.get_area(arr)
    assert np.round(area, 2) == 0.16


def test_boxes_overlap1():
    arr1 = np.array(((0.4, 0.4), (0.8, 0.8))).astype(np.float64)
    arr2 = np.array(((0.5, 0.5), (0.9, 0.9))).astype(np.float64)
    overlap = box.boxes_overlap(arr1, arr2)
    assert overlap


def test_boxes_overlap2():
    arr1 = np.array(((0.4, 0.4), (0.8, 0.8))).astype(np.float64)
    arr2 = np.array(((0.9, 0.9), (0.95, 0.95))).astype(np.float64)
    overlap = box.boxes_overlap(arr1, arr2)
    assert not overlap
