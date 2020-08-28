import pytest
import numpy as np

from geometry import point


def test_assert_point1():
    arr = np.array((0.48, 0.82)).astype(np.float64)
    point.assert_point(arr, True)


def test_assert_point2():
    arr = np.array((0.48, 0.82)).astype(np.float32)
    with pytest.raises(point.AssertPointError):
        point.assert_point(arr, True)


def test_assert_point3():
    arr = np.array((0.48,)).astype(np.float64)
    with pytest.raises(point.AssertPointError):
        point.assert_point(arr, True)


def test_assert_point4():
    arr = np.array((0.48, 0.82, 0.91)).astype(np.float64)
    with pytest.raises(point.AssertPointError):
        point.assert_point(arr, True)


def test_assert_point5():
    arr = np.array((0.48, 1.02)).astype(np.float64)
    with pytest.raises(point.AssertPointError):
        point.assert_point(arr, True)


def test_assert_point6():
    arr = np.array((-0.21, 0.82)).astype(np.float64)
    with pytest.raises(point.AssertPointError):
        point.assert_point(arr, True)
