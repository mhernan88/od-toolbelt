import pytest
import numpy as np

from geometry import cube


def test_assert_cube1():
    box1 = np.array(((0.48, 0.82), (0.82, 0.48))).astype(np.float64)
    box2 = np.array(((0.14, 0.55), (0.55, 0.14))).astype(np.float64)
    arr = np.zeros((2, 2, 2)).astype(np.float64)
    arr[0, :, :] = box1
    arr[1, :, :] = box2
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