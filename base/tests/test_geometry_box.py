import pytest
import numpy as np

from geometry import box


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