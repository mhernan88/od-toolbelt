import pytest
import numpy as np


@pytest.fixture
def overlap1():
    bounding_box1 = np.array(((0.10, 0.30), (0.20, 0.40)), dtype=np.float64)
    bounding_box2 = np.array(((0.11, 0.31), (0.21, 0.41)), dtype=np.float64)
    return bounding_box1, bounding_box2


@pytest.fixture
def no_overlap1():
    bounding_box1 = np.array(((0.10, 0.30), (0.20, 0.40)), dtype=np.float64)
    bounding_box2 = np.array(((0.40, 0.60), (0.21, 0.41)), dtype=np.float64)
    return bounding_box1, bounding_box2


@pytest.fixture
def no_overlap2():
    bounding_box1 = np.array(((0.10, 0.10), (0.20, 0.20)), dtype=np.float64)
    bounding_box2 = np.array(((0.30, 0.30), (0.40, 0.40)), dtype=np.float64)
    return bounding_box1, bounding_box2
