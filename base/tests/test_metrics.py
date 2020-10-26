import pytest
import numpy as np
from od_toolbelt.nms.metrics import DefaultIntersectionOverTheUnion


def setup_metrics_overlap():
    bounding_box1 = np.array(((0.10, 0.30), (0.20, 0.40)), dtype=np.float64)
    bounding_box2 = np.array(((0.11, 0.31), (0.21, 0.41)), dtype=np.float64)
    return bounding_box1, bounding_box2


def setup_metrics_no_overlap():
    bounding_box1 = np.array(((0.10, 0.30), (0.20, 0.40)), dtype=np.float64)
    bounding_box2 = np.array(((0.40, 0.60), (0.21, 0.41)), dtype=np.float64)
    return bounding_box1, bounding_box2


def setup_metrics_no_overlap2():
    bounding_box1 = np.array(((0.10, 0.10), (0.20, 0.20)), dtype=np.float64)
    bounding_box2 = np.array(((0.30, 0.30), (0.40, 0.40)), dtype=np.float64)
    return bounding_box1, bounding_box2


def test_diou_compute_overlap():
    bounding_box1, bounding_box2 = setup_metrics_overlap()
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    metric_value = metric.compute(bounding_box1, bounding_box2)
    assert metric_value == pytest.approx(0.6806, 0.01)


def test_diou_compute_no_overlap():
    bounding_box1, bounding_box2 = setup_metrics_no_overlap()
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    metric_value = metric.compute(bounding_box1, bounding_box2)
    assert metric_value == pytest.approx(0, 0.01)


def test_diou_compute_no_overlap2():
    bounding_box1, bounding_box2 = setup_metrics_no_overlap2()
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    metric_value = metric.compute(bounding_box1, bounding_box2)
    assert metric_value == pytest.approx(0, 0.01)


def test_diou_within_range_overlap():
    bounding_box1, bounding_box2 = setup_metrics_overlap()
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    # Metric returns True if computed value is greater than or equal to 0.01
    # (i.e. if the two boxes overlap), else False.
    metric_flag = metric.overlap(bounding_box1, bounding_box2)
    assert metric_flag


def test_diou_within_range_no_overlap():
    bounding_box1, bounding_box2 = setup_metrics_no_overlap()
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    # Metric returns True if computed value is greater than or equal to 0.01
    # (i.e. if the two boxes overlap), else False.
    metric_flag = metric.overlap(bounding_box1, bounding_box2)
    assert not metric_flag
