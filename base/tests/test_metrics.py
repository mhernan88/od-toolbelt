import pytest
import numpy as np
from od_toolbelt.nms.metrics import DefaultIntersectionOverTheUnion


def test_diou_compute_overlap(overlap1):
    bounding_box1, bounding_box2 = overlap1
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    metric_value = metric.compute(bounding_box1, bounding_box2)
    assert metric_value == pytest.approx(0.6806, 0.01)


def test_diou_compute_no_overlap(no_overlap1):
    bounding_box1, bounding_box2 = no_overlap1
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    metric_value = metric.compute(bounding_box1, bounding_box2)
    assert metric_value == pytest.approx(0, 0.01)


def test_diou_compute_no_overlap2(no_overlap2):
    bounding_box1, bounding_box2 = no_overlap2
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    metric_value = metric.compute(bounding_box1, bounding_box2)
    assert metric_value == pytest.approx(0, 0.01)


def test_diou_within_range_overlap(overlap1):
    bounding_box1, bounding_box2 = overlap1
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    # Metric returns True if computed value is greater than or equal to 0.01
    # (i.e. if the two boxes overlap), else False.
    metric_flag = metric.overlap(bounding_box1, bounding_box2)
    assert metric_flag


def test_diou_within_range_no_overlap(no_overlap1):
    bounding_box1, bounding_box2 = no_overlap1
    metric = DefaultIntersectionOverTheUnion(0.01, "gte")
    # Metric returns True if computed value is greater than or equal to 0.01
    # (i.e. if the two boxes overlap), else False.
    metric_flag = metric.overlap(bounding_box1, bounding_box2)
    assert not metric_flag
