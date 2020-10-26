# Copyright 2020 Michael Hernandez

import random  # type: ignore
import logging  # type: ignore
from copy import copy  # type: ignore
from .test_utils.setup_tests import setup_test_case, jitter_boxes  # type: ignore

import od_toolbelt as od  # type: ignore

random.seed(7171)

IOU_THRESHOLD = 0.001
LOGGER = logging.getLogger()


def run_test_end_to_end(n_boxes: int, suppressor: od.nms.suppression.Suppressor):
    bounding_boxes, confidences, labels, bbids = setup_test_case()
    n = copy(bounding_boxes.shape[0])

    bounding_boxes, confidences, labels, _ = jitter_boxes(
        bounding_boxes,
        confidences,
        labels,
        bbids,
        range_n_new_boxes=(n_boxes, n_boxes),
    )

    data_payload = od.BoundingBoxArray(
        bounding_boxes=bounding_boxes, confidences=confidences, labels=labels
    )

    filtered_boxes = suppressor.transform(bounding_box_array=data_payload)
    assert len(filtered_boxes) == n


def run_test_end_to_end_cp_nms(n_boxes: int):
    iou_metric = od.nms.metrics.DefaultIntersectionOverTheUnion(
        threshold=IOU_THRESHOLD, direction="gte"
    )
    random_selector = od.nms.selection.RandomSelector()

    suppressor = od.nms.suppression.CartesianProductSuppression(
        metric=iou_metric, selector=random_selector
    )

    run_test_end_to_end(n_boxes, suppressor)


def run_test_end_to_end_sb_nms(n_boxes: int):
    iou_metric = od.nms.metrics.DefaultIntersectionOverTheUnion(
        threshold=IOU_THRESHOLD, direction="gte"
    )
    random_selector = od.nms.selection.RandomSelector()

    suppressor = od.nms.suppression.SectorSuppression(
        metric=iou_metric, selector=random_selector, sector_divisions=1
    )

    run_test_end_to_end(n_boxes, suppressor)


def test_end_to_end_cp_nms1():
    run_test_end_to_end_cp_nms(1)


def test_end_to_end_cp_nms2():
    run_test_end_to_end_cp_nms(2)


def test_end_to_end_cp_nms3():
    run_test_end_to_end_cp_nms(3)


def test_end_to_end_cp_nms4():
    run_test_end_to_end_cp_nms(4)


def test_end_to_end_cp_nms5():
    run_test_end_to_end_cp_nms(5)


def test_end_to_end_sb_nms1():
    run_test_end_to_end_sb_nms(1)


def test_end_to_end_sb_nms2():
    run_test_end_to_end_sb_nms(2)


def test_end_to_end_sb_nms3():
    run_test_end_to_end_sb_nms(3)


def test_end_to_end_sb_nms4():
    run_test_end_to_end_sb_nms(4)


def test_end_to_end_sb_nms5():
    run_test_end_to_end_sb_nms(5)
