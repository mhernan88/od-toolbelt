import numpy as np

from od_toolbelt import BoundingBoxArray
from od_toolbelt.nms.suppression.cartesian_product_suppression import CartesianProductSuppression
from od_toolbelt.nms.metrics.iou import DefaultIntersectionOverTheUnion
from od_toolbelt.nms.selection.random_selector import RandomSelector
from .test_utils.test_setup import setup


def test_smoke1():
    bounding_boxes, confidences, labels = setup()

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes, confidences=confidences, labels=labels
    )
    bb.check()

    metric = DefaultIntersectionOverTheUnion(threshold=0.35, direction="lt")
    selector = RandomSelector()

    suppressor = CartesianProductSuppression(metric=metric, selector=selector)
    suppressor.transform(bounding_boxes, confidences, labels)