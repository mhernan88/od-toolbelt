import pytest
from datetime import datetime
import numpy as np
import loguru
from enhance.generic.non_maximum_suppression.default import DefaultNonMaximumSuppression
from exceptions.decorators import log_exception

l = loguru.logger
l.add(f"test_{datetime.now().strftime('%Y%m%d')}.log")


@log_exception(l)
def test_filter_by_confidence():
    # 3 is number of observations,
    # 4 is number of coordinates per observation,
    # 2 is each x, y value for a given coordinate.
    coordinates_shape = (3, 4, 2)

    coordinates = np.zeros(coordinates_shape)
    confidences = np.asarray([0.9, 0.4, 0.7])

    nms = DefaultNonMaximumSuppression(0, 0.5)
    out_coordinates, out_confidences = nms.filter_by_confidence(
        coordinates, confidences
    )

    l.info(f"out_coordinates has a 0th dimension of shape {out_coordinates.shape[0]}")
    assert out_coordinates.shape[0] == 2
    l.info(f"out_confidences has a 0th dimension of shape {out_coordinates.shape[0]}")
    assert out_confidences.shape[0] == 2
    l.info(f"out_confidences has an index-0 value of {out_confidences[0]}")
    assert out_confidences[0] == 0.9
    l.info(f"out_confidences has an index-1 value of {out_confidences[1]}")
    assert out_confidences[1] == 0.7


@log_exception(l)
def test_multi_filter_by_confidence():
    coordinates_shape = (3, 4, 2)

    coordinates = [
        np.zeros(coordinates_shape),
        np.zeros(coordinates_shape),
        np.zeros(coordinates_shape),
    ]
    confidences = [
        np.asarray([0.9, 0.4, 0.7]),
        np.asarray([0.1, 0.3, 0.9]),
        np.asarray([0.9, 0.9, 0.9]),
    ]

    nms = DefaultNonMaximumSuppression(0, 0.5)
    (
        out_coordinates,
        out_confidences,
        out_coordinates_combined,
        out_confidences_combined,
    ) = nms.multi_filter_by_confidence(coordinates, confidences)

    assert len(out_confidences) == 3


@log_exception(l)
def test_ious():
    coordinates1 = np.asarray([[1, 1], [1, 3], [3, 1], [3, 3]])
    coordinates1 = np.expand_dims(coordinates1, 0)
    coordinates1_area = 4

    coordinates12 = np.asarray([[2, 2], [2, 4], [4, 2], [4, 4]])
    coordinates2 = np.expand_dims(coordinates12, 0)
    coordinates2_area = 4

    iou = 1
    expected_iou = iou / (coordinates1_area + coordinates2_area - iou)

    nms = DefaultNonMaximumSuppression(0, 0)
    actual_iou = nms._find_overlap(coordinates1, coordinates2)

    assert np.abs(actual_iou, expected_iou) < 1e-5
