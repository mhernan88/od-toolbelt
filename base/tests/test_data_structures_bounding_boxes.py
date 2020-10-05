import pytest
import numpy as np
from data_structures.bounding_boxes import BoundingBoxArray


def setup():
    bounding_boxes = np.array((
        ((0.04, 0.19), (0.14, 0.29)),
        ((0.11, 0.15), (0.21, 0.25))
    ), dtype=np.float32)
    confidences = np.array((0.4, 0.6), dtype=np.float64)
    labels = np.array((0, 1), dtype=np.int64)

    return bounding_boxes, confidences, labels


def one_additional():
    bounding_box = np.array((
        (0.09, 0.44), (0.11, 0.50)
    ))
    confidence = 0.7
    label = 1

    return bounding_box, confidence, label


def test_check1():
    """Test check without arguments"""
    bounding_boxes, confidences, labels = setup()

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    bb.check()


def test_check2():
    """Test check with arguments"""
    bounding_boxes, confidences, labels = setup()
    bounding_box, confidence, label = one_additional()

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    bb.check(
        bounding_box=bounding_box,
        confidence=confidence,
        label=label
    )


def test_check3():
    """Test check without arguments - warn"""
    bounding_boxes, confidences, labels = setup()
    bounding_boxes = bounding_boxes.astype(np.int64)

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.warns(SyntaxWarning):
        bb.check()


def test_check4():
    """Test check without arguments - warn"""
    bounding_boxes, confidences, labels = setup()
    confidences = confidences.astype(np.int64)

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.warns(SyntaxWarning):
        bb.check()


def test_check5():
    """Test check without arguments - warn"""
    bounding_boxes, confidences, labels = setup()
    labels = labels.astype(np.float64)

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.warns(SyntaxWarning):
        bb.check()


def test_check6():
    """Test check with arguments - warn"""
    bounding_boxes, confidences, labels = setup()
    bounding_box, confidence, label = one_additional()
    bounding_box = bounding_box.astype(np.int64)

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.warns(SyntaxWarning):
        bb.check(
            bounding_box=bounding_box,
            confidence=confidence,
            label=label
        )


def test_check7():
    """Test check without arguments - fail bb range"""
    bounding_boxes, confidences, labels = setup()
    bounding_boxes[0, 0, 0] = 1.01

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.raises(AssertionError):
        bb.check()


def test_check8():
    """Test check without arguments - fail bb type"""
    bounding_boxes, confidences, labels = setup()
    bounding_boxes = (
        ((0.04, 0.19), (0.14, 0.29)),
        ((0.11, 0.15), (0.21, 0.25))
    )

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.raises(AssertionError):
        bb.check()


def test_check9():
    """Test check without arguments - fail bb shape len"""
    bounding_boxes, confidences, labels = setup()
    bounding_boxes = np.array((
        ((0.04, 0.19, 0.12), (0.14, 0.29, 0.81)),
        ((0.11, 0.15, 0.02), (0.21, 0.25, 0.14))
    ), dtype=np.float64)

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.raises(AssertionError):
        bb.check()


def test_check10():
    """Test check without arguments - fail confidence type"""
    bounding_boxes, confidences, labels = setup()
    confidences = (0.4, 0.6)

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.raises(AssertionError):
        bb.check()


def test_check11():
    """Test check without arguments - fail labels"""
    bounding_boxes, confidences, labels = setup()
    labels = (1, 0)

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    with pytest.raises(AssertionError):
        bb.check()


def test_append1():
    """Test append without arguments"""
    bounding_boxes, confidences, labels = setup()
    before_length = bounding_boxes.shape[0]

    bounding_box, confidence, label = one_additional()

    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes,
        confidences=confidences,
        labels=labels
    )

    bb.append(
        bounding_box=bounding_box,
        confidence=confidence,
        label=label
    )
    assert bb.bounding_boxes.shape[0] == before_length + 1
