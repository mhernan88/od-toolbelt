import itertools
import od_toolbelt as od

from .test_utils import setup_tests


def test_evaluate_overlap1():
    # Tests method without any overlapping boxes
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()
    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )
    bbid_cp = itertools.product(bb.bounding_box_ids.tolist(), bb.bounding_box_ids.tolist())

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    selected_bids, _ = suppressor._evaluate_overlap(bb, bbid_cp)
    assert len(selected_bids) == 2


def test_evaluate_overlap2():
    # Tests method 1 overlapping box each
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    boxes, confidences, labels, bbids = setup_tests.jitter_boxes(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bbids=bbids,
        range_n_new_boxes=(1, 1)
    )

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )
    bbid_cp = itertools.product(bb.bounding_box_ids.tolist(), bb.bounding_box_ids.tolist())

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    selected_bids, _ = suppressor._evaluate_overlap(bb, bbid_cp, symmetric=True)
    assert len(selected_bids) == 2

    resulting_box1 = bb.lookup_box(int(selected_bids[0]))
    resulting_box2 = bb.lookup_box(int(selected_bids[1]))
    assert not metric.overlap(resulting_box1, resulting_box2)


def test_evaluate_overlap3():
    # Test method 5 overlapping boxes each
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    boxes, confidences, labels, bbids = setup_tests.jitter_boxes(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bbids=bbids,
        range_n_new_boxes=(5, 5)
    )

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )
    bbid_cp = itertools.product(bb.bounding_box_ids.tolist(), bb.bounding_box_ids.tolist())

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    selected_bids, _ = suppressor._evaluate_overlap(bb, bbid_cp, symmetric=True)
    assert len(selected_bids) == 2

    resulting_box1 = bb.lookup_box(int(selected_bids[0]))
    resulting_box2 = bb.lookup_box(int(selected_bids[1]))
    assert not metric.overlap(resulting_box1, resulting_box2)


def test_cp_transform1():
    # Test None
    bb = None
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor._cp_transform(bb)
    assert resulting_bb is None


def test_cp_transform2():
    # Test w/ 0 overlapping boxes
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor._cp_transform(bb)
    assert len(resulting_bb.bounding_box_ids) == 2
    assert resulting_bb.bounding_boxes.shape[0] == 2

    resulting_box1 = resulting_bb.bounding_boxes[0, :, :]
    resulting_box2 = resulting_bb.bounding_boxes[1, :, :]
    assert not metric.overlap(resulting_box1, resulting_box2)


def test_cp_transform3():
    # Test w/ 1 overlapping boxes
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    boxes, confidences, labels, bbids = setup_tests.jitter_boxes(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bbids=bbids,
        range_n_new_boxes=(1, 1)
    )

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor._cp_transform(bb)
    assert len(resulting_bb.bounding_box_ids) == 2
    assert resulting_bb.bounding_boxes.shape[0] == 2

    resulting_box1 = resulting_bb.bounding_boxes[0, :, :]
    resulting_box2 = resulting_bb.bounding_boxes[1, :, :]
    assert not metric.overlap(resulting_box1, resulting_box2)


def test_cp_transform4():
    # Test w/ 1 overlapping boxes
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    boxes, confidences, labels, bbids = setup_tests.jitter_boxes(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bbids=bbids,
        range_n_new_boxes=(5, 5)
    )

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor._cp_transform(bb)
    assert len(resulting_bb.bounding_box_ids) == 2
    assert resulting_bb.bounding_boxes.shape[0] == 2

    resulting_box1 = resulting_bb.bounding_boxes[0, :, :]
    resulting_box2 = resulting_bb.bounding_boxes[1, :, :]
    assert not metric.overlap(resulting_box1, resulting_box2)


def test_transform1():
    # Test None
    bb = None
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor.transform(bb)
    assert resulting_bb is None


def test_transform2():
    # Test w/ 0 overlapping boxes
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor.transform(bb)
    assert len(resulting_bb.bounding_box_ids) == 2
    assert resulting_bb.bounding_boxes.shape[0] == 2

    resulting_box1 = resulting_bb.bounding_boxes[0, :, :]
    resulting_box2 = resulting_bb.bounding_boxes[1, :, :]
    assert not metric.overlap(resulting_box1, resulting_box2)


def test_transform3():
    # Test w/ 1 overlapping boxes
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    boxes, confidences, labels, bbids = setup_tests.jitter_boxes(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bbids=bbids,
        range_n_new_boxes=(1, 1)
    )

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor.transform(bb)
    assert len(resulting_bb.bounding_box_ids) == 2
    assert resulting_bb.bounding_boxes.shape[0] == 2

    resulting_box1 = resulting_bb.bounding_boxes[0, :, :]
    resulting_box2 = resulting_bb.bounding_boxes[1, :, :]
    assert not metric.overlap(resulting_box1, resulting_box2)


def test_transform4():
    # Test w/ 1 overlapping boxes
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()

    boxes, confidences, labels, bbids = setup_tests.jitter_boxes(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bbids=bbids,
        range_n_new_boxes=(5, 5)
    )

    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.CartesianProductSuppression(metric=metric, selector=selector)

    resulting_bb = suppressor.transform(bb)
    assert len(resulting_bb.bounding_box_ids) == 2
    assert resulting_bb.bounding_boxes.shape[0] == 2

    resulting_box1 = resulting_bb.bounding_boxes[0, :, :]
    resulting_box2 = resulting_bb.bounding_boxes[1, :, :]
    assert not metric.overlap(resulting_box1, resulting_box2)