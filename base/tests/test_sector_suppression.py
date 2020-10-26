import random
import pytest
import od_toolbelt as od

from .test_utils import setup_tests

random.seed(7171)


def test_create_sectors1():
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    for n in range(10):
        print(f"Running test_create_sectors for {n+1} division.")
        suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector, sector_divisions=n+1)
        sectors, _ = suppressor._create_sectors()
        assert len(sectors) == 2**(n+1)


def create_sectors_dividing_lines(sector_divisions):
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector, sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    return dividing_lines[-1]


def test_create_sectors2():
    dividing_lines = create_sectors_dividing_lines(1)
    assert dividing_lines[0] == pytest.approx(0.5, 0.0001)
    assert dividing_lines[1]


def test_create_sectors3():
    dividing_lines = create_sectors_dividing_lines(2)
    assert dividing_lines[0] == pytest.approx(0.5, 0.0001)
    assert not dividing_lines[1]


def test_create_sectors4():
    dividing_lines = create_sectors_dividing_lines(3)
    assert dividing_lines[0] == pytest.approx(0.25, 0.0001) or dividing_lines[0] == pytest.approx(0.75, 0.0001)
    assert dividing_lines[1]


def test_create_sectors5():
    dividing_lines = create_sectors_dividing_lines(4)
    assert dividing_lines[0] == pytest.approx(0.25, 0.0001) or dividing_lines[0] == pytest.approx(0.75, 0.0001)
    assert not dividing_lines[1]


def test_handle_boundaries1():
    # Handle boundaries - no boxes on boundaries.
    boxes, confidences, labels, bbids = setup_tests.setup_test_case()
    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    sector_divisions = 1
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector, sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    result = suppressor._handle_boundaries(bb, dividing_lines)
    assert result is None


def test_handle_boundaries2():
    # Handle boundaries - two non-overlapping boxes on boundaries w/ 1 division.
    boxes, confidences, labels, bbids = setup_tests.setup_test_case_on_boundary()
    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    sector_divisions = 1
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector, sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    result = suppressor._handle_boundaries(bb, dividing_lines)
    assert len(result) == 1


def test_handle_boundaries3():
    # Handle boundaries - two non-overlapping boxes on boundaries w/ 2 divisions
    boxes, confidences, labels, bbids = setup_tests.setup_test_case_on_boundary()
    bb = od.BoundingBoxArray(
        bounding_boxes=boxes,
        confidences=confidences,
        labels=labels,
        bounding_box_ids=bbids
    )

    sector_divisions = 2
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector,
                                                      sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    result = suppressor._handle_boundaries(bb, dividing_lines)
    assert len(result) == 2


def test_handle_boundaries4():
    # Handle boundaries - two sets of boxes (1 overlap each) on boundaries w/ 1 division
    boxes, confidences, labels, bbids = setup_tests.setup_test_case_on_boundary()

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
    bb.check()

    sector_divisions = 1
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector,
                                                      sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    result = suppressor._handle_boundaries(bb, dividing_lines)
    assert len(result) == 1


def test_handle_boundaries5():
    # Handle boundaries - two sets of boxes (1 overlap each) on boundaries w/ 2 division
    boxes, confidences, labels, bbids = setup_tests.setup_test_case_on_boundary()

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
    bb.check()

    sector_divisions = 2
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector,
                                                      sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    result = suppressor._handle_boundaries(bb, dividing_lines)
    assert len(result) == 2


# TODO: Not passing
def test_handle_boundaries6():
    # Handle boundaries - two sets of boxes (5 overlap each) on boundaries w/ 1 division
    boxes, confidences, labels, bbids = setup_tests.setup_test_case_on_boundary()

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
    bb.check()

    sector_divisions = 1
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector,
                                                      sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    result = suppressor._handle_boundaries(bb, dividing_lines)
    assert len(result) == 1


def test_handle_boundaries7():
    # Handle boundaries - two sets of boxes (5 overlap each) on boundaries w/ 2 division
    boxes, confidences, labels, bbids = setup_tests.setup_test_case_on_boundary()

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
    bb.check()

    sector_divisions = 2
    metric, selector = setup_tests.get_default_sector_suppressor_components()
    suppressor = od.nms.suppression.SectorSuppression(metric=metric, selector=selector,
                                                      sector_divisions=sector_divisions)
    _, dividing_lines = suppressor._create_sectors()
    result = suppressor._handle_boundaries(bb, dividing_lines)
    assert len(result) == 2