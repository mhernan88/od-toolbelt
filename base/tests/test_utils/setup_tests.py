import numpy as np
import od_toolbelt as od
from typing import List, Tuple, Any
from nptyping import NDArray

IOU_THRESHOLD = 0.001


def get_default_sector_suppressor_components():
    metric = od.nms.metrics.DefaultIntersectionOverTheUnion(threshold=IOU_THRESHOLD, direction="gte")
    selector = od.nms.selection.RandomSelector()
    return metric, selector


def setup_test_case() -> Tuple[
    NDArray[(2, 2, 2), np.float64],
    NDArray[(2,), np.float64],
    NDArray[(2,), np.int64],
    NDArray[(2,), np.int64]
]:
    bounding_boxes = np.array(
        (((0.04, 0.19), (0.14, 0.29)), ((0.75, 0.79), (0.82, 0.91))), dtype=np.float64
    )
    confidences = np.array((0.4, 0.6), dtype=np.float64)
    labels = np.array((0, 1), dtype=np.int64)
    bbids = np.array((1, 2), dtype=np.int64)

    return bounding_boxes, confidences, labels, bbids


def setup_test_case_on_boundary() -> Tuple[
    NDArray[(2, 2, 2), np.float64],
    NDArray[(2,), np.float64],
    NDArray[(2,), np.int64],
    NDArray[(2,), np.int64]
]:
    bounding_boxes = np.array(
        (((0.44, 0.19), (0.54, 0.29)), ((0.75, 0.49), (0.82, 0.51))), dtype=np.float64
    )
    confidences = np.array((0.4, 0.6), dtype=np.float64)
    labels = np.array((0, 1), dtype=np.int64)
    bbids = np.array((1, 2), dtype=np.int64)

    return bounding_boxes, confidences, labels, bbids



def setup_test_case_multi() -> List[
    Tuple[
        NDArray[(Any, 2, 2), np.float64],
        NDArray[(Any,), np.float64],
        NDArray[(Any,), np.int64],
        NDArray[(Any,), np.int64]
    ]
]:
    boxes1 = np.array(
        (
            ((0.10, 0.10), (0.20, 0.20)),
            ((0.30, 0.30), (0.40, 0.40)),
        ),
        dtype=np.float64,
    )
    confidences1 = np.array((0.6, 0.65), dtype=np.float64)
    labels1 = np.array((1, 1), dtype=np.int64)
    bbids1 = np.array((1, 2), dtype=np.int64)
    arr1 = (boxes1, confidences1, labels1, bbids1,)


    boxes2 = np.array(
        (
            ((0.11, 0.11), (0.21, 0.21)),
            ((0.35, 0.35), (0.50, 0.50)),
            ((0.60, 0.60), (0.70, 0.70)),
        ),
        dtype=np.float64,
    )
    confidences2 = np.array((0.7, 0.65, 0.55), dtype=np.float64)
    labels2 = np.array((1, 1, 1), dtype=np.int64)
    bbids2 = np.array((3, 4, 5), dtype=np.int64)
    arr2 = (boxes2, confidences2, labels2, bbids2,)

    return [arr1, arr2]


def jitter_boxes(
        bounding_boxes: NDArray[(Any, 2, 2), np.float64],
        confidences: NDArray[(Any,), np.float64],
        labels: NDArray[(Any,), np.int64],
        bbids: NDArray[(Any,), np.int64],
        range_n_new_boxes: Tuple[int, int],
        box_standard_dev: float = 0.0005,
        confidence_standard_dev: float = 0.001,
        label_classes: List[int] = (0, 1),
        label_probabilities: List[float] = (0.5, 0.5),
) -> Tuple[
    NDArray[(Any, 2, 2), np.float64],
    NDArray[(Any,), np.float64],
    NDArray[(Any,), np.int64],
    NDArray[(Any,), np.int64],
]:
    assert range_n_new_boxes[0] <= range_n_new_boxes[1]

    label_classes = [int(x) for x in label_classes]
    label_probabilities = [float(x) for x in label_probabilities]

    new_bounding_boxes = []
    new_confidences = []
    new_labels = []
    new_bbids = []

    for i in np.arange(0, bounding_boxes.shape[0]):  # Iterating over existing boxes.
        if range_n_new_boxes[0] != range_n_new_boxes[1]:
            n_new_boxes = np.random.randint(*range_n_new_boxes)
        else:
            n_new_boxes = range_n_new_boxes[0]
        for _ in np.arange(
            0, n_new_boxes
        ):  # Iterating over number of new boxes to create around existing box.
            new_box_pt1_x = bounding_boxes[i, 0, 0] + np.random.normal(
                loc=0, scale=box_standard_dev
            )
            new_box_pt1_y = bounding_boxes[i, 0, 1] + np.random.normal(
                loc=0, scale=box_standard_dev
            )
            new_box_pt2_x = bounding_boxes[i, 1, 0] + np.random.normal(
                loc=0, scale=box_standard_dev
            )
            new_box_pt2_y = bounding_boxes[i, 1, 1] + np.random.normal(
                loc=0, scale=box_standard_dev
            )

            new_bounding_boxes.append(
                np.expand_dims(
                    np.array(
                        ((new_box_pt1_x, new_box_pt1_y), (new_box_pt2_x, new_box_pt2_y))
                    ),
                    0,
                )
            )

            new_confidences.append(
                (
                    confidences[i]
                    + np.random.normal(loc=0, scale=confidence_standard_dev)
                )
            )

            new_labels.append((np.random.choice(label_classes, p=label_probabilities)))

            if len(new_bbids) == 0:
                new_bbids.append(np.max(bbids) + 1)
            else:
                new_bbids.append(np.max(new_bbids) + 1)

    new_bounding_boxes.append(bounding_boxes)
    new_confidences.extend(confidences.tolist())
    new_labels.extend(labels.tolist())
    new_bbids.extend(bbids)

    new_bounding_boxes = np.concatenate(new_bounding_boxes)
    new_confidences = np.asarray(new_confidences, dtype=np.float64)
    new_labels = np.asarray(new_labels, dtype=np.int64)
    new_bbids = np.asarray(new_bbids, dtype=np.int64)

    try:
        assert new_bounding_boxes.shape[0] == new_confidences.shape[0]
    except AssertionError:
        raise AssertionError(
            f"new_bounding_boxes length of {new_bounding_boxes.shape[0]} did not match "
            f"new_confidences length of {new_confidences.shape[0]}"
        )
    try:
        assert new_bounding_boxes.shape[0] == new_labels.shape[0]
    except AssertionError:
        raise AssertionError(
            f"new_bounding_boxes length of {new_bounding_boxes.shape[0]} did not match "
            f"new_labels length of {new_labels.shape[0]}"
        )

    try:
        assert new_bounding_boxes.shape[0] == new_bbids.shape[0]
    except AssertionError:
        raise AssertionError(
            f"new_bounding_boxes length of {new_bounding_boxes.shape[0]} did not match "
            f"new_bbids length of {new_bbids.shape[0]}"
        )

    return new_bounding_boxes, new_confidences, new_labels, new_bbids


def one_additional() -> Tuple[
    NDArray[(2, 2), np.float64],
    float,
    int,
]:
    bounding_box = np.array(((0.09, 0.44), (0.11, 0.50)))
    confidence = 0.7
    label = 1

    return bounding_box, confidence, label
