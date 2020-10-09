import numpy as np
from typing import List, Tuple, Any
from nptyping import NDArray


def setup_test_case() -> Tuple[
    NDArray[(2, 2, 2), np.float64],
    NDArray[(2,), np.float64],
    NDArray[(2,), np.int64],
]:
    bounding_boxes = np.array((
        ((0.04, 0.19), (0.14, 0.29)),
        ((0.75, 0.79), (0.82, 0.91))
    ), dtype=np.float64)
    confidences = np.array((0.4, 0.6), dtype=np.float64)
    labels = np.array((0, 1), dtype=np.int64)

    return bounding_boxes, confidences, labels


def jitter_boxes(
        bounding_boxes: NDArray[(Any, 2, 2), np.float64],
        confidences: NDArray[(Any,), np.float64],
        labels: NDArray[(Any,), np.int64],
        range_n_new_boxes: Tuple[int, int],
        box_standard_dev: float = 0.0005,
        confidence_standard_dev: float = 0.001,
        label_classes: List[int] = (0, 1),
        label_probabilities: List[float] = (0.5, 0.5),
) -> Tuple[
    NDArray[(Any, 2, 2), np.float64],
    NDArray[(Any,), np.float64],
    NDArray[(Any,), np.int64],
]:
    assert range_n_new_boxes[0] <= range_n_new_boxes[1]

    label_classes = [int(x) for x in label_classes]
    label_probabilities = [float(x) for x in label_probabilities]

    new_bounding_boxes = []
    new_confidences = []
    new_labels = []

    for i in np.arange(0, bounding_boxes.shape[0]):  # Iterating over existing boxes.
        if range_n_new_boxes[0] != range_n_new_boxes[1]:
            n_new_boxes = np.random.randint(*range_n_new_boxes)
        else:
            n_new_boxes = range_n_new_boxes[0]
        for _ in np.arange(0, n_new_boxes):  # Iterating over number of new boxes to create around existing box.
            new_box_pt1_x = bounding_boxes[i, 0, 0] + np.random.normal(loc=0, scale=box_standard_dev)
            new_box_pt1_y = bounding_boxes[i, 0, 1] + np.random.normal(loc=0, scale=box_standard_dev)
            new_box_pt2_x = bounding_boxes[i, 1, 0] + np.random.normal(loc=0, scale=box_standard_dev)
            new_box_pt2_y = bounding_boxes[i, 1, 1] + np.random.normal(loc=0, scale=box_standard_dev)

            new_bounding_boxes.append(np.expand_dims(np.array(
                ((new_box_pt1_x, new_box_pt1_y), (new_box_pt2_x, new_box_pt2_y))
            ), 0))

            new_confidences.append((
                confidences[i] + np.random.normal(loc=0, scale=confidence_standard_dev)
            ))

            new_labels.append((
                np.random.choice(label_classes, p=label_probabilities)
            ))

    new_bounding_boxes.append(bounding_boxes)
    new_confidences.extend(confidences.tolist())
    new_labels.extend(labels.tolist())

    new_bounding_boxes = np.concatenate(new_bounding_boxes)
    new_confidences = np.asarray(new_confidences, dtype=np.float64)
    new_labels = np.asarray(new_labels, dtype=np.int64)

    try:
        assert new_bounding_boxes.shape[0] == new_confidences.shape[0]
    except AssertionError:
        raise AssertionError(f"new_bounding_boxes length of {new_bounding_boxes.shape[0]} did not match "
                             f"new_confidences length of {new_confidences.shape[0]}")
    try:
        assert new_bounding_boxes.shape[0] == new_labels.shape[0]
    except AssertionError:
        raise AssertionError(f"new_bounding_boxes length of {new_bounding_boxes.shape[0]} did not match "
                             f"new_labels length of {new_labels.shape[0]}")

    return new_bounding_boxes, new_confidences, new_labels


def one_additional() -> Tuple[
    NDArray[(2, 2), np.float64],
    float,
    int,
]:
    bounding_box = np.array(((0.09, 0.44), (0.11, 0.50)))
    confidence = 0.7
    label = 1

    return bounding_box, confidence, label
