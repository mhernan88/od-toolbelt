import numpy as np


def setup():
    bounding_boxes = np.array(
        (((0.04, 0.19), (0.14, 0.29)), ((0.11, 0.15), (0.21, 0.25))), dtype=np.float32
    )
    confidences = np.array((0.4, 0.6), dtype=np.float64)
    labels = np.array((0, 1), dtype=np.int64)

    return bounding_boxes, confidences, labels


def one_additional():
    bounding_box = np.array(((0.09, 0.44), (0.11, 0.50)))
    confidence = 0.7
    label = 1

    return bounding_box, confidence, label