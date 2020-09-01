import logging
import rootpath
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import numpy as np
from nptyping import NDArray
from typing import Any, List, Tuple

from nms.suppressors.default import NonMaximumSuppression
from box_selectors.random_selector import random_selector

logger = logging.getLogger("nms_example")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup() -> Path:
    # return Path(rootpath.detect(), "tests", "base/tests/test_data")
    return Path("C:/dev/detection-enhancer/base/tests/test_data")


def to_box(x: List[Tuple[int, int]], shape: Tuple[int, int]) -> (NDArray[(2, 2), np.float64], NDArray[(Any,), np.float64]):
    return np.array(((x[0][1] / shape[0], x[0][0] / shape[1]), (x[1][1] / shape[0], x[1][0] / shape[1])))


def to_box_multi(x: List[List[Tuple[int, int]]], shape: Tuple[int, int]) -> (NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]):
    boxes_arr = np.zeros((len(x), 2, 2), dtype=np.float64)
    confidences_arr = np.zeros(len(x), dtype=np.float64)
    for i, this_box in enumerate(x):
        boxes_arr[i, :, :] = to_box(this_box, shape)
        confidences_arr[i] = np.random.uniform(0, 1)
    return boxes_arr, confidences_arr


def to_list(x: NDArray[(2, 2), np.float64], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    assert len(shape) == 2
    assert len(x.shape) == 2
    return [(x[0, 1] * shape[1], x[0, 0] * shape[0]), (x[1, 1] * shape[1], x[1, 0] * shape[0])]


def to_list_multi(x: NDArray[(Any, 2, 2), np.float64], shape: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
    assert len(shape) == 2
    assert len(x.shape) == 3
    out = []
    for i in np.arange(0, x.shape[0]):
        out.append(to_list(x[i, :, :], shape))
    return out


def draw(boxes: List[List[Tuple[int, int]]], img: Image.Image):
    img_draw = ImageDraw.Draw(img)
    for b in boxes:
        img_draw.rectangle(b, outline="red", width=5)
    img.show()


def jitter_boxes(boxes, n_boxes, img, std):
    new_boxes = []
    for i in range(n_boxes):
        for j, this_box in enumerate(boxes):
            print(f"Adding new box {i} around existing box {j}")
            new_box_pt1_x = this_box[0][0] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[0])
            new_box_pt1_y = this_box[0][1] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[1])
            new_box_pt2_x = this_box[1][0] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[0])
            new_box_pt2_y = this_box[1][1] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[1])
            new_box = [(new_box_pt1_x, new_box_pt1_y), (new_box_pt2_x, new_box_pt2_y)]
            new_boxes.append(new_box)
    boxes.extend(new_boxes)
    return new_boxes


def get_data(n_boxes: int, std: float, show=False):
    p = setup()
    image1 = Path(p, "IMG_1663.jpg")
    if not image1.is_file():
        resp = requests.get("https://od-toolbox.s3.amazonaws.com/images/IMG_1663.jpg")
        if resp.status_code != 200:
            raise Exception("failed to download test image")
        with open(image1, "wb") as f:
            f.write(resp.content)

    print(f"Opening file at {image1}")
    img = Image.open(image1)
    shape = np.asarray(img).shape[:2]

    boxes = [
        [(575, 975), (1225, 1650)],
        [(1800, 300), (2700, 1060)],
        [(1950, 2100), (2950, 2925)]
    ]

    new_boxes = jitter_boxes(boxes, n_boxes, img, std)
    boxes_arr, confidences_arr = to_box_multi(new_boxes, shape)

    if show:
        draw(boxes, img)
    return boxes_arr, confidences_arr, img, shape


def apply_nms():
    logger.debug("beginning example")
    boxes, confs, img, shape = get_data(n_boxes=7, std=0.03, show=False)

    nms = NonMaximumSuppression(iou_threshold=0.01, selection_func=random_selector, logger=logger, confidence_threshold=0.3, selection_kwargs=None, exact=False)
    pboxes, pconfs = nms.transform(boxes, confs)

    bboxes = to_list_multi(np.asarray(pboxes), shape)
    draw(bboxes, img)

# get_data(n_boxes=3, std=0.03, show=True)
apply_nms()