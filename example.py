import logging
import rootpath
import requests
from pathlib import Path
from PIL import Image, ImageDraw
from itertools import product
import copy

import numpy as np
from nptyping import NDArray
from typing import Any, List, Tuple

# from nms.suppressors.default import NonMaximumSuppression
# from nms.iou import DefaultNonMaximumSuppression
from suppression.cartesian_product_suppression import DefaultNonMaximumSuppression
from metrics.iou import DefaultIntersectionOverTheUnion
from selection.random_selector import RandomSelector

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
    print("STARTING DRAW")
    img_draw = ImageDraw.Draw(img)
    for b in boxes:
        img_draw.rectangle(b, outline="red", width=5)
    img.show()


def draw_cartesian(boxes: List[List[Tuple[int, int]]], img_raw: Image.Image):
    boxes_cart = product(boxes, boxes)
    iou = DefaultIntersectionOverTheUnion()
    shp = np.asarray(img_raw).shape
    print(len([b for b in boxes_cart]))
    boxes_cart = [x for x in product(boxes, boxes)]
    print(f"LEN BOXES_CART: {len(boxes_cart)}")
    for i in range(len(boxes_cart)):
        img = copy.copy(img_raw)
        img_draw = ImageDraw.Draw(img)
        print(f"STARTING LOOP ITER {i}")
        try:
            box1 = np.expand_dims(to_box(boxes_cart[i][0], shp), 0)
            box2 = np.expand_dims(to_box(boxes_cart[i][1], shp), 0)
            this_iou = iou.compute_cube(box1, box2)
        except Exception as e:
            print(boxes_cart[0])
            print(len(boxes_cart[0]))
            print(i)
            print("FAILED TO COMPLETE LOOP!!!!!!!!!!!!!!!!!!!!!!!")
            raise e
        # if this_iou > 0:
        if True:
            print(f"DRAWING LOOP ITER {i}")
            img_draw.rectangle(boxes_cart[i][0], outline="purple", width=5)
            img_draw.rectangle(boxes_cart[i][1], outline="purple", width=5)
            print(f"IOU: {this_iou}")
            print(f"BOX1: {box1}")
            print(f"BOX2: {box2}")
            print(f"ARR1: {boxes_cart[i][0]}")
            print(f"ARR2: {boxes_cart[i][1]}")
        if 0 < this_iou <= 1:
            # img.show()
            print(f"OK {i+1} of {len(boxes_cart)}")
        elif 0 <= this_iou:
            img.show()
            print(f"Skipped {i+1} of {len(boxes_cart)} because it was > 1")
        elif this_iou < 1:
            print(f"Skipped {i} of {len(boxes_cart)} because it was <= 0")
        else:
            print(f"Skipped {i+1} of {len(boxes_cart)}")
    print("DONE")


def jitter_boxes(this_boxes, n_boxes, img, std):
    new_boxes = []
    for i in range(n_boxes):
        for j, this_box in enumerate(this_boxes[:len(this_boxes)-1]):
            print(f"Adding new box {i} around existing box {j}")
            new_box_pt1_x = this_box[0][0] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[0])
            new_box_pt1_y = this_box[0][1] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[1])
            new_box_pt2_x = this_box[1][0] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[0])
            new_box_pt2_y = this_box[1][1] + np.random.normal(loc=0, scale=std * np.asarray(img).shape[1])
            new_box = [(new_box_pt1_x, new_box_pt1_y), (new_box_pt2_x, new_box_pt2_y)]
            new_boxes.append(new_box)
    this_boxes.extend(new_boxes)
    return this_boxes


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

    print(f"ORIG LEN BOXES = {len(boxes)}")
    new_boxes = jitter_boxes(boxes, n_boxes, img, std)
    boxes_arr, confidences_arr = to_box_multi(new_boxes, shape)

    # TODO: Number of boxes decreases during to_box_multi. Fix this
    print(f"LEN BOXES = {len(boxes)}")
    print(f"LEN NEW_BOXES = {len(new_boxes)}")
    print(f"LEN BOXES ARR = {boxes_arr.shape}")

    if show:
        draw(boxes, img)
    return boxes_arr, confidences_arr, img, shape

# TODO: MIGRATE TO OPENCV


def apply_nms():
    logger.debug("beginning example")
    boxes, confs, img, shape = get_data(n_boxes=3, std=0.03, show=False)
    print(f"STARTING WITH {len(boxes)} boxes")
    # draw(to_list_multi(boxes, np.asarray(img)[:, :, 0].shape), img)

    selector = RandomSelector()
    metric = DefaultIntersectionOverTheUnion()

    nms = DefaultNonMaximumSuppression(metric_threshold=0.1, selector=selector, metric=metric)
    # pboxes, pconfs = nms.transform(boxes, confs)
    # boxes = boxes[confs > 0.5, :, :]
    ixs = nms.transform(boxes)

    try:
        pboxes = boxes[ixs, :, :]
    except Exception as e:
        print(ixs)
        raise e


    print(len(pboxes))
    bboxes = to_list_multi(np.asarray(pboxes), shape)
    # draw_cartesian(bboxes, img)
    draw(bboxes, img)
# get_data(n_boxes=3, std=0.03, show=True)
apply_nms()