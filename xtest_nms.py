import logging
import rootpath
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from nms.suppressors.default import NonMaximumSuppression
from box_selectors.random_selector import random_selector

logger = logging.getLogger("nms_example")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup():
    # return Path(rootpath.detect(), "tests", "base/tests/test_data")
    return Path("C:/dev/detection-enhancer/base/tests/test_data")


def to_box(x, shape):
    return np.array(((x[0][1] / shape[0], x[0][0] / shape[1]), (x[1][1] / shape[0], x[1][0] / shape[1])))


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

    boxes = [
        [(575, 975), (1225, 1650)],
        [(1800, 300), (2700, 1060)],
        [(1950, 2100), (2950, 2925)]
    ]

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

    boxes_arr = np.zeros((len(new_boxes), 2, 2), dtype=np.float64)
    confidences_arr = np.zeros(len(new_boxes), dtype=np.float64)
    for i, this_box in enumerate(new_boxes):
        boxes_arr[i, :, :] = to_box(this_box, np.asarray(img).shape)
        confidences_arr[i] = np.random.uniform(0, 1)

    if show:
        img_draw = ImageDraw.Draw(img)
        for this_box in boxes:
            img_draw.rectangle(this_box, outline="red", width=5)
        img.show()
    return boxes_arr, confidences_arr


def apply_nms():
    logger.debug("beginning example")
    boxes, confidences = get_data(n_boxes=3, std=0.03, show=False)
    nms = NonMaximumSuppression(iou_threshold=0.01, selection_func=random_selector, logger=logger, confidence_threshold=0.5, selection_kwargs=None, exact=False)
    result = nms.transform(boxes, confidences)
    print(result)



# get_data(n_boxes=3, std=0.03, show=True)
apply_nms()