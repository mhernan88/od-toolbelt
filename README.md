# OD-Toolbelt

![badge](https://github.com/mhernan88/od-toolbelt/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/mhernan88/od-toolbelt/branch/master/graph/badge.svg)](https://codecov.io/gh/mhernan88/od-toolbelt)
[![Requirements Status](https://requires.io/github/mhernan88/od-toolbelt/requirements.svg?branch=master)](https://requires.io/github/mhernan88/od-toolbelt/requirements/?branch=master)

**This library is still under active development and is not yet producing stable results. Use at your own risk.**

OD-Toolbelt is a suite of tools for use with image object detection models.

OD-Toolbelt takes raw bounding box predictions from the object detection model of your choosing, post-processes them,
and returns enhanced and filtered predictions. Think of it as a post-processing step you can add onto your object
detection workflow to boost performance.

Modules:
- Suppressors: Different non-maximum suppression methodologies to select from.
- Selectors: A modular way to create selection logic for image suppression.
- Metrics: A modular way to create measures of bounding box overlap.

## Concept

### Related Work
- [ASAP-NMS](https://arxiv.org/pdf/2007.09785.pdf) (arXiv:2007.09785v2 [cs.CV])

### Explanation

#### Non-Maximum Suppression Definition
In object detection machine learning models, typically several sets of overlapping 
bounding boxes are predicted for each given object in the image. Only a fraction of
these bounding boxes are valid, and the others must be removed from the final prediction.

Given the following photo...  
![Raw Image](https://od-toolbox.s3.amazonaws.com/images/raw_image.jpg)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

... we would normally expect the following predictions from an object detector trained to
detect birds.  
![Raw Image with Detections](https://od-toolbox.s3.amazonaws.com/images/raw_image_with_predictions.jpg)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

But, before non-maximum suppression, the predictions typically look something like the below.  
![Raw Image with Detections + False Positives](https://od-toolbox.s3.amazonaws.com/images/raw_image_with_predictions_and_false_positives.jpg)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

What non-maximum suppression does is take us from a raw model output that has false positives to a refined model output
with only one bounding box per object.  
![From False Positives to Only Detections](https://od-toolbox.s3.amazonaws.com/images/from_false_positives_to_filtered.JPG)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

#### Sector-Based Non-Maximum Suppression Explanation
Although non-maximum suppression (NMS) does significantly reduce the number of bounding boxes and typically provides a 
satisfactory result, it is an expensive algorithm. Greedy NMS compares each bounding box against
every other bounding box in the image. In our example, we have 22 predictions (8 correct predictions and 14 false positives). 
In order to compare each bounding box against every other bounding box, we need to perform (n^2 - n) / 2 comparisons. 
In our case, that would be (22 ^ 2 - 22) / 2 operations (where N represents the number of bounding boxes being evaluated), 
which comes out to 231 comparisons. As we scale up n, the number of comparisons becomes very large very fast.
![Raw Image with Detections + False Positives](https://od-toolbox.s3.amazonaws.com/images/raw_image_with_predictions_and_false_positives.jpg)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

One alternative is to use a "sector-based" approach (SB-NMS). With this approach, instead of comparing each image to every other
image, we divide our image up into "sectors", by recursively splitting it in half. In our example, we have decided to
bisect our image twice into four sectors. This results in M + 2NM + (M + 1) * (N/M) ^ 2 operations (where N represents
the number of bounding boxes being evaluated and M represents the number of sectors that our image has been split
into). This assumes a roughly equal distribution of images over sectors. In reality, the actual efficiency of this
algorithm will be somewhere between slightly slower than Greedy NMS (in worst case scenarios) and up to around 80%
faster than Greedy-NMS in more optimistic scenarios.
![Raw Image with Detections, False Positives, and Sectors](https://od-toolbox.s3.amazonaws.com/images/raw_image_with_predictions_and_false_positives_and_sectors.jpg)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

Note that in our simplified example, with 22 total bounding boxes, Greedy NMS performs fewer operations than SB-NMS 
and is more efficient. SB-NMS becomes more (theoretically) efficient than Greedy NMS at about 50 total bounding boxes
being evaluated (which is far lower than what object detectors typically predict).

In this example, we can see that Greedy NMS makes 231 comparisons. SB-NMS makes 202 comparisons. Although SB-NMS makes
fewer comparisons, it does have some up-front costs (e.g. creating sectors, assigning boxes to sectors), which
makes it worthwhile only after 50 bounding boxes in its current state.

**Theoretical Number of Operations at Different Numbers of Bounding Boxes**  

| Number of Bounding Boxes | Operations Greedy-NMS  | Operations SB-NMS  | Approximate Reduction in Operations  |
|--------------------------|------------------------|--------------------|--------------------------------------|
| 50                       | 1,225                  | 1,160              | 0%                                   |
| 100                      | 4,950                  | 3,015              | 40%                                  |
| 500                      | 124,750                | 32,618             | 74%                                  |
| 1,000                    | 499,500                | 96,259             | 80%                                  |

**Actual Performance Comparisons at Different Numbers of Bounding Boxes**  
To be completed...

We then identify which bounding boxes are overlapping with boundaries. 
![Raw Images with Detections, False Positives, Sectors, and Detections on Boundaries1](https://od-toolbox.s3.amazonaws.com/images/raw_image_with_predictions_and_false_positives_and_sectors_highlighted1.JPG)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

Additionally, we also identify any bounding boxes that overlap with any of our boundary-intersecting bounding boxes.
Our first set of selected bounding boxes is picked from these boxes using Greedy NMS. After this round of selecting
bounding boxes, the candidate boxes are then removed from consideration from downsteram comparisons. This guarantees
that all of the remaining boxes will fall entirely within a single sector.

*Note, that the above means SB-NMS prioritizes bounding boxes that fall on boundaries.*
![Raw Images with Detections, False Positives, Sectors, and Detections on Boundaries2](https://od-toolbox.s3.amazonaws.com/images/raw_image_with_predictions_and_false_positives_and_sectors_highlighted2.JPG)  
*[Feb 19, 2015 - morning sunrise, 02](https://www.flickr.com/photos/72098626@N00/16043767804) by [Ed Yourdon](https://www.flickr.com/photos/72098626@N00) is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/?ref=ccsearch&atype=rich)*

Next, we iterate over each sector and perform Greedy NMS on the bounding boxes in each sector. We append the results
of each iteration of Greedy NMS to our existing set of selected boxes. At the end of the entire process, we end up with
a full set of selected bounding boxes.

## Getting started
**Installation:**
```shell script
python setup.py install
```

**Quick Start:**

The first step is to get some bounding box predictions to process. All bounding boxes
 must be measured in relative coordinates (i.e. scaled from 0 to 1). In this example, 
 we will be creating four bounding boxes. Note that all labels have already been encoded.
```python
import numpy as np
boxes = np.array((
    ((0.190146, 0.241815), (0.405093, 0.409226)),
    ((0.595238, 0.074405), (0.892857, 0.262897)),
    ((0.644841, 0.520833), (0.975529, 0.725446)),
    ((0.941701, 0.991324), (0.110437, 0.241192))
))

confidences = np.array((0.65, 0.89, 0.70, 0.14))

labels = np.array((0, 1, 1, 1))
```

In this example, we'll be simulating having multiple ("n_boxes") overlapping bounding boxes by
jittering the coordinates of our three ground truth boxes.
```python
n_boxes = 5  # The number of boxes to simulate per original box.
std = 0.005  # The standard deviation for normal random sampling.
new_boxes = []
new_confidences = []
new_labels = []
for i in range(n_boxes):
    for j in range(boxes.shape[0]):
        print(f"Adding new box {i} around existing box {j}")
        new_box_pt1_x = boxes[j, 0, 0] + np.random.normal(loc=0, scale=std)
        new_box_pt1_y = boxes[j, 0, 1] + np.random.normal(loc=0, scale=std)
        new_box_pt2_x = boxes[j, 1, 0] + np.random.normal(loc=0, scale=std)
        new_box_pt2_y = boxes[j, 1, 1] + np.random.normal(loc=0, scale=std)
        new_box = np.array(((new_box_pt1_x, new_box_pt1_y), (new_box_pt2_x, new_box_pt2_y)), dtype=np.float64)
        new_boxes.append(new_box)
        new_confidence = confidences[j] + np.random.uniform(-0.1, 0.1)
        new_confidences.append(new_confidence)
        new_labels.append(labels[j])

new_boxes.append(boxes)
boxes = np.concatenate(new_boxes, axis = 0)
```

In most cases, we will want to filter out bounding boxes below our confidence threshold before applying our
non-maximum suppression algorithm. Here, we are filtering out the bounding boxes below 50% confidence.
```python
confidence_threshold = 0.5
confidences = confidences[confidences > confidence_threshold]
labels = labels[confidences > confidence_threshold]
boxes = boxes[confidences > confidence_threshold]
```

Now that we have a set of boxes to run out non-maximum suppression algorithm on, we can create instances of
our metric and selector. We can then pass those onto our suppression algorithm, along with our iou_threshold value.
Once we have an instance of our suppression algorithm, we run the transform() method to apply non-maximum suppression.
```python
import od_toolbelt as od

# All pairs of bounding boxes with intersection over the union of 0.25 or less will be considered separate boxes.
METRIC_THRESHOLD = 0.25

metric = od.nms.metrics.DefaultIntersectionOverTheUnion(threshold=METRIC_THRESHOLD, direction="lte")
selector = od.nms.selection.RandomSelector()

nms = od.nms.suppression.CartesianProductSuppression(
    metric=metric, selector=selector
)

data_payload = od.BoundingBoxArray(
    bounding_boxes=boxes,
    confidences=confidences,
    labels=labels,
)

resulting_payload = nms.transform(data_payload)
```

Alternatively, we can use other non-maximum suppression algorithms, such as the SectorSuppression algorithm. We can
runt he same transform() method.
```python
sector_divisions = 1
nms = od.nms.suppression.SectorSuppression(
    metric=metric, selector=selector, sector_divisions=sector_divisions
)

resulting_payload = nms.transform(data_payload)
```

Also, note that the metric and selector can be changed out for others (or use-defined metrics and selectors).

## Technical Specifications
### Suppressors
Suppressors are our non-maximum suppression algorithms. They are what developer creates and runs in order to apply
non-maximum suppression.

They are inherited off of a base class. To use them, the developer should create an instance of a suppressor class. 

**Individual Photo:**  
After creating an instance of the class, the developer simply needs to run the transform() method in order to apply
non-maximum suppression to all bounding boxes and associated confidence values in a given photo.

**Burst Mode:**  
After creating an instance of the class, the developer can run the burst() method in order to apply non-maximum
suppression to a "burst" of photos. This effectively concatenates all arrays of bounding boxes into a single
contiguous array. The bounding boxes in this array are all evaluated concurrently. This, in most cases, leads to
a higher precision, but can also lead to lower recall depending on the upstream object detection model.

This is intended to be run on bounding box predictions of multiple photos in sequence. This is intended to allow
non-maximum suppression algorithms to have multiple sets of predictions to pull bounding boxes from in case one
photo is missing a bounding box that is in other similar photos.

Note that passing a single photo to burst() is effectively the same as running transform() on that photo.

**Batch Mode:**  
After creating an instance of the class, the developer can run the batch() method in order to apply non-maximum
suppression to a "batch" of "bursts". This simply loops the burst() method over multiple different sets of photos.

### Metrics
Metrics are how we measure the degree to which two bounding boxes overlap. Intersection over the union is one such
metric, but other metrics can be used as well. To use them, the developer should create an instance of a metric class.

After creating an instance of a metric class, the developer simply needs to pass two bounding boxes to the compute()
method. A float metric value is returned.

### Selectors
Selectors are how non-maximum suppression algorithms select a single bounding box from many. Random selection is one
such selector, but other selectors can be used as well. To use them, the developer should create an instance of a
selector class.

After creating an instance of a selector class, the developer simply needs to pass a list of bounding box identifiers
to the select() method. A single identifier is returned.

## Roadmap
1. Python code optimizations.
2. Create 'recipes' (wrappers for simple use cases like iou-based greedy non-maximum suppression).
3. Add additional selectors (e.g. first selector, average selector, median selector, etc.).
4. Add additional metrics (e.g. IOU^2, etc.).
5. Complete refactorization (version 0.1).
6. Create PyPi package.
7. Convert codebase to optimized Cython.
8. Add soft-NMS as an option.
9. Integration with object detection frameworks (e.g. Tensorflow, etc.).
10. Integration with object detection services (e.g. AWS SageMaker, Azure Custom Vision, etc.).