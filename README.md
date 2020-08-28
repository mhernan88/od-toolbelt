# OD-Toolbelt

![badge](https://github.com/mhernan88/od-toolbelt/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/mhernan88/od-toolbelt/branch/master/graph/badge.svg)](https://codecov.io/gh/mhernan88/od-toolbelt)
[![Requirements Status](https://requires.io/github/mhernan88/od-toolbelt/requirements.svg?branch=master)](https://requires.io/github/mhernan88/od-toolbelt/requirements/?branch=master)

---

OD-Toolbelt is a suite of tools for use with image object detection models.

OD-Toolbelt takes raw bounding box predictions from the object detection model of your choosing, post-processes them,
and returns enhanced and filtered predictions. Think of it as a post-processing step you can add onto your object
detection workflow to boost performance.

Modules:
- Multi-image non-maximum suppression: This module applies the well known non-maximum suppression on a series of images as
opposed to an individual image. This yields a higher overall precision than what could be achieved with a single photo.
- More to be added with additional enhancements.

---
## Multi-Image Non-maximum Suppression:
As object detection models become more commonplace in industry, the challenges of applying a digital model in the
physical world become more present.

One such challenge is inconsistency of the performance of object detection algorithms due to inconsistencies in camera
position, lighting, and other factors. This can lead to several photographs of the same exact object from the same
perspective yielding different predictions.

Multi-image non-maximum suppression takes a "burst" of image predictions, and processes those as a group, yielding a
higher (or, at minimum, equal) precision than any individual photo.

One other enhancement this module applies is modularizing the selection methodology of the non-maximum suppression
algorithm. This allows developers to pick their own methodology for selecting between overlapping bounding boxes.
See the "selectors" package module.

## Geometry "structures":
This package relies heavily on the speed and flexibility of Numpy. As such, custom classes are generally (with 
exceptions) avoided in favor of Numpy arrays.

With the above in mind, there are three primary "structures" used in this package:
1. Point: This is a Numpy array with shape (2, ) and data type np.float64. See the assert_point() function in
geometry/point.py for a comprehensive set of requirements around what constitutes a "point". A point is just a set of
y, x coordinates (Note: Points expected *relative* coordinates between 0 and 1, inclusive).
2. Box: This is a Numpy array with shape (2, 2) and data type np.float64. See the assert_box() function in 
geometry/box.py for a comprehensive set of requirements around what constitutes a "box". A box
is just two points (an "upper-left" point and a "bottom-right" point).
3. Cube: This is a Numpy array with shape (-1, 2, 2) and data type np.float64. See the assert_cube() function in
geometry/cube.py for a comprehensive set of requirements around what constitutes a "cube". A cube is just a set of
boxes.

These "structures" are referenced throughout the documentation.