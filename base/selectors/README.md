# Selectors

Selectors are designed to provide the developer the freedom to "plug and play" different ways of having the 
non-maximum suppression algorithm choose between overlapping boxes. A selector is passed as an argument to the 
NonMaximumSuppression class in the enhance/nms.py file.

## Built-in selectors include:
1. random_selector: This selects a bounding box from the provided bounding boxes at random.

## Specifications for custom selectors:
Custom selectors can easily be created. They must follow a specific convention of arguments and returns values.

Required arguments (argument names are arbitrary):
1. cube1 - This is a Numpy array with shape (-1, 2, 2) with dtype numpy.float64.
2. cube2 - This is a Numpy array with shape (-1, 2, 2) with dtype numpy.float64.
3. confidences1 - This is a Numpy array with shape (-1, ) with dtype numpy.float64.
4. confidences2 - This is a Numpy array with shape (-1, ) with dtype numpy.float64.
5. kwargs - This is a dictionary of any keyword arguments you would like for your selector to accept. If there are no
keyword arguments, then this can be an unused argument.

Required returns:
1. A Numpy array with shape (-1, 2, 2) with dtype numpy.float64.
2. A Numpy array with shape (-1, ) with dtype numpy.float64.