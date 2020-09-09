# Metrics

Metrics are measures of overlap in an image. One such metric is non-maximum suppression.
Metrics are designed to provide the developer with the freedom to "plug-and-play" different
metrics to measure overlap between boxes.

## Built-in metrics include:
1. DefaultIntersectionOverTheUnion: This measures the intersection over the union between two
bounding boxes.

## Specifications for custom metrics:
Custom metrics can easily be created. They should follow the specifications laid out in the base class
(metrics.base.Metric).