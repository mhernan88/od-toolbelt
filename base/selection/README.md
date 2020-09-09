# Selectors

Selectors are designed to provide the developer the freedom to "plug and play" different 
selection algorithms choose between overlapping boxes.

## Built-in selectors include:
1. RandomSelector: This selects a bounding box from the provided bounding boxes at random.

## Specifications for custom selectors:
Custom selectors can easily be created. They should follow the specifications laid out in the base class
(selection.base.Selector).
