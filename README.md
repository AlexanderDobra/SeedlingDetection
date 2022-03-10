# Improving Seedling Detection by Predicting Height Information

## Description

Improving Seedling Detection of conifer seedlings along seismic lines with addition of predicted height information to drone images.

CNN Models used:
- Object Detection:
  - Vanilla Model (used for comparison to Pre-RPN):
    - Basic Faster R-CNN with ResNet50-FPN-Backbone
    - **Inputs/Outputs**: Images ⮕ Detections
  - Pre-Region Proprosal Network (Pre-RPN):
    - Modified Vanilla Model derived from [here](https://github.com/JasonJooste/seedlings_height)
    - **Inputs/Outputs**: Images + Depth Maps ⮕ Detections
- Monocular Depth Estimation:
  - MDE Network:
    - Modified ResNet101-FPN
    - **Input/Outputs**: Images ⮕ Depth Prediction Maps

<img src="images/bachelor%20depth%20prediction.PNG" width="500">

<img src="images/bachelor%20predictions.PNG" width="500">

# Results

<img src="images/last%20top%20table.PNG" width="500">
<img src="images/last%20first%20table.PNG" width="500">
<img src="images/last%20middle%20table.PNG" width="500">
<img src="images/last%20last%20table.PNG" width="500">



