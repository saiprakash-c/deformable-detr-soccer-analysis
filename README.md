# Deformable DETR Soccer Analysis

## Overview

This project implements and fine-tunes the Deformable DETR (DEtection TRansformer) model to detect and classify soccer game entities - players, referees, and the ball - using bounding boxes. By leveraging the SoccerNet dataset, this implementation demonstrates how advanced object detection transformers can be applied to sports analytics.

## Features

- **Multi-class Detection**: Accurately identifies players, referees, and the ball with distinct bounding boxes
- **Transformer-based Architecture**: Uses Deformable DETR, an improved version of DETR with deformable attention mechanisms
- **PyTorch Implementation**: Built on PyTorch with HuggingFace transformers integration
- **Coordinate Transformation**: Includes utility functions for converting between bbox formats (XYXY ↔ CCWH)
- **Performance Metrics**: Calculates mAP (mean Average Precision) to evaluate detection quality

## Dataset

The SoccerNet dataset provides comprehensive annotated soccer match footage, including:
- High-resolution broadcast video frames
- Precise bounding box annotations for players, referees, and the ball
- Diverse game scenarios across multiple camera angles

Our implementation samples frames at regular intervals (every 10th frame) to reduce training on highly correlated images.

## Technical Implementation

1. **Data Pipeline**: Custom `SoccerNetDataset` class handles data loading and preprocessing
2. **Model Architecture**: Deformable DETR with ResNet backbone
3. **Training Process**: Implemented in PyTorch with customized loss functions
4. **Visualization Tools**: Functions to render predictions and ground truth boxes

## Getting Started

1. Install dependencies: `pip install lightning timm transformers torchmetrics`
2. Run notebook: `jupyter notebook deformable_detr_soccer_analysis.ipynb`

## References

- [Deformable DETR Paper](https://arxiv.org/abs/2010.04159)
- [SoccerNet Dataset](https://www.soccer-net.org/)

## Development Roadmap

- [x] Parse the SoccerNet dataset and extract images and annotations
- [x] Implement the `SoccerNetDataset` class for data loading and preprocessing
- [x] Configure Deformable DETR model from HuggingFace
- [x] Solve training issues:
  - [x] Fix loss not decreasing problem
  - [x] Resolve metric calculation on validation dataset
  - [x] Sample every 10th frame to reduce redundancy
- [x] Implement visualization for detection results
- [ ] Document training parameters and evaluation process
- [ ] Optimize model performance and hyperparameters
- [ ] Explore additional applications (player tracking, tactical analysis)