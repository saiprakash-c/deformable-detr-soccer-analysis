# About the Project

## Overview

This project focuses on fine-tuning the Deformable DETR (Deformable Detection Transformer) model to detect players, referees, and the ball in soccer games using bounding boxes. The dataset used for this task is the SoccerNet dataset, which contains annotated images of soccer matches.

## Objectives

- **Fine-tune Deformable DETR**: Adapt the Deformable DETR model to accurately detect and classify objects specific to soccer games.
- **Detect Multiple Classes**: Identify and classify different entities such as players, referees, and the ball.
- **Utilize SoccerNet Dataset**: Leverage the SoccerNet dataset for training and evaluation.

## Dataset

The SoccerNet dataset is a comprehensive collection of annotated soccer match images. It includes bounding box annotations for various entities such as players, referees, and the ball. The dataset is structured to facilitate training and evaluation of object detection models.

## Methodology

1. **Data Preprocessing**: Prepare the SoccerNet dataset by organizing images and annotations.
2. **Model Fine-tuning**: Fine-tune the Deformable DETR model using the preprocessed dataset.
3. **Evaluation**: Assess the performance of the fine-tuned model on a validation set.

## Results

The fine-tuned Deformable DETR model is expected to achieve high accuracy in detecting and classifying players, referees, and the ball in soccer match images. Detailed results and performance metrics will be provided upon completion of the training and evaluation phases.

## Conclusion

This project demonstrates the effectiveness of the Deformable DETR model in the context of sports analytics, specifically for detecting key entities in soccer matches. The fine-tuned model can be utilized for various applications such as game analysis, player tracking, and automated highlight generation.

## References

- [Deformable DETR](https://arxiv.org/abs/2010.04159)
- [SoccerNet Dataset](https://www.soccer-net.org/)

## TODO List

- [x] Parse the SoccerNet dataset and extract images and annotations.
- [x] Implement the `SoccerNetDataset` class to handle data loading and preprocessing.
- [x] Set up the Deformable DETR model from hugging face for object detection.
- [ ] Fine-tune the Deformable DETR model using the SoccerNet dataset.
    - [x] Solve the loss not decreasing issue 
    - [x] Solve the issue with metric calculation on val dataset
    - [ ] Select every 10th frame since the frames are highly correlated
- [ ] Visualize detection results on sample images.
- [ ] Document the training and evaluation process.
- [ ] Optimize the model for better performance.
