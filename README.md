# RFA Thermal Effect Prediction Neural Network

## Overview
This repository contains the implementation of a neural network designed to predict the thermal effects of Radiofrequency Ablation (RFA) on liver tumors. RFA is a medical procedure used to ablate, or destroy, liver tumors using heat generated by radiofrequency energy. Understanding the thermal impact of RFA is crucial for effective treatment planning and outcome prediction.

## Features
- 3D Encoder-decoder-based convolutional neural network (EDCNN), U-Net, and Attention U-Net architecture for accurate prediction of ablated area and temperature distribution.
- Custom loss functions that combine MSE, weighted MSE, and dice loss to enhance prediction accuracy.
- Data pre-processing and loading modules for efficient handling of RFA simulation data.
- Evaluation metrics to assess model performance, including MSE, RMSE, Dice coefficient, and Jaccard index.

## Installation
To set up the project environment:
- Clone the repository: `git clone https://github.com/iangilan/RFANet.git`

## Dataset
The dataset used in this project consists of:

-Temperature distribution data post-RFA treatment.
-Electrode location and orientation data.
-MRI scans indicating the damaged area post-treatment.

> Note: Due to patient privacy, the dataset is not publicly available in this repository.

## Usage
1. Load and preprocess the data using `data_loader.py`.
2. Train the model using `python train.py`.
3. Evaluate the model's performance on test data using `python test.py`.

## Model Architecture
- The `TemperatureEDCNN` model is a 3D EDCNN comprising of encoder and decoder blocks for feature extraction and temperature distribution prediction.
- The `TemperatureUNet` model is a 3D U-Net comprising of encoder and decoder blocks for feature extraction and temperature distribution prediction.
- The `TemperatureAttUNet` model is a 3D Attention U-Net comprising of encoder and decoder blocks for feature extraction and temperature distribution prediction. 
The architectures of all models are defined in `models.py`.

## Custom Loss Function
The model uses a combined loss function (`new_combined_loss` in `utils.py`) incorporating MSE, weighted MSE, and Dice loss to cater to the specific challenges in RFA thermal effect prediction.

## Evaluation
The model is evaluated based on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Dice Coefficient, providing a comprehensive assessment of its prediction accuracy.

## Citation
If you use this tool in your research, please cite the following paper:
- [Paper](link-to-paper)

## Contact
For any queries, please reach out to [Minwoo Shin](mjmj0210@gmail.com).
