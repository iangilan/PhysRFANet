# PhyRFANet: Physics-Guided Neural Network for Real-Time Prediction of Thermal Effect During Radiofrequency Ablation Treatment

## Overview
This repository contains the implementation of a neural network designed to predict the thermal effects of Radiofrequency Ablation (RFA) on liver tumors. RFA is a medical procedure used to ablate, or destroy, liver tumors using heat generated by radiofrequency energy. Understanding the thermal impact of RFA is crucial for effective treatment planning and outcome prediction.

## Features
- 3D Encoder-decoder-based convolutional neural network (EDCNN), U-Net, and Attention U-Net architecture for accurate prediction of ablated area and temperature distribution.
- Custom loss functions that combine MSE, weighted MSE, and weighted Dice loss to enhance prediction accuracy.
- Data pre-processing and loading modules for efficient handling of RFA simulation data.
- Evaluation metrics to assess model performance, including MSE, RMSE, dice coefficient, and Jaccard index.

## Requirements
- python 3.9
- torch-1.11.0+cu113
- cudatoolkit-11.3.1

## Installation
To set up the project environment:
- Clone the repository: `git clone https://github.com/iangilan/RFANet.git`

## Dataset
The dataset used in this project consists of:
- Temperature distribution data post-RFA treatment (https://drive.google.com/file/d/1F7OFzfXZdc6jGWc_qIpxW5WrgBBzYh_o/view?usp=sharing)
- Ablated area data post-RFA treatment (https://drive.google.com/file/d/1CDLMCfDLaI5SfMdX6DV5EJflgjRwjj9D/view?usp=sharing)
- Electrode location and geometry data during RFA treatment (https://drive.google.com/file/d/18rzSAqrPdOKl7YipzP73VnS7oK9d_-Ua/view?usp=sharing).
- Segmented breast tumor data obtained from MR images (https://drive.google.com/file/d/1O85XRSbVJly1kMyxfzIvbwV-84xo0nxS/view?usp=sharing).
> Note: Due to patient privacy, the original breast tumor MRI dataset is not publicly available in this repository.

## Usage
1. Locate your RFA dataset in your local storage.
2. Edit config.py according to the user's need.
3. Edit the data loaders `data_loader_Temp.py` and `data_loader_Dmg.py` for temperature distribution and damaged area, respectively, to load and preprocess the data. 
4. Train the model using `python train_Temp.py` or `python train_Dmg.py`.
5. Evaluate the model's performance on test data using `python test_Temp.py` or `python test_Dmg.py`.

## Model Architecture
- The `RFACNN` model is a 3D EDCNN that consists of encoder and decoder blocks, designed for extracting features and predicting both temperature distribution and damaged (ablated) areas.
- The `RFAUNet` model is a 3D U-Net, designed for extracting features and predicting both temperature distribution and damaged (ablated) areas.
- The `RFAAttUNet` model is a 3D Attention U-Net, designed for extracting features and predicting both temperature distribution and damaged (ablated) areas.
- The architectures of all models are defined in `models.py`.

## Custom Loss Function
The model uses a combined loss function (`new_combined_loss` in `utils.py`) incorporating MSE, weighted MSE, and weighted Dice loss to cater to the specific challenges in RFA thermal effect prediction.

## Evaluation
The model is evaluated based on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Dice Coefficient, providing a comprehensive assessment of its prediction accuracy.

## Citation
If you use this tool in your research, please cite the following paper:
- [M. Shin, M. Seo, S. Cho, J. Park, J. Kwon, D. Lee, K. Yoon. "PhyRFANet: Physics-Guided Neural Network for Real-Time Prediction of Thermal Effect During Radiofrequency Ablation Treatment." *TBD*](link-to-paper)

## Contact
For any queries, please reach out to [Minwoo Shin](mjmj0210@gmail.com).
