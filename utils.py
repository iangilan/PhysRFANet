import torch
import torch.nn as nn

def weighted_mse_loss(pred, target, weight=1.0, threshold=50.0):
    """
    Custom weighted MSE loss which applies a weight to errors on target values above a threshold.

    Args:
    - pred (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - weight (float): Weight to apply to the loss for values above the threshold.
    - threshold (float): Threshold above which the weight is applied.

    Returns:
    - torch.Tensor: Computed weighted MSE loss.
    """
    mse_loss = nn.MSELoss()(pred, target)
    mask = target > threshold
    weighted_loss = mse_loss * (weight * mask.float() + (1.0 - mask.float()))
    return torch.mean(weighted_loss)

def dice_loss(pred, target, threshold=50):
    """
    Dice loss computation for binary classification problems.

    Args:
    - pred (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - threshold (float): Threshold for binarization of predictions and targets.

    Returns:
    - torch.Tensor: Computed Dice loss.
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    dice_coef = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    return 1 - dice_coef

def new_combined_loss(pred, target, alpha, beta, gamma):
    """
    Combined loss function using a mix of MSE, weighted MSE, and Dice loss.

    Args:
    - pred (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - alpha (float): Weight for MSE loss.
    - beta (float): Weight for weighted MSE loss.
    - gamma (float): Weight for Dice loss.

    Returns:
    - torch.Tensor: Computed combined loss.
    """
    mse = nn.MSELoss()(pred, target)
    wmse = weighted_mse_loss(pred, target)
    dice = dice_loss(pred, target)
    return alpha * mse + beta * wmse + gamma * dice

def mse_3d(pred, label):
    """
    Calculate the Mean Squared Error (MSE) for a 3D array.
    :param pred: 3D Numpy array of predictions.
    :param label: 3D Numpy array of labels.
    :return: MSE value for the 3D array.
    """
    mse = np.mean(np.square(pred - label))
    return mse

def rmse_3d(pred, label):
    """
    Calculate the Root Mean Squared Error (RMSE) for each element in the prediction and label arrays.
    :param pred: Numpy array of predictions.
    :param label: Numpy array of labels.
    :return: Numpy array of RMSE values for each element.
    """
    rmse = np.sqrt(np.mean(np.square(pred - label)))
    return rmse

def mae_3d(pred, label):
    """
    Calculate the Mean Absolute Error (MAE) for each element in the prediction and label arrays.
    :param pred: Numpy array of predictions.
    :param label: Numpy array of labels.
    :return: Numpy array of MAE values for each element.
    """
    # Element-wise absolute differences
    mae = np.mean(np.abs(pred - label))
    return mae

def dice_score(pred, target, threshold):
    """
    Compute Dice Loss for values greater than the threshold.

    Args:
    - pred (torch.Tensor): the network's raw output.
    - target (torch.Tensor): the ground truth.
    - threshold (float): the threshold value to consider in the loss computation. Default is 50.

    Returns:
    - dice_loss_val (torch.Tensor): computed Dice loss.
    """

    # Binarize tensors
    pred = (pred > threshold).astype(int)
    target = (target > threshold).astype(int)

    # Compute Dice coefficient
    intersection = (pred * target).sum()
    dice_coef = (2. * intersection) / (pred.sum() + target.sum())  # Adding a small epsilon to avoid division by zero

    # Return Dice loss
    return dice_coef


def calculate_metrics(all_predictions, all_labels, folder_name):
    """
    Calculate MSE, RMSE, MAE, and Dice scores for each pair of 3D arrays in the given lists.
    Save the results to text files and print the mean of each metric.
    :param all_predictions: List of 3D Numpy arrays (predictions).
    :param all_labels: List of 3D Numpy arrays (labels).
    :param folder_name: Directory to save the result files.
    """
    # Extract 3D arrays from the lists
    pred = [item[:, :, :] for item in all_predictions]
    label = [item[:, :, :] for item in all_labels]

    # Calculate and save MSE
    mse_per_item = [mse_3d(p, l) for p, l in zip(pred, label)]
    mse_nparray = np.array(mse_per_item)
    np.savetxt(f"{folder_name}/mse.txt", mse_nparray, fmt='%.4f')

    # Calculate and save RMSE
    rmse_per_item = [rmse_3d(p, l) for p, l in zip(pred, label)]
    rmse_nparray = np.array(rmse_per_item)
    np.savetxt(f"{folder_name}/rmse.txt", rmse_nparray, fmt='%.4f')

    # Calculate and save MAE
    mae_per_item = [mae_3d(p, l) for p, l in zip(pred, label)]
    mae_nparray = np.array(mae_per_item)
    np.savetxt(f"{folder_name}/mae.txt", mae_nparray, fmt='%.4f')

    # Calculate and save Dice scores
    # Assuming dice_score function takes a threshold as the third argument
    dice_scores_40 = [dice_score(p, l, 40) for p, l in zip(pred, label)]
    dice_scores_nparray_40 = np.array(dice_scores_40).reshape(-1, 1)
    np.savetxt(f"{folder_name}/dice_40.txt", dice_scores_nparray_40, fmt='%.4f')

    dice_scores_50 = [dice_score(p, l, 50) for p, l in zip(pred, label)]
    dice_scores_nparray_50 = np.array(dice_scores_50).reshape(-1, 1)
    np.savetxt(f"{folder_name}/dice_50.txt", dice_scores_nparray_50, fmt='%.4f')

    mse = np.mean(mse_nparray)
    rmse = np.mean(rmse_nparray)
    mae = np.mean(mae_nparray)
    dice40 = np.mean(dice_scores_nparray_40)
    dice50 = np.mean(dice_scores_nparray_50)
    # Print the mean of each metric
    print('mse: '   , mse   ) 
    print('rmse: '  , rmse  ) 
    print('mae: '   , mae   ) 
    print('dice40: ', dice40)
    print('dice50: ', dice50)

    file_path = f"{folder_name}/results.txt"
    with open(file_path, "w") as file:
        file.write(f"Mean Squared Error: {mse}\n")
        file.write(f"Root Mean Squared Error: {rmse}\n")
        file.write(f"Mean Absolute Error: {mae}\n")
        file.write(f"Dice Coefficient>40: {dice40}\n")
        file.write(f"Dice Coefficient>50: {dice50}\n")

def save_plot(all_predictions, all_labels, all_Ninput, folder_name):
    """
    Save plots of slices from predictions, labels, and Ninput arrays.
    :param all_predictions: List of 3D numpy arrays (predictions).
    :param all_labels: List of 3D numpy arrays (labels).
    :param all_Ninput: List of 3D numpy arrays (Ninput data).
    :param folder_name: Directory to save the plots.
    """
    # Extract slices from the 3D arrays
    center = 21
    z_slices = [item[center, :, :] for item in all_predictions]
    y_slices = [item[:, center, :] for item in all_predictions]
    x_slices = [item[:, :, center] for item in all_predictions]

    zz_slices = [item[center, :, :] for item in all_labels]
    yy_slices = [item[:, center, :] for item in all_labels]
    xx_slices = [item[:, :, center] for item in all_labels]

    zzz_slices = [item[center, :, :] for item in all_Ninput]
    yyy_slices = [item[:, center, :] for item in all_Ninput]
    xxx_slices = [item[:, :, center] for item in all_Ninput]

    sample_count = len(all_predictions)
    
    for i in range(sample_count):
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        
        # Plot Z, Y, X slices for predictions, labels, and Ninput
        for ax, slice_data, title in zip(axes, [z_slices[i], zz_slices[i], y_slices[i], yy_slices[i], x_slices[i], xx_slices[i]], 
                                         ["Z-slice (pred)", "Z-slice (gt)", "Y-slice (pred)", "Y-slice (gt)", "X-slice (pred)", "X-slice (gt)"]):
            ax.imshow(slice_data, cmap='jet')
            ax.set_title(f'Sample {i+1}: {title}')
            ax.axis('off')

        plt.savefig(f"{folder_name}/sample_{i+1}.png")
        plt.close() 
