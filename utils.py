import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

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

def dice_loss(pred, target, epsilon=1e-6):
    """
    Compute the Dice Loss.

    Args:
    - pred (torch.Tensor): the network's prediction output, with values in [0, 1].
    - target (torch.Tensor): the ground truth targets, with values in [0, 1].
    - epsilon (float): a small value to avoid division by zero.

    Returns:
    - dice_loss (torch.Tensor): the computed Dice loss.
    """
    pred = torch.sigmoid(pred)
    # Flatten the tensors to simplify the computation
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Compute the intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Compute Dice coefficient and Dice loss
    dice_coef = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice_coef

    return dice_loss

def dice_loss50(pred, target, threshold=50):
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
    dice = dice_loss50(pred, target)
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

def dice_score(pred, target, epsilon=1e-6): # this metric is used for the Damage data
    """
    Compute the Dice Loss.

    Args:
    - pred (torch.Tensor): the network's prediction output, with values in [0, 1].
    - target (torch.Tensor): the ground truth targets, with values in [0, 1].
    - epsilon (float): a small value to avoid division by zero.

    Returns:
    - dice_loss (torch.Tensor): the computed Dice loss.
    """
    # Convert numpy arrays to PyTorch tensors if necessary
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)    
    
    pred = torch.sigmoid(pred) # Needed for smoothing the boundary (binary data)
    # Flatten the tensors to simplify the computation
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Compute the intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Compute Dice coefficient and Dice loss
    dice_coef = (2. * intersection + epsilon) / (union + epsilon)

    return dice_coef

def dice_score_threshold(pred, target, threshold): # this metric is used for the Temp data
    epsilon=1e-6
    # Convert numpy arrays to PyTorch tensors if necessary
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)    
    pred = (pred > threshold).float()
    target = (target > threshold).float()
        
    # Flatten the tensors to simplify the computation
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Compute the intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Compute Dice coefficient and Dice loss
    dice_coef = (2. * intersection + epsilon) / (union + epsilon)

    return dice_coef

def jaccard_score(pred, target, epsilon=1e-6):
    """
    Calculate the Jaccard score (Intersection over Union) for binary arrays.

    :param pred: Numpy array of predictions (binary).
    :param target: Numpy array of true targets (binary).
    :return: Jaccard score as a float.
    """
    # Convert numpy arrays to PyTorch tensors if necessary
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)    
    
    pred = torch.sigmoid(pred)    
    # Binarize tensors
    pred = pred.view(-1)
    target = target.view(-1)      

    # Intersection and Union
    intersection = (pred * target).sum()
    total = (pred + target).sum()
    union = total - intersection

    # Jaccard score calculation
    jaccard = (intersection + epsilon) / (union + epsilon)

    return jaccard

def hausdorff_distance(pred, target, threshold = 0):
    """
    Calculate the Hausdorff distance between two binary arrays.

    :param pred: Numpy array of predictions (binary).
    :param target: Numpy array of true targets (binary).
    :return: Hausdorff distance as a float.
    """
    pred = np.where(pred > threshold, 1, 0)
    # Extract the indices of the non-zero points
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    # Compute the directed Hausdorff distances and take the maximum
    forward_hausdorff = directed_hausdorff(pred_points, target_points)[0]
    reverse_hausdorff = directed_hausdorff(target_points, pred_points)[0]

    hausdorff_dist = max(forward_hausdorff, reverse_hausdorff)
    
    return hausdorff_dist


def calculate_metrics_Temp(all_predictions, all_labels, folder_name):
    """
    Calculate MSE, RMSE, MAE, and Dice scores for each pair of 3D arrays in the given lists.
    Save the results to text files and print the mean of each metric.
    :param all_predictions: List of 3D Numpy arrays (predictions).
    :param all_labels: List of 3D Numpy arrays (labels).
    :param folder_name: Directory to save the result files.
    """
    # Extract 3D arrays from the lists
    pred  = [item[:, :, :] for item in all_predictions]
    label = [item[:, :, :] for item in all_labels]

    # Calculate and save MSE
    mse_per_item = [mse_3d(p, l) for p, l in zip(pred, label)]
    mse_nparray  = np.array(mse_per_item)
    np.savetxt(f"{folder_name}/mse.txt", mse_nparray, fmt='%.4f')

    # Calculate and save RMSE
    rmse_per_item = [rmse_3d(p, l) for p, l in zip(pred, label)]
    rmse_nparray  = np.array(rmse_per_item)
    np.savetxt(f"{folder_name}/rmse.txt", rmse_nparray, fmt='%.4f')

    # Calculate and save MAE
    mae_per_item = [mae_3d(p, l) for p, l in zip(pred, label)]
    mae_nparray  = np.array(mae_per_item)
    np.savetxt(f"{folder_name}/mae.txt", mae_nparray, fmt='%.4f')

    # Calculate and save Dice scores
    dice_scores_30 = [dice_score_threshold(p, l, 30) for p, l in zip(pred, label)]
    dice_scores_nparray_30 = np.array(dice_scores_30).reshape(-1, 1)
    np.savetxt(f"{folder_name}/dice_30.txt", dice_scores_nparray_30, fmt='%.4f')  

    dice_scores_40 = [dice_score_threshold(p, l, 40) for p, l in zip(pred, label)]
    dice_scores_nparray_40 = np.array(dice_scores_40).reshape(-1, 1)
    np.savetxt(f"{folder_name}/dice_40.txt", dice_scores_nparray_40, fmt='%.4f')

    dice_scores_50 = [dice_score_threshold(p, l, 50) for p, l in zip(pred, label)]
    dice_scores_nparray_50 = np.array(dice_scores_50).reshape(-1, 1)
    np.savetxt(f"{folder_name}/dice_50.txt", dice_scores_nparray_50, fmt='%.4f')

    mse    = np.mean(mse_nparray)
    rmse   = np.mean(rmse_nparray)
    mae    = np.mean(mae_nparray)
    dice30 = np.mean(dice_scores_nparray_30)
    dice40 = np.mean(dice_scores_nparray_40)
    dice50 = np.mean(dice_scores_nparray_50)
    
    # Print the mean of each metric
    print(f'mse:    {mse:.4f}') 
    print(f'rmse:   {rmse:.4f}') 
    print(f'mae:    {mae:.4f}') 
    print(f'dice30: {dice30:.4f}')
    print(f'dice40: {dice40:.4f}')
    print(f'dice50: {dice50:.4f}')
    
    file_path = f"{folder_name}/results.txt"
    with open(file_path, "w") as file:
        file.write(f"Mean Squared Error: {mse}\n")
        file.write(f"Root Mean Squared Error: {rmse}\n")
        file.write(f"Mean Absolute Error: {mae}\n")
        file.write(f"Dice Coefficient>30: {dice30}\n")
        file.write(f"Dice Coefficient>40: {dice40}\n")
        file.write(f"Dice Coefficient>50: {dice50}\n")

def calculate_metrics_Dmg(all_predictions, all_targets, folder_name):
    """
    Calculate MSE, RMSE, MAE, and Dice scores for each pair of 3D arrays in the given lists.
    Save the results to text files and print the mean of each metric.
    :param all_predictions: List of 3D Numpy arrays (predictions).
    :param all_targets: List of 3D Numpy arrays (targets).
    :param folder_name: Directory to save the result files.
    """        
    
    # Extract 3D arrays from the lists
    pred   = [item[:, :, :] for item in all_predictions]
    target = [item[:, :, :] for item in all_targets]

    # Calculate and save Dice scores
    dice_scores = [dice_score(p, l) for p, l in zip(pred, target)]
    dice_scores_nparray = np.array(dice_scores).reshape(-1, 1)
    np.savetxt(f"{folder_name}/dice.txt", dice_scores_nparray, fmt='%.4f')

    jaccard_scores = [jaccard_score(p, l) for p, l in zip(pred, target)]
    jaccard_scores_nparray = np.array(jaccard_scores).reshape(-1, 1)
    np.savetxt(f"{folder_name}/jaccard.txt", jaccard_scores_nparray, fmt='%.4f')

    hausdorff = [hausdorff_distance(p, l) for p, l in zip(pred, target)]
    hausdorff_nparray = np.array(hausdorff).reshape(-1, 1)
    np.savetxt(f"{folder_name}/hausdorff.txt", hausdorff_nparray, fmt='%.4f')

    dice = np.mean(dice_scores_nparray)
    jaccard = np.mean(jaccard_scores_nparray)
    hausdorff = np.mean(hausdorff_nparray)
    
    # Print the mean of each metric
    print(f'dice:      {dice:.4f}')
    print(f'jaccard:   {jaccard:.4f}')
    print(f'hausdorff: {hausdorff:.4f}')

    file_path = f"{folder_name}/results.txt"
    with open(file_path, "w") as file:
        file.write(f"Dice: {dice}\n")
        file.write(f"Jaccard: {jaccard}\n")
        file.write(f"Hausdorff: {hausdorff}\n")

def save_plot_Temp(all_predictions, all_labels, all_Ninput, folder_name):
    """
    Save plots of slices from predictions, labels, and Ninput arrays.
    :param all_predictions: List of 3D numpy arrays (predictions).
    :param all_labels: List of 3D numpy arrays (labels).
    :param all_Ninput: List of 3D numpy arrays (Ninput data).
    :param folder_name: Directory to save the plots.
    """
    # Extract slices from the 3D arrays
    center = 21
    x_slices = [item[center, :, :] for item in all_predictions]
    y_slices = [item[:, center, :] for item in all_predictions]
    z_slices = [item[:, :, center] for item in all_predictions]

    xx_slices = [item[center, :, :] for item in all_labels]
    yy_slices = [item[:, center, :] for item in all_labels]
    zz_slices = [item[:, :, center] for item in all_labels]

    xxx_slices = [item[center, :, :] for item in all_Ninput]
    yyy_slices = [item[:, center, :] for item in all_Ninput]
    zzz_slices = [item[:, :, center] for item in all_Ninput]

    sample_count = len(all_predictions)
    
    for i in range(sample_count):
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        
        # Plot Z, Y, X slices for predictions, labels, and Ninput
        for ax, slice_data, title in zip(axes, [z_slices[i], zz_slices[i], x_slices[i], xx_slices[i], y_slices[i], yy_slices[i]], 
                                         ["Z-slice (pred)", "Z-slice (gt)", "X-slice (pred)", "X-slice (gt)", "Y-slice (pred)", "Y-slice (gt)"]):
            ax.imshow(slice_data, cmap='jet')
            ax.set_title(f'Sample {i+1}: {title}')
            ax.axis('off')

        plt.savefig(f"{folder_name}/sample_{i+1}.png")
        plt.close() 
        
def save_plot_Dmg(all_predictions, all_targets, all_Ninput, folder_name, threshold = 0):
    """
    Save plots of slices from predictions, targets, and Ninput arrays.
    :param all_predictions: List of 3D numpy arrays (predictions).
    :param all_targets: List of 3D numpy arrays (targets).
    :param all_Ninput: List of 3D numpy arrays (Ninput data).
    :param folder_name: Directory to save the plots.
    """
    
    # Binarize all_predictions
    binarized_predictions = [(item > threshold).astype(int) for item in all_predictions]
    
    # Extract slices from the 3D arrays
    center = 21
    x_slices = [item[center, :, :] for item in binarized_predictions]
    y_slices = [item[:, center, :] for item in binarized_predictions]
    z_slices = [item[:, :, center] for item in binarized_predictions]

    xx_slices = [item[center, :, :] for item in all_targets]
    yy_slices = [item[:, center, :] for item in all_targets]
    zz_slices = [item[:, :, center] for item in all_targets]

    xxx_slices = [item[center, :, :] for item in all_Ninput]
    yyy_slices = [item[:, center, :] for item in all_Ninput]
    zzz_slices = [item[:, :, center] for item in all_Ninput]

    sample_count = len(all_predictions)
    
    for i in range(sample_count):
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        
        # Plot Z, Y, X slices for predictions, targets, and Ninput
        for ax, slice_data, title in zip(axes, [z_slices[i], zz_slices[i], x_slices[i], xx_slices[i], y_slices[i], yy_slices[i]], 
                                         ["Z-slice (pred)", "Z-slice (gt)", "X-slice (pred)", "X-slice (gt)", "Y-slice (pred)", "Y-slice (gt)"]):
            ax.imshow(slice_data)
            ax.set_title(f'Sample {i+1}: {title}')
            ax.axis('off')

        plt.savefig(f"{folder_name}/sample_{i+1}.png")
        plt.close()
#=======================================================================================        
def save_plot_Temp_each(all_predictions, all_labels, all_Ninput, folder_name):
    """
    Save plots of slices from predictions, labels, and Ninput arrays.
    :param all_predictions: List of 3D numpy arrays (predictions).
    :param all_labels: List of 3D numpy arrays (labels).
    :param all_Ninput: List of 3D numpy arrays (Ninput data).
    :param folder_name: Directory to save the plots.
    """
    # Extract slices from the 3D arrays
    center = 21
    x_slices = [item[center, :, :] for item in all_predictions]
    y_slices = [item[:, center, :] for item in all_predictions]
    z_slices = [item[:, :, center] for item in all_predictions]

    xx_slices = [item[center, :, :] for item in all_labels]
    yy_slices = [item[:, center, :] for item in all_labels]
    zz_slices = [item[:, :, center] for item in all_labels]

    xxx_slices = [item[center, :, :] for item in all_Ninput]
    yyy_slices = [item[:, center, :] for item in all_Ninput]
    zzz_slices = [item[:, :, center] for item in all_Ninput]

    sample_count = len(all_predictions)
    
    for i in range(sample_count):
        # Save Z-slices
        plt.imshow(z_slices[i], cmap='jet')
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Z_pred.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(zz_slices[i], cmap='jet')
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Z_gt.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save X-slices
        plt.imshow(x_slices[i], cmap='jet')
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_X_pred.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(xx_slices[i], cmap='jet')
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_X_gt.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save Y-slices
        plt.imshow(y_slices[i], cmap='jet')
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Y_pred.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(yy_slices[i], cmap='jet')
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Y_gt.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
def save_plot_Dmg_each(all_predictions, all_targets, all_Ninput, folder_name, threshold = 0):
    """
    Save plots of slices from predictions, targets, and Ninput arrays.
    :param all_predictions: List of 3D numpy arrays (predictions).
    :param all_targets: List of 3D numpy arrays (targets).
    :param all_Ninput: List of 3D numpy arrays (Ninput data).
    :param folder_name: Directory to save the plots.
    """
    
    # Binarize all_predictions
    binarized_predictions = [(item > threshold).astype(int) for item in all_predictions]
    
    # Extract slices from the 3D arrays
    center = 21
    x_slices = [item[center, :, :] for item in binarized_predictions]
    y_slices = [item[:, center, :] for item in binarized_predictions]
    z_slices = [item[:, :, center] for item in binarized_predictions]

    xx_slices = [item[center, :, :] for item in all_targets]
    yy_slices = [item[:, center, :] for item in all_targets]
    zz_slices = [item[:, :, center] for item in all_targets]

    xxx_slices = [item[center, :, :] for item in all_Ninput]
    yyy_slices = [item[:, center, :] for item in all_Ninput]
    zzz_slices = [item[:, :, center] for item in all_Ninput]

    sample_count = len(all_predictions)
    
    for i in range(sample_count):
        # Save Z-slices
        plt.imshow(z_slices[i])
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Z_pred.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(zz_slices[i])
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Z_gt.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save X-slices
        plt.imshow(x_slices[i])
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_X_pred.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(xx_slices[i])
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_X_gt.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save Y-slices
        plt.imshow(y_slices[i])
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Y_pred.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(yy_slices[i])
        plt.axis('off')
        plt.savefig(f"{folder_name}/sample_{i+1}_Y_gt.png", bbox_inches='tight', pad_inches=0)
        plt.close()           
