import torch
from models import RFACNN, RFAUNet, RFAAttUNet
from data_loader_Dmg import DmgDataset, load_data
from torch.utils.data import Dataset, DataLoader
from utils import dice_loss, calculate_metrics_Dmg, save_plot_Dmg
import config
import utils
from config import num_epochs, batch_size, model_path_Dmg, file_paths, figure_path_Dmg, model_name, use_foreseen
import os

def get_model(choice):
    if choice == "1":
        return RFACNN()
    elif choice == "2":
        return RFAUNet()
    elif choice == "3":
        return RFAAttUNet()
    else:
        raise ValueError(f"Unknown model choice: {choice}")

def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    all_predictions = []
    all_labels = []
    all_Ninput = []

    with torch.no_grad():
        for batch in test_loader:
            Ninput_test_data_batch, MR_test_data_batch, Dmg_test_data_batch = batch

            Ninput_test_data_batch = Ninput_test_data_batch.unsqueeze(1).cuda()
            MR_test_data_batch     = MR_test_data_batch.unsqueeze(1).cuda()
            Dmg_test_data_batch    = Dmg_test_data_batch.unsqueeze(1).cuda()      

            # Forward pass
            outputs = model(Ninput_test_data_batch, MR_test_data_batch)

            # Store predictions and labels for evaluation
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(Dmg_test_data_batch.cpu().numpy())
            all_Ninput.extend(Ninput_test_data_batch.cpu().numpy()) 

            # Calculate the loss for this batch
            loss = dice_loss(outputs, Dmg_test_data_batch)
            total_loss += loss.item()

    # Calculate average loss over all batches
    avg_loss = total_loss / len(test_loader)
    print('Average Test Loss:', avg_loss)

    # Determine the figure path based on the dataset type
    if dataset_type == "foreseen":
        figure_path = f"{figure_path_Dmg}/{model.__class__.__name__}_seen"
    elif dataset_type == "unforeseen":
        figure_path = f"{figure_path_Dmg}/{model.__class__.__name__}_unseen"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Check if the directory already exists
    if not os.path.exists(figure_path):
        # If it doesn't exist, create the directory
        os.makedirs(figure_path)
        print(f"Directory '{figure_path}' created successfully.")
    else:
        print(f"Directory '{figure_path}' already exists.")

    ## Flatten the lists for plotting
    all_predictions = [item for sublist in all_predictions for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]
    all_Ninput = [item for sublist in all_Ninput for item in sublist]
       
    # calculate metrics
    calculate_metrics_Dmg(all_predictions, all_labels, figure_path)

    # save plots
    save_plot_Dmg(all_predictions, all_labels, all_Ninput, figure_path)

    return all_predictions, all_labels, avg_loss

if __name__ == "__main__":
    # Load data from data_loader
    Dmg_train_data, Ninput_train_data, MR_train_data, Dmg_valid_data, Ninput_valid_data, MR_valid_data, Dmg_test_data_foreseen, Dmg_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)
    
    # Load the training dataset
    Dmg_test_dataset_foreseen = DmgDataset(Ninput_test_data_foreseen, MR_test_data_foreseen, Dmg_test_data_foreseen)
    Dmg_test_dataset_unforeseen = DmgDataset(Ninput_test_data_unforeseen, MR_test_data_unforeseen, Dmg_test_data_unforeseen)  
    
    # Set test_loader and dataset_type based on config.use_foreseen
    if use_foreseen:
        test_loader = DataLoader(Dmg_test_dataset_foreseen, batch_size=batch_size, shuffle=False)
        dataset_type = "foreseen"
    else:
        test_loader = DataLoader(Dmg_test_dataset_unforeseen, batch_size=batch_size, shuffle=False)
        dataset_type = "unforeseen"

    # Initialize the model and load weights
    model = get_model(model_name)
    model_name = model.__class__.__name__    
    model.load_state_dict(torch.load(f'{model_path_Dmg}/{model_name}_Dmg_{num_epochs}epoch.pth'))
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Test the model
    predictions, labels, avg_loss = test_model(model, test_loader)


