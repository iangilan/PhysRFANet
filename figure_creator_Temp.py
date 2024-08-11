import torch
from models import RFACNN, RFAUNet, RFAAttUNet
from data_loader_Temp import TemperatureDataset, DataLoader, load_data
from utils import new_combined_loss, save_plot_Temp_each
import config
import utils
from config import num_epochs, batch_size, alpha, beta, gamma, model_path_Temp, file_paths, figure_path_Temp, model_name, use_foreseen
import numpy as np
import os 
import matplotlib.pyplot as plt

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
            Ninput_test_data_batch, MR_test_data_batch, Temp_test_data_batch = batch

            Ninput_test_data_batch = Ninput_test_data_batch.unsqueeze(1).cuda()
            MR_test_data_batch     = MR_test_data_batch.unsqueeze(1).cuda()
            Temp_test_data_batch   = Temp_test_data_batch.unsqueeze(1).cuda()     

            # Forward pass
            outputs = model(Ninput_test_data_batch, MR_test_data_batch)

            # Store predictions and labels for evaluation
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(Temp_test_data_batch.cpu().numpy())
            all_Ninput.extend(Ninput_test_data_batch.cpu().numpy()) 

            # Calculate the loss for this batch
            loss = new_combined_loss(outputs, Temp_test_data_batch, alpha, beta, gamma)
            total_loss += loss.item()

    # Calculate average loss over all batches
    avg_loss = total_loss / len(test_loader)
    print('Average Test Loss:', avg_loss)

    # Determine the figure path based on the dataset type
    if dataset_type == "foreseen":
        figure_path = f"{figure_path_Temp}/{model.__class__.__name__}_seen/fig_each"
    elif dataset_type == "unforeseen":
        figure_path = f"{figure_path_Temp}/{model.__class__.__name__}_unseen/fig_each"
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
       
    # save plots
    save_plot_Temp_each(all_predictions, all_labels, all_Ninput, figure_path)

    return all_predictions, all_labels, avg_loss

if __name__ == "__main__":
    # Load data from data_loader
    Temp_train_data, Ninput_train_data, MR_train_data,Temp_valid_data, Ninput_valid_data, MR_valid_data, Temp_test_data_foreseen, Temp_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)
    # Load the test dataset
    Temp_test_dataset_foreseen = TemperatureDataset(Ninput_test_data_foreseen, MR_test_data_foreseen, Temp_test_data_foreseen)
    Temp_test_dataset_unforeseen = TemperatureDataset(Ninput_test_data_unforeseen, MR_test_data_unforeseen, Temp_test_data_unforeseen)
    
    # Set test_loader and dataset_type based on config.use_foreseen
    if use_foreseen:
        test_loader = DataLoader(Temp_test_dataset_foreseen, batch_size=batch_size, shuffle=False)
        dataset_type = "foreseen"
    else:
        test_loader = DataLoader(Temp_test_dataset_unforeseen, batch_size=batch_size, shuffle=False)
        dataset_type = "unforeseen"

    # Initialize the model and load weights
    model = get_model(model_name)
    model_name = model.__class__.__name__    
    model.load_state_dict(torch.load(f'{model_path_Temp}/{model_name}_Temp_{num_epochs}epoch.pth'))
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Test the model
    predictions, labels, avg_loss = test_model(model, test_loader)

    # Post-process predictions and labels for further analysis or visualization
    # ...

