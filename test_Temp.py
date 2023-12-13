import torch
from models import RFACNN, RFAUNet, RFAAttUNet
from data_loader_Temp import TemperatureDataset, DataLoader, load_data
from utils import new_combined_loss
import config
from config import num_epochs, batch_size, alpha, beta, gamma, model_path_Temp, model_path_Dmg, file_paths

def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    all_predictions = []
    all_labels = []
    all_Ninput = []

    with torch.no_grad():
        for batch in test_loader:
            Ninput_test_data_batch, MR_test_data_batch, Temp_test_data_batch = batch

            Ninput_test_data_batch = Ninput_test_data_batch.unsqueeze(1)
            MR_test_data_batch     = MR_test_data_batch.unsqueeze(1)
            Temp_test_data_batch   = Temp_test_data_batch.unsqueeze(1)

            if torch.cuda.is_available(): # Move data to GPU if available
                Ninput_test_data_batch = Ninput_test_data_batch.cuda()
                MR_test_data_batch     = MR_test_data_batch.cuda()
                Temp_test_data_batch   = Temp_test_data_batch.cuda()          

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

    # Additional evaluation metrics can be calculated here
    # ...

    return all_predictions, all_labels, avg_loss

if __name__ == "__main__":
    # Load data from data_loader
    Temp_train_data, Ninput_train_data, MR_train_data, Temp_test_data_foreseen, Temp_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)
    # Load the training dataset
    Temp_test_dataset_foreseen = TemperatureDataset(Ninput_test_data_foreseen, MR_test_data_foreseen, Temp_test_data_foreseen)
    Temp_test_dataset_unforeseen = TemperatureDataset(Ninput_test_data_unforeseen, MR_test_data_unforeseen, Temp_test_data_unforeseen)    
    test_loader = DataLoader(Temp_test_dataset_foreseen, batch_size=batch_size, shuffle=True) # or choose Temp_test_dataset_unforeseen for unforeseen test dataset

    # Initialize the model and load weights
    model = RFACNN()
    model.load_state_dict(torch.load(f'{model_path_Temp}/temperature_model.pth'))
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Test the model
    predictions, labels, avg_loss = test_model(model, test_loader)

    # Post-process predictions and labels for further analysis or visualization
    # ...

