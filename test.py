import torch
from models import TemperatureCNN
from data_loader import TemperatureDataset, DataLoader
from utils import new_combined_loss
from config import alpha, beta, gamma

def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            Ninput_test_data_batch, MR_test_data_batch, Temp_test_data_batch = batch

            # Move data to GPU if available
            if torch.cuda.is_available():
                Ninput_test_data_batch = Ninput_test_data_batch.cuda()
                MR_test_data_batch = MR_test_data_batch.cuda()
                Temp_test_data_batch = Temp_test_data_batch.cuda()

            # Forward pass
            outputs = model(Ninput_test_data_batch, MR_test_data_batch)

            # Store predictions and labels for evaluation
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(Temp_test_data_batch.cpu().numpy())

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
    # Example usage

    # Load the test dataset (you might want to implement a function for this in data_loader.py)
    test_dataset = TemperatureDataset( ... )  # Fill with your data
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the model and load weights
    model = TemperatureCNN()
    model.load_state_dict(torch.load('path_to_your_saved_model.pt'))
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Test the model
    predictions, labels, avg_loss = test_model(model, test_loader)

    # Post-process predictions and labels for further analysis or visualization
    # ...

