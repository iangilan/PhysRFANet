import torch
import torch.optim as optim
from data_loader import TemperatureDataset, create_data_loaders
from models import TemperatureCNN
from train import train_model
from test import test_model
from config import num_epochs, batch_size, alpha, beta, gamma

def main():
    # Load Data
    # Note: You will need to implement the create_data_loaders function in data_loader.py
    train_loader, test_loader = create_data_loaders(batch_size)

    # Initialize the model
    model = TemperatureCNN().cuda() if torch.cuda.is_available() else TemperatureCNN()

    # Define the loss function and optimizer
    criterion = new_combined_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, num_epochs)

    # Evaluate the model
    test_model(model, test_loader)

if __name__ == "__main__":
    main()

