import torch
import torch.optim as optim
from tqdm import tqdm
from models import TemperatureCNN
from data_loader import TemperatureDataset, DataLoader
from utils import new_combined_loss
from config import num_epochs, batch_size, alpha, beta, gamma

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            Ninput_train_data_batch, MR_train_data_batch, Temp_train_data_batch = batch

            # Move data to GPU if available
            if torch.cuda.is_available():
                Ninput_train_data_batch = Ninput_train_data_batch.cuda()
                MR_train_data_batch = MR_train_data_batch.cuda()
                Temp_train_data_batch = Temp_train_data_batch.cuda()

            # Forward pass
            outputs = model(Ninput_train_data_batch, MR_train_data_batch)

            # Calculate loss
            loss = criterion(outputs, Temp_train_data_batch, alpha, beta, gamma)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        # Print the average loss for the epoch
        print(f'Epoch: {epoch+1}, Average Loss: {epoch_loss / len(train_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), 'temperature_model.pth')

if __name__ == "__main__":
    # Example usage

    # Initialize the model
    model = TemperatureCNN() # Choose your model
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Define the loss function and optimizer
    criterion = new_combined_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load the training dataset (you might want to implement a function for this in data_loader.py)
    train_dataset = TemperatureDataset( ... )  # Fill with your data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, num_epochs)

