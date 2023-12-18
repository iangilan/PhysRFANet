import torch
import torch.optim as optim
from tqdm import tqdm
from models import RFACNN, RFAUNet, RFAAttUNet
from data_loader_Dmg import DmgDataset, load_data, DataLoader, batch_size
from utils import new_combined_loss
from config import num_epochs, batch_size, alpha, beta, gamma, model_path_Dmg, file_paths

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            Ninput_train_data_batch, MR_train_data_batch, Dmg_train_data_batch = batch

            # Move data to GPU if available
            if torch.cuda.is_available():
                Ninput_train_data_batch = Ninput_train_data_batch.unsqueeze(1).cuda()
                MR_train_data_batch     = MR_train_data_batch.unsqueeze(1).cuda()
                Dmg_train_data_batch   = Dmg_train_data_batch.unsqueeze(1).cuda()

            # Forward pass
            outputs = model(Ninput_train_data_batch, MR_train_data_batch)

            # Calculate loss
            loss = criterion(outputs, Dmg_train_data_batch, alpha, beta, gamma)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        # Print the average loss for the epoch
        print(f'Epoch: {epoch+1}, Average Loss: {epoch_loss / len(train_loader)}')

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No gradients required for validation
            val_loss = 0

            for val_batch in Dmg_valid_loader:
                # Forward pass
                Ninput_val_data_batch, MR_val_data_batch, Dmg_val_data_batch = val_batch
                Dmg_val_data_batch    = Dmg_val_data_batch.unsqueeze(1).cuda()
                MR_val_data_batch     = MR_val_data_batch.unsqueeze(1).cuda()
                Ninput_val_data_batch = Ninput_val_data_batch.unsqueeze(1).cuda()

                val_outputs = model(Ninput_val_data_batch, MR_val_data_batch)

                # Calculate the loss
                loss = criterion(val_outputs, Dmg_val_data_batch,alpha,beta,gamma)
                val_loss += loss.item()

            # Calculate average validation loss and update the scheduler
            avg_val_loss = val_loss / len(Dmg_valid_loader)
            scheduler.step(avg_val_loss)

            # Print validation loss
            print(f'Epoch: {epoch+1}, Validation Loss: {val_loss / len(Dmg_valid_loader)}')    
    
    # Save the trained model
    torch.save(model.state_dict(), f'{model_path_Dmg}/dmg_model.pth')

if __name__ == "__main__":
    # Load data from data_loader
    Dmg_train_data, Ninput_train_data, MR_train_data, Dmg_test_data_foreseen, Dmg_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)

    # Initialize the model
    model = RFACNN() # Choose your model
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Define the loss function and optimizer
    criterion = new_combined_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load the training dataset
    Dmg_train_dataset = DmgDataset(Ninput_train_data, MR_train_data, Dmg_train_data)    
    train_loader = DataLoader(Dmg_train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    #train_model(model, criterion, optimizer, train_loader, num_epochs)
    train_model(model, criterion, optimizer, train_loader, num_epochs)

