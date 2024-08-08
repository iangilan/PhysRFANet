import torch
import torch.optim as optim
from tqdm import tqdm
from models import RFACNN, RFAUNet, RFAAttUNet
from data_loader_Dmg import DmgDataset, load_data
from torch.utils.data import Dataset, DataLoader
import config
from utils import dice_loss
from config import num_epochs, batch_size, model_path_Dmg, file_paths, model_name
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

# Use the configurations
file_paths = config.file_paths

def get_model(choice):
    if choice == "1":
        return RFACNN()
    elif choice == "2":
        return RFAUNet()
    elif choice == "3":
        return RFAAttUNet()
    else:
        raise ValueError(f"Unknown model choice: {choice}")

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0


def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    model.train()
    
    train_loss = []
    valid_loss = []
    
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            Ninput_train_data_batch, MR_train_data_batch, Dmg_train_data_batch = batch
            Ninput_train_data_batch = Ninput_train_data_batch.unsqueeze(1).cuda()
            MR_train_data_batch     = MR_train_data_batch.unsqueeze(1).cuda()
            Dmg_train_data_batch    = Dmg_train_data_batch.unsqueeze(1).cuda()

            # Forward pass
            outputs = model(Ninput_train_data_batch, MR_train_data_batch)

            # Calculate loss
            loss = criterion(outputs, Dmg_train_data_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()
            
        # Print the average loss for the epoch
        print(f'Epoch: {epoch+1}, Average Loss: {epoch_loss / len(train_loader)}')
        train_loss.append(epoch_loss / len(train_loader))
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No gradients required for validation
            val_loss = 0

            for val_batch in valid_loader:
                # Forward pass
                Ninput_val_data_batch, MR_val_data_batch, Dmg_val_data_batch = val_batch
                Dmg_val_data_batch    = Dmg_val_data_batch.unsqueeze(1).cuda()
                MR_val_data_batch     = MR_val_data_batch.unsqueeze(1).cuda()
                Ninput_val_data_batch = Ninput_val_data_batch.unsqueeze(1).cuda()

                val_outputs = model(Ninput_val_data_batch, MR_val_data_batch)

                # Calculate the loss
                loss = criterion(val_outputs, Dmg_val_data_batch)
                val_loss += loss.item()

            # Calculate average validation loss and update the scheduler
            avg_val_loss = val_loss / len(valid_loader)
            scheduler.step(avg_val_loss)

            # Print validation loss
            print(f'Epoch: {epoch+1}, Validation Loss: {val_loss / len(valid_loader)}')    
            valid_loss.append(val_loss / len(valid_loader))

            # Early stopping
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # Load the best model
    model.load_state_dict(early_stopping.best_model)
                              
    return train_loss, valid_loss
    

if __name__ == "__main__":
    # Load data from data_loader
    Dmg_train_data, Ninput_train_data, MR_train_data, Dmg_valid_data, Ninput_valid_data, MR_valid_data, Dmg_test_data_foreseen, Dmg_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)

    # Initialize the model
    model = get_model(model_name)
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Define the loss function and optimizer
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # Load the training dataset
    Dmg_train_dataset = DmgDataset(Ninput_train_data, MR_train_data, Dmg_train_data)    
    train_loader = DataLoader(Dmg_train_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the validation dataset
    Dmg_valid_dataset = DmgDataset(Ninput_valid_data, MR_valid_data, Dmg_valid_data)
    valid_loader = DataLoader(Dmg_valid_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    #train_model(model, criterion, optimizer, train_loader, num_epochs)
    train_loss, valid_loss = train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs)
    
    # Save the trained model
    model_name = model.__class__.__name__
    torch.save(model.state_dict(), f'{model_path_Dmg}/{model_name}_Dmg_{num_epochs}epoch.pth')    

    # Set font sizes
    plt.rcParams.update({'font.size': 20}) 
    #plt.rcParams['axes.titlesize'] = 18
    #plt.rcParams['axes.labelsize'] = 16
    #plt.rcParams['xtick.labelsize'] = 14
    #plt.rcParams['ytick.labelsize'] = 14
    #plt.rcParams['legend.fontsize'] = 14
    
    # Plot training graph
    plt.figure(figsize=(10,5))
    plt.title("Training Loss")
    plt.plot(train_loss,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'train_graph/{model_name}_Dmg_train.png', dpi=600, pad_inches=0)
    plt.close()
    
    
    plt.figure(figsize=(10,5))
    plt.title("Validation Loss")
    plt.plot(valid_loss,label="val")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'train_graph/{model_name}_Dmg_valid.png', dpi=600, pad_inches=0)
    plt.close()
    
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(valid_loss,label="val")
    plt.plot(train_loss,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'train_graph/{model_name}_Dmg_train&valid.png', dpi=600, pad_inches=0)
    plt.close()
