import torch
from models import RFACNN, RFAUNet, RFAAttUNet
from data_loader_Dmg import DmgDataset, load_data
from torch.utils.data import Dataset, DataLoader
from utils import dice_loss
import config
from config import num_epochs, batch_size, model_path_Dmg, file_paths

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

    # Additional evaluation metrics can be calculated here
    # ...

    return all_predictions, all_labels, avg_loss

if __name__ == "__main__":
    # Load data from data_loader
    Dmg_train_data, Ninput_train_data, MR_train_data, Dmg_valid_data, Ninput_valid_data, MR_valid_data, Dmg_test_data_foreseen, Dmg_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)
    
    # Load the training dataset
    Dmg_test_dataset_foreseen = DmgDataset(Ninput_test_data_foreseen, MR_test_data_foreseen, Dmg_test_data_foreseen)
    Dmg_test_dataset_unforeseen = DmgDataset(Ninput_test_data_unforeseen, MR_test_data_unforeseen, Dmg_test_data_unforeseen)    
    test_loader = DataLoader(Dmg_test_dataset_foreseen, batch_size=batch_size, shuffle=True) # or choose Dmg_test_dataset_unforeseen for unforeseen test dataset

    # Initialize the model and load weights
    model = RFACNN()
    model.load_state_dict(torch.load(f'{model_path_Dmg}/dmg_model.pth'))
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Test the model
    predictions, labels, avg_loss = test_model(model, test_loader)


