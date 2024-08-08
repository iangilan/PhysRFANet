import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config

# Use the configurations
batch_size = config.batch_size
file_paths = config.file_paths

class TemperatureDataset(Dataset):
    def __init__(self, Ninput_data, MR_data, Temp_data):
        self.Ninput_data = Ninput_data
        self.MR_data = MR_data
        self.Temp_data = Temp_data

    def __len__(self):
        return len(self.Ninput_data)

    def __getitem__(self, index):
        return self.Ninput_data[index], self.MR_data[index], self.Temp_data[index]

def load_data(file_paths):
    # Loading the .npy files using numpy
    Temp_train_data = np.load(file_paths['Temp_train'])
    Temp_test_data = np.load(file_paths['Temp_test'])
    Ninput_train_data = np.load(file_paths['Ninput_train'])
    Ninput_test_data = np.load(file_paths['Ninput_test'])
    MR_train_data = np.load(file_paths['MR_train'])
    MR_test_data = np.load(file_paths['MR_test'])

    # Convert to PyTorch tensors and preprocess
    Temp_train_data = torch.from_numpy(Temp_train_data).float()
    Ninput_train_data = torch.from_numpy(Ninput_train_data).float()
    MR_train_data = torch.from_numpy(MR_train_data).float()
    MR_train_data = MR_train_data.repeat_interleave(500, dim=0)

    Temp_test_data = torch.from_numpy(Temp_test_data).float()
    Ninput_test_data = torch.from_numpy(Ninput_test_data).float()
    MR_test_data = torch.from_numpy(MR_test_data).float()
    MR_test_data = MR_test_data.repeat_interleave(500, dim=0)

    # Remove the first 500 samples from the test data
    Temp_test_data = Temp_test_data[500:, :, :, :]
    Ninput_test_data = Ninput_test_data[500:, :, :, :]
    MR_test_data = MR_test_data[500:, :, :, :]
    
    # Split the data into training and test sets
    Temp_train_data, Temp_test_data_foreseen = train_test_split(Temp_train_data, test_size=500, random_state=42)
    Temp_test_data_unforeseen, Temp_test_data_dummy = train_test_split(Temp_test_data, test_size=500, random_state=42)
    Temp_train_data, Temp_valid_data = train_test_split(Temp_train_data, test_size=200, random_state=42)

    # Split the data into training and test sets
    Ninput_train_data, Ninput_test_data_foreseen = train_test_split(Ninput_train_data, test_size=500, random_state=42)
    Ninput_test_data_unforeseen, Ninput_test_data_dummy = train_test_split(Ninput_test_data, test_size=500, random_state=42)
    Ninput_train_data, Ninput_valid_data = train_test_split(Ninput_train_data, test_size=200, random_state=42)

    # Split the data into training and test sets
    MR_train_data, MR_test_data_foreseen = train_test_split(MR_train_data, test_size=500, random_state=42)
    MR_test_data_unforeseen, MR_test_data_dummy = train_test_split(MR_test_data, test_size=500, random_state=42)
    MR_train_data, MR_valid_data = train_test_split(MR_train_data, test_size=200, random_state=42)
    
    return Temp_train_data, Ninput_train_data, MR_train_data, Temp_valid_data, Ninput_valid_data, MR_valid_data, Temp_test_data_foreseen, Temp_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen


if __name__ == "__main__":
    Temp_train_data, Ninput_train_data, MR_train_data, 
    Temp_valid_data, Ninput_valid_data, MR_valid_data, 
    Temp_test_data_foreseen, Temp_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)

    # Create the TemperatureDataset object
    Temp_train_dataset = TemperatureDataset(Ninput_train_data, MR_train_data, Temp_train_data)
    Temp_valid_dataset = TemperatureDataset(Ninput_valid_data, MR_valid_data, Temp_valid_data)
    Temp_test_dataset_foreseen  = TemperatureDataset(Ninput_test_data_foreseen, MR_test_data_foreseen, Temp_test_data_foreseen)
    Temp_test_dataset_unforeseen  = TemperatureDataset(Ninput_test_data_unforeseen, MR_test_data_unforeseen, Temp_test_data_unforeseen)

    # Create the data loader
    Temp_train_loader = DataLoader(Temp_train_dataset, batch_size=batch_size, shuffle=False)
    Temp_valid_loader = DataLoader(Temp_valid_dataset, batch_size=batch_size, shuffle=False)
    Temp_test_loader_foreseen = DataLoader(Temp_test_dataset_foreseen, batch_size=batch_size, shuffle=False)    
    Temp_test_loader_unforeseen = DataLoader(Temp_test_dataset_foreseen, batch_size=batch_size, shuffle=False)    

