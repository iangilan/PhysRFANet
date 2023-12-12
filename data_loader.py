import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
    # Load the data from .npy files
    # Example: 'file_paths' can be a dictionary with keys 'Temp_train', 'Temp_test', 'Ninput_train', etc.
    data = {key: torch.from_numpy(np.load(path)).float() for key, path in file_paths.items()}
    
    # Additional processing if necessary
    # ...

    return data

def create_data_loaders(data, batch_size, test_size=0.2):
    # Split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Create Dataset objects
    train_dataset = TemperatureDataset(train_data['Ninput_train'], train_data['MR_train'], train_data['Temp_train'])
    test_dataset = TemperatureDataset(test_data['Ninput_test'], test_data['MR_test'], test_data['Temp_test'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example file paths
data_path = "your/data/path"
file_paths = {
    'Temp_train'  : 'data_path/Temp_train.npy',
    'Temp_test'   : 'data_path/Temp_test.npy',
    'Ninput_train': 'data_path/Ninput_train.npy',
    'Ninput_test' : 'data_path/Ninput_test.npy',
    'MR_train'    : 'data_path/MR_train.npy',
    'MR_test'     : 'data_path/MR_test.npy'
}

# Example usage
data = load_data(file_paths)
train_loader, test_loader = create_data_loaders(data, batch_size=1)

