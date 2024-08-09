# Configuration settings
num_epochs = 100
batch_size = 2 #16 for RFACNN, RFAUNet # Batch size for DataLoader

# weights for the combined loss function (0.6, 0.2, 0.2) 
alpha = 0.6
beta  = 0.2
gamma = 0.2

# model path
model_path_Temp = "model_Temp" # Replace with your actual Temperature model path
model_path_Dmg  = "model_Dmg"  # Replace with your actual Damage model path

# figure path
figure_path_Temp = "fig_Temp" # Replace with your actual Temperature figure path
figure_path_Dmg  = "fig_Dmg"  # Replace with your actual Damage figure path

# training/test data path
data_path = "/media/mws/Data/data_RFA" #"data"  # Replace with your actual data path
#data_path = "/media/mws/usb/data_RFA" #"data"  # Replace with your actual data path
#data_path = "data"  # Replace with your actual data path

file_paths = {
    'Temp_train'  : data_path + '/data_Temp/Temp_train.npy',
    'Temp_test'   : data_path + '/data_Temp/Temp_test.npy',
    'Dmg_train'   : data_path + '/data_Dmg/Dmg_train.npy',
    'Dmg_test'    : data_path + '/data_Dmg/Dmg_test.npy',
    'Ninput_train': data_path + '/data_Ninput/Ninput_train.npy',
    'Ninput_test' : data_path + '/data_Ninput/Ninput_test.npy',
    'MR_train'    : data_path + '/data_MR/MRdata_train.npy',
    'MR_test'     : data_path + '/data_MR/MRdata_test.npy'
}

# Model selection
model_name = "1"  # Choose 1, 2, or 3 for models: "(1) RFACNN", "(2) RFAUNet", "(3) RFAAttUNet"

# Foreseen or unforeseen test dataset flag
use_foreseen = True  # Change this to False to use unforeseen dataset
