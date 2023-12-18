# Configuration settings
num_epochs = 200
batch_size = 16 # Batch size for DataLoader

# weights for the combined loss function
alpha = 0.6
beta  = 0.1
gamma = 0.3

# model path
model_path_Temp = "models_Temp" # Replace with your actual Temperature model path
model_path_Dmg  = "models_Dmg"  # Replace with your actual Damage model path

# training/test data path
data_path = "data"  # Replace with your actual data path
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
