import torch
from models import RFACNN, RFAUNet, RFAAttUNet
from data_loader_Temp import TemperatureDataset, DataLoader, load_data
from utils import new_combined_loss, calculate_metrics_Temp, save_plot_Temp
import config
import utils
from config import alpha, beta, gamma, model_path_Temp, file_paths, figure_path_Temp, model_name, use_foreseen, num_epochs
import numpy as np
import os 


def get_model(choice):
    if choice == "1":
        return RFACNN()
    elif choice == "2":
        return RFAUNet()
    elif choice == "3":
        return RFAAttUNet()
    else:
        raise ValueError(f"Unknown model choice: {choice}")
     
        
     
        
def measure_time(model, Ninput, MR):
    
    total_infer_time = 0
        
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        
        for i in range(len(Ninput)):
            
            Ninput_input = Ninput[i,:,:,:].unsqueeze(0).unsqueeze(0).cuda()
            MR_input = MR[i,:,:,:].unsqueeze(0).unsqueeze(0).cuda()
            
            starter.record()
            outputs = model(Ninput_input, MR_input)
            ender.record()
            
            torch.cuda.synchronize()
            infer_time = starter.elapsed_time(ender)
            total_infer_time += infer_time
    
    avg_infer_time = total_infer_time / len(Ninput)
    
    return avg_infer_time



            
if __name__ == "__main__":
    
    
    Temp_train_data, Ninput_train_data, MR_train_data,Temp_valid_data, Ninput_valid_data, MR_valid_data, Temp_test_data_foreseen, Temp_test_data_unforeseen, Ninput_test_data_foreseen, Ninput_test_data_unforeseen, MR_test_data_foreseen, MR_test_data_unforeseen = load_data(file_paths)
    Ninput_test_data = torch.cat((Ninput_test_data_foreseen, Ninput_test_data_unforeseen))
    MR_test_data = torch.cat((MR_test_data_foreseen, MR_test_data_unforeseen)) 
    model = get_model(model_name)
    model_name = model.__class__.__name__    
    model.load_state_dict(torch.load(f'{model_path_Temp}/{model_name}_Temp_{num_epochs}epoch.pth'))
    model.cuda() if torch.cuda.is_available() else model.cpu()
    avg_time = measure_time(model, Ninput_test_data, MR_test_data)
    
    print(avg_time)