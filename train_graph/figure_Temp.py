import numpy as np
import matplotlib.pyplot as plt

# Load the data from the provided files
attunet_train_loss = np.loadtxt('RFAAttUNet_Temp_train_loss.txt')
attunet_valid_loss = np.loadtxt('RFAAttUNet_Temp_valid_loss.txt')
cnn_train_loss     = np.loadtxt('RFACNN_Temp_train_loss.txt')
cnn_valid_loss     = np.loadtxt('RFACNN_Temp_valid_loss.txt')
unet_train_loss    = np.loadtxt('RFAUNet_Temp_train_loss.txt')
unet_valid_loss    = np.loadtxt('RFAUNet_Temp_valid_loss.txt')

# Plotting all losses in one figure
plt.figure(figsize=(12, 8))

# Plot RFACNN losses
plt.plot(cnn_train_loss, label='EDCNN Train Loss')
plt.plot(cnn_valid_loss, label='EDCNN Valid Loss')

# Plot RFAUNet losses
plt.plot(unet_train_loss, label='U-Net Train Loss')
plt.plot(unet_valid_loss, label='U-Net Valid Loss')

# Plot RFAAttUNet losses
plt.plot(attunet_train_loss, label='Attention U-Net Train Loss')
plt.plot(attunet_valid_loss, label='Attention U-Net Valid Loss')

# Adding labels and legend
plt.xlabel('Epochs', fontsize=24)
plt.ylabel('Loss', fontsize=24)
#plt.title('Training and Validation Losses for Different Models', fontsize=24)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.ylim(0,3)

# Save the figure
plt.savefig('Temp_losses_plot.png')

# Show the plot
plt.show()
