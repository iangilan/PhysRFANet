import matplotlib.pyplot as plt

# Function to read data from a text file
def read_data(file_name):
    with open(file_name, 'r') as file:
        return [float(line.strip()) for line in file]

# Folder names
folders = ['RFACNN_seen', 'RFAUNet_seen', 'RFAAttUNet_seen', 'RFACNN_unseen', 'RFAUNet_unseen', 'RFAAttUNet_unseen']

# File name to be read from each folder
file_name = 'rmse.txt'

# Construct full paths to the mse.txt files in each folder and read data
data = [read_data(f'{folder}/{file_name}') for folder in folders]

# Define boxplot properties
boxprops = dict(linestyle='-', linewidth=2, color='k')
medianprops = dict(linestyle='-', linewidth=2, color='red')
colors = ['lightcyan', 'lightgreen', 'lightpink']

# Create the boxplot with adjusted figure size for vertical length
fig, ax = plt.subplots(figsize=(6, 8))  # Width, Height in inches
positions = [1, 1.5, 2, 3, 3.5, 4]

bplot = ax.boxplot(data, positions=positions, patch_artist=True, boxprops=boxprops, medianprops=medianprops)

# Coloring the boxes
for patch, color in zip(bplot['boxes'], colors * 2):  # Multiply the color list by 2 for both groups
    patch.set_facecolor(color)

# Adding labels
ax.set_xticks([1.5, 3.5])
ax.set_xticklabels(['(Foreseen test)\nEDCNN, U-Net, Att. U-Net', '(Unforeseen test)\nEDCNN, U-Net, Att. U-Net'], fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)

# Save the figure with 600 DPI
plt.savefig('boxplot.png', dpi=600)

# Show the plot
plt.show()

