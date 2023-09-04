import os
import pickle
import matplotlib.pyplot as plt

# Initialize an empty dictionary to store the figures
figs = {}

# Get a list of all files in the "plots" directory
files = os.listdir("./plots")

# Loop through each file
for filename in files:
    # Construct the full file path
    filepath = os.path.join("./plots", filename)
    
    # Open the file and load the figure
    with open(filepath, 'rb') as file:
        fig = pickle.load(file)
    
    # Save the figure in the dictionary
    figs[filename] = fig

# Now, figs is a dictionary where the keys are filenames and the values are the corresponding figures

# Display all figures
for filename, fig in figs.items():
    print(f"Displaying figure: {filename}")
    #set title to the fig as the filename without the extension
    fig.suptitle(filename.split('.')[0])    
        
    #save fig to file
    fig.set_size_inches(19.2, 10.8)
    fig.savefig(f"./plots/{filename}.png", dpi=100)
    fig.show()

plt.show()
