import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import time
import os

# Initialize truncation times
vanilla_times = []
optimized_times = []

# Define the sparsity levels
sparsity_levels = [0.1, 0.15, 0.2, 0.25, 0.3]

# Directory to save the sliced models
save_base_dir = "dir/to/save/sliced_models"

# Model name
model_name = "microsoft/phi-2"

# Device to run the script
device = "cuda:0"

# Ensure the base directory exists
os.makedirs(save_base_dir, exist_ok=True)

# Loop through each sparsity level and perform vanilla SliceGPT
for sparsity in sparsity_levels:
    # Define the save directory for the current sparsity level
    save_dir = os.path.join(save_base_dir, f"sparsity_{sparsity}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the command
    command = (
        f"python run_slicegpt.py "
        f"--model {model_name} "
        f"--save-dir {save_dir} "
        f"--sparsity {sparsity} "
        f"--device {device} "
        f"--eval-baseline "
        f"--no-wandb"
    )
    
    # Print the command (for debugging purposes)
    print(f"Running command: {command}")
    
    start_vanilla = time.time()
    # Execute the command
    os.system(command)
    end_vanilla = time.time()
    vanilla_times.append(end_vanilla - start_vanilla)
    
    
# Loop through each sparsity level and perform optimized SliceGPT
for sparsity in sparsity_levels:
    # Define the save directory for the current sparsity level
    save_dir = os.path.join(save_base_dir, f"sparsity_{sparsity}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the command
    command = (
        f"python run_optimized_slicegpt.py "
        f"--model {model_name} "
        f"--save-dir {save_dir} "
        f"--sparsity {sparsity} "
        f"--device {device} "
        f"--eval-baseline "
        f"--no-wandb"
    )
    
    # Print the command (for debugging purposes)
    print(f"Running command: {command}")
    
    start_optimized = time.time()
    # Execute the command
    os.system(command)
    end_optimized = time.time()
    optimized_times.append(end_optimized - start_optimized)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sparsity_levels, optimized_times, marker='o', label="Optimized SliceGPT SVD Times")
plt.plot(sparsity_levels, vanilla_times, color='r', marker='o', label="Vanilla SliceGPT Times")
plt.xlabel("Number of Singular Values (k)")
plt.ylabel("Time (seconds)")
plt.title("Time to Compute Truncated SVD vs. Truncation Level")
plt.legend()
plt.grid()
plt.show()
