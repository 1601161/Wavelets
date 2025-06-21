#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reads PSNR data from `psnr_lists.txt`, extracts individual image 
PSNR curves, and generates a separate line plot for each using matplotlib. 
Each plot shows how PSNR varies with wavelet coefficient compression for a 
given image.

INPUTS
    - 'psnr_lists.txt': A text file with:
        Discarded_percentage: List of % coefficients discarded
        Multiple `PSNR_<image_id>` variables containing PSNR values at each discard level

OUTPUTS
    - Folder `psnr_plots/` with one PNG file per PSNR curve 
            e.g. `PSNR_Paciente_100_TC_84.png`, etc.
"""

import matplotlib.pyplot as plt  # For plotting
import re  # Regular expressions for text parsing
import os  # File and folder operations

# Read the psnr_lists.txt content
with open("psnr_lists.txt", "r") as file:
    content = file.read()

# Extract the common discard percentage values
# Regex match: extract the list from "Discarded_percentage = [ ... ]"
discarded_match = re.search(r"Discarded_percentage\s*=\s*\[([^\]]+)\]", content)
# Convert the matched string of numbers into a list of floats
discarded_percentage = [float(x.strip()) for x in discarded_match.group(1).split(",")]

# Extract all individual PSNR arrays
# Find all matches of the form:
#   PSNR_Paciente_100_TC_84 = [ ... ]
# Returns a list of (var_name, list_string) pairs
psnr_matches = re.findall(r"(PSNR_Paciente_\d+_TC_\d+)\s*=\s*\[([^\]]+)\]", content)

# Create a folder to store output plots
# Ensure the folder "psnr_plots" exists (create it if not)
os.makedirs("psnr_plots", exist_ok=True)

for var_name, psnr_data in psnr_matches:
    # Convert PSNR string into a list of float values
    psnr_values = [float(x.strip()) for x in psnr_data.split(",")]

    # Start a new figure
    plt.figure(figsize=(10, 5))
    
    # Plot PSNR vs discarded %
    plt.plot(discarded_percentage[:len(psnr_values)], psnr_values, marker='o', label=var_name)
    
    # Add title and labels
    plt.title(f"{var_name} vs Discarded Percentage")
    plt.xlabel("Discarded Percentage (%)")
    plt.ylabel("PSNR")
    plt.grid(False)
    plt.legend()

    # Save plot as PNG file inside the folder
    filename = os.path.join("psnr_plots", f"{var_name}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()  # Close the figure to free memory

# Print confirmation message
print("All individual PSNR plots have been saved in the 'psnr_plots' folder.")
