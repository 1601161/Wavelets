#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads the results from wavelet-based compression experiments 
(using Daubechies wavelets db2–db10) at different decomposition levels and 
retention ratios. It generates line plots showing the average PSNR for each 
wavelet across decomposition levels, for each fixed retention ratio.

INPUT: 
    - 'db_level_results.txt' (tab-separated, with columns Wavelet, Level, Retention, Avg_PSNR)
OUTPUT: 
    - One PNG file per retention ratio, named 'psnr_retention_XX.png' (e.g. psnr_retention_27.png)
        Each plot shows PSNR vs Wavelet for various decomposition levels at fixed retention
"""

import pandas as pd                      # For reading and handling tabular data
import matplotlib.pyplot as plt          # For creating visualizations
import seaborn as sns                    # For enhanced plot aesthetics

# Load the data, Read the tab-separated results from previous experiments
df = pd.read_csv('db_level_results.txt', sep='\t')

# Convert Wavelet column to ordered categorical (db2, db3, ..., db10)
df['Wavelet'] = pd.Categorical(df['Wavelet'], categories=[f'db{i}' for i in range(2, 11)], ordered=True)

# Plot PSNR curves for each retention ratio
# Iterate over each unique retention ratio (e.g., 0.01, 0.03, 0.05, 0.10)
for retention in sorted(df['Retention'].unique()):
    plt.figure(figsize=(12, 6)) # Set the size of the plot canvas
    # Filter the DataFrame to include only the current retention ratio
    subset = df[df['Retention'] == retention]
    
    # Set default font sizes for better readability
    plt.rcParams.update({
        'font.size': 16,             # Default font size
        'axes.titlesize': 20,        # Title font size
        'axes.labelsize': 18,        # Axis label font size
        'xtick.labelsize': 18,       # X tick label font size
        'ytick.labelsize': 18,       # Y tick label font size
        'legend.fontsize': 18,       # Legend text font size
        'legend.title_fontsize': 18  # Legend title font size
    })

    # Create a line plot showing PSNR for each wavelet and level
    sns.lineplot(
        data=subset,
        x='Wavelet',            # X-axis: Wavelet type (db2 to db10)
        y='Avg_PSNR',           # Y-axis: Average PSNR in dB
        hue='Level',            # Different colors for each decomposition level
        marker='o',             # Add circular markers to each data point
        palette='viridis',      # Use a perceptually uniform colormap
        linewidth=2             # Make lines thicker for clarity
    )
    
    # Add plot title, axis labels, grid and legend
    plt.title(f'Wavelet Compression Comparison\nRetention: {int(retention * 100)}%')
    plt.xlabel('Wavelet (Daubechies dbN)')
    plt.ylabel('Average PSNR (dB)')
    plt.grid(True)
    plt.legend(title='Decomposition Level')
    # Optimize layout to prevent overlaps
    plt.tight_layout()
    
    # Save the plot as a PNG file named by retention percentage
    filename = f'psnr_retention_{int(retention * 100)}.png'
    plt.savefig(filename)
    plt.close() # Close figure to avoid memory issues in loops
    
    # Print confirmation message
    print(f"Saved plot for {int(retention * 100)}% retention to: {filename}")
