#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reads precomputed PSNR values from a text file (created by a 
prior code), fits a smoothing spline to the PSNR vs. coefficient-discard 
percentage data, and analyzes the rate of PSNR drop. The goal is to find 
the "sweet spot" the point where each additional discarded coefficient 
causes a sudden drop in quality.

INPUTS:
    - 'psnr_lists.txt': A text file containing:
        'Discarded_percentage': List of % coefficients discarded
        'PSNR_average': List of average PSNR values at each discard level

OUTPUTS:
    - 'spline_fit.png': Plot of original PSNR data, spline fit, and identified sweet spot
    - 'derivative.png': Plot of spline derivative (rate of change of PSNR)
    - Console output showing sweet spot (% discarded where PSNR drops most rapidly)
"""

import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
from scipy.interpolate import UnivariateSpline  # For fitting and differentiating spline curves

# Open and read the text file containing PSNR data
with open("psnr_lists.txt", "r") as file:
    content = file.read()

# Use a separate dictionary namespace to safely execute and load data
namespace = {}
exec(content, {}, namespace)  # Execute the content in an isolated dictionary

# Extract the relevant lists
discarded = np.array(namespace["Discarded_percentage"])  # x-axis: % of discarded coefficients
psnr_values = np.array(namespace["PSNR_average"])  # y-axis: average PSNR values

# Fit a smoothing spline
# Fit a univariate spline with a high smoothing factor (s=10000) to smooth the curve
spline = UnivariateSpline(discarded, psnr_values, s=10000)
fitted_psnr = spline(discarded)  # Evaluate the fitted spline at all discard levels

# Analyse the derivative
# Compute the first derivative of the spline (rate of PSNR change)
derivative = spline.derivative()(discarded)

# Only consider points where more than 20% of coefficients are discarded
mask = discarded > 20
filtered_discarded = discarded[mask]
filtered_derivative = derivative[mask]

# Find index of minimum derivative value (steepest PSNR drop)
min_deriv_idx = np.argmin(filtered_derivative)
sweet_spot_discarded = filtered_discarded[min_deriv_idx]  # Discard % at sweet spot
sweet_spot_psnr = spline(sweet_spot_discarded)  # Corresponding PSNR value

# Print results to console
print(f"Sweet spot at {sweet_spot_discarded}% discarded (PSNR ≈ {sweet_spot_psnr:.2f} dB)")

# Plot PSNR with spline
# Set consistent, readable font sizes for plots
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 22
})

# Plot original data and fitted spline
plt.figure(figsize=(10, 5))
plt.plot(discarded, psnr_values, 'o', label='Original Data', alpha=0.5)  # Scatter: raw data
plt.plot(discarded, fitted_psnr, '-', label='Spline Fit', linewidth=2)  # Line: smoothed spline
plt.axvline(sweet_spot_discarded, color='red', linestyle='--', label=f'Sweet Spot ≈ {sweet_spot_discarded}%')
plt.title("Spline Fit of PSNR vs Compression")
plt.xlabel("Discarded Coefficients (%)")
plt.ylabel("Average PSNR (dB)")
plt.legend()
plt.grid(False)
plt.tight_layout()

# Save the PSNR curve plot
filename = 'spline_fit.png'
plt.savefig(filename)
plt.close()


# Plot derivative
plt.figure(figsize=(10, 5))
plt.plot(discarded, derivative, '-b', label='d(PSNR)/d(Discarded%)')
plt.axvline(sweet_spot_discarded, color='red', linestyle='--', label='Min Derivative (Sweet Spot)')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Derivative of PSNR vs Discarded Coefficients")
plt.xlabel("Discarded Coefficients (%)")
plt.ylabel("Derivative of PSNR")
plt.legend()
plt.grid(False)
plt.tight_layout()

# Save the derivative plot
filename = 'derivative.png'
plt.savefig(filename)
plt.close()
