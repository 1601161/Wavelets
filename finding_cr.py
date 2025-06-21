#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads the medical images in NIfTI format and applies wavelet-based 
compression by discarding a percentage of wavelet coefficients. It evaluates 
the compression quality using Peak Signal-to-Noise Ratio (PSNR) and outputs
results in both text and plot formats.

INPUTS:
    - The folder 'RadioLUNG' containing `.nii` files

OUTPUTS:
    - 'psnr_lists.txt': text file with PSNR values for each image and the average.
    - A plot of average PSNR vs. percentage of wavelet coefficients discarded.
"""

import os  # For file and folder handling
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results
import pywt  # For wavelet transforms
import nibabel as nib  # For loading NIfTI medical image files
from skimage.metrics import peak_signal_noise_ratio as psnr  # For measuring image reconstruction quality

# Load and normalize 2D slices from NIfTI
def load_nii_images(folder_path):
    images = []  # Store normalized 2D slices
    filenames = []  # Store corresponding filenames
    for i, fname in enumerate(sorted(os.listdir(folder_path))):  # Iterate through sorted files in the folder
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):  # Only process NIfTI files
            print(f"Loading image {i+1}: {fname}")
            path = os.path.join(folder_path, fname)  # Full file path
            nii = nib.load(path)  # Load NIfTI image
            data = nii.get_fdata()  # Extract image data as NumPy array
            mid_slice = int(data.shape[2] // 2)  # Take middle slice along z-axis
            slice_2d = data[:, :, mid_slice]  # Extract 2D slice
            norm_slice = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())  # Normalize to [0, 1]
            images.append(norm_slice)  # Store normalized image
            filenames.append(fname)  # Store filename
    print(f"Finished loading {len(images)} images.\n")
    return images, filenames  # Return all images and filenames

# Wavelet compression and PSNR computation
def wavelet_compress_psnr(image, wavelet='db4', level=6, discard_percents=None):
    if discard_percents is None:
        discard_percents = [round(p, 2) for p in np.arange(0, 100, 1)] + [99]  # Default discard levels

    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)  # Perform 2D wavelet decomposition
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)  # Flatten coefficients to a single array
    psnr_values = []  # Store PSNR values at each compression level

    for percent in discard_percents:
        thresh = np.percentile(np.abs(coeff_arr), percent)  # Find threshold to discard 'percent' of smallest values
        compressed_arr = coeff_arr * (np.abs(coeff_arr) >= thresh)  # Zero out coefficients below threshold
        compressed_coeffs = pywt.array_to_coeffs(compressed_arr, coeff_slices, output_format='wavedec2')  # Reconstruct coeffs
        reconstructed = pywt.waverec2(compressed_coeffs, wavelet=wavelet)  # Reconstruct image
        reconstructed = np.clip(reconstructed, 0, 1)  # Clip values to [0, 1]

        value = psnr(image, reconstructed, data_range=1.0)  # Compute PSNR
        psnr_values.append(value)  # Store result

    return psnr_values  # Return all PSNR values for the image

# Compute PSNR matrix [images x discard levels]
def compute_psnr_matrix(images, wavelet='db4', level=6, discard_percents=None):
    if discard_percents is None:
        discard_percents = [round(p, 2) for p in np.arange(0, 100, 1)] + [99]  # Default discards

    psnr_matrix = []  # Store PSNR lists for all images

    for idx, image in enumerate(images):
        print(f"Processing image {idx + 1}/{len(images)}")
        psnrs = wavelet_compress_psnr(image, wavelet=wavelet, level=level, discard_percents=discard_percents)
        psnr_matrix.append(psnrs)  # Append image's PSNR list

    psnr_matrix = np.array(psnr_matrix)  # Convert to NumPy array
    avg_psnr = np.mean(psnr_matrix, axis=0)  # Average PSNR across all images
    return discard_percents, psnr_matrix, avg_psnr  # Return results

# Save PSNR results as list-style txt
def save_lists_as_txt(discard_percents, psnr_matrix, avg_psnr, filenames, filename="psnr_lists.txt"):
    with open(filename, 'w') as f:
        percent_list = [int(p) for p in discard_percents]
        f.write("Discarded_percentage = " + repr(percent_list) + "\n\n")  # Write discard levels

        for fname, psnrs in zip(filenames, psnr_matrix):
            base_name = os.path.splitext(os.path.basename(fname))[0]  # Extract base name without extension
            psnr_list = [float(p) for p in psnrs]
            f.write(f"PSNR_{base_name} = " + repr(psnr_list) + "\n")  # Write per-image PSNR

        avg_list = [float(p) for p in avg_psnr]
        f.write("\nPSNR_average = " + repr(avg_list) + "\n")  # Write average PSNR
        # Print confirmation message
    print(f"Saved list-style PSNR data to '{filename}'.")

# Main
if __name__ == "__main__":
    folder_path = "RadioLUNG"  # Folder containing .nii
    images, filenames = load_nii_images(folder_path)  # Load and normalize slices

    discard_percents = [round(p, 2) for p in np.arange(0, 100, 1)] + [99]  # Compression levels
    discard_percents, psnr_matrix, avg_psnr = compute_psnr_matrix(
        images, wavelet='db4', level=6, discard_percents=discard_percents
    )

    save_lists_as_txt(discard_percents, psnr_matrix, avg_psnr, filenames)  # Save results

    # Plot average PSNR curve
    plt.figure(figsize=(8, 5))
    plt.plot(discard_percents, avg_psnr, marker='o', color='purple')
    plt.xlabel("Percentage of Coefficients Discarded")
    plt.ylabel("Average PSNR (dB)")
    plt.title("Average PSNR vs Compression over Radiographic Images")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
