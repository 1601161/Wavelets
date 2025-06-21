#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs wavelet-based image compression on medical NIfTI images. 
Each image is compressed using the Daubechies-4 wavelet and applying 6-levels
of the Fast Wavelet Transform. Then, 27% of the coefficients are discarded and
the reconstruction is ssaved as a PNG file. The PSNR (Peak Signal-to-Noise 
Ratio) is also computed to evaluate compression quality.

INPUTS:
    - A folder (`RadioLUNG/`) containing 3D `.nii` files

OUTPUTS:
    - Compressed PNG images saved to: 'final compressed results/'
    - A text file ('final_compressed_data.txt') summarizing PSNR for each 
    compressed image
"""

import os
import numpy as np
import pywt  # Wavelet transforms
import nibabel as nib  # For reading NIfTI medical images
from skimage.metrics import peak_signal_noise_ratio as psnr  # Quality metric
from skimage.io import imsave  # Save PNG images

# Load and normalize the center slice from each NIfTI file
def load_nii_images(folder_path):
    images = []
    filenames = []
    for i, fname in enumerate(sorted(os.listdir(folder_path))):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            print(f"Loading image {i+1}: {fname}")
            path = os.path.join(folder_path, fname)
            
            # Load the 3D volume
            nii = nib.load(path)
            data = nii.get_fdata()
            
            # Get the middle slice along the z-axis
            mid_slice = int(data.shape[2] // 2)
            slice_2d = data[:, :, mid_slice]
            
            # Normalize to [0, 1]
            norm_slice = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
            
            images.append(norm_slice)
            filenames.append(os.path.splitext(fname)[0])  # Remove file extension
    print(f"Finished loading {len(images)} images.\n")
    return images, filenames

# Compress an image, save it as PNG, and return its PSNR
def wavelet_compress_and_save(image, filename, output_dir, wavelet='db4', level=6, discard_percent=27):
    # Decompose the image using wavelets
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # Apply hard thresholding to discard lowest-magnitude coefficients
    threshold = np.percentile(np.abs(coeff_arr), discard_percent)
    compressed_arr = coeff_arr * (np.abs(coeff_arr) >= threshold)

    # Reconstruct image from compressed coefficients
    compressed_coeffs = pywt.array_to_coeffs(compressed_arr, coeff_slices, output_format='wavedec2')
    reconstructed = pywt.waverec2(compressed_coeffs, wavelet=wavelet)
    
    # Clip reconstructed image to [0, 1]
    reconstructed = np.clip(reconstructed, 0, 1)

    # Save the reconstructed image as a PNG
    output_path = os.path.join(output_dir, f"{filename}_compressed.png")
    imsave(output_path, (reconstructed * 255).astype(np.uint8))

    # Return PSNR between original and reconstructed
    return psnr(image, reconstructed, data_range=1.0)

# Main
if __name__ == "__main__":
    input_dir = "RadioLUNG"  # Input folder with NIfTI images
    output_dir = "final compressed results"  # Output folder for PNGs and results
    
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load images and their filenames
    images, filenames = load_nii_images(input_dir)

    # File to log PSNR results
    results_path = os.path.join(output_dir, "final_compressed_data.txt")
    
    with open(results_path, "w") as f:
        # Compress each image and write its PSNR
        for image, name in zip(images, filenames):
            psnr_value = wavelet_compress_and_save(image, name, output_dir)
            f.write(f"{name}: PSNR = {psnr_value:.2f} dB\n")
    # Print confirmation message       
    print(f"\nAll PSNR values saved to: {results_path}")
    
    
    
    
    
    
    