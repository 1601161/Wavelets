#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs wavelet-based image compression on CT scan slices 
stored in NIfTI format using the Fast Wavelet Transform (FWT). It evaluates 
how compression performance varies across different Daubechies wavelets 
(db2–db10), decomposition levels and coefficient retention ratios.

Each application of the FWT constitutes one "level" of decomposition, 
where the approximation (low-pass) coefficients from the previous level 
are further decomposed. 

Steps:
1. Load and normalize the central axial slice from each 3D CT scan.
2. Visualize wavelet decompositions at increasing levels (e.g., level 1–5).
3. Apply FWT up to a chosen level, retain a fraction of the largest 
   wavelet coefficients, and reconstruct the image.
4. Save the original, reconstructed, and difference images.
5. Report average PSNR across all test images.

Compression parameters include:
- Wavelet type (db2 to db10)
- Decomposition level (1 to max allowed by image size)
- Retention ratio (e.g., 10%, 25%, 50% of coefficients)


INPUT:
   - RadioLUNG Folder of NIfTI files (.nii)
   - List of Daubechies wavelets: db2 through db10
   - Retention ratios (fraction of coefficients retained): [0.01, 0.03, 0.05, 0.10]
   - Parameters like maximum decomposition level (max_cap), PSNR threshold

OUTPUT:
    - 'daubechies_level_results.txt'
        Tab-separated file with columns:  Wavelet | Vanishing_Moments | Level | Retention | Avg_PSNR

   - PNG images saved to folder 'visual_results_db_level/':
       For each wavelet–level–retention–image configuration:
           • Original image (only saved for level=1):
               'imgXX_wavelet-dbN_L1_rYY_original.png'
           • Reconstructed image:
               'imgXX_wavelet-dbN_LX_rYY_reconstructed.png'
           • Absolute error image (|original - recon|):
               'imgXX_wavelet-dbN_LX_rYY_diff.png'
       where:
           - XX: image index (e.g. 00, 01, ...)
           - dbN: Daubechies wavelet (e.g. db4)
           - LX: decomposition level (e.g. L1, L2, ...)
           - rYY: retention percentage (e.g. r03 for 3%)

   - Console output:
       For each tested configuration:
           • Wavelet and retention being tested
           • Level being tested
           • Average PSNR achieved
           • Messages if PSNR improvement is below threshold or if max level is exceeded
       Display of FWT decomposition (cA, cH, cV, cD) for the 11th image 
"""

import nibabel as nib                          # For loading NIfTI (.nii/.nii.gz) medical image files
import os                                      # For file and directory operations
import pywt                                    # For performing the Fast Wavelet Transform (FWT)
import numpy as np                             # For numerical operations and arrays
import matplotlib.pyplot as plt                # For image visualization
import matplotlib.image as mpimg               # For saving images using matplotlib
from skimage.metrics import peak_signal_noise_ratio as compute_psnr  # For computing PSNR metric

# Create results folder if it doesn't exist
os.makedirs("visual_results_db_level", exist_ok=True)

# Load and normalize 2D slices from NIfTI
def load_nii_images(folder_path):
    images = []  # List to store the 2D slices
    for i, fname in enumerate(sorted(os.listdir(folder_path))):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):  # Only process NIfTI files
            print(f"Loading image {i+1}: {fname}")
            path = os.path.join(folder_path, fname)
            nii = nib.load(path)              # Load the .nii or .nii.gz file
            data = nii.get_fdata()            # Extract the 3D array from the file
            mid_slice = int(data.shape[2] // 2)  # Get the index of the middle axial slice
            slice_2d = data[:, :, mid_slice]     # Extract the middle 2D slice
            norm_slice = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())  # Normalize to [0, 1]
            images.append(norm_slice)            # Add to the list of images
    print(f"Finished loading {len(images)} images.\n")
    return images

# Show decompositions (cA, cH, cV, cD) for first image, all wavelets and levels
def show_decompositions_for_first_image(img, wavelets, max_level=5):
    for wavelet in wavelets:
        print(f"\nWavelet: {wavelet}")
        max_possible = pywt.dwt_max_level(min(img.shape), pywt.Wavelet(wavelet).dec_len)  # Max levels for this image
        for level in range(1, min(max_possible, max_level) + 1):
            coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)  # Multilevel 2D decomposition
            cA, details = coeffs[0], coeffs[1:]  # cA is final approximation; details = list of (cH, cV, cD)
            cH, cV, cD = details[-1]            # Show only the last level's details
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(cA, cmap='gray'); axes[0].set_title(f'{wavelet} - Level {level} Approximation (cA)')
            axes[1].imshow(cH, cmap='gray'); axes[1].set_title('Horizontal Detail (cH)')
            axes[2].imshow(cV, cmap='gray'); axes[2].set_title('Vertical Detail (cV)')
            axes[3].imshow(cD, cmap='gray'); axes[3].set_title('Diagonal Detail (cD)')
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

# Compress with fixed retention
def compress_with_retention(img, wavelet_name, level, retention_ratio):
    coeffs = pywt.wavedec2(img, wavelet=wavelet_name, level=level)  # Decompose image into wavelet coeffs
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)  # Flatten the coefficient tree into a single array
    N = arr.size
    k = int(N * retention_ratio)                      # Number of coefficients to retain
    flat = np.abs(arr).flatten()
    threshold = np.partition(flat, -k)[-k]            # Get cutoff value to retain largest-k coefficients
    arr_thresh = np.where(np.abs(arr) >= threshold, arr, 0)  # Zero out all below threshold
    coeffs_thresh = pywt.array_to_coeffs(arr_thresh, coeff_slices, output_format='wavedec2')  # Repack into tree
    img_reconstructed = pywt.waverec2(coeffs_thresh, wavelet=wavelet_name)  # Reconstruct image from pruned coeffs
    img_reconstructed = np.clip(img_reconstructed, 0, 1)  # Clip to [0, 1]
    img_reconstructed = img_reconstructed[:img.shape[0], :img.shape[1]]  # Crop in case of padding
    return img_reconstructed

# Run adaptive compression experiment
def run_full_experiment(images, wavelets, retention_ratios, log_file_path='daubechies_level_results.txt', psnr_threshold=0.05, max_cap=10):
    results = []  # To store summary of all tests
    config_count = 1  # Counter for configurations
    with open(log_file_path, 'w') as log_file:
        log_file.write("Wavelet\tVanishing_Moments\tLevel\tRetention\tAvg_PSNR\n")  # Header for results
        for wavelet in wavelets:
            try:
                vanishing_moments = int(wavelet.replace("db", ""))  # Extract number of vanishing moments
            except ValueError:
                vanishing_moments = 0
            for ratio in retention_ratios:
                print(f"\nTesting wavelet '{wavelet}', retention {int(ratio * 100)}%...")
                previous_psnr = -np.inf  # To compare improvements
                for level in range(1, max_cap + 1):
                    print(f"  Level {level}...")
                    psnrs = []
                    for i, img in enumerate(images):
                        if i == 0:
                            max_possible_level = pywt.dwt_max_level(min(img.shape), pywt.Wavelet(wavelet).dec_len)
                        try:
                            if level > max_possible_level:
                                print(f"    Skipping: level {level} exceeds max {max_possible_level}")
                                continue
                            recon = compress_with_retention(img, wavelet, level, ratio)  # Compress and reconstruct
                            psnr = compute_psnr(img, recon)  # Compute PSNR

                            # Save visualization (only original once)
                            img_name = f"img{i:02d}_wavelet-{wavelet}_L{level}_r{int(ratio*100)}"
                            orig_path = f"visual_results_db_level/{img_name}_original.png"
                            recon_path = f"visual_results_db_level/{img_name}_reconstructed.png"
                            diff_path = f"visual_results_db_level/{img_name}_diff.png"
                            if level == 1:
                                mpimg.imsave(orig_path, img, cmap='gray')  # Save original image
                            mpimg.imsave(recon_path, recon, cmap='gray')  # Save reconstructed image
                            diff_img = np.abs(img - recon)
                            mpimg.imsave(diff_path, diff_img, cmap='hot')  # Save heatmap of differences

                            psnrs.append(psnr)
                        except Exception as e:
                            print(f"    Error: {e} — skipping image {i+1}")
                    if not psnrs:
                        print(f"  No PSNR results at level {level} — stopping.")
                        break
                    avg_psnr = np.mean(psnrs)
                    results.append((wavelet, vanishing_moments, level, ratio, avg_psnr))  # Store summary
                    log_file.write(f"{wavelet}\t{vanishing_moments}\t{level}\t{ratio:.2f}\t{avg_psnr:.4f}\n")
                    log_file.flush()
                    print(f"Avg PSNR: {avg_psnr:.2f}")
                    if avg_psnr - previous_psnr < psnr_threshold:  # Stop if gain is too small
                        print(f"PSNR gain < {psnr_threshold} dB — stopping at level {level}.")
                        break
                    previous_psnr = avg_psnr
                    config_count += 1
    print("\nExperiment finished. Results saved to:", log_file_path)
    return results

# Run everything
folder = 'RadioLUNG'
images = load_nii_images(folder)

if images:
    # Show one image
    print("Showing original image...")
    plt.imshow(images[10], cmap='gray')
    plt.title("Original First Image")
    plt.axis('off')
    plt.show()

    # Show decompositions for first image
    print("Showing FWT decompositions for first image...")
    wavelets = [f'db{i}' for i in range(2, 11)]  # db2 to db10
    show_decompositions_for_first_image(images[10], wavelets, max_level=6)

    # Run compression tests
    print("Running compression experiments...")
    retention_ratios = [0.01, 0.03, 0.05, 0.1]  # Percent of coeffs retained
    results = run_full_experiment(images, wavelets, retention_ratios, max_cap=10)
