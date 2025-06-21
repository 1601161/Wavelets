#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script converts a folder of 3D CT scans in NIfTI format (.nii)
into 2D PNG images. It extracts the middle axial slice from each scan, normalizes 
it to 8-bit grayscale, and saves it to the 'original cts' folder.

Input:  RadioLUNG Folder of NIfTI files (.nii)
Output: PNG images in 'original cts/'
"""

# Import necessary libraries
import os                            # For file and directory operations
import numpy as np                   # For numerical array manipulation
import nibabel as nib                # For reading .nii/.nii.gz medical image files
from skimage.io import imsave        # For saving images as PNG

# Function to convert NIfTI to PNG by extracting the middle slice
def convert_nifti_to_png(input_folder="RadioLUNG", output_folder="original cts"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all files in the input folder, sorted alphabetically
    for i, fname in enumerate(sorted(os.listdir(input_folder))):
        # Only process files with NIfTI extensions
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            print(f"[{i+1:02d}/92] Processing: {fname}")  # Status update in console
            
            # Build full file path and load the NIfTI image
            path = os.path.join(input_folder, fname)
            nii = nib.load(path)
            data = nii.get_fdata()  # Get image data as a 3D NumPy array
            
            # Extract the middle slice along the axial (z) direction
            mid_slice = int(data.shape[2] // 2)
            slice_2d = data[:, :, mid_slice]

            # Normalize the 2D slice to [0, 255] for 8-bit PNG
            norm_slice = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
            slice_uint8 = (norm_slice * 255).astype(np.uint8)

            # Clean up filename by removing double extension if present
            name, _ = os.path.splitext(fname)
            if name.endswith('.nii'):
                name = name.replace('.nii', '')

            # Construct output path and save the PNG image
            output_path = os.path.join(output_folder, f"{name}.png")
            imsave(output_path, slice_uint8)

    # Final message after all conversions are done
    print("\nAll CT slices converted and saved to 'original cts/'.")

# Main
if __name__ == "__main__":
    convert_nifti_to_png()  # Run the conversion when the script is executed
