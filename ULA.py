
# -*- coding:utf-8 -*-

import os
import nibabel as nib
import numpy as np


def load_nii_to_numpy(folder_path):
    numpy_arrays = []
    for file in os.listdir(folder_path):
        if file.endswith('.nii'):
            file_path = os.path.join(folder_path, file)
            print(f"load {file_path}")
            nii_data = nib.load(file_path)
            data_array = nii_data.get_fdata()
            numpy_arrays.append(data_array)
    return numpy_arrays


def ULA_process(
    labels: np.array,
    ) -> np.array:
    """
    The process of uncertainty label assignment
    
    Args:
    labels: The origin annotation.
    
    Returns:
    uncertainty_labels: The uncertainty annotation.
    """
    uncertainty_labels = []
    for label in labels:
        new_label = np.copy(label).astype(np.int32)
        for i in range(1, label.shape[0] - 1):
            for j in range(1, label.shape[1] - 1):
                for k in range(1, label.shape[2] - 1):
                    current_label = label[i, j, k]
                    if current_label == 2 or current_label == 3:
                        neighbors = [label[i+1, j, k], label[i-1, j, k], 
                                    label[i, j+1, k], label[i, j-1, k], 
                                    label[i, j, k+1], label[i, j, k-1]]
                        if (current_label == 2 and 3 in neighbors) or (current_label == 3 and 2 in neighbors):
                            new_label[i, j, k] = 4
                        
        uncertainty_labels.append(new_label)
    return uncertainty_labels


def save_nii_from_numpy(numpy_arrays, original_files, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for array, orig_file in zip(numpy_arrays, original_files):
        original_nii = nib.load(orig_file)
        new_nii = nib.Nifti1Image(array.astype(original_nii.get_data_dtype()), original_nii.affine, original_nii.header)
        # new_nii = nib.Nifti1Image(array, original_nii.affine, original_nii.header)
        new_file_path = os.path.join(output_folder, os.path.basename(orig_file))
        nib.save(new_nii, new_file_path)
        print(f"文件 {os.path.basename(orig_file)} save in  {new_file_path}")

label_folder_path = ./raw_test_data_4label/label'  
labels = load_nii_to_numpy(label_folder_path)

modified_labels = ULA_process(labels)
original_label_files = [os.path.join(label_folder_path, f) for f in os.listdir(label_folder_path) if f.endswith('.nii')]
output_folder = './raw_test_data_5label/label'  
save_nii_from_numpy(modified_labels, original_label_files, output_folder)
