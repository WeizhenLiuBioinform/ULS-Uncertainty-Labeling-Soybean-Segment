import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
from patchify import patchify
import random
import nibabel as nib



class SOY_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, window_size, threshold_min, threshold_max) :
        self.raw_dataset_path = raw_dataset_path
        self.fixed_dataset_path = fixed_dataset_path
        self.classes = 5 
        self.valid_rate = 0.2
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.window_size = window_size

    def fix_data(self):
        if not os.path.exists(self.fixed_dataset_path):   
            os.makedirs(join(self.fixed_dataset_path,'ct'))
            os.makedirs(join(self.fixed_dataset_path, 'label'))
        file_list = os.listdir(join(self.raw_dataset_path,'ct'))
       

        Numbers = len(file_list)
        print('Total numbers of samples is :',Numbers)

        for ct_file , i in zip(file_list , range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i+1,Numbers))
            ct_path = os.path.join(raw_dataset_path, 'ct', ct_file)
            seg_path = os.path.join(raw_dataset_path, 'label', ct_file.replace('volume', 'segmentation'))
            ct_patch_list, seg_patch_list = self.process(ct_path, seg_path, classes = self.classes)

            for k, (ct_slice, seg_slice) in enumerate(zip(ct_patch_list, seg_patch_list)):
                    ct_affine = nib.load(ct_path).affine
                    seg_affine = nib.load(seg_path).affine
                
                    ct_nii = nib.Nifti1Image(ct_slice, affine=ct_affine)
                    seg_nii = nib.Nifti1Image(seg_slice, affine=seg_affine)

                    ct_save_name = os.path.splitext(ct_file)[0]
                    ct_save_path = os.path.join(self.fixed_dataset_path, 'ct', f'{ct_save_name}_{k}.nii')
                    seg_file = ct_file.replace('volume', 'segmentation')
                    seg_save_name = os.path.splitext(seg_file)[0]
                    seg_save_path = os.path.join(self.fixed_dataset_path, 'label', f'{seg_save_name}_{k}.nii')

                    nib.save(ct_nii, ct_save_path)
                    nib.save(seg_nii, seg_save_path)
                  




    def process(self, ct_path, seg_path, classes):

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_patch_list, seg_patch_list = self.slice_cut(ct_array, seg_array , self.window_size, self.threshold_min, self.threshold_max)

        return ct_patch_list,seg_patch_list

    

    def slice_cut(self, ct_array, seg_array, window_size, threshold_min, threshold_max):
        data_shape = ct_array.shape

        seg_array_test = seg_array.copy()
        seg_array_test[(0 < seg_array_test) & (seg_array_test < 5)] = 1



        flag = False
        max_label_ratio = 0
        max_ct_patch = None
        max_seg_patch = None

        new_ct_list = []
        new_seg_list = []


        for x in range(0, data_shape[0], window_size[0] // 2):
            for y in range(0, data_shape[1], window_size[1] // 2):
                for z in range(0, data_shape[2], window_size[2] // 2):
                    ct_patch = ct_array[x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]] 
                    seg_patch = seg_array[x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]]
                    seg_patch_test = seg_array_test[x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]]
                    
                    label_ratio = np.mean(seg_patch_test)
                    print(f"Label ratio for patch starting at ({x}, {y}, {z}): {label_ratio}")
                    if not flag and label_ratio > max_label_ratio:
                        max_label_ratio = label_ratio
                        
                        max_ct_patch = ct_patch
                        max_seg_patch = seg_patch

                    #if threshold_min <= label_ratio <= threshold_max:
                    if threshold_min <= label_ratio:
                        flag = True
                        ct_patch = ct_patch.astype(np.uint16)
                        seg_patch = seg_patch.astype(np.uint8)

                        if ct_patch.shape == window_size:
                            new_ct_list.append(ct_patch)
                            new_seg_list.append(seg_patch)
                    
        if not flag:
            print("False")
            ct_patch = max_ct_patch.astype(np.uint16)
            seg_patch = max_seg_patch.astype(np.uint8)

            if ct_patch.shape == window_size:
                new_ct_list.append(ct_patch)
                new_seg_list.append(seg_patch)

        print(f"slice num: {len(new_ct_list)}")

        return new_ct_list, new_seg_list
    


    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_dataset_path, "ct"))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)  

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num*(1-self.valid_rate))]
        val_name_list = data_name_list[int(data_num*(1-self.valid_rate)):int(data_num*((1-self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")

    
    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_dataset_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_dataset_path, 'ct', name)
            seg_path = os.path.join(self.fixed_dataset_path, 'label', name.replace('volume', 'segmentation'))           
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()





if __name__ == '__main__':
    raw_dataset_path = './dataset/raw_data_4label'
    fixed_dataset_path = './dataset/fixed_raw_data_4label'
    window_size = (128 ,128 ,128)  
    threshold_min = 0
    threshold_max = 0.046
    tool = SOY_preprocess(raw_dataset_path, fixed_dataset_path, window_size, threshold_min, threshold_max)
    tool.fix_data()
    tool.write_train_val_name_list()     