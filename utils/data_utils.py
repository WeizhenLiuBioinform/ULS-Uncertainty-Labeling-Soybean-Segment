import math
import os

import numpy as np
import torch

from monai import data, transforms
import nibabel as nib
from monai.data.meta_tensor import MetaTensor


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def load_data_list(file_path):
    with open(file_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    return file_names

def create_dataset(data_dir, file_list):
    data = []
    for file_name in file_list:
        ct_file = file_name.split("/")[-1]
        label_file = ct_file.replace("ct", "label")
        
        ct_path = os.path.join(data_dir, "ct", ct_file)
        label_path = os.path.join(data_dir, "label", label_file)
        
        data.append({"image": ct_path, "label": label_path})
    
    return data


def print_shape(x, message):
    shape = x.shape if isinstance(x, np.ndarray) else x.size()
    print(f"{message} - shape: {shape}")
    return x




def load_data(data_dir):
    ct_dir = os.path.join(data_dir, "ct")
    label_dir = os.path.join(data_dir, "label")
    
    data = []
    for file_name in sorted(os.listdir(ct_dir)):
        if file_name.endswith(".nii"):
            ct_path = os.path.join(ct_dir, file_name)
            label_path = os.path.join(label_dir, file_name)  # Assuming the label has the same name
            label_img = nib.load(label_path)
            label_meta_dict = {
                "affine": label_img.affine,
                "original_shape": label_img.shape,
            }
            image_meta_dict = {
                "filename_or_obj": ct_path
            }
            data.append({
                "image": ct_path, 
                "label": label_path, 
                "label_meta_dict": label_meta_dict,
                "image_meta_dict": image_meta_dict
            })
    
    return data

def print_data(data, msg=""):
    if data is None:
        print(f"{msg}: data is None")
        return data
    if isinstance(data, dict):
        for k, v in data.items():
            if v is not None:
                if isinstance(v, MetaTensor):
                    print(f"{msg} - {k} MetaTensor data:\n{v.as_tensor()}")
                else:
                    print(f"{msg} - {k} data:\n{v}")
            else:
                print(f"{msg} - {k}: None")
    elif isinstance(data, MetaTensor):
        print(f"{msg} MetaTensor data:\n{data.as_tensor()}")
    else:
        print(f"{msg} data:\n{data}")
    return data


def get_loader(data_dir, test_data_dir,train_list_path=None, val_list_path=None, args=None):
    data_dir = args.data_dir
    # datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: print_shape(x, "After LoadImaged")),
            transforms.EnsureChannelFirstD(keys=["image", "label"]),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: print_shape(x, "After AddChanneld")),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: print_shape(x, "After Orientationd")),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            # transforms.Lambdad(keys=["image", "label"], func=lambda x: print_shape(x, "After Spacingd")),
            # transforms.ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            # transforms.Lambdad(keys=["image", "label"], func=lambda x: print_shape(x, "After ScaleIntensityRanged")),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: print_data(x, "33Dddddata before RandCropByPosNegLabeld")),
            transforms.CropForegroundd(keys=["image", "label"], source_key="label"),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: print_shape(x, "After CropForegroundd")),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: print_data(x, "44Dddddata before RandCropByPosNegLabeld")),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: print_shape(x, "After RandCropByPosNegLabeld")),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstD(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            # transforms.ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="label"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstD(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            # transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            # transforms.ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            transforms.ToTensord(keys=["image", "label"]),
            transforms.EnsureTyped(keys=["image", "label", "label_meta_dict", "image_meta_dict"]),
        ]
    )

    if args.test_mode:
        # test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_files = load_data(test_data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )

        # # 添加调试信息
        # for i, batch in enumerate(test_loader):
        #     print(f"Batch {i} keys: {batch.keys()}")
        #     if 'label_meta_dict' in batch:
        #         print(f"Batch {i} label_meta_dict: {batch['label_meta_dict']}")
        #     if 'image_meta_dict' in batch:
        #         print(f"Batch {i} image_meta_dict: {batch['image_meta_dict']}")
        #     else:
        #         print(f"Batch {i} is missing 'image_meta_dict'")
        
        return test_loader
    else:
        train_file_list = load_data_list(train_list_path)
        val_file_list = load_data_list(val_list_path)
        
        train_files = create_dataset(data_dir, train_file_list)
        val_files = create_dataset(data_dir, val_file_list)
        for file in train_files:
            print(file)
        for file in val_files:
            print(file)
        train_ds = data.CacheDataset(
            data=train_files, transform=train_transform, cache_num=len(train_files), cache_rate=1.0, num_workers=args.workers
        )
        train_sampler = None 
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = None 
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]
        return loader
        
