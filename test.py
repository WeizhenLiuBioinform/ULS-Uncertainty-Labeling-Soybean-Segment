import argparse
import logging
import os

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
from utils.utils import iou, resample_3d
from unet import *
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR,VNet
from monai import transforms

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    

def main():
    parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
    parser.add_argument("--data_dir", default="./dataset/raw_test_data_4label", type=str, help="dataset directory")
    parser.add_argument("--exp_name", default="total", type=str, help="experiment name")
    parser.add_argument("--test_data_dir", default="./dataset/raw_test_data_5label", type=str, help="test dataset directory")
    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=5, type=int, help="number of output channels")
    parser.add_argument("--roi_x", default=64, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=64, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--logdir", default="experiments", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--log_file", default="./runs/total/test.log", type=str, help="log file")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logging.info("Starting test...")

    args.test_mode = True
    if args.test_mode:
        args.train_list_path = None
        args.val_list_path = None

    output_directory = os.path.join("./dadou/outputs" , args.exp_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    model_path = './runs/VNET_uncertain/model.pt'
    print(f"Loading model weights from {model_path}")
    if not os.path.isfile(model_path):
        print(f"Model weights not found at {model_path}, please check the path.")
        return
    
    val_loader = get_loader(args.data_dir, args.test_data_dir, args=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VNet(spatial_dims= 3,in_channels = args.in_channels,out_channels= args.out_channels,)
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {model_path}")
    # logging.info(f"Loaded model weights from {model_path}")
    
    model.eval()

    torch.backends.cudnn.enabled = False  


    # torch.backends.cudnn.enabled = False  

    with torch.no_grad():
        iou_list_case = []
        overall_iou_scores = {f"iou_Class_{i}": [] for i in range(0, 4)}
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            if "label_meta_dict" not in batch:
                print(f"Batch {i} is missing 'label_meta_dict', skipping this batch.")
                continue

            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            print(f"val_outputs shape: {val_outputs.shape}, val_labels shape: {val_labels.shape}")

            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)

                
            iou_list_sub = []
            iou_scores = {}
            for class_id in range(0, 4):
                organ_iou = iou(val_outputs == class_id, val_labels == class_id)
                iou_list_sub.append(organ_iou)
                iou_scores[f"iou_Class_{class_id}"] = organ_iou
                overall_iou_scores[f"iou_Class_{class_id}"].append(organ_iou)
            
            mean_iou = np.mean(iou_list_sub)
            # logging.info(f"Mean Organ iou: {mean_iou}")
            print("Mean Organ iou: {}".format(mean_iou))
            iou_list_case.append(iou_scores)

            print(", ".join([f"iou_Class_{class_id}: {iou_scores[f'iou_Class_{class_id}']}" for class_id in range(0, 4)]))
            logging.info(", ".join([f"iou_Class_{class_id}: {iou_scores[f'iou_Class_{class_id}']}" for class_id in range(0, 4)]))
            combined_dict = {
                "image": val_outputs[np.newaxis, ...],
                "label": val_labels[np.newaxis, ...],
            }
            combined_dict = transforms.Orientationd(keys=["image", "label"], axcodes="LPS")(combined_dict)
            val_outputs = combined_dict["image"][0] 
            val_labels = combined_dict["label"][0]

            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
            )

        overall_mean_iou = np.mean([np.mean(list(case.values())) for case in iou_list_case])
        print("Overall Mean iou: {}".format(overall_mean_iou))
        logging.info(f"Overall Mean iou: {overall_mean_iou}")

        for class_id in range(0, 4):
            class_ious = overall_iou_scores[f"iou_Class_{class_id}"]
            valid_class_ious = [d for d in class_ious if d > 0]
            if valid_class_ious:  
                mean_class_iou = np.mean(valid_class_ious)
                print(f"Overall iou for Class {class_id}: {mean_class_iou}")
                logging.info(f"Overall iou for Class {class_id}: {mean_class_iou}")
            else:
                print(f"Overall iou for Class {class_id}: No valid entries")
                logging.info(f"Overall iou for Class {class_id}: No valid entries")

if __name__ == "__main__":
    main()


