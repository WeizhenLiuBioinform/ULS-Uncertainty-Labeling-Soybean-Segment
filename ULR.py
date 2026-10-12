import glob
import os
import SimpleITK as sitk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from utils.utils import iou


def read_txt_file(
    txt_file_path: str,
    ) -> str:
    """
    Read each line in the text file records the IoU values for each sample.
    
    Args:
    txt_file_path: The file path to record the IoU results in a text file.
    
    Returns:
    lines: Each line in the text file records the IoU values for each sample.
    """
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def calculate_average_iou_per_category(
    lines: str,
    ) -> list:
    """
    Calculate the mean IoU of each class.
    
    Args:
    lines: Each line in the text file records the IoU values for each sample.
    
    Returns:
    average_iou_per_category: List of mean IoU values for each class.
    """
    iou_sums = {}
    iou_counts = {}
    
    for line in lines:
        parts = line.strip().split(' ')
        for part in parts[1:]:  
            category_iou = part.split(':')
            if len(category_iou) == 2:  
                category, iou_str = category_iou
                try:
                    iou = float(iou_str)
                    if not np.isnan(iou) and iou != 0: 
                        if category in iou_sums:
                            iou_sums[category] += iou
                            iou_counts[category] += 1
                        else:
                            iou_sums[category] = iou
                            iou_counts[category] = 1
                except ValueError:
                    continue  
    average_iou_per_category = {category: iou_sums[category] / iou_counts[category] for category in iou_sums}
    
    return average_iou_per_category

def load_image(path):
    sitk_image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(sitk_image)

def ULR_process(
    label: np.array,
    img: np.array,
    pred: np.array,
    ) -> np.array:
    """
    Perform uncertainty label reassignment on the network output results for each scan.
    
    Args:
    label: Ground truth label.
    img: Input CT scan
    pred: output of segmentation network.
    
    Returns:
    pred: The final predicted result.
    label: Ground truth label.
    """
    label4_indices = np.where(pred == 4)
    if len(label4_indices[0]) == 0:
        return pred, label  
    label4_intensities = img[label4_indices].reshape(-1, 1)
    label23_indices = np.where((label == 2) | (label == 3))
    label23_intensities = img[label23_indices].reshape(-1, 1)

    if len(label23_intensities) == 0:
        return pred, label
    if len(label23_intensities) < 2:
        return pred, label
    kmeans = KMeans(n_clusters=2, random_state=0).fit(label23_intensities)
    label23_kmeans_labels = kmeans.predict(label23_intensities)

    X_train = np.hstack([label23_intensities, label23_kmeans_labels.reshape(-1, 1),
                        label23_indices[0].reshape(-1, 1),label23_indices[1].reshape(-1, 1),label23_indices[2].reshape(-1, 1)])
    y_train = label[label23_indices].flatten()   
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    label4_kmeans_labels = kmeans.predict(label4_intensities)
    X_test = np.hstack([label4_intensities, label4_kmeans_labels.reshape(-1, 1),
                        label4_indices[0].reshape(-1, 1),label4_indices[1].reshape(-1, 1),label4_indices[2].reshape(-1, 1)])
    label4_new_labels = knn.predict(X_test)

    for i, new_label in zip(range(len(label4_indices[0])), label4_new_labels):
        pred[label4_indices[0][i], label4_indices[1][i], label4_indices[2][i]] = new_label

    return pred, label

def calculate_iou(
    pred: np.array,
    label: np.array, 
    num_classes: int,
    ) -> list:
    """
    Calculate the IoU.
    
    Args:
    pred: The final predicted result.
    label: The ground truth label.
    num_classes: Total number of classes.
    
    Returns:
    iou_per_class: List of IoU values for each class.
    """
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        iou_score = iou(pred_inds, label_inds)
        iou_per_class.append(iou_score)
    
    return iou_per_class

    

def process_files_in_directory(
    pred_path: str, 
    directory_path: str, 
    txt_save_path: str, 
    save_path: str,
    ):
    """
    Create folders and files to save the results.
    
    Args:
    pred_path: The folder path to store the output of segmentation network.
    directory_path: The folder path to store the ground truth labels.
    txt_save_path: The file path to record the IoU results in a text file.
    save_path: The folder path to store the final prediction results.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pred_files = glob.glob(os.path.join(pred_path, '*.nii'))

    pred_files.sort()
    for pred_file in pred_files:
        common_part = os.path.basename(pred_file)  
        common_part = os.path.splitext(common_part)[0]  
        common_part = os.path.splitext(common_part)[0] 
        
        img_file = os.path.join(directory_path, 'ct', common_part + '.nii')
        label_file = os.path.join(directory_path, 'label', common_part + '.nii')

        if os.path.exists(img_file) and os.path.exists(label_file):
            pred = load_image(pred_file)
            img = load_image(img_file)
            label = load_image(label_file)
            processed_label, label = ULR_process(label, img, pred)
            iou_per_class = calculate_iou(processed_label, label, num_classes=4)

            with open(txt_save_path, 'a') as file:
                file.write(f"{common_part}: ")
                for index, iou in enumerate(iou_per_class, start=0):
                    file.write(f"类别{index}:{iou} ")
                file.write("\n")
            processed_pred_sitk = sitk.GetImageFromArray(processed_label)
            processed_label_path = os.path.join(save_path, 'processed_label_new', common_part +  '_processed.nii')
            os.makedirs(os.path.dirname(processed_label_path), exist_ok=True)
            sitk.WriteImage(processed_pred_sitk, processed_label_path)
        else:
            print(f"Missing image or label for {common_part}")

def main():
    txt_save_path = './output_post/result.txt'
    save_path = './output_post/'
    directory_path = 'fixed_test_data_4label'
    pred_path = './output/'
    process_files_in_directory(pred_path, directory_path, txt_save_path,save_path)
    lines = read_txt_file(txt_save_path)
    average_iou_per_category = calculate_average_iou_per_category(lines)

    with open(txt_save_path, 'a') as file:
        for category, average_iou in average_iou_per_category.items():
            print(f"category {category} mean IOU: {average_iou}")
            file.write(f"category {category} mean IOU: {average_iou}")
            file.write("\n")

if __name__ == "__main__":
    main()


