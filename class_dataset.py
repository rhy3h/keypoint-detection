import os, json, cv2, shutil

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision.transforms import functional as F

class ClassDataset(Dataset):
    def __init__(self, root, transform = None, num_keypoints = None):
        self.root = root
        self.transform = transform
        self.num_keypoints = num_keypoints
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']

            bboxes_labels_original = ['Model' for _ in bboxes_original]

        if self.transform:
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]

            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']
            
            keypoints_transformed_unflattened = np.reshape(
                np.array(transformed['keypoints']),
                (-1, self.num_keypoints, 2)
            ).tolist()

            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):
                obj_keypoints = []
                for k_idx, kp in enumerate(obj):
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)

        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)
        target["image_id"] = idx
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.imgs_files)

def split_data_set(root, ratio: float):
    all_file_list = sorted(os.listdir(root))

    image_file_list = [f for f in all_file_list if f.endswith('.jpeg')]
    annotation_file_list = [f for f in all_file_list if f.endswith('.json')]

    result = []

    for image_file in image_file_list:
        annotation_file = os.path.splitext(image_file)[0] + '.json'
        if annotation_file in annotation_file_list:
            result.append([
                image_file,
                annotation_file
            ])

    total_size = len(result)
    first_part_size = int(total_size * ratio)

    first_part = result[:first_part_size]
    second_part = result[first_part_size:]

    return first_part, second_part

def copy_data_set_to_dir(source_root, file_list, target_root):
    print(f"Creating '{target_root}' folder")
    os.makedirs(target_root, exist_ok=True)

    images_folder_path = os.path.join(target_root, 'images')
    print(f"Creating '{images_folder_path}' folder")
    os.makedirs(images_folder_path, exist_ok=True)

    annotations_folder_path = os.path.join(target_root, 'annotations')
    print(f"Creating '{annotations_folder_path}' folder")
    os.makedirs(annotations_folder_path, exist_ok=True)

    idx = 0
    len_file_list = len(file_list)
    for image_file, annotation_file in file_list:
        image_file_path = os.path.join(source_root, image_file)
        annotation_file_path = os.path.join(source_root, annotation_file)

        target_image_file_path = os.path.join(target_root, 'images', image_file)
        target_annotation_file_path = os.path.join(target_root, 'annotations', annotation_file)

        shutil.copy(image_file_path, target_image_file_path)
        shutil.copy(annotation_file_path, target_annotation_file_path)

        print(f"Copied {idx + 1} / {len_file_list}")
        idx += 1

    return
