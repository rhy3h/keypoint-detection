import json

import cv2
from torchvision.transforms import functional
import numpy as np
import torch
import matplotlib.pyplot as plt

def trans_img_2_np(img):
    img = functional.to_tensor(img)

    return (img.permute(1,2,0).numpy() * 255).astype(np.uint8)

def trans_bboxes_2_np(bboxes):
    bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
    bboxes = bboxes.detach().cpu().numpy().astype(np.int32).tolist()

    return bboxes

def trans_keypoints_2_np(keypoints):
    keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
    keypoints = keypoints.detach().cpu().numpy().astype(np.int32).tolist()

    result = []
    for kps in keypoints:
        result.append([kp[:2] for kp in kps])

    return result

def visualize(image, bboxes, keypoints):
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)

    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " , tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    plt.imshow(image)

def visualize_by_predict(image, bboxes, keypoints):
    image = trans_img_2_np(image)
    bboxes = trans_bboxes_2_np(bboxes)
    keypoints = trans_keypoints_2_np(keypoints)

    visualize(image, bboxes, keypoints)

def visualize_by_file(image_path, annotation_path):
    data = {}

    data['img'] = cv2.imread(image_path)
    data['img'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)

    with open(annotation_path) as f:
        json_data = json.load(f)
        data['bboxes'] = json_data['bboxes']
        data['keypoints'] = json_data['keypoints']

    img = trans_img_2_np(data['img'])
    bboxes = trans_bboxes_2_np(data['bboxes'])
    keypoints = trans_keypoints_2_np(data['keypoints'])

    visualize(img, bboxes, keypoints)
