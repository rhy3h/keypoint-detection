from model import get_model

import torch
from torchvision.transforms import functional as F

import numpy as np
import cv2

import torchvision

class Predictor:
    def __init__(self, model_path: str, num_keypoints):
        self.model = get_model(num_keypoints, model_path)

        if torch.cuda.is_available():
            print('Using Cuda')
            self.model.cuda()

    def predict_with_image_path(self, image_path: str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = F.to_tensor(image).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        bboxes, keypoints = self.predict(image_tensor)

        return image, bboxes, keypoints

    def predict(self, image_tensor: torch.Tensor):
        with torch.no_grad():
            self.model.eval()
            output = self.model(image_tensor)

        scores = output[0]['scores'].detach().cpu().numpy()

        threshold = 0.7
        high_scores_idxs = np.where(scores > threshold)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))

        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        return bboxes, keypoints
