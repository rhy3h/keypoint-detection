import os

import torch
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn, ResNet50_Weights

def get_model(num_keypoints, weight_root = None):
    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0)
    )
    model = keypointrcnn_resnet50_fpn(
        weights = None,
        weights_backbone = ResNet50_Weights.DEFAULT,
        num_keypoints = num_keypoints,
        num_classes = 2,
        rpn_anchor_generator = anchor_generator
    )

    if weight_root:
        last_weight_path = os.path.join(weight_root, 'keypoints_rcnn_weights_last.pth')

        if os.path.isfile(last_weight_path):
            print(f"loading weight '{last_weight_path}'")
            state_dict = torch.load(last_weight_path)
            model.load_state_dict(state_dict)

    return model
