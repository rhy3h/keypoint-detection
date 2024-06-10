from datetime import datetime
import os
import shutil

import torch

from engine import train_one_epoch, evaluate

def start_train(model, data_loader_train, data_loader_test, weight_root = ''):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 5

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device)

    now = datetime.now().strftime("%m%d_%H%M")
    weight_file_name = f'{now}_keypoints_rcnn_weights.pth'
    weight_path = os.path.join(weight_root, weight_file_name)

    # Save model weights after training
    torch.save(model.state_dict(), weight_path)
    print(f"saving weight '{weight_path}'")

    last_weight_file_name = 'last_keypoints_rcnn_weights.pth'
    last_weight_path = os.path.join(weight_root, last_weight_file_name)
    shutil.copy(weight_path, last_weight_path)
    print(f"saving weight '{last_weight_path}'")
