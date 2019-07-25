import os
import time

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision.models as models
import numpy as np


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet152 = models.resnet18(pretrained=True)
        dim_in = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(dim_in,10)
        #self.BackNone = nn.Sequential(*list(self.resnet152.children())[:-1])

        # self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # self.conv1_1 = nn.Conv2d(32, 32, 3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        # self.conv4 = nn.Conv2d(128, 512, 3, padding=1)
        # self.conv4_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        # self.conv5_1 = nn.Conv2d(1024, 1024, 3, padding=1)
        # self.batchnormal2D32 = nn.BatchNorm2d(32)
        # self.batchnormal2D64 = nn.BatchNorm2d(64)
        # self.batchnormal2D128 = nn.BatchNorm2d(128)
        # self.batchnormal2D512 = nn.BatchNorm2d(512)
        # self.batchnormal2D1024 = nn.BatchNorm2d(1024)
        #
        # self.lin1 = nn.Linear(1024 * 1 * 1, 128)
        # self.lin2 = nn.Linear(128, 10)

        #self.finish = nn.Linear(128, 10)


    def forward(self, images: Tensor) -> Tensor:
        # x = F.relu(self.conv1(images))
        # x = F.relu(self.conv1_1(x))
        # x = F.max_pool2d(x, 2)
        # x = self.batchnormal2D32(x)
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv2_1(x))
        # x = F.max_pool2d(x, 2)
        # x = self.batchnormal2D64(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv3_1(x))
        # x = F.max_pool2d(x, 2)
        # x = self.batchnormal2D128(x)
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv4_1(x))
        # x = F.max_pool2d(x, 2)
        # x = self.batchnormal2D512(x)
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv5_1(x))
        # x = F.max_pool2d(x, 2)
        # x = self.batchnormal2D1024(x)
        # x = F.dropout2d(x)
        # x = x.view(-1, 1024)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x)
        # x = F.relu(self.lin2(x))

        x = self.resnet152(images)


        return x

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:

        return F.cross_entropy(logits, labels)

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
