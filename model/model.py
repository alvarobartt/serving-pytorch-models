# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock


class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)

        self.fc = nn.Sequential(
            nn.Linear(self.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, self.num_classes),
            nn.LogSoftmax(dim=1)
        )