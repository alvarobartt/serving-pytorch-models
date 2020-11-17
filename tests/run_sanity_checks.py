import io
import json
import os
from random import choice

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import BasicBlock, ResNet

TEST_DIR = "dataset/test"

ID2LABEL = {
    0: 'chicken_curry',
    1: 'chicken_wings',
    2: 'fried_rice',
    3: 'grilled_salmon',
    4: 'hamburger',
    5: 'ice_cream',
    6: 'pizza',
    7: 'ramen',
    8: 'steak',
    9: 'sushi'
}


class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)

        self.fc = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

image_processing = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(
    root=test_data_path,
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=False
)

model = ImageClassifier()
model.load_state_dict(torch.load("model/foodnet_resnet18.pth", map_location=torch.device('cpu')))

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels)
    
loss = running_loss / len(test_dataset)
acc = running_corrects.double() / len(test_dataset)

with open("results.txt", "w") as f:
    f.write(pd.DataFrame({'accuracy': acc, 'loss': loss}).to_markdown())

# for key, value in ID2LABEL.items():
#     path = f"{TEST_DIR}/{value}"
#     random_image = choice(os.listdir(path))
#     random_image = Image.open(f"{path}/{random_image}")
#     random_image = image_processing(random_image)
    
#     with torch.no_grad():
#         outputs = model(random_image.unsqueeze(0))
#         _, preds = torch.max(outputs, 1)
#         predicted_class = ID2LABEL[preds.numpy()[0]]