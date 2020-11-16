import os

from random import choice

from PIL import Image

import pandas as pd

import torch
import torch.nn as nn

from torchvision import transforms as T 
from torchvision.models.resnet import ResNet, BasicBlock

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

model = ImageClassifier()
model.load_state_dict(torch.load("model/foodnet_resnet18.pth"))

results = list()

for key, value in ID2LABEL.items():
    path = f"{TEST_DIR}/{value}"
    random_image = choice(os.listdir(path))
    random_image = Image.open(f"{path}/{random_image}")
    random_image = image_processing(random_image)
    

    with torch.no_grad():
        outputs = model(random_image.unsqueeze(0))
        _, preds = torch.max(outputs, 1)
        predicted_class = ID2LABEL[preds.numpy()[0]]

    results.append(f"{path} image was {value}, but the model predicted {predicted_class}")

with open('results.txt', 'w') as f:
    f.write('## Results')
    for result in results:
        f.write(result)