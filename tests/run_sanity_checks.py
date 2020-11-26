import io
import os
from random import choice

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import BasicBlock, ResNet

SANITY_DIR = "dataset/sanity"

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

sanity_dataset = ImageFolder(
    root=SANITY_DIR,
    transform=image_processing
)

sanity_loader = DataLoader(
    sanity_dataset,
    batch_size=8,
    num_workers=0,
    shuffle=True
)

model = ImageClassifier()
model.load_state_dict(torch.load("model/foodnet_resnet18.pth", map_location=torch.device('cpu')))
model.eval();

criterion = nn.CrossEntropyLoss()

running_corrects, running_loss = .0, .0
all_preds = torch.Tensor()
shuffled_labels = torch.Tensor()

for inputs, labels in sanity_loader:
    inputs, labels = inputs.to('cpu'), labels.to('cpu')

    shuffled_labels = torch.cat((shuffled_labels, labels), dim=0)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

    all_preds = torch.cat((all_preds, preds), dim=0)

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels)

stacks = torch.stack((shuffled_labels.type(torch.int32), all_preds.type(torch.int32)), dim=1)
conf_mat = torch.zeros(len(ID2LABEL), len(ID2LABEL), dtype=torch.int32)

for stack in stacks:
    true_label, pred_label = stack.tolist()
    conf_mat[true_label, pred_label] += 1

with open("confusion_matrix.txt", "w") as f:
    f.write(pd.DataFrame(conf_mat.numpy(), index=list(ID2LABEL.values()), columns=list(ID2LABEL.values())).to_markdown())

loss = running_loss / len(sanity_dataset)
acc = running_corrects.double() / len(sanity_dataset)

with open("results.txt", "w") as f:
    f.write(pd.DataFrame([{'accuracy': acc, 'loss': loss}]).to_markdown())