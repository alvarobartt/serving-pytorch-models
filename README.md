# Serving PyTorch models with TorchServe :fire: 

![PyTorch Logo](https://miro.medium.com/max/1024/1*KKADWARPMxHb-WMxCgW_xA.png)

__TorchServe is the ML model serving framework developed by PyTorch__.

Along this repository, the procedure so as to train and deploy a transfer learning CNN model which
classifies images retrieved from a slice of a well known food dataset, named 
[Food101](https://www.tensorflow.org/datasets/catalog/food101).

__WARNING__: TorchServe is experimental and subject to change.

---

## :closed_book: Table of Contents

- [Requirements](#hammer_and_wrench-requirements)
- [Dataset](#open_file_folder-dataset)
- [Modelling](#robot-modelling)
- [Deployment](#rocket-deployment)
- [Usage](#mage_man-usage)
- [Credits](#computer-credits)

---

## :hammer_and_wrench: Requirements

- torch, torchvision, torchserve, torch-model-archiver
- Java, JDK 11

---

## :open_file_folder: Dataset

TODO: explain what does the dataset contain
TODO: include an overview with a sample image per class

---

## :robot: Modelling

As the modelling is not the most relevant part/section that aims to be covered along this repository, we 
will just be using transfer learning from a pre-trained [ResNet](https://arxiv.org/abs/1512.03385) as it is 
the SOTA when it comes to image classification.

In this case, as we want to serve a PyTorch model, we will be using [PyTorch's implementation of ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
and more concretely, ResNet18, where the 18 stands for the number of layers that it contains.

TODO: Explain how to load the model and some considerations towards preparing the model for TorchServe.

TODO: Regarding the training process, ...

TODO: Finally, in order to dump ...

Once the state_dict has been generated from the pre-trained model, you need to make sure that it can be loaded properly.
But before checking that, you need to define the model's architecture as a Python class, so that the pre-trained set of 
weights is being loaded into that architecture, which means that the keys should match between the model and the weights.

As we used transfer learning from a pre-trained model and we just modified the last fully connected layer (fc), we need to
modify the original ResNet18 class. You can find the original class for this model at 
[torchvision/models/segmentation](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L268-L277)
and for the rest of the PyTorch pre-trained models at [torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models).

The code for the ResNet18 model looks like:

```python
def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
```

Which translated to our model file it should look like:

```python
import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock


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
```

As you can see we are creating a new class named `ImageClassifier` which inherits from the base `ResNet` class defined in
that file. We then need to initialize that class with our architecture, which in this case is the same one as the ResNet18,
including the `BasicBlock`, specifying the ResNet18 layers `[2,2,2,2]` and then we modify the number of classes, which for 
our case is 10 as we previously mentioned.

Finally, so as to make the state_dict match with the model class, we need to override the `self.fc` layer, which is the last
layer of the network. As we use that sequential layer while training the model, the final weights have been optimized for our
dataset over that layer, so just overriding it we will get the model's architecture with our modifications.

Then in order to check that the model can be loaded into the `ImageClassifier` class, you should just need to define the class and
load the weights using the following piece of code:

```python
model = ImageClassifier()
model.load_state_dict(torch.load("../foodnet/foodnet_resnet18.pth"))
```

Whose expected output should be `<All keys matched successfully>`.

You can find more Image Classification pre-trained PyTorch models at 
[PyTorch Image Classification Models](https://pytorch.org/docs/stable/torchvision/models.html#classification).

__Note__: the model has been trained on a NVIDIA GeForce GTX 1070 8GB GPU using CUDA 11. If you want to get you GPU specs, just
use the `nvidia-smi` command on your console, but make sure that you have your NVIDIA drivers properly installed. So as 
to check whether PyTorch is using the GPU you can just use the following piece of code which will tell you whether there's
a GPU (or more) available or not and, if so, which is the name of that device depending on its ID if there's more than 
one GPU.

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.get_device_name(0)
```

---

## :rocket: Deployment

In order to deploy the model you will need to reproduce the following steps once you installed all the requirements
as described in the section above.

  ---

__1. Generate MAR file:__ first of all you will need to generate the MAR file, which is the servable archive of the model
generated with `torch-model-archiver`. So on, in order to do so, you will need to use the following command:

  ```bash
  torch-model-archiver --model-name foodnet_resnet18 \
                       --version 1.0 \
                       --model-file foodnet/model.py \
                       --serialized-file foodnet/foodnet_resnet18.pth \
                       --handler foodnet/handler.py \
                       --extra-files foodnet/index_to_name.json
  ```

  Where the flag `--model-name` stands for the name that the generated MAR servable file will have, the `--version` is optional
  but it's a nice practice to include the version of the models so as to keep a proper tracking over them and finally you will need
  to specify the model's architecture file with the flag `--model-file`, the dumped state_dict of the trained model with the flag
  `--serialized-file` and the handler which will be in charge of the data preprocessing, inference and postprocessing with `--handler`,
  but you don't need to create custom ones as you can use the available handlers at TorchServe. Additionally, as this is a classification
  problem you can include the dictionary/json containing the relationships between the IDs (model's target) and the labels/names and/or 
  also additional files required by the model-file to work properly, with the flag `--extra-files`, separating the different files with 
  commas.

  More information regarding `torch-model-archiver` available at [Torch Model Archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md).

  ---

__2. Deploy TorchServe:__ once you create the MAR servable model, you just need to serve it. The serving process
of a pre-trained PyTorch model as a MAR file, starts with the deployment of the TorchServe REST APIs, which are the
Inference API, Management API and Metrics API, deployed by default on `localhost` (of if you prefer `127.0.0.1`) in the
ports 8080, 8081 and 8082, respectively. While deploying TorchServe, you can also specify the directory where the MAR files
are stored, so that they are deployed within the API at startup.

  So on, the command to deploy the current MAR model stored under model-store/ is the following:

  ```bash
  torchserve --start --ncs --model-store model-store --models foodnet=foodnet_resnet18.mar
  ```

  Where the flag `--start` means that you want to start the TorchServe service (deploy the APIs), the flag `--ncs`
  means that you want to disable the snapshot feature (optional), `--model-store` is the directory where the MAR files
  are stored and `--models` is/are the name/s of the model/s that will be served on the startup, including both an alias 
  which will be the API endpoint of that concrete model and the filename of that model, with format `endpoint=model_name.mar`.

  __Note__: another procedure can be deploying TorchServe first using the command 
  `torchserve --start --ncs --model-store model-store` (without defining the models) and then registering the model
  using the Management API via a HTTP POST request like `curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=foodnet_resnet18.mar"` 
  and later you can also scale the workers using `curl -X PUT "http://localhost:8081/models/foodnet?min_worker=3"`.

  More information regarding `torchserve` available at [TorchServe CLI](https://pytorch.org/serve/server.html#command-line-interface).

  ---

__3. Check its status:__ in order to check the availability of the deployed TorchServe API, you can just send a HTTP GET
request to the Inference API deployed by default in the `8080` port, but you should check the `config.properties` file, which
specifies `inference_address` including the port.

  ```bash
  curl http://localhost:8080/ping
  ```

  If everything goes as expected, it should output the following response:

  ```json
  {
    "status": "Healthy"
  }
  ```

  __Note__: If the status of the health-check request was `"Unhealthy"`, you should check the logs either from the console from where
  you did run the TorchServe deployment or from the `logs/` directory that is created automatically while deploying TorchServe from
  the same directory where you deployed it.

  ---

__4. Stop TorchServe:__ once you are done and you no longer need TorchServe, you can gracefully shut it down with the
following command:
  
  ```bash
  torchserve --stop
  ```

  Then the next time you deploy TorchServe, it will take less time than the first one if the models to be server were already
  registered/loaded, as TorchServe keeps them cached under a `/tmp` directory so it won't need to load them again if neither the name nor 
  the version changed. On the other hand, if you register a new model, TorchServe will have to load it and it may take a little 
  bit more of time depending on your machine specs. 

---

## :mage_man: Usage

Once you completed all the steps above, you can send a sample request to the deployed model so as to see its performance
and make the inference. In this case, as the problem we are facing is an image classification problem, we will use a sample
image as the one provided below and then send it as a file on the HTTP request's body as it follows:

```bash
wget https://raw.githubusercontent.com/alvarobartt/pytorch-model-serving/master/images/sample.jpg
curl -X POST http://localhost:8080/predictions/foodnet -T sample.jpg
```

Which should output something similar to:

```json
{
  "hamburger": 0.6911126375198364,
  "grilled_salmon": 0.11039528995752335,
  "pizza": 0.039219316095113754,
  "steak": 0.03642556071281433,
  "chicken_curry": 0.03306535258889198,
  "sushi": 0.028345594182610512,
  "chicken_wings": 0.027532529085874557,
  "fried_rice": 0.01296720840036869,
  "ice_cream": 0.012180349789559841,
  "ramen": 0.008756187744438648
}
```

__Remember__: that the original inference's output is the dict with the identifier of each class, not the class names,
in this case as we included `index_to_name.json` as an extra-file while creating the MAR, TorchServe is automatically 
assigning the identifiers with the class names so that the prediction is clearer.

  ---

The commands above translated into Python code looks like:

```python
# Download a sample image from the available samples at alvarobartt/pytorch-model-serving/images
import urllib
url, filename = ("https://raw.githubusercontent.com/alvarobartt/pytorch-model-serving/master/images/sample.jpg", "sample.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Transform the input image into a bytes object
import cv2
from PIL import Image
from io import BytesIO

image = Image.fromarray(cv2.imread(filename))
image2bytes = BytesIO()
image.save(image2bytes, format="PNG")
image2bytes.seek(0)
image_as_bytes = image2bytes.read()

# Send the HTTP POST request to TorchServe
import requests

req = requests.post("http://localhost:8080/predictions/foodnet", data=image_as_bytes)
if req.status_code == 200: res = req.json()
```

__Note__: that to execute the piece of sample code above you will need more requirements than the ones specified in the
[Requirements section](#hammer_and_wrench-requirements) so just run the following command so as to install them:

  ```bash
  pip install opencv-python pillow requests --upgrade
  ```
  
## :computer: Credits

Credits for the dataset slice go to @mrdbourke, as he nicely provided me the information via Twitter DM.

Credits for the tips on how to serve a PyTorch transfer learningn model using TorchServe go to 
@prashantsail as he properly explained in [this comment](https://github.com/pytorch/serve/issues/620#issuecomment-674971664).
