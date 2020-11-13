# Serving PyTorch models with TorchServe :fire: 

![PyTorch Logo](https://miro.medium.com/max/1024/1*KKADWARPMxHb-WMxCgW_xA.png)

TorchServe is the ML model serving framework developed by PyTorch. Along this repository, the fundamentals
will be explained so as to deploy a sample CNN model trained to classify images from a food dataset
which is called [Food101](https://www.tensorflow.org/datasets/catalog/food101),
which contains images of up to 101 food classes, but in this case we will just use a "slice" of that
dataset which contains just 10 classes. Please, find the dataset [here](dataset/).

Credits for the dataset slice go to @mrdbourke, as he nicely provided me the information at 
Twitter, and credits for the tips on how to serve a PyTorch model using TorchServe go to 
@pranshantsail as he explained in [this comment](https://github.com/pytorch/serve/issues/620#issuecomment-674971664).

__WARNING__: TorchServe is experimental and subject to change.

---

## :closed_book: Table of Contents

- [Requirements](#hammer_and_wrench-requirements)
- [Modelling](#robot-modelling)
- [Deployment](#rocket-deployment)
- [Usage](#mage_man-usage)

---

## :hammer_and_wrench: Requirements

- torch, torchvision, torchserve, torch-model-archiver
- Java, JDK 11

---

## :robot: Modelling

As the modelling is not the most relevant part/section that aims to be covered along this repository, we 
will just be using transfer learning from a pre-trained [ResNet](https://arxiv.org/abs/1512.03385) as it is 
the SOTA when it comes to image classification.

In this case, as we want to serve a PyTorch model, we will be using [PyTorch's implementation of ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
and more concretely, ResNet18, where the 18 stands for the number of layers that it contains.

...

Explain how to load the model and some considerations towards preparing the model for TorchServe.

Find more Image Classification pre-trained PyTorch models at: https://pytorch.org/docs/stable/torchvision/models.html#classification

---

## :rocket: Deployment

In order to deploy the model you will need to reproduce the following steps once you installed all the requirements
as described in the section above.

__1. Generate MAR file:__ first of all you will need to generate the MAR file, which is the servable archive of the model
generated with `torch-model-archiver`. So on, in order to do so, you will need to use the following command:

  ```bash
  torch-model-archiver --model-name foodnet_resnet18 --version 1.0 --model-file foodnet/model.py --serialized-file foodnet/foodnet_resnet18.pth --handler foodnet/handler.py --extra-files foodnet/index_to_name.json
  ```

  More information regarding `torch-model-archiver` available at [Torch Model Archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md).

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

  ...

  __Note__: another procedure can be deploying TorchServe first using the command 
  `torchserve --start --ncs --model-store model-store` (without defining the models) and then registering the model
  using the Management API via a HTTP POST request like `curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=foodnet_resnet18.mar"` 
  and later you can also scale the workers using `curl -X PUT "http://localhost:8081/models/foodnet?min_worker=3"`.

  More information regarding `torchserve` available at [TorchServe CLI](https://pytorch.org/serve/server.html#command-line-interface).

__3. Check its status:__ in order to check the availability of the deployed TorchServe API, you can just send a HTTP GET
request to the Inference API deployed by default in the `8080` port, but you should check the `config.properties` file, which
specifies `inference_address` including the port.

  ```bash
  curl http://localhost:8080/ping
  ```

  If everything goes as expected, it should output the following response:

  ```json
  {
    'status': 'Healthy'
  }
  ```

  __Note__: If the status of the health-check request was `"Unhealthy"`, you should check the logs either from the console from where
  you did run the TorchServe deployment or from the `logs/` directory that is created automatically while deploying TorchServe from
  the same directory where you deployed it.

---

## :mage_man: Usage

Once you completed all the steps above, you can send a sample request to the deployed model so as to see its performance
and make the inference. In this case, as the problem we are facing is an image classification problem, we will use a sample
image as the one provided below and then send it as a file on the HTTP request's body as it follows:

```bash
wget https://assets.epicurious.com/photos/57c5c6d9cf9e9ad43de2d96e/master/pass/the-ultimate-hamburger.jpg # TODO: update with GitHub URL
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

The commands above translated into Python code looks like:

```python
# Download a sample image from the available samples at alvarobartt/pytorch-model-serving/images
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg") # TODO: update with GitHub URL
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