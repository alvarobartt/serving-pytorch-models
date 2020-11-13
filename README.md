# Serving PyTorch models with TorchServe :fire: 

![PyTorch Logo](https://miro.medium.com/max/1024/1*KKADWARPMxHb-WMxCgW_xA.png)

TorchServe is the ML model serving framework developed by PyTorch. Along this repository, the fundamentals
will be explained so as to deploy a sample CNN model trained to classify images from a food dataset
which is called [Food101](https://www.tensorflow.org/datasets/catalog/food101),
which contains images of up to 101 food classes, but in this case we will just use a "slice" of that
dataset which contains just 10 classes. Please, find the dataset [here](dataset/). Credits for the dataset
slice go to @mrdbourke (as he nicely provided me the information at Twitter).

__WARNING__: TorchServe is experimental and subject to change.

---

## :closed_book: Contents

...

---

## :hammer_and_wrench: Requirements

...

---

## :robot: Modelling

As the modelling is not the most relevant part/section that aims to be covered along this repository, we 
will just be using transfer learning from a pre-trained [ResNet](https://arxiv.org/abs/1512.03385) as it is 
the SOTA when it comes to image classification.

In this case, as we want to serve a PyTorch model, we will be using [PyTorch's implementation of ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
and more concretely, ResNet50, where the 50 stands for the number of layers that it contains, which in this case 
is 50.

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
  torch-model-archiver --model-name foodnet_resnet50 --version 1.0 --model-file model/model.py --serialized-file model/foodnet_resnet50.pth --handler model/foodnet_handler.py --extra-files model/...,model/...
  ```

__2. Deploy TorchServe:__

  ```bash
  torchserve --start --ncs --ts-config model/config.properties --model-store model/model-store --models foodnet=foodnet_resnet50.mar
  ```

__3. Register the model:__

__4. Check its status:__ in order to check the availability of the deployed TorchServe API, you can just send a HTTP GET
request to the Inference API deployed by deafult in the `8080` port, but you should check the `config.properties` file, which
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

---

## :mage_man: Usage

Once you completed all the steps above, you can send a sample request to the deployed model so as to see its performance
and make the inference. In this case, as the problem we are facing is an image classification problem, we will use a sample
image as the one provided below and then send it as a file on the HTTP request's body as it follows:

```bash
wget https://assets.epicurious.com/photos/57c5c6d9cf9e9ad43de2d96e/master/pass/the-ultimate-hamburger.jpg
curl -X POST http://localhost:8080/predictions/foodnet -T the-ultimate-hamburger.jpg
```

The commands above translated into Python code looks like:

```python
urllib ...
from io import BytesIO ...
import requests ...
```