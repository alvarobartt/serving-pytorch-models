# Servir modelos de PyTorch con TorchServe :fire:

![Logo de PyTorch](https://miro.medium.com/max/1024/1*KKADWARPMxHb-WMxCgW_xA.png)

__TorchServe es el framework para servir modelos de ML desarrollado por PyTorch__.

A lo largo de este repositorio podrás encontrar una guía sobre cómo entrenar y desplegar/servir
un modelo de _transfer learning_ basado en una red neuronal convolucional (CNN) con 
[ResNet](https://arxiv.org/abs/1512.03385) como _backbone_, cuyo objetivo es clasificar imágenes,
del popular conjunto de datos [Food101](https://www.tensorflow.org/datasets/catalog/food101), 
en categorías.

__AVISO__: TorchServe está aún en fase experimental y, por tanto, sujeto a cambios.

![sanity-checks](https://github.com/alvarobartt/serving-pytorch-models/workflows/sanity-checks/badge.svg?branch=master)

---

## :closed_book: Lista de Contenidos

- [Requisitos](#hammer_and_wrench-requisitos)
- [Conjunto de Datos](#open_file_folder-conjunto-de-datos)
- [Modelado](#robot-modelado)
- [Despliegue](#rocket-despliegue)
- [Docker](#whale2-docker)
- [Uso](#mage_man-uso)
- [Créditos](#computer-créditos)

---

## :hammer_and_wrench: Requisitos

Antes de comenzar, tendrás que asegurarte de que tienes todas las dependencias necesarias instaladas
o, en caso de no tenerlas, instalarlas.

Primero tienes que comprobar que tienes el JDK 11 de Java instalado, ya que es un requisito
de `torchserve` a la hora de desplegar los modelos, ya que expone las APIs utilizando Java.

```bash
sudo apt install --no-install-recommends -y openjdk-11-jre-headless
```

A continuación, puedes proceder con la instalación de los paquetes de Python necesarios tanto para
entrenar como para servir el modelo de PyTorch. De este modo, para instalarlo puedes utilizar el siguiente
comando:

```bash
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torchserve==0.2.0 torch-model-archiver==0.2.0
```

O bien puedes instalarlo desde el fichero de requisitos llamado `requirements.txt`, con el comando:

```bash
pip install -r requirements.txt
```

En caso de que tengas algún problema durante la instalación, visita
[PyTorch - Get Started Locally](https://pytorch.org/get-started/locally/).

---

## :open_file_folder: Conjunto de Datos

El conjunto de datos a utilizar para el entrenamiento del modelo para la clasifacación de imágenes
en categorías es [Food101](https://www.tensorflow.org/datasets/catalog/food101). En este caso, dado que
esto es una guía, no se utilizará el conjunto de datos completo, sino que se utilizará un fragmento reducido del mismo,
en este caso, aproximadamente el 10% del total, abarcando tan solo 10 clases/categorías de las 101 disponibles.

El conjunto de datos original contiene imágenes de 101 categorías de comida distintas, con un total de 
101000 imágenes. Así, para cada clase, hay 750 imágenes para el entrenamiento y 250 imágenes para la 
evaluación del modelo. Las imágenes del conjunto de datos de evaluación han sido etiquetadas manualmente,
mientras que en las de entrenamiento puede existir algo de ruido, principalmente en forma de imágenes con colores
intensos o etiquetas erróneas. Por último mencionar que todas las imágenes han sido rescaladas con el fin de que
tengan un tamaño máximo de 512 píxeles (bien de largo o bien de ancho).

![](https://raw.githubusercontent.com/alvarobartt/serving-pytorch-models/master/images/data.jpg)

---

## :robot: Modelado

We will proceed with a transfer learning approach using [ResNet](https://arxiv.org/abs/1512.03385) as its backbone
with a pre-trained set of weights trained on [ImageNet](http://www.image-net.org/), as it is the SOTA when it 
comes to image classification.

In this case, as we want to serve a PyTorch model, we will be using 
[PyTorch's implementation of ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
and more concretely, ResNet18, where the 18 stands for the number of layers that it contains.

As we are going to use transfer learning from a pre-trained PyTorch model, we will load the ResNet18 model
and freeze it's weights using the following piece of code:

```python
from torchvision import models

model = models.resnet18(pretrained=True)
model.eval()

for param in model.parameters():
    param.requires_grad = False
```

Once loaded, we need to update the `fc` layer, which stands for fully connected and it's the last 
layer of the model, and over the one that the weights will be calculated to optimize the network 
for our dataset.

En este caso concreto, incluimos una capa secuencial que se añadirá tras las capas convolucionales
del modelo original. Así la capa secuencial incluida es la que se muestra en el siguiente bloque de código:

```python
import torch.nn as nn

sequential_layer = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(.2),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
)

model.fc = sequential_layer
```

Then we will train the model with the TRAIN dataset which contains 750 images and that has been 
splitted as 80%-20% for training and validation, respectively. And tested over the TEST dataset 
which contains 2500 images.

__Note__: for more details regarding the model training process, feel free to check it at 
[notebooks/transfer-learning.ipynb](notebooks/transfer-learning.ipynb)

After training the model you just need to dump the state_dict into a `.pth` file, which contains
the pre-trained set of weights, with the following piece of code:

```python
torch.save(model.state_dict(), '../model/foodnet_resnet18.pth')
```

Once the state_dict has been generated from the pre-trained model, you need to make sure that it can be loaded properly.
But before checking that, you need to define the model's architecture as a Python class, so that the pre-trained set of 
weights is being loaded into that architecture, which means that the keys should match between the model and the weights.

As we used transfer learning from a pre-trained model and we just modified the last fully connected layer (fc), we need to
modify the original ResNet18 class. You can find the original class for this model at 
[torchvision/models/segmentation](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L268-L277)
and for the rest of the PyTorch pre-trained models at [torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models).

El código original de PyTorch del modelo ResNet18 es el siguiente:

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

Que, traducido a la arquitectura completa de nuestro modelo, será de la forma:

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
model.load_state_dict(torch.load("../model/foodnet_resnet18.pth"))
```

Así, la salida del fragmento de código anterior tras la carga del modelo, ha de ser `<All keys matched successfully>`, en caso
de haber tenido éxito.

Puedes encontrar más modelos pre-entrenados de PyTorch para la clasificación de imágenes en 
[PyTorch Image Classification Models](https://pytorch.org/docs/stable/torchvision/models.html#classification), y probar
así distintos modelos de `transfer learning`.

__Nota__: el modelo ha sido entrenado con una tarjeta gráfica NVIDIA GeForce GTX 1070 8GB GPU con CUDA 11. En caso de no conocer
los requisitos de la tarjeta gráfica de tu sistema, puedes utilizar el comando `nvidia-smi`, que también te indicará si los _drivers_
de NVIDIA y CUDA están correctamente instalados. Además, para comprobar si PyTorch está haciendo uso de la GPU disponible en el sistema
o no, puedes utilizar el fragmento de código presentado a continuación:

    ```python
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.get_device_name(0)
    ```

---

## :rocket: Despliegue

Finalmente, de cara a desplegar el modelo necesitarás reproducir la siguiente secuencia de pasos tras haber instalado todos los 
requisitos previamente mencionados y disponer del modelo entrenado, tal y como se ha descrito previamente.

### 1. Generar el fichero MAR

Inicialmente se ha de generar el fichero MAR, que es un fichero listo para servir y que contiene el modelo completo
generado con `torch-model-archiver` a partir del `state_dict` exportado previamente. Para generar así el fichero MAR
tienes que utilizar el siguiente comando:

```bash
torch-model-archiver --model-name foodnet_resnet18 \
                     --version 1.0 \
                     --model-file model/model.py \
                     --serialized-file model/foodnet_resnet18.pth \
                     --handler model/handler.py \
                     --extra-files model/index_to_name.json
```

Una vez generado el fichero MAR, tienes que moverlo al directorio [_deployment/model-store_](deployment/model-store) que
contendrá tanto este modelo como el resto de modelos puesto que a TorchServe se le indica el directorio sobre el cual ha de 
leer los modelos de PyTorch para servirlos más adelante.

```bash
mv foodnet_resnet18.mar deployment/model-store/
```

Where the flag `--model-name` stands for the name that the generated MAR servable file will have, the `--version` is optional
but it's a nice practice to include the version of the models so as to keep a proper tracking over them and finally you will need
to specify the model's architecture file with the flag `--model-file`, the dumped state_dict of the trained model with the flag
`--serialized-file` and the handler which will be in charge of the data preprocessing, inference and postprocessing with `--handler`,
but you don't need to create custom ones as you can use the available handlers at TorchServe. Additionally, as this is a classification
problem you can include the dictionary/json containing the relationships between the IDs (model's target) and the labels/names and/or 
also additional files required by the model-file to work properly, with the flag `--extra-files`, separating the different files with 
commas.

Puedes encontrar más información sobre `torch-model-archiver` en 
[Torch Model Archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md).

### 2. Desplegar TorchServe

Once you create the MAR servable model, you just need to serve it. The serving process
of a pre-trained PyTorch model as a MAR file, starts with the deployment of the TorchServe REST APIs, which are the
Inference API, Management API and Metrics API, deployed by default on `localhost` (of if you prefer `127.0.0.1`) in the
ports 8080, 8081 and 8082, respectively. While deploying TorchServe, you can also specify the directory where the MAR files
are stored, so that they are deployed within the API at startup.

De este modo, el comando para desplegar TorchServe junto con el modelo MAR generado previamente,
disponible en el directorio [_deployment/model-store/_](deployment/model-store/), es el siguiente:

```bash
torchserve --start --ncs --ts-config deployment/config.properties --model-store deployment/model-store --models foodnet=foodnet_resnet18.mar
```

Where the flag `--start` means that you want to start the TorchServe service (deploy the APIs), the flag `--ncs`
means that you want to disable the snapshot feature (optional), `--ts-config` to include the configuration file
which is something optional too, `--model-store` is the directory where the MAR files are stored and 
`--models` is(are) the name(s) of the model(s) that will be served on the startup, including both an alias 
which will be the API endpoint of that concrete model and the filename of that model, with format `endpoint=model_name.mar`.

__Nota__: otra forma de proceder en el despliegue consiste en desplegar primero TorchServe sin ningún modelo indicado en
tiempo de despliegue y, en su defecto, registrar el modelo o modelos a través de la API de _management_ (que también permite
gestionar los _workers_ asignados a cada modelo entre otras cosas).

    ```bash
    torchserve --start --ncs --ts-config deployment/config.properties --model-store deployment/model-store
    curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=foodnet_resnet18.mar"
    curl -X PUT "http://localhost:8081/models/foodnet?min_worker=3"
    ```

Puedes encontrar más información sobre `torchserve` en [TorchServe CLI](https://pytorch.org/serve/server.html#command-line-interface).

### 3. Comprobar el estado de TorchServe

Para comprobar la disponibilidad de TorchServe tras el despliegue, puedes enviar una petición HTTP GET a la API para la inferencia
desplegada por defecto en el puerto 8080 con el comando presentado a continuación. Para conocer el puerto o puertos en los que se ha 
desplegado cada una de las APIs, puedes comprobar el fichero [_config.properties_](deployment/config.properties) que contiene dicha
información sobre los servicios desplegados por TorchServe.

```bash
curl http://localhost:8080/ping
```

Si todo ha ido como se esperaba, debería de mostrar una salida similar a la siguiente:

```json
{
  "status": "Healthy"
}
```

__Nota__: si el estado del _health check_ es `"Unhealthy"`, deberías de comprobar los _logs_ de TorchServe para 
comprobar que el despliegue fue correctamente y, en caso de no haber ido bien, identificar el error e intentar resolverlo.
Al desplegar o intentar desplegar TorchServe, se creará automáticamente un directorio, desde donde se usó el comando, 
llamado `logs/`, donde poder comprobar si el despliegue ha ido como se esperaba.

### 4. Parar TorchServe

Una vez se "termine" de utilizar TorchServe con idea de no utilizarlo más, puedes pararlo de forma elegante con el 
siguiente comando:
  
```bash
torchserve --stop
```

Así la próxima vez que despliegues TorchServe, tardará menos tiempo puesto que en el primer despliegue, tanto los modelos 
especificados durante el despliegue como los modelos registrados más adelante, serán cacheados, de modo que en los siguientes
despliegues de TorchServe, dichos modelos no requerirán ser registrados de nuevo por estar ya en caché.

---

## :whale2: Docker

Con el fin de reproducir el despliegue de PyTorch, tal y como se ha descrito antes, en una imagen de Docker sobre Ubuntu, 
tendrás que asegurarte de tener Docker instalado en tu máquina y proceder con la ejecución de los comandos presentados a continuación:

```bash
docker build -t ubuntu-torchserve:latest deployment/
docker run --rm --name torchserve_docker \
           -p8080:8080 -p8081:8081 -p8082:8082 \
           ubuntu-torchserve:latest \
           torchserve --model-store /home/model-server/model-store/ --models foodnet=foodnet_resnet18.mar
```

Para más información en lo referente al despliegue desde Docker, puedes acudir a la documentación de TorchServe 
en la que se detalla el uso de Docker para el despliegue y notas adicionales disponible en 
[pytorch/serve/docker](https://github.com/pytorch/serve/tree/master/docker).

---

## :mage_man: Uso

Una vez que has completado con éxito todos los pasos descritos previamente, puedes probar las APIs desplegadas por TorchServe
enviando peticiones de ejemplo al modelo que está siendo servido. En este caso, dado que es un problema de clasificación de imágenes, 
se utilizará una imagen que se pueda englobar en alguna de las categorías sobre las que hace la inferencia el modelo. De este modo, 
una vez dispongamos de una imagen válida, podremos enviar la petición HTTP POST con el contenido de la imagen en el cuerpo de la 
petición de la forma:

```bash
wget https://raw.githubusercontent.com/alvarobartt/pytorch-model-serving/master/images/sample.jpg
curl -X POST http://localhost:8080/predictions/foodnet -T sample.jpg
```

Que, si todo ha ido bien, debería de devolver una salida en formato JSON como la que se muestra a continuación:

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

__Recuerda__: el hecho de que la respuesta de la petición HTTP POST a TorchServe esté formateada con los nombres originales de 
cada una de las categorías de comida, se debe a que durante la creación del fichero MAR se especifico el índice al que correspondía
cada categoría, en el fichero `index_to_name.json`. Así TorchServe realiza la asignación de índices a categorías de forma automática
al responder a la petición a la API para la inferencia, de modo que es más clara.

  ---

Así los comandos presentados anteriormente se traducen en código de Python de la forma presentada en el siguiente bloque:

```python
# Descarga una imagen de ejemplo de alvarobartt/pytorch-model-serving/images
import urllib
url, filename = ("https://raw.githubusercontent.com/alvarobartt/pytorch-model-serving/master/images/sample.jpg", "sample.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Transforma la imagene en un objeto de bytes
import cv2
from PIL import Image
from io import BytesIO

image = Image.fromarray(cv2.imread(filename))
image2bytes = BytesIO()
image.save(image2bytes, format="PNG")
image2bytes.seek(0)
image_as_bytes = image2bytes.read()

# Envía la petición HTTP POST a TorchServe
import requests

req = requests.post("http://localhost:8080/predictions/foodnet", data=image_as_bytes)
if req.status_code == 200: res = req.json()
```

__Nota__: en caso de querer ejecutar un _script_ con el código proporcionado anteriormente, se requiren más requisitos
de los mencionados previamente en la [sección de requisitos](#hammer_and_wrench-requisitos), con lo que para instalarlos, 
puedes utilizar el siguiente comando:

  ```bash
  pip install opencv-python pillow requests --upgrade
  ```

---

## :computer: Créditos

Los créditos del conjunto de datos utilizados son para [@mrdbourke](https://github.com/mrdbourke), quien
amablemente me proporciono el fragmento del conjunto de datos Food101 a través de mensaje directo de Twitter.

Por otro lado, también dar créditos a [@prashantsail](https://github.com/prashantsail), por su colaboración 
en TorchServe y, en concreto, por la guía disponible en [este comentario](https://github.com/pytorch/serve/issues/620#issuecomment-674971664)
utilizada como base para servir los modelos.
