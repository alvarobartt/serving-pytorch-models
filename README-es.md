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

Comenzaremos con la creación de un modelo de _transfer learning_ a partir del _backbone_ [ResNet](https://arxiv.org/abs/1512.03385) 
con un conjunto pre-entrenado de pesos, entrenado y evaluado con el conjunto de datos [ImageNet](http://www.image-net.org/), que
es el estado del arte en lo que a clasificación de imágenes se refiere.

En este caso, queremos servir un modelo de PyTorch por lo que partiremos de 
[la implementación de ResNet de PyTorch](https://pytorch.org/hub/pytorch_vision_resnet/) y, más concretamente,
ResNet18, que es la implementación de ResNet que contiene 18 capas convolucionales.

Por tanto, cargaremos dicho modelo a partir de los pesos pre-entrenados desde el Hub de PyTorch y los congelaremos puesto 
que no nos interesa modificarlos dado que la idea del _transfer learning_ es que ya se han ajustado para obtener el mejor
resultado posible sobre el conjunto de datos ImageNet. Para cargar el modelo de PyTorch desde el Hub podemos utilizar el siguiente
fragmento de código:

```python
from torchvision import models

model = models.resnet18(pretrained=True)
model.eval()

for param in model.parameters():
    param.requires_grad = False
```

Una vez cargado el modelo, necesitamos actualizar la capa `fc`, cuyas siglas del inglés significan _Fully Connected_, y
es la última capa del modelo que define las neuronas de salida. En este caso concreto, incluimos una capa 
secuencial que se añadirá tras las capas convolucionales del modelo original, dado que el objetivo es optimizar los pesos
de dicha capa para obtener los mejores resultados sobre el conjunto de datos de evaluación que estamos utilizando. Así 
la capa secuencial incluida es la que se muestra en el siguiente bloque de código:

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

Tras determinar la arquitectura de la red, se procederá a entrenar dicho modelo con el conjunto de datos de
entrenamiento que contiene 750 imágenes y que ha sido divido en dos sub-conjuntos, uno para el entrenamiento y uno 
para la validación con una separación del 80-20%, respectivamente. Dicha separación se realiza para poder estimar
durante el entrenamiento del modelo cómo se comportará el modelo ante ejemplos no vistos previamente y, por tanto, 
para poder estimar como funcionará el modelo cuando se le pase el conjunto de prueba que contiene 2500 imágenes.

__Nota__: para más detalles en lo que a la creación y entrenamiento del modelo de _transfer learning_ se refiere, puedes
observar el código desarrollado en [notebooks/transfer-learning.ipynb](notebooks/transfer-learning.ipynb) a modo de ejemplo.

Tras entrenar el modelo se procederá a exportar el modelo desde el _state\_dict_ a un fichero ".pth", el cual 
contiene el conjunto pre-entrenado de pesos que más adelante se podrá cargar de nuevo, con el código mostrado 
a continuación:

```python
torch.save(model.state_dict(), '../model/foodnet_resnet18.pth')
```

Tras generar fichero exportable del modelo ya entrenado, por lo que tenemos a asegurarnos de que se ha exportado 
previamente, comprobando que se carga correctamente. Para poder realizar esta comprobación es importante que la 
arquitectura del modelo esté propiamente definida, dado que se requiere de la arquitectura de la red para poder
cargar los pesos pre-entrenados sobre dicha red.

Puesto que hemos utilizado _transfer learning_ a partir de un modelo para clasificación de imágenes pre-entrenado pero
modificando tanto la última capa como ajustado los pesos de la misma, tenemos que modificar también la arquitectura original 
de ResNet18 definida en una clase de Python en 
[torchvision/models/segmentation](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L268-L277). El 
código original de PyTorch del modelo ResNet18 es el siguiente:

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

De este modo creamos una nueva clase de Python llamada `ImageClassifier` que hereda de la clase base de `ResNet`
definida por PyTorch en `torchvision`. Tenemos que inicializar dicha clase con nuestra arquitectura, puesto que modificamos
la última capa de la arquitectura de la red original ResNet18, pero antes hay que definir la arquitectura base de ResNet18
aunque modifiquemos el número clases de salida, que en este caso serán 10 clases de Food101 como se ha mencionado previamente.

Finalmente, tras definir la arquitectura base de ResNet18, procedemos a sobreescribir la capa `self.fc` con la definida previamente
y sobre la cual hemos optimizado los pesos de la red para nuestro conjunto de datos. En este caso, la arquitectura resultante será la
de ResNet pero con la última capa _Fully Connected_ modificada con la capa secuencial pre-definida.

Así ahora ya podremos comprobar si los pesos del modelo que hemos exportado se pueden cargar propiamente en la clase
`ImageClassifier` con el siguiente fragmento de código:

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

A continuación, se presenta el significado de cada uno de los flags utilizados por __torch-model-archiver__:

- `--model-name`: indica el nombre del modelo en formato MAR que vamos a generar.
- `--version`: se refiere a la version del modelo, lo cual es una buena práctica a la hora de mantener los modelos
puesto que se degradan con el tiempo.
- `--model-file`: contiene el fichero de Python que contiene la clase con la arquitectura del modelo que vamos a servir.
- `--serialized-file`: contiene el _state dict_ con los pesos del modelo ya entrenado.
- `--handler`: especifica el cómo se van a manejar los datos en las llamadas a dicho modelo, por lo que incluye tanto el preprocesamiento
como el postprocesamiento.
- `--extra-files`: dado que este es un problema de clasificación de imágenes, se puede incluir un fichero JSON que contenga 
las relaciones entre los IDs que asigna el modelo con los nombres de las categorías o etiquetas asignadas a cada uno de los IDs.

Mencionar que no se requiere crear _handlers_ personalizados puesto que los disponibles en TorchServe
son bastante útiles, pero en caso de necesitar redefinir cualquiera de los procesos, preprocesamiento o postprocesamiento, se 
podrá crear uno personalizado como el presentado en este proyecto.

Una vez generado el fichero MAR, tienes que moverlo al directorio [_deployment/model-store_](deployment/model-store) que
contendrá tanto este modelo como el resto de modelos puesto que a TorchServe se le indica el directorio sobre el cual ha de 
leer los modelos de PyTorch para servirlos más adelante.

```bash
mv foodnet_resnet18.mar deployment/model-store/
```

Puedes encontrar más información sobre `torch-model-archiver` en 
[Torch Model Archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md).

### 2. Desplegar TorchServe

Una vez que se haya generado un fichero MAR para servir, tan solo se necesita proceder con el despliegue de TorchServe. Así el 
proceso de servir un modelo pre-entrenado de PyTorch en formato MAR comienza con el despliegue de las APIs REST de TorchServe, que 
son las llamadas: _Inference API_, _Management API_ y _Metris API_, que se despliegan en el localhost o, lo que es lo mismo, la IP
127.0.0.1, en los puertos 8080, 8081 y 8082, respectivamente.

De este modo, el comando para desplegar TorchServe junto con el modelo MAR generado previamente,
disponible en el directorio [_deployment/model-store/_](deployment/model-store/), es el siguiente:

```bash
torchserve --start --ncs --ts-config deployment/config.properties --model-store deployment/model-store --models foodnet=foodnet_resnet18.mar
```

A continuación, se presenta el significado de cada uno de los flags utilizados por __torchserve__:

- `--start`: indica que el servicio de TorchServe se va a desplegar (es decir, las APIs).
- `--ncs`: indica que se desactivará el _snapshot_ para no hacer una copia del contenido de la API, 
lo cual reduce los tiempos de despliegue, pero por seguridad se puede activar (sin poner dicho flag).
- `--ts-config`: especifica la configuración del despliegue a utilizar desde el fichero de configuración.
- `--model-store`: indica el directorio desde el cual se van a leer los ficheros MAR listos para ser 
servidos (expuestos como un endpoint de las APIs).
- `--models`: especifica los nombres de los modelos a utilizar, de modo que a cada uno de los modelos 
disponibles, bajo el directorio mencionado previamente, se les podrá asignar un alias que será a través 
del cual se creará el endpoint para acceder a dicho modelo desde las APIs REST. Sino, se utilizarían los 
nombres por defecto del fichero, pero lo recomendable es especificar manualmente el nombre de todos los 
modelos a servir con el formato: _endpoint=model_name.mar_.

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

Con el fin de reproducir el despliegue de TorchServe, tal y como se ha descrito antes, en una imagen de Docker sobre Ubuntu, 
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
