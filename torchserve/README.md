# TorchServe

TorchServe is the ML model serving framework developed by PyTorch. In this section, the fundamentals
will be explained so as to deploy a sample CNN model trained to classify images from a food dataset
which is called [FoodX-251](https://www.groundai.com/project/foodx-251-a-dataset-for-fine-grained-food-classification/1),
which contains images of up to 251 food classes, but in this case we will just use a "slice" of that
dataset which contains just 10 classes. Please, find the dataset [here](dataset/).

---

## Installation

...

---

## Deployment

In order to deploy the model you will need to reproduce the following steps once you installed all the requirements
as described in the section above.

__1. Generate MAR file:__ first of all you will need to generate the MAR file, which is the servable archive of the model
generated with `torch-model-archiver`. So on, in order to do so, you will need to use the following command:

    ```bash
    torch-model-archiver ...
    ```

__2. Deploy TorchServe:__

__3. Register the model:__

__4. Check its status:__

---

## Usage

Once you completed all the steps above, you can send a sample request to the deployed model so as to see its performance
and make the inference. In this case, as the problem we are facing is an image classification problem, we will use a sample
image as the one provided below and then send it as a file on the HHTP request's body as it follows:

```bash
curl ...
```

Note that you can also send the requests to the deployed models with TorchServe as it follows:

```python

```