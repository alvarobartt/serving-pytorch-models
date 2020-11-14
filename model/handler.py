# Reference: https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py

from ts.torch_handler.image_classifier import ImageClassifier as ImgClsfr


class ImageClassifier(ImgClsfr):
    topk = 10