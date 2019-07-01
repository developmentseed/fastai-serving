import torch
import torchvision
from fastai.vision import ImageDataBunch

data = ImageDataBunch.from_csv('fixtures')
learner = cnn_learner(data, torchvision.models.resnet34)
learner.export()
