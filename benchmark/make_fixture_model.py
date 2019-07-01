import torchvision
from fastai.vision import ImageDataBunch, cnn_learner

data = ImageDataBunch.from_csv('fixtures')
learner = cnn_learner(data, torchvision.models.resnet34)
learner.export()
