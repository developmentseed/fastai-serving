import torchvision
from fastai.vision import ImageDataBunch, cnn_learner, unet_learner, SegmentationItemList

# data = ImageDataBunch.from_csv('fixtures/classification')
# learner = cnn_learner(data, torchvision.models.resnet34)
# learner.export()

data = (SegmentationItemList.from_folder('fixtures/segmentation')
    .split_none()
    .label_from_func(lambda x: f'fixtures/segmentation/{x.stem}.jpg', classes=[0, 1, 2])
    .databunch())
learner = unet_learner(data, torchvision.models.resnet50)
learner.export()
