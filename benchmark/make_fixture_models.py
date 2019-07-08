import torchvision
from fastai.vision import ImageDataBunch, cnn_learner, unet_learner, SegmentationItemList, imagenet_stats

data = ImageDataBunch.from_csv('fixtures/classification').normalize(imagenet_stats)
learner = cnn_learner(data, torchvision.models.resnet34)
learner.export()

data = (SegmentationItemList.from_folder('fixtures/segmentation/images')
    .split_none()
    .label_from_func(lambda x: f'fixtures/segmentation/masks/{x.stem}.jpg', classes=[0, 1, 2])
    .databunch()
    .normalize(imagenet_stats))
learner = unet_learner(data, torchvision.models.resnet50)
learner.export('../export.pkl')
