import numpy as np
import torch

from torch.utils.data import DataLoader

from torchvision.transforms import transforms

import tests
from utils.ImageManager import ImageManager
from nn.FlowerDataset import FlowerDataset
import nn.transforms as Transforms
import nn.learning as Learning
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops
import albumentations as alb
import torchmetrics

# hyperparameters
seed = 42
train_dataset_ratio = 0.8
test_dataset_ratio = 0.1
validation_dataset_ratio = 0.1
number_of_classes = 4

batch_size = 8
epochs = 20
learning_rate = 0.05
images_per_class = 80

# model = smp.Unet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,  # model output channels (number of classes in your dataset)
# ).to(Device.get_default_device())

model = smp.MAnet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,  # model output channels (number of classes in your dataset)
).to(Device.get_default_device())
model_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/models/94.88Manet"
# image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/browneyedsusan.jpg"
# image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/testFlower.png"
# image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/rose.jpg"

model.load_state_dict(torch.load(model_path))
dataset = FlowerDataset("../datasets/17flowers/jpg/",
                        "../datasets/trimaps/",
                        None,
                        None,
                        None)
train_dataset_size = int(train_dataset_ratio * len(dataset))
test_dataset_size = int(test_dataset_ratio * len(dataset))
validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size
torch.manual_seed(seed)  # to ensure creating same sets are created
train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
    dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])
test_dataloader = DataLoader(test_dataset, batch_size)
network_result = Learning.evaluate(model, test_dataloader, metric=torchmetrics.JaccardIndex(num_classes=4, average="none"))
print("Network result: ", network_result)
# Accuracy
# accuracy: 0.963104248046875
# accuracy: 0.988800048828125
# accuracy: 0.92803955078125
# accuracy: 0.969696044921875
# accuracy: 0.9241943359375
# accuracy: 0.934356689453125
# accuracy: 0.9588623046875
# accuracy: 0.95733642578125
# accuracy: 0.977783203125
# accuracy: 0.989471435546875

# jaccard index all classes
# accuracy: 0.869756281375885
# accuracy: 0.9469850063323975
# accuracy: 0.7292043566703796
# accuracy: 0.8997483849525452
# accuracy: 0.7717071175575256
# accuracy: 0.8301178216934204
# accuracy: 0.8555188179016113
# accuracy: 0.8773785829544067
# accuracy: 0.8418269157409668
# accuracy: 0.9488279819488525
