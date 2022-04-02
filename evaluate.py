import numpy as np
import torch
import sys
import cv2

from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
import nn.transforms as Transforms
import utils.Helpers as Helpers
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops
from nn.FlowerDataset import FlowerDataset
import nn.learning as Learning

number_of_classes = 3
batch_size = 8
seed = 42
train_dataset_ratio = 0.8
test_dataset_ratio = 0.1
validation_dataset_ratio = 0.1

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,  # model output channels (number of classes in your dataset)
).to(Device.get_default_device())

if len(sys.argv) != 2:
    raise Exception("Invalid number of parameters (need 2)")

model_path = sys.argv[1]

model.load_state_dict(torch.load(model_path))
dataset = FlowerDataset("../datasets/17flowers/jpg/",
                        "../datasets/trimaps/",
                        transforms.Compose([Transforms.Resize((256, 128)), transforms.ToTensor()]),
                        transforms.Compose([Transforms.ChangeColor(np.array([0, 0, 0]), np.array([128, 128, 128])),
                                            Transforms.Resize((256, 128)), Transforms.ToMask(),
                                            transforms.ToTensor()]))
train_dataset_size = int(train_dataset_ratio * len(dataset))
test_dataset_size = int(test_dataset_ratio * len(dataset))
validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size

torch.manual_seed(seed)  # to ensure creating same sets
_, test_dataset, _ = random_split(dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])

test_dataloader = DataLoader(test_dataset, batch_size)
avg_accuracy = Learning.evaluate(model, test_dataloader)
print("Average accuracy: ", "{:.2f}".format(avg_accuracy))
