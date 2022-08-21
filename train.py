import numpy as np
import torch

from torch.utils.data import DataLoader

from torchvision.transforms import transforms

from utils.ImageManager import ImageManager
from nn.FlowerDataset import FlowerDataset
import nn.transforms as Transforms
import nn.learning as Learning
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops

# hyperparameters
seed = 42
train_dataset_ratio = 0.8
test_dataset_ratio = 0.1
validation_dataset_ratio = 0.1
number_of_classes = 4

batch_size = 8
epochs = 20
learning_rate = 0.05
model_path = "models/"
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


# manager = ImageManager("../datasets/17flowers/jpg", "../datasets/trimaps", "../datasets/trimaps/imlist.mat")
# manager.load()
# # manager.set_image_dimensions()
# print("Flowers: ", manager.count_flower_types())
dataset = FlowerDataset("../datasets/17flowers/jpg/",
                        "../datasets/trimaps/",
                        transforms.Compose([transforms.ToTensor()]),
                        transforms.Compose([Transforms.ToMask(), transforms.ToTensor()]),
                        transforms.Compose([Transforms.Resize((256, 128))]))
# dataset = FlowerDataset("../datasets/17flowers/jpg/",
#                         "../datasets/trimaps/",
#                         transforms.Compose([Transforms.Resize((256, 128)), transforms.ToTensor()]),
#                         transforms.Compose([Transforms.ChangeColor(np.array([0, 0, 0]), np.array([128, 128, 128])),
#                                             Transforms.Resize((256, 128)), Transforms.ToMask(),
#                                             transforms.ToTensor()]),
#                         None)  # (864, 480)
train_dataset_size = int(train_dataset_ratio * len(dataset))
test_dataset_size = int(test_dataset_ratio * len(dataset))
validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size

torch.manual_seed(seed)  # to ensure creating same sets
train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
    dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])
# to_image_transform = Transforms.ToImage()
# imops.displayImage(to_image_transform(test_dataset[0][1]))
# imops.displayImage(train_dataset[0][0].numpy())
print(f"train dataset length: {len(train_dataset)}")
print(f"validation dataset length: {len(validation_dataset)}")
print(f"test dataset length: {len(test_dataset)}")
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size)
Learning.train(model, epochs, learning_rate, train_dataloader, validation_dataloader)
avg_accuracy = Learning.evaluate(model, test_dataloader)
print("Average accuraccy: ", avg_accuracy)
torch.save(model.state_dict(), model_path + "{:.2f}".format(avg_accuracy) + "Manet")
