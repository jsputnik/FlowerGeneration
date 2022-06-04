import torch
import cv2
import numpy as np
from nn.FlowerCenterDataset import FlowerCenterDataset
from torchvision.transforms import transforms
import nn.transforms as customtrans
from torch.utils.data import DataLoader
import nn.transforms as flowertransforms
import nn.learning as learning
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops
import utils.Helpers as Helpers



def train_flower_centers():
    batch_size = 2
    epochs = 10
    learning_rate = 0.05
    model_path = "models/center/"

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    ).to(Device.get_default_device())

    train_dataset = FlowerCenterDataset("../datasets/centerflowers/originals/",
                                        "../datasets/centerflowers/segmaps/",
                                        transforms.ToTensor(),
                                        transforms.Compose([flowertransforms.ToCenterMask(), transforms.ToTensor()]))
    print("Files: ", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_dataloader = DataLoader(train_dataset, batch_size)
    learning.train(model, epochs, learning_rate, train_dataloader, validation_dataloader)
    avg_accuracy = learning.evaluate(model, train_dataloader)
    torch.save(model.state_dict(), model_path + "{:.2f}".format(avg_accuracy) + "centerflower")


def segment_flower_parts(image_path):
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]
    flower_segmap = learning.segment(image_path,
                                     model_path="C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/models/95.38flower",
                                     number_of_classes=4)
    center_segmap = learning.segment(image_path,
                                     model_path="C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/models/center/96.86centerflower",
                                     number_of_classes=3)
    mask = np.all(center_segmap == np.array([128, 128, 128]), axis=-1)
    center_result = Helpers.apply_boolean_mask(flower_segmap, mask, new_color=np.array([0, 128, 0]))
    to_original_size = customtrans.RestoreOriginalSize((width, height))
    enlarged_segmap = to_original_size(center_result)
    return enlarged_segmap



