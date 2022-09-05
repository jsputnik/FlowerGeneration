import torch
import cv2
import numpy as np
from nn.FlowerCenterDataset import FlowerCenterDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import nn.transforms as flowertransforms
import nn.learning as learning
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops
import utils.Helpers as Helpers
import albumentations as alb


def train_flower_centers():
    batch_size = 2
    epochs = 200
    learning_rate = 0.05
    model_path = "models/center/"

    model = smp.DeepLabV3(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output chanznels (number of classes in your dataset)
    ).to(Device.get_default_device())

    image_transforms = alb.Compose([
        # alb.ToGray()
    ])

    shared_transforms = alb.Compose([
        alb.Rotate(limit=180)
    ], additional_targets={"image": "image", "mask": "mask"})
    train_dataset = FlowerCenterDataset("../datasets/centerflowers/originals/",
                                        "../datasets/centerflowers/segmaps/",
                                        None,
                                        None,
                                        shared_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_dataloader = DataLoader(train_dataset, batch_size)
    learning.train(model, epochs, learning_rate, train_dataloader, validation_dataloader)
    avg_accuracy = learning.evaluate(model, train_dataloader)
    print("Accuracy: ", avg_accuracy)
    torch.save(model.state_dict(), model_path + "{:.2f}".format(avg_accuracy) + "centerasd")


def segment_flower_parts(image_path):
    general_model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,  # model output channels (number of classes in your dataset)
    )
    general_model_path = "./static/models/95.35UnetShiftScaleRotate"
    general_model.load_state_dict(torch.load(general_model_path))
    center_model = smp.DeepLabV3(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    )
    center_model_path = "./static/models/96.02centerDeeplabRotate"
    center_model.load_state_dict(torch.load(center_model_path))
    flower_segmap = learning.segment(image_path,
                                     model=general_model,
                                     number_of_classes=4)
    center_segmap = learning.segment(image_path,
                                     model=center_model,
                                     number_of_classes=3)
    mask = np.all(center_segmap == np.array([128, 128, 128]), axis=-1)
    center_result = Helpers.apply_boolean_mask(flower_segmap, mask, new_color=np.array([0, 128, 0]))
    return center_result
