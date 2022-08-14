import sys
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as ptransforms
from utils.ImageManager import ImageManager
import utils.Helpers as Helpers
from nn.FlowerDataset import FlowerDataset
from nn.FlowerCenterDataset import FlowerCenterDataset
import nn.transforms as Transforms
import nn.learning as Learning
import utils.Device as Device
import utils.image_operations as imops
import segmentation_models_pytorch as smp
from skimage.draw import line
import decomposition.algorithm as dec
import segmentation.segmentation as seg


def test_image_modifiers():
    # manager = ImageManager("../datasets/17flowers/jpg", "../datasets/trimaps", "../datasets/trimaps/imlist.mat")
    # manager.load()
    # manager.set_image_dimensions()
    seed = 42
    train_dataset_ratio = 0.8
    test_dataset_ratio = 0.1
    dataset = FlowerDataset("../datasets/17flowers/jpg/",
                            "../datasets/trimaps/",
                            None, None,
                            # ptransforms.Compose([ptransforms.ToTensor()]),
                            # ptransforms.Compose([Transforms.ToMask(), ptransforms.ToTensor()]),
                            ptransforms.Compose([Transforms.CenterCrop(), Transforms.Resize((256, 128)), Transforms.RandomRotate(180)]))
    train_dataset_size = int(train_dataset_ratio * len(dataset))
    test_dataset_size = int(test_dataset_ratio * len(dataset))
    validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size

    torch.manual_seed(seed)  # to ensure creating same sets
    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])
    imops.displayImagePair(train_dataset[7][0], train_dataset[7][1])


def test_center_image_modifiers():
    train_dataset = FlowerCenterDataset("../datasets/centerflowers/originals/",
                                        "../datasets/centerflowers/segmaps/",
                                        None,
                                        None,
                                        ptransforms.Compose([Transforms.CenterCrop(), Transforms.Resize((256, 128))]))
    imops.displayImagePair(train_dataset[7][0], train_dataset[7][1])


def test_center_segmentation():
    # original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/17flowers/jpg/image_0014.jpg"
    original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/testFlower.png"
    segmap = seg.segment_flower_parts(original_image_path)
    imops.displayImage(segmap)
