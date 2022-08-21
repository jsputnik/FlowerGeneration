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
import nn.learning as learning


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
                                        Transforms.ColorJitter(0.5),
                                        None,
                                        ptransforms.Compose([Transforms.CenterCrop(), Transforms.Resize((256, 128))]))
    imops.displayImagePair(train_dataset[7][0], train_dataset[7][1])


def test_center_segmentation():
    image_names = ["image_0004.jpg", "image_0325.jpg", "image_0730.jpg", "image_0980.jpg", "image_1238.jpg", "image_1308.jpg", "testFlower.png", "browneyedsusan.jpg"]
    folder_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/thesis assets/center/data/"
    # image_names = ["image_0325.jpg"]
    # folder_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/17flowers/jpg/"
    for name in image_names:
        segmap = seg.segment_flower_parts(folder_path + name)
        imops.displayImage(segmap)
        # cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/thesis assets/center/results_architectures/Manet" + name, segmap)


def get_flower_part():
    original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/testFlower.png"
    original_image = cv2.imread(original_image_path)
    # imops.displayImage(new_image)
    image_transform = Transforms.Resize((256, 128))
    resized_image = image_transform(original_image)
    general_model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,  # model output channels (number of classes in your dataset)
    )
    general_model_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/models/95.38flower"
    general_model.load_state_dict(torch.load(general_model_path))
    segmap = learning.segment(original_image_path,
                              model=general_model,
                              number_of_classes=4)
    print((segmap == np.array([128, 128, 128])).all(axis=-1).sum())
    # imops.displayImage(segmap)
    # mask = np.all(segmap != np.array([128, 128, 128]), axis=-1)
    mask = np.all(segmap == np.array([0, 128, 128]), axis=-1)
    result = Helpers.apply_boolean_mask(resized_image, mask)
    # mask2 = np.all(segmap == np.array([128, 128, 128]), axis=- 1)
    # result = Helpers.apply_boolean_mask(result, mask2)
    # imops.displayImagePair(original_image, result)
    cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/thesis assets/general/masked_results/grayyestestFlower.png", result)
    sys.exit()

