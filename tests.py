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
import albumentations as alb


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


def test_alb_image_modifiers():
    seed = 42
    train_dataset_ratio = 0.8
    test_dataset_ratio = 0.1

    # image_transforms = alb.Compose([
    #     # alb.CenterCrop(width=256, height=128),
    #     # alb.HorizontalFlip(p=0.5),
    #     # alb.RandomBrightnessContrast(p=0.2),
    #     alb.Emboss()
    # ])

    shared_transforms = alb.Compose([
        # alb.CenterCrop(width=256, height=128),
        # alb.HorizontalFlip(p=0.5),
        # alb.RandomBrightnessContrast(p=0.2),
        alb.Rotate(limit=180)
    ], additional_targets={"image": "image", "mask": "mask"})
    dataset = FlowerDataset("../datasets/17flowers/jpg/",
                            "../datasets/trimaps/",
                            None,
                            None,
                            shared_transforms)
    train_dataset_size = int(train_dataset_ratio * len(dataset))
    test_dataset_size = int(test_dataset_ratio * len(dataset))
    validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size

    torch.manual_seed(seed)  # to ensure creating same sets
    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])
    image, mask = train_dataset[5]
    imops.displayImagePair(image, mask)


def test_center_image_modifiers():
    image_transforms = alb.Compose([
        # alb.CenterCrop(width=256, height=128),
        # alb.HorizontalFlip(p=0.5),
        # alb.RandomBrightnessContrast(p=0.2),
        alb.ColorJitter()
    ])

    shared_transforms = alb.Compose([
        # alb.CenterCrop(width=256, height=128),
        # alb.HorizontalFlip(p=0.5),
        # alb.RandomBrightnessContrast(p=0.2),
        alb.Rotate(limit=180)
    ], additional_targets={"image": "image", "mask": "mask"})
    train_dataset = FlowerCenterDataset("../datasets/centerflowers/originals/",
                                        "../datasets/centerflowers/segmaps/",
                                        None,
                                        None,
                                        shared_transforms)
    #
    # train_dataset = FlowerCenterDataset("../datasets/centerflowers/originals/",
    #                                     "../datasets/centerflowers/segmaps/",
    #                                     Transforms.ColorJitter(0.5),
    #                                     None,
    #                                     ptransforms.Compose([Transforms.CenterCrop(), Transforms.Resize((256, 128))]))
    image, mask = train_dataset[3]
    imops.displayImagePair(image, mask)


def test_center_segmentation():
    image_names = ["image_0004.jpg", "image_0325.jpg", "image_0730.jpg", "image_0980.jpg", "image_1238.jpg", "image_1308.jpg", "testFlower.png", "browneyedsusan.jpg"]
    folder_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/thesis assets/center/data/"
    # image_names = ["image_0325.jpg"]
    # folder_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/17flowers/jpg/"
    for name in image_names:
        segmap = seg.segment_flower_parts(folder_path + name)
        # imops.displayImage(segmap)
        cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/thesis assets/center/results_architectures/Manet" + name, segmap)


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
    # cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/thesis assets/general/masked_results/grayyestestFlower.png", result)
    sys.exit()


def list_images_in_dataset(dataset):
    array = []
    for index in dataset.indices:
        _, _, image_name = dataset.dataset[index]
        array.append(image_name)
    array.sort()
    # print("Array: ", array)
    # for elem in array:
        # print(elem)
    print("array len: ", len(array))
    return array


def count_flower_types(image_names):
        daffodils = 0
        snowdrops = 0
        lily_valleys = 0
        bluebells = 0
        crocuses = 0
        irises = 0
        tigerlilies = 0
        tulips = 0
        fritillaries = 0
        sunflowers = 0
        daisies = 0
        colts_foots = 0
        dandellions = 0
        cowslips = 0
        buttercups = 0
        windflowers = 0
        pansies = 0
        for im in image_names:
            if "image_0000.jpg" <= im <= "image_0080.jpg":
                daffodils = daffodils + 1
            elif "image_0081.jpg" <= im <= "image_0160.jpg":
                snowdrops = snowdrops + 1
            elif "image_0161.jpg" <= im <= "image_0240.jpg":
                lily_valleys = lily_valleys + 1
            elif "image_0241.jpg" <= im <= "image_0320.jpg":
                bluebells = bluebells + 1
            elif "image_0321.jpg" <= im <= "image_0400.jpg":
                crocuses = crocuses + 1
            elif "image_0401.jpg" <= im <= "image_0480.jpg":
                irises = irises + 1
            elif "image_0481.jpg" <= im <= "image_0560.jpg":
                tigerlilies = tigerlilies + 1
            elif "image_0561.jpg" <= im <= "image_0640.jpg":
                tulips = tulips + 1
            elif "image_0641.jpg" <= im <= "image_0720.jpg":
                fritillaries = fritillaries + 1
            elif "image_0721.jpg" <= im <= "image_0800.jpg":
                sunflowers = sunflowers + 1
            elif "image_0801.jpg" <= im <= "image_0880.jpg":
                daisies = daisies + 1
            elif "image_0881.jpg" <= im <= "image_0960.jpg":
                colts_foots = colts_foots + 1
            elif "image_0961.jpg" <= im <= "image_1040.jpg":
                dandellions = dandellions + 1
            elif "image_1041.jpg" <= im <= "image_1120.jpg":
                cowslips = cowslips + 1
            elif "image_1121.jpg" <= im <= "image_1200.jpg":
                buttercups = buttercups + 1
            elif "image_1201.jpg" <= im <= "image_1280.jpg":
                windflowers = windflowers + 1
            elif "image_1281.jpg" <= im <= "image_1360.jpg":
                pansies = pansies + 1
        return daffodils, snowdrops, lily_valleys, bluebells, crocuses, irises, tigerlilies, tulips, \
            fritillaries, sunflowers, daisies, colts_foots, dandellions, cowslips, buttercups, windflowers, \
            pansies
