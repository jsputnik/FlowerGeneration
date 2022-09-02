from torch.utils.data import Dataset
import torch
import os
import numpy as np
from scipy.io import loadmat
import cv2
from utils import Helpers
import nn.transforms as Transforms
from torchvision.transforms import transforms as ptransforms
import utils.image_operations as imops


class FlowerDataset(Dataset):

    def __init__(self, images_root: str, trimaps_root: str, imageTransform, trimapTransform, sharedTransform):
        self.images_root = images_root
        self.trimaps_root = trimaps_root
        self.imageTransform = imageTransform
        self.trimapTransform = trimapTransform
        self.sharedTransform = sharedTransform
        self.config_file = "imlist.mat"
        self.image_names, self.trimap_names = self.get_valid_images()

    # loads data and trimaps for 17flowers dataset, only the data that have corresponding trimaps
    def get_valid_images(self):
        trimap_indexes = loadmat(self.trimaps_root + self.config_file)  # to assign data to trimaps
        trimap_indexes = list(map(int, trimap_indexes["imlist"][0]))
        # trimap_indexes = [x - 1 for x in trimap_indexes]
        image_names = os.listdir(self.images_root)  # only difference is format jpg and png
        del image_names[:2]
        valid_image_names = [self.find_by_id(os.listdir(self.images_root), i) for i in trimap_indexes]
        # valid_image_names = [image_names[i] for i in trimap_indexes]
        valid_trimap_names = [self.find_by_id(os.listdir(self.trimaps_root), i) for i in trimap_indexes]
        return valid_image_names, valid_trimap_names

    def find_by_id(self, names: [str], id: int):
        for name in names:
            if len(name) == 14 and name[0:6] == "image_" and int(name[6:10]) == id:
                return name
        return ""

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = cv2.imread(self.images_root + "/" + self.image_names[index])
        trimap = cv2.imread(self.trimaps_root + "/" + self.trimap_names[index])
        change_color = Transforms.ChangeColor(np.array([0, 0, 0]), np.array([128, 128, 128]))
        trimap = change_color(trimap)
        if self.sharedTransform is not None:
            transformed = self.sharedTransform(image=image, mask=trimap)
            image = transformed["image"]
            trimap = transformed["mask"]
            # image = self.sharedTransform(image)
            # trimap = self.sharedTransform(trimap)
        resize_transform = Transforms.Resize((256, 128))
        image = resize_transform(image)
        trimap = resize_transform(trimap)
        if self.imageTransform is not None:
            image = self.imageTransform(image=image)
            image = image["image"]
            # image = self.imageTransform(image)
        if self.trimapTransform is not None:
            trimap = self.trimapTransform(trimap)
        # tensor_transform = ptransforms.ToTensor()
        # image = tensor_transform(image)
        # mask_transform = ptransforms.Compose([Transforms.ToMask(), ptransforms.ToTensor()])
        # trimap = mask_transform(trimap).long()
        return image, trimap
