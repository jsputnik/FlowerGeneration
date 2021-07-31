from torch.utils.data import Dataset
import torch
import os
from scipy.io import loadmat
import cv2


class FlowerDataset(Dataset):

    def __init__(self, images_root: str, trimaps_root: str, transform=None):
        self.images_root = images_root
        self.trimaps_root = trimaps_root
        self.transform = transform
        self.config_file = "imlist.mat"
        self.image_names = self.get_valid_images()
        # self.imgs = os.listdir(images_root)
        # self.trimaps = os.listdir(trimaps_root)


    # loads data and trimaps for 17flowers dataset, only the data that have corresponding trimaps
    def get_valid_images(self):
        trimap_indexes = loadmat(self.trimaps_root + self.config_file)  # to assign data to trimaps
        # print("Trimaps type: ", type(trimap_indexes["imlist"]))  #uint16
        trimap_indexes = list(map(int, trimap_indexes["imlist"][0]))
        trimap_indexes = [x - 1 for x in trimap_indexes]
        image_names = os.listdir(self.images_root)  # only difference is format jpg and png
        del image_names[:2]
        return [image_names[i] for i in trimap_indexes]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = cv2.imread(self.images_root + "/" + self.image_names[index])
        if self.transform:
            image = self.transform(image)
        return image



