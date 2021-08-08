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
        self.image_names, self.trimap_names = self.get_valid_images()

    # loads data and trimaps for 17flowers dataset, only the data that have corresponding trimaps
    def get_valid_images(self):
        trimap_indexes = loadmat(self.trimaps_root + self.config_file)  # to assign data to trimaps
        # print("Trimaps type: ", type(trimap_indexes["imlist"]))  #uint16
        trimap_indexes = list(map(int, trimap_indexes["imlist"][0]))
        # trimap_indexes = [x - 1 for x in trimap_indexes]
        image_names = os.listdir(self.images_root)  # only difference is format jpg and png
        del image_names[:2]
        valid_image_names = [self.find_by_id(os.listdir(self.images_root), i) for i in trimap_indexes]
        # valid_image_names = [image_names[i] for i in trimap_indexes]
        valid_trimap_names = [self.find_by_id(os.listdir(self.trimaps_root), i) for i in trimap_indexes]
        return valid_image_names, valid_trimap_names

    # move this to another class and make it check if valid there
    def find_by_id(self, names: [str], id: int):
        for name in names:
            if len(name) == 14 and name[0:6] == "image_" and int(name[6:10]) == id:
                print("Found match for: ", name)
                return name
        print("no match")
        return ""

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = cv2.imread(self.images_root + "/" + self.image_names[index])
        trimap = cv2.imread(self.trimaps_root + "/" + self.trimap_names[index])
        if self.transform:
            image = self.transform(image)
            trimap = self.transform(trimap)
        return image, trimap
