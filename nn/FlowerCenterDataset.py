from torch.utils.data import Dataset
import torch
import os
import cv2


class FlowerCenterDataset(Dataset):

    def __init__(self, images_root: str, trimaps_root: str, image_transform, trimap_transform, shared_transform):
        self.images_root = images_root
        self.trimaps_root = trimaps_root
        self.image_transform = image_transform
        self.trimap_transform = trimap_transform
        self.shared_transform = shared_transform
        self.image_names = os.listdir(images_root)
        self.trimap_names = os.listdir(trimaps_root)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = cv2.imread(self.images_root + "/" + self.image_names[index])
        trimap = cv2.imread(self.trimaps_root + "/" + self.trimap_names[index])
        # image = self.image_transform(image)
        # trimap = self.trimap_transform(trimap).long()
        if self.shared_transform is not None:
            image = self.shared_transform(image)
            trimap = self.shared_transform(trimap)
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.trimap_transform is not None:
            trimap = self.trimap_transform(trimap).long()
        return image, trimap
