from torch.utils.data import Dataset
import os

class FlowerDataset(Dataset):
    def __init__(self, imgs_root, trimaps_root, transforms):
        self.imgs_root = imgs_root
        self.trimaps_root = trimaps_root
        self.transforms = transforms
        self.imgs = os.listdir(imgs_root)
        self.trimaps = os.listdir(trimaps_root)

    def print(self):
        print("Hello")



