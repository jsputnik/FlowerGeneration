import sys

import numpy as np
import torch
import cv2

from torch.utils.data import DataLoader

from torchvision.transforms import transforms

from utils.ImageManager import ImageManager
import utils.Helpers as Helpers
from nn.FlowerDataset import FlowerDataset
import nn.transforms as Transforms
import nn.learning as Learning
import utils.Device as Device
import utils.image_operations as imops
import segmentation_models_pytorch as smp
from skimage.draw import line
import decomposition.algorithm as dec
import segmentation.segmentation as seg
import tests as tests

print("Start")
# tests.test_alb_image_modifiers()
# tests.test_center_image_modifiers()
# seg.train_flower_centers()
tests.test_center_segmentation()
# tests.get_flower_part()
sys.exit()