import cv2
import numpy as np
import nn.learning as learning
import utils.image_operations as imops
import nn.transforms as Transforms
from torchvision.transforms import transforms
import decomposition.algorithm as dec
import utils.Helpers as Helpers
import sys
import os
import segmentation.segmentation as seg
from PIL import Image
import decomposition.algorithm as dec
import math

# original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/browneyedsusan.jpg"
original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/17flowers/jpg/image_0008.jpg"
segmap = seg.segment_flower_parts(original_image_path)
# black_white_transform = transforms.Compose([Transforms.ChangeColor(np.array([128, 128, 128]), np.array([0, 0, 0])),
#                                             Transforms.ChangeColor(np.array([0, 0, 128]), np.array([0, 0, 0])),
#                                             Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
#                                             Transforms.ChangeColor(np.array([0, 128, 0]), np.array([255, 255, 255]))])
# segmap = black_white_transform(segmap)
center_point = dec.get_center_point(segmap)
imops.displayImage(segmap)
segmap[center_point[1]][center_point[0]] = np.array([255, 0, 0])
segmap[center_point[1]+1][center_point[0]] = np.array([255, 0, 0])
segmap[center_point[1]-1][center_point[0]] = np.array([255, 0, 0])
segmap[center_point[1]+2][center_point[0]] = np.array([255, 0, 0])
segmap[center_point[1]-2][center_point[0]] = np.array([255, 0, 0])
imops.displayImage(segmap)

sys.exit()


# seg.train_flower_centers()
# original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/testFlowerRotated.png"
# original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/browneyedsusan.jpg"
original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/17flowers/jpg/image_0008.jpg"
original_image = cv2.imread(original_image_path)
# imops.displayImage(new_image)
image_transform = Transforms.Resize((256, 128))
image = image_transform(original_image)
segmap = learning.segment(original_image_path, model_path="C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/models/center/96.86centerflower", number_of_classes=3)
print((segmap == np.array([128, 128, 128])).all(axis=-1).sum())
imops.displayImage(segmap)
# mask = np.all(segmap != np.array([128, 128, 128]), axis=-1)
mask = np.all(segmap == np.array([128, 128, 128]), axis=-1)
result = Helpers.apply_boolean_mask(image, mask)
imops.displayImagePair(original_image, result)
sys.exit()


masks_folder_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/centerflowers_modified/center_marked"
originals_folder_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/centerflowers_modified/cropped"
mask_images, mask_filenames = Helpers.read_images(masks_folder_path)
original_images, original_filenames = Helpers.read_images(originals_folder_path)
segmaps = Helpers.create_center_segmaps(mask_images, original_images)
# Helpers.save_images(segmaps, mask_filenames)
sys.exit()

original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/browneyedsusan.jpg"
# original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/17flowers/jpg/image_1221.jpg"
# original_segmap_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/trimaps/image_1254.png"
segmap = learning.segment(original_image_path)
# segmap = cv2.imread(original_segmap_path)
image_transform = Transforms.Resize((256, 128))
original_image = cv2.imread(original_image_path)
image = image_transform(original_image)
# cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/centerflowers_modified/image_1221.jpg", image)

# hard to classify pixels treated as background
black_white_transform = transforms.Compose([Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
                                            Transforms.ChangeColor(np.array([0, 0, 128]),
                                                                   np.array([255, 255, 255])),
                                            Transforms.ChangeColor(np.array([128, 128, 128]),
                                                                   np.array([0, 0, 0]))])

black_white_image = black_white_transform(segmap)
thresholded_image = dec.threshold_image(black_white_image)

image1 = Helpers.apply_mask(image.copy(), thresholded_image)

black_white_transform2 = transforms.Compose([Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
                                            Transforms.ChangeColor(np.array([0, 0, 128]),
                                                                   np.array([255, 255, 255])),
                                            Transforms.ChangeColor(np.array([128, 128, 128]),
                                                                   np.array([255, 255, 255]))])
black_white_image2 = black_white_transform2(segmap)
thresholded_image2 = dec.threshold_image(black_white_image2)
print("Shape1: ", thresholded_image2.shape)
image2 = Helpers.apply_mask(image, thresholded_image2)

# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
imops.displayImagePair(image1, image2)
