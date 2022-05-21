import cv2
import numpy as np
import nn.learning as learning
import utils.image_operations as imops
import nn.transforms as Transforms
from torchvision.transforms import transforms
import decomposition.algorithm as dec
import utils.Helpers as Helpers
import sys

original_image = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/centerflowers_modified/cropped/image_0581.jpg")
modified_image = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/centerflowers_modified/image_0581.png")
print((modified_image == np.array([0, 0, 255])).all(axis=2).sum())
imops.displayImage(modified_image)
segmap = learning.segment_image(original_image)
black_white_transform = transforms.Compose([Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
                                            Transforms.ChangeColor(np.array([0, 0, 128]),
                                                                   np.array([255, 255, 255])),
                                            Transforms.ChangeColor(np.array([128, 128, 128]),
                                                                   np.array([0, 0, 0]))])

black_white_image = black_white_transform(segmap)
imops.displayImage(black_white_image)
print("Shape1: ", black_white_image.shape)
print("Shape2: ", modified_image.shape)
mask = np.all(modified_image == np.array([0, 0, 255]), axis=-1)
print((np.all(modified_image == np.array([0, 0, 255]), axis=-1)).sum())
# print(mask)
print("Shape3: ", mask.shape)
# imops.displayImage(mask)
result = Helpers.apply_boolean_mask(black_white_image, mask, new_color=np.array([128, 128, 128]))
imops.displayImage(result)
sys.exit()

original_image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/17flowers/jpg/image_1248.jpg"
# original_segmap_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/trimaps/image_1254.png"
segmap = learning.segment(original_image_path)
# segmap = cv2.imread(original_segmap_path)
image_transform = Transforms.Resize((256, 128))
original_image = cv2.imread(original_image_path)
image = image_transform(original_image)
# cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/centerflowers_modified/image_1248.jpg", image)

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
