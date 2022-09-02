import cv2
import numpy as np


def displayTensor(tensor):
    image = np.copy(tensor.numpy().transpose((1, 2, 0))).astype(np.uint8)
    cv2.imshow("Image", image)
    # cv2.imshow("Image", tensor.numpy.transpose((1, 2, 0)))
    cv2.waitKey(0)


# takes [H, W, C = 3]
def displayImage(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def displayImagePair(first_image, second_image):
    cv2.imshow("1", first_image)
    cv2.imshow("2", second_image)
    cv2.waitKey(0)
