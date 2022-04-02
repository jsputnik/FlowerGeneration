import numpy as np
import cv2
import torch


class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.target_width = output_size[0]
        self.target_height = output_size[1]

    def __call__(self, image):
        height, width = image.shape[:2]
        ratio = height / width
        if height > self.target_height:
            height = self.target_height
            width = int(self.target_height // ratio)
            image = cv2.resize(image, (width, height))
        result = np.zeros((self.target_height, self.target_width, 3), np.uint8)
        start_x = (self.target_width - width) // 2
        start_y = (self.target_height - height) // 2
        result[start_y:start_y + height, start_x:start_x + width] = image
        return result

        # image, landmarks = sample['image'], sample['landmarks']
        #
        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size
        #
        # new_h, new_w = int(new_h), int(new_w)
        #
        # img = transform.resize(image, (new_h, new_w))
        #
        # # h and w are swapped for landmarks because for images,
        # # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]
        #
        # return {'image': img, 'landmarks': landmarks}


class ChangeColor:
    def __init__(self, old_color, new_color):
        self.old_color = old_color
        self.new_color = new_color

    def __call__(self, image):
        recolored_image = np.copy(image)
        recolored_image[(image == self.old_color).all(axis=2)] = self.new_color
        return recolored_image


class ToMask:
    def __call__(self, image):
        black = np.array([0, 0, 0])
        beige = np.array([(0, 128, 128)])
        red = np.array([(0, 0, 128)])
        gray = np.array([(128, 128, 128)])

        mask = np.zeros((image.shape[:2]), dtype=np.int)
        mask[(image == black).all(axis=2)] = 0
        mask[(image == beige).all(axis=2)] = 1
        mask[(image == red).all(axis=2)] = 2
        mask[(image == gray).all(axis=2)] = 3
        return mask


class ToImage:
    def __call__(self, mask):

        black = np.array([0])
        beige = np.array([1])
        red = np.array([2])
        gray = np.array([3])
        # black = 0
        # beige = 1
        # red = 2
        # gray = 3
        mask = mask.cpu().numpy()
        mask = mask.transpose((1, 2, 0))
        height, width = mask.shape[:2]

        # image = np.zeros((mask.shape[:2]), dtype=np.uint8)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[(mask == black).all(axis=2)] = np.array([0, 0, 0])
        image[(mask == beige).all(axis=2)] = np.array([(0, 128, 128)])
        image[(mask == red).all(axis=2)] = np.array([0, 0, 128])
        image[(mask == gray).all(axis=2)] = np.array([(128, 128, 128)])
        return image


# class ToOriginalSize:
#     def __init__(self, width, height):
#         self.target_width = width
#         self.target_height = height
#
#     def __call__(self, image):
#         height, width = image.shape[:2]
#         ratio = height / width
#         if height > self.target_height:
#             height = self.target_height
#             width = int(self.target_height // ratio)
#             image = cv2.resize(image, (width, height))
#         result = np.zeros((self.target_height, self.target_width, 3), np.uint8)
#         start_x = (self.target_width - width) // 2
#         start_y = (self.target_height - height) // 2
#         result[start_y:start_y + height, start_x:start_x + width] = image
#         return result


class ToTensor(object):
    def __call__(self, image):
        return image.transpose((2, 0, 1))
