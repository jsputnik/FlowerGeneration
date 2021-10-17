import numpy as np
import cv2


class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.default_width = output_size[0]
        self.default_height = output_size[1]

    def __call__(self, image):
        height, width = image.shape[:2]
        ratio = height / width
        if height > self.default_height:
            height = self.default_height
            width = int(self.default_height // ratio)
            image = cv2.resize(image, (width, height))
        result = np.zeros((self.default_height, self.default_width, 3), np.uint8)
        start_x = (self.default_width - width) // 2
        start_y = (self.default_height - height) // 2
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


class ToTensor(object):
    def __call__(self, image):
        return image.transpose((2, 0, 1))
