import numpy as np
import cv2
from torchvision.transforms import transforms as ptransforms
import torchvision.transforms.functional as ptf
import PIL
import random


class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.target_width = output_size[0]
        self.target_height = output_size[1]

    def __call__(self, image):
        height, width = image.shape[:2]
        ratio = height / width
        if ratio < self.target_height / self.target_width:  # in case of very wide images
            if width > self.target_width:
                width = self.target_width
                height = int(self.target_width * ratio)
                image = cv2.resize(image, (width, height))
        else:
            if height > self.target_height:
                height = self.target_height
                width = int(self.target_height // ratio)
                image = cv2.resize(image, (width, height))
        result = np.zeros((self.target_height, self.target_width, 3), np.uint8)
        start_x = (self.target_width - width) // 2
        start_y = (self.target_height - height) // 2
        result[start_y:start_y + height, start_x:start_x + width] = image
        return result


class RestoreOriginalSize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.target_width = output_size[0]
        self.target_height = output_size[1]

    def __call__(self, image):
        height, width = image.shape[:2]
        ratio = height / width
        if ratio > self.target_height / self.target_width:  # if original is wider
            height = int(self.target_width * ratio)
            # resize leaving black area on sides
            image = cv2.resize(image, (self.target_width, height), interpolation=cv2.INTER_NEAREST)
            up = (height - self.target_height) / 2
            down = up + self.target_height
            result = image[int(up):int(down), 0:int(self.target_width)]
            return result
        else:
            # else
            width = int(self.target_height // ratio)
            # resize leaving black area on sides
            image = cv2.resize(image, (width, self.target_height), interpolation=cv2.INTER_NEAREST)
            left = (width - self.target_width) / 2
            right = left + self.target_width
            result = image[0:int(self.target_height), int(left):int(right)]
            return result


class RandomRotate:
    def __init__(self, degrees):
        self.angle = random.randint(-degrees, degrees)

    def __call__(self, image):
        image = PIL.Image.fromarray(np.uint8(image))
        result = ptf.rotate(image, self.angle)
        result = np.array(result)
        return result


class CenterCrop:
    def __call__(self, image):
        height, width = image.shape[:2]
        image = PIL.Image.fromarray(np.uint8(image))
        center_crop = ptransforms.CenterCrop((int(height // 1.5), int(width // 1.5)))
        result = center_crop(image)
        result = np.array(result)
        return result


class ColorJitter:
    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, image):
        image = PIL.Image.fromarray(np.uint8(image))
        color_jitter = ptransforms.ColorJitter(self.brightness)
        result = color_jitter(image)
        result = np.array(result)
        return result


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
        mask = mask.cpu().numpy()
        mask = mask.transpose((1, 2, 0))
        height, width = mask.shape[:2]

        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[(mask == black).all(axis=2)] = np.array([0, 0, 0])
        image[(mask == beige).all(axis=2)] = np.array([(0, 128, 128)])
        image[(mask == red).all(axis=2)] = np.array([0, 0, 128])
        image[(mask == gray).all(axis=2)] = np.array([(128, 128, 128)])
        return image


class ToCenterMask:
    def __call__(self, image):
        black = np.array([0, 0, 0])
        white = np.array([(255, 255, 255)])
        gray = np.array([(128, 128, 128)])

        mask = np.zeros((image.shape[:2]), dtype=np.int)
        mask[(image == black).all(axis=2)] = 0
        mask[(image == white).all(axis=2)] = 1
        mask[(image == gray).all(axis=2)] = 2
        return mask
