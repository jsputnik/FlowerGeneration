import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import nn.transforms as Transforms
import utils.Device as Device


def decode_segmap(image_tensor, number_of_classes):
    image = torch.argmax(image_tensor.squeeze(), dim=0).detach().cpu().numpy()
    label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])
    red = np.zeros_like(image).astype(np.uint8)
    green = np.zeros_like(image).astype(np.uint8)
    blue = np.zeros_like(image).astype(np.uint8)
    for label in range(0, number_of_classes):
        idx = image == label  # all indexes of pixels, where class corresponds to given pixel
        red[idx] = label_colors[label, 0]
        green[idx] = label_colors[label, 1]
        blue[idx] = label_colors[label, 2]
    return cv2.cvtColor(np.stack([red, green, blue], axis=2), cv2.COLOR_RGB2BGR)
    # return np.stack([red, green, blue], axis=2)


def predict(model, image):
    image_transform = transforms.Compose([Transforms.Resize((256, 128)), transforms.ToTensor()])
    image_tensor = image_transform(image)
    input_tensor = image_tensor.unsqueeze(0).to(Device.get_default_device())
    output = model(input_tensor)
    return output


def remove_file_extension(filename):
    separator = "."
    return filename.split(separator, 1)[0]


def apply_mask(image, mask, color=np.array([0, 0, 0]), new_color=np.array([0, 0, 0])):
    image[mask == color.all(-1)] = new_color
    return image


def apply_boolean_mask(image, mask, new_color=np.array([0, 0, 0])):
    image[mask] = new_color
    return image

