import os
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import nn.transforms as Transforms
import utils.Device as Device
import nn.learning as learning
import utils.image_operations as imops
from utils.Color import Color
from skimage.measure import regionprops


def decode_segmap(image_tensor, label_colors):
    image = torch.argmax(image_tensor.squeeze(), dim=0).detach().cpu().numpy()
    red = np.zeros_like(image).astype(np.uint8)
    green = np.zeros_like(image).astype(np.uint8)
    blue = np.zeros_like(image).astype(np.uint8)
    for label in range(0, len(label_colors)):
        idx = image == label  # all indexes of pixels, where class corresponds to given pixel
        red[idx] = label_colors[label, 0]
        green[idx] = label_colors[label, 1]
        blue[idx] = label_colors[label, 2]
    return cv2.cvtColor(np.stack([red, green, blue], axis=2), cv2.COLOR_RGB2BGR)


def predict(model, image):
    model.eval()
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
    image = image.copy()
    image[mask] = new_color
    return image


def separate_flower_parts(image, mask, petal_number):
    if petal_number > 15:
        petal_number = 15
    height, width = image.shape[:2]
    result_image_width = 2 * width
    result_image_height = 10 * height
    result_image = np.zeros((result_image_height, result_image_width, 3)).astype(np.uint8)
    current_color: np.ndarray = Color.first_petal_color.copy()
    x = 0
    y = 0
    max_part_height = 0
    for index in range(petal_number):
        try:
            part_image = separate_single_part(image, mask, current_color)
            y, x, max_part_height = update_result(result_image, part_image, y, x, max_part_height)
            current_color -= Color.color_difference
        except Exception as e:
            current_color -= Color.color_difference
    # now center
    try:
        part_image = separate_single_part(image, mask, Color.center_color)
        y, _, max_part_height = update_result(result_image, part_image, y, x, max_part_height)
    except Exception as e:
        pass
    cropped_result_height = y + max_part_height
    if y + max_part_height == 0:
        cropped_result_height = result_image_height
    cropped_result = result_image[0:cropped_result_height, 0:result_image_width]
    return cropped_result


def separate_single_part(image, mask, color):
    masked_part = np.all(mask == color, axis=-1).astype(int)
    properties = regionprops(masked_part, image)
    part_image = properties[0].intensity_image
    return part_image


def update_result(result_image, part_image, y, x, max_part_height):
    spacing = 10
    result_image_height, result_image_width = result_image.shape[:2]
    part_height, part_width = part_image.shape[:2]
    if part_height > max_part_height:
        if part_height > result_image_height:
            return y, x, max_part_height
        max_part_height = part_height
    if x + part_width + spacing > result_image_width:
        x = 0
        y += max_part_height + spacing
        max_part_height = part_height
        if y + part_height + spacing > result_image_height:
            return y, x, max_part_height
    result_image[y:y + part_height, x:x + part_width] = part_image
    x += part_width + spacing
    return y, x, max_part_height


def create_center_segmaps(masks, originals):
    if len(masks) != len(originals):
        return
    results = []
    for index in range(len(masks)):
        original = originals[index]
        mask = masks[index]
        boolean_mask = np.all(mask == np.array([0, 0, 255]), axis=-1)
        segmap = learning.segment_image(original)
        black_white_transform = transforms.Compose(
            [Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
             Transforms.ChangeColor(np.array([0, 0, 128]),
                                    np.array([255, 255, 255])),
             Transforms.ChangeColor(np.array([128, 128, 128]),
                                    np.array([0, 0, 0]))])

        black_white_image = black_white_transform(segmap)
        result = apply_boolean_mask(black_white_image, boolean_mask, new_color=np.array([128, 128, 128]))
        results.append(result)
    return results


def read_images(folder_path):
    images = []
    filenames = os.listdir(folder_path)
    for filename in filenames:
        full_path = os.path.join(folder_path, filename)
        images.append(cv2.imread(full_path))
    return images, filenames

