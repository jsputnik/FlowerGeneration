import os
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import nn.transforms as Transforms
import utils.Device as Device
import nn.learning as learning
import utils.image_operations as imops


# def decode_segmap(image_tensor, number_of_classes):
#     image = torch.argmax(image_tensor.squeeze(), dim=0).detach().cpu().numpy()
#     label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])
#     red = np.zeros_like(image).astype(np.uint8)
#     green = np.zeros_like(image).astype(np.uint8)
#     blue = np.zeros_like(image).astype(np.uint8)
#     for label in range(0, number_of_classes):
#         idx = image == label  # all indexes of pixels, where class corresponds to given pixel
#         red[idx] = label_colors[label, 0]
#         green[idx] = label_colors[label, 1]
#         blue[idx] = label_colors[label, 2]
#     return cv2.cvtColor(np.stack([red, green, blue], axis=2), cv2.COLOR_RGB2BGR)
#     # return np.stack([red, green, blue], axis=2)


def decode_segmap(image_tensor, number_of_classes, label_colors=np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])):
    image = torch.argmax(image_tensor.squeeze(), dim=0).detach().cpu().numpy()
    # label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])
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

# def decode_segmap(image_tensor, number_of_classes):
#     image = torch.argmax(image_tensor.squeeze(), dim=0).detach().cpu().numpy()
#     label_colors = np.array([(0, 0, 0), (255, 255, 255), (128, 128, 128)])
#     red = np.zeros_like(image).astype(np.uint8)
#     green = np.zeros_like(image).astype(np.uint8)
#     blue = np.zeros_like(image).astype(np.uint8)
#     for label in range(0, number_of_classes):
#         idx = image == label  # all indexes of pixels, where class corresponds to given pixel
#         red[idx] = label_colors[label, 0]
#         green[idx] = label_colors[label, 1]
#         blue[idx] = label_colors[label, 2]
#     return cv2.cvtColor(np.stack([red, green, blue], axis=2), cv2.COLOR_RGB2BGR)
#     # return np.stack([red, green, blue], axis=2)


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
    image = image.copy()
    image[mask] = new_color
    return image


def create_center_segmaps(masks, originals):
    if len(masks) != len(originals):
        print("Number of masks and originals are different")
        return
    results = []
    for index in range(len(masks)):
        original = originals[index]
        mask = masks[index]
        boolean_mask = np.all(mask == np.array([0, 0, 255]), axis=-1)
        # imops.displayImage(mask)
        segmap = learning.segment_image(original)
        black_white_transform = transforms.Compose(
            [Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
             Transforms.ChangeColor(np.array([0, 0, 128]),
                                    np.array([255, 255, 255])),
             Transforms.ChangeColor(np.array([128, 128, 128]),
                                    np.array([0, 0, 0]))])

        black_white_image = black_white_transform(segmap)
        result = apply_boolean_mask(black_white_image, boolean_mask, new_color=np.array([128, 128, 128]))
        imops.displayImage(result)
        results.append(result)
    return results


def read_images(folder_path):
    images = []
    filenames = os.listdir(folder_path)
    for filename in filenames:
        full_path = os.path.join(folder_path, filename)
        images.append(cv2.imread(full_path))
    return images, filenames


def save_images(images, filenames):
    if len(images) != len(filenames):
        print("Number of images and filenames are different")
        return
    save_folder_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/centerflowers_modified/segmaps"
    for index in range(len(images)):
        cv2.imwrite(os.path.join(save_folder_path, filenames[index]), images[index])
