import cv2
import numpy as np
import torch


def decode_segmap(imageTensor, number_of_classes):
    print("imageTensor: ", imageTensor)
    print("image.shape: ", imageTensor.shape)
    image = torch.argmax(imageTensor.squeeze(), dim=0).detach().cpu().numpy()
    print("type(image squeezed): ", type(image))
    print("image squeezed.shape: ", image.shape)
    label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0)])
    red = np.zeros_like(image).astype(np.uint8)
    green = np.zeros_like(image).astype(np.uint8)
    blue = np.zeros_like(image).astype(np.uint8)
    for label in range(0, number_of_classes):
        # print("image in decode_segmap(): ", image)
        # print("label in decode_segmap(): ", label)
        idx = image == label  # all indexes of pixels, where class corresponds to given pixel
        red[idx] = label_colors[label, 0]
        green[idx] = label_colors[label, 1]
        blue[idx] = label_colors[label, 2]
    print("finished decoding segmap")
    return cv2.cvtColor(np.stack([red, green, blue], axis=2), cv2.COLOR_RGB2BGR)
    # return np.stack([red, green, blue], axis=2)


