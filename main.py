import sys

import numpy as np
import torch
import cv2

from torch.utils.data import DataLoader

from torchvision.transforms import transforms

from utils.ImageManager import ImageManager
import utils.Helpers as Helpers
from nn.FlowerDataset import FlowerDataset
import nn.transforms as Transforms
import nn.learning as Learning
import utils.Device as Device
import utils.image_operations as imops
import segmentation_models_pytorch as smp
from skimage.draw import line
import decomposition.algorithm as dec

print("Start")

# segmap = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/segmented_testFlower.png")
# segmap = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/segmented_flower93.png")
# segmap = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/segmented_flower250.png")
# segmap = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/segmented_rose.png")
# segmap = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/trimaps/image_0017.png")
# print((segmap == np.array([128, 128, 128])).all(axis=2).sum())  # 1124
# print((segmap == np.array([0, 0, 0])).all(axis=2).sum())  # 15609
# print((segmap == np.array([0, 0, 128])).all(axis=2).sum())  # 4979
# print((segmap == np.array([0, 128, 128])).all(axis=2).sum())  # 11056
# print(segmap.all(axis=2).sum())  # 4979
# imops.displayImage(segmap)
original = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/upload/image_0014.jpg")
height, width = original.shape[:2]
segmap = cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/segment/image_0014.png")
decomposed_image, number_of_petals = dec.decomposition_algorithm(segmap)
to_original_size = Transforms.RestoreOriginalSize((width, height))
result_image = to_original_size(decomposed_image)
imops.displayImage(result_image)
result_image = Helpers.separate_flower_parts(original, result_image, number_of_petals)
imops.displayImage(result_image)
sys.exit()


# detecting contours
black_white_transform = transforms.Compose([Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
                                            Transforms.ChangeColor(np.array([0, 0, 128]), np.array([255, 255, 255])),
                                            Transforms.ChangeColor(np.array([128, 128, 128]), np.array([255, 255, 255]))])
black_white_image = black_white_transform(segmap)
image_gray = cv2.cvtColor(black_white_image, cv2.COLOR_BGR2GRAY)
ret, image_gray = cv2.threshold(image_gray, 127, 255, 0)
# imops.displayImagePair(image_gray, segmap)
contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_TREE cv2.CHAIN_APPROX_SIMPLE

# big_contours = filter(lambda c: (cv2.contourArea(c) > 1000), contours)

big_contours = []
for contour in contours:
    if cv2.contourArea(contour) > 1000:  # add to parameters
        print("Contour len: ", len(contour))
        big_contours.append(contour)
print("BC len: ", len(big_contours))
cv2.drawContours(black_white_image, big_contours, 0, (0, 255, 0))
black_white_image[60][130] = np.array([255, 0, 0])
print("Image stats after drawPixels():")
print((black_white_image == np.array([255, 255, 0])).all(axis=2).sum())
print((black_white_image == np.array([0, 255, 0])).all(axis=2).sum())
print((black_white_image == np.array([0, 0, 0])).all(axis=2).sum())
print((black_white_image == np.array([255, 255, 255])).all(axis=2).sum())
imops.displayImage(black_white_image)  # [60][130]
worm_length = 21
min_distance = 4.5
for contour in big_contours:
    # current_index = 0
    contour = contour.squeeze()
    intersection_counter = 0
    intersections = []
    worm = np.append(contour, contour[:worm_length], axis=0)
    previous_pixel = np.array([0, 0])
    previous_distance = 0
    potential_intersection = False
    for current_index in range(len(contour)):
        currently_analyzed_pixels = worm[current_index:current_index + worm_length]
        head = currently_analyzed_pixels[0]
        tail = currently_analyzed_pixels[-1]
        middle = currently_analyzed_pixels[worm_length // 2]
        # print(head)
        # print(tail)
        # print(middle)
        # head[1] = 127 - head[1]
        # tail[1] = 127 - tail[1]
        # middle[1] = 127 - middle[1]
        # for pixel in currently_analyzed_pixels:
        #     black_white_image[pixel[1]][pixel[0]] = np.array([255, 0, 0])

        black_white_image[middle[1]][middle[0]] = np.array([255, 255, 255])
        print(head)
        print(tail)
        print(middle)
        # imops.displayImage(black_white_image)
        rr, cc = line(head[1], head[0], tail[1], tail[0])
        print("Rows: ", rr)
        print("Columns: ", cc)
        print("Asa: ", black_white_image[rr[0]][cc[0]])
        print("Asa2: ", type(black_white_image[rr[0]][cc[0]]))
        print("Asa3: ", black_white_image[rr[0]][cc[0]].shape)
        print("Asa4: ", np.array([255, 255, 255]))
        if (black_white_image[rr[1]][cc[1]] == np.array([255, 255, 255])).all() or (black_white_image[rr[-2]][cc[-2]] == np.array([255, 255, 255])).all():
            continue
        distance = np.linalg.norm(np.cross(head - tail, tail - middle)) / np.linalg.norm(head - tail)
        if potential_intersection and distance <= previous_distance:
            intersection_counter += 1
            intersections.append(previous_pixel)
            potential_intersection = False
            black_white_image[previous_pixel[1]][previous_pixel[0]] = np.array([255, 255, 0])
        if distance > min_distance and distance > previous_distance:  # adjust
            potential_intersection = True
            # print("poss intersection point:")
            # print(middle[1])
            # print(middle[0])
        print("Distance: ", distance)
        previous_pixel = middle
        previous_distance = distance
    print("Counter: ", intersection_counter)
    result_image = black_white_image.copy()
    for point in intersections:
        x, y = point.ravel()
        cv2.line(result_image, (x, y), (130, 60), (0, 0, 255))
# magnify_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 512))])
# magnified_image = magnify_transform(black_white_image)
# magnified_image = np.array(magnified_image)
# imops.displayImage(magnified_image)
# print((black_white_image == np.array([255, 255, 0])).all(axis=2).sum())
# print((black_white_image == np.array([255, 0, 0])).all(axis=2).sum())
# print((black_white_image == np.array([0, 0, 0])).all(axis=2).sum())
# print((black_white_image == np.array([255, 255, 255])).all(axis=2).sum())
print("Intersection points: ", intersections)
imops.displayImage(result_image)
# cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/divided_flower14.png", result_image)
sys.exit()
#######################

# hyperparameters
seed = 42
# set_ratio = 0.2
train_dataset_ratio = 0.8
test_dataset_ratio = 0.1
validation_dataset_ratio = 0.1
number_of_classes = 4

batch_size = 8
epochs = 4#30
learning_rate = 0.05
model_path = "./models/"
images_per_class = 80

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,  # model output channels (number of classes in your dataset)
).to(Device.get_default_device())

flower_model_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/models/95.38flower"  # 90.18
image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/17flower_dataset/17flowers/jpg/image_1043.jpg"
# image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/rose.jpg"
trimap_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/trimaps/image_0014.png"

model.load_state_dict(torch.load(flower_model_path))
test_image = cv2.imread(image_path)
test_trimap = cv2.imread(trimap_path)
custom_transform = transforms.Compose([Transforms.ChangeColor(np.array([0, 0, 0]), np.array([128, 128, 128])), Transforms.Resize((256, 128))])
recolored_image = custom_transform(test_trimap)
# color_transform = Transforms.ChangeColor(np.array([0, 0, 0]), np.array([128, 128, 128]))
# recolored_image = color_transform(test_trimap)
# imops.displayImagePair(test_trimap, recolored_image)
original_height, original_width = test_image.shape[:2]
output = Helpers.predict(model, test_image)
segmap = Helpers.decode_segmap(output, number_of_classes)
imops.displayImagePair(test_trimap, segmap)
# cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/segmented_flower1043.png", segmap)
print((segmap == np.array([128, 128, 128])).all(axis=2).sum())  # 1124
print((segmap == np.array([0, 0, 0])).all(axis=2).sum())  # 15609
print((segmap == np.array([0, 0, 128])).all(axis=2).sum())  # 4979
print((segmap == np.array([0, 128, 128])).all(axis=2).sum())  # 11056

# detecting contours
black_white_transform = transforms.Compose([Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
                                            Transforms.ChangeColor(np.array([0, 0, 128]), np.array([255, 255, 255])),
                                            Transforms.ChangeColor(np.array([128, 128, 128]), np.array([255, 255, 255]))])
black_white_image = black_white_transform(segmap)
image_gray = cv2.cvtColor(black_white_image, cv2.COLOR_BGR2GRAY)
ret, image_gray = cv2.threshold(image_gray, 127, 255, 0)
# imops.displayImagePair(image_gray, segmap)
contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

big_contours = []
for contour in contours:
    print("Area: ", cv2.contourArea(contour))
    print("Contour length: ", len(contour))
    if cv2.contourArea(contour) > 1000:  # add to parameters
        big_contours.append(contour)
cv2.drawContours(black_white_image, big_contours, -1, (0, 255, 0))
imops.displayImagePair(black_white_image, image_gray)




# manager = ImageManager("../datasets/17flowers/jpg", "../datasets/trimaps", "../datasets/trimaps/imlist.mat")
# manager.load()
# manager.set_image_dimensions()
# dataset = FlowerDataset("../datasets/17flowers/jpg/",
#                         "../datasets/trimaps/",
#                         transforms.Compose([Transforms.Resize((256, 128)), transforms.ToTensor()]),
#                         transforms.Compose([Transforms.Resize((256, 128)), Transforms.ToMask(),
#                                             transforms.ToTensor()]))  # (864, 480)
# train_dataset_size = int(train_dataset_ratio * len(dataset))
# test_dataset_size = int(test_dataset_ratio * len(dataset))
# validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size
#
# torch.manual_seed(seed)  # to ensure creating same sets
# train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
#     dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])
# print(f"train dataset length: {len(train_dataset)}")
# print(f"train dataset length: {len(validation_dataset)}")
# print(f"train dataset length: {len(test_dataset)}")
# train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size)
# validation_dataloader = DataLoader(validation_dataset, batch_size)
# # show_batch(test_dataloader)
# Learning.train(model, epochs, learning_rate, train_dataloader, validation_dataloader)
# Learning.evaluate(model, test_dataloader)
# torch.save(model.state_dict(), model_path + "flower1")
# model.load_state_dict(torch.load(model_path + model_name))

# output = Helpers.predict(model, train_dataset[3][0], train_dataset[3][1])
# print("image.type: ", type(train_dataset[3][0]))
# print("Trimaps.type: ", type(train_dataset[3][1]))
# # rgb = output.squeeze().detach().numpy().transpose((1, 2, 0))
# rgb = Helpers.decode_segmap(output, number_of_classes)
# print("RGB shape: ", rgb.shape)
# # print(dataset[3][1])
# image.displayTensor(dataset[3][0])
# image.displayImage(dataset[3][1])
# print("Trimap: ", dataset[3][1])
# image.displayImage(rgb)

print("End")


