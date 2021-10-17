import torch
import cv2
import numpy as np

import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torchvision import models

import matplotlib.pyplot as plt

from nn.FlowerNet import FlowerNet
from utils.ImageManager import ImageManager
from nn.FlowerDataset import FlowerDataset
from nn.transforms import Resize
from nn.loss_functions import DiceLoss
# from nn.transforms import ToTensor
# import torch.utils.data
# from torch.utils.data import random_split


def show_img(img, label, ds):
    print(f"Label: {ds.classes[label]}")
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


# def show_batch(dl):
#     for images, labels in dl:
#         print(f"labels: {labels}")
#         fig, ax = plt.subplots(figsize=(8, 8))
#         plt.axis("off")
#         plt.imshow(make_grid(images, nrow=10).permute(1, 2, 0))
#         plt.show()
#         break

def show_batch(dl):
    for images, labels in dl:
        # print(f"labels: {labels}")
        print("images[0] type: ", type(images[0]))
        print("images shape: ", images.shape)
        print("images[0] shape: ", images[0].shape)
        print("permuted images.shape: ", images.shape)
        grid_img = torchvision.utils.make_grid(images, nrow=4)
        print("grid shape: ", grid_img.shape)
        plt.axis("off")
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        break


# def evaluate(model, validation_dataloader, loss_function=nn.CrossEntropyLoss()):
#     model.eval()  # turn off regularisation
#     print("Evaluating...")
#     with torch.no_grad():
#         total = 0
#         correct = 0
#         for batch in validation_dataloader:
#             inputs, trimaps = batch
#             outputs = model(inputs)
#             loss = loss_function(outputs, trimaps)  # calculate loss on the batch
#             _, predicted = torch.max(outputs, 1)
#             # _, predicted = torch.max(outputs.data, 1)
#             total += trimaps.size(0)
#             correct += (predicted == trimaps).sum().item()
#             # print(f"Outputs: {outputs}")
#             # print(f"Predicted: {predicted}")
#             # print(f"Labels: {labels}")
#             # print(f"Total: {total}")
#             # print(f"Correct: {correct}")
#         print(f"Final total: {total}")
#         print(f"Final correct: {correct}")
#         grade = 100 * correct / total
#         print(f"Loss: {loss}")
#         print(f"Accuraccy: {grade}")


# def train(model, num_epochs, learning_rate, train_dataloader, validation_dataloader, loss_fun=nn.CrossEntropyLoss(), opt_fun=torch.optim.SGD):
#     opt = opt_fun(model.parameters(), learning_rate)
#     for epoch in range(num_epochs):
#         model.train()  # turn on regularisation, to prevent overfitting
#         print(f"Training... Epoch {epoch}...")
#         i = 1
#         for batch in train_dataloader:
#             # print(f"batch: {batch}")
#             inputs, trimaps = batch
#             opt.zero_grad()  # reset gradients
#             outputs = model(inputs)  # put input batch through the model
#             loss = loss_fun(outputs, trimaps)  # calculate loss on the batch
#             loss.backward()  # calculate gradients
#             opt.step()  # update parameters?
#             # print(f"Loss: {loss.item()}")
#             # print(f"Outputs: {outputs}")
#             # print(f"Labels: {labels}")
#             if (i % 10 == 0):
#                 print(f"Batch {i}")
#             i += 1
#         evaluate(model, validation_dataloader)

def evaluate(model, validation_dataloader, loss_function=DiceLoss()):
    model.eval()  # turn off regularisation
    print("Evaluating...")
    with torch.no_grad():
        total = 0
        correct = 0
        for batch in validation_dataloader:
            print("evaluating batch")
            inputs, trimaps = batch
            outputs = model(inputs)["out"]
            # outputs = model(inputs)
            loss = loss_function(outputs, trimaps)  # calculate loss on the batch
            _, predicted = torch.max(outputs, 1)
            # _, predicted = torch.max(outputs.data, 1)
            total += trimaps.size(0)
            correct += (predicted == trimaps).sum().item()
            # print(f"Outputs: {outputs}")
            # print(f"Predicted: {predicted}")
            # print(f"Labels: {labels}")
            # print(f"Total: {total}")
            # print(f"Correct: {correct}")
        print(f"Final total: {total}")
        print(f"Final correct: {correct}")
        grade = 100 * correct / total
        print(f"Loss: {loss}")
        print(f"Accuraccy: {grade}")


def train(model, epochs, learning_rate, train_dataloader, validation_dataloader, loss_fun=nn.CrossEntropyLoss(), opt_fun=torch.optim.SGD):
    opt = opt_fun(model.parameters(), learning_rate)
    for epoch in range(epochs):
        model.train()  # turn on regularisation, to prevent overfitting
        print(f"Training... Epoch {epoch}...")
        i = 1
        for batch in train_dataloader:
            # print(f"batch: {batch}")
            inputs, trimaps = batch
            opt.zero_grad()  # reset gradients
            outputs = model(inputs)  # put input batch through the model
            loss = loss_fun(outputs, trimaps)  # calculate loss on the batch
            loss.backward()  # calculate gradients
            opt.step()  # update parameters?
            # print(f"Loss: {loss.item()}")
            # print(f"Outputs: {outputs}")
            # print(f"Labels: {labels}")
            if (i % 10 == 0):
                print(f"Batch {i}")
            i += 1
        evaluate(model, validation_dataloader)


def predict(model, image, trimap):
    input = image.unsqueeze(0)
    output = model(input)["out"]
    print("Output.shape(): ", output.shape)
    print("Output: ", output)
    return output
    # _, predicted = torch.max(output, 1)
    # return classes[predicted.item()]


# put this in evaluation before loss function?
def decode_segmap(image, number_of_classes):
    image = torch.argmax(image.squeeze(), dim=0).detach().cpu().numpy()
    label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0)])
    # label_colors = np.array([(0, 0, 0),
    #                          (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    #                          (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    #                          (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    #                          (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    red = np.zeros_like(image).astype(np.uint8)
    green = np.zeros_like(image).astype(np.uint8)
    blue = np.zeros_like(image).astype(np.uint8)
    for label in range(0, number_of_classes):
        idx = image == label
        red[idx] = label_colors[label, 0]
        green[idx] = label_colors[label, 1]
        blue[idx] = label_colors[label, 2]
    return cv2.cvtColor(np.stack([red, green, blue], axis=2), cv2.COLOR_RGB2BGR)
    # return cv2.cvtColor(rgb_segmap, cv2.COLOR_RGB2BGR)
    # return np.stack([red, green, blue], axis=2)


print("Start")

# hyperparameters
seed = 42
set_ratio = 0.2
train_dataset_ratio = 0.8
test_dataset_ratio = 0.1
validation_dataset_ratio = 0.1
number_of_classes = 3

batch_size = 1#20
epochs = 2#30
learning_rate = 0.05
model_path = "./models/"
images_per_class = 80

manager = ImageManager("../datasets/17flowers/jpg", "../datasets/trimaps", "../datasets/trimaps/imlist.mat")
# manager.displayImage(cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/trimaps/" + "image_0043.png"))
# print(cv2.imread("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/datasets/trimaps/" + "image_0026.png"))
manager.load()
manager.set_image_dimensions()
print(manager.default_width)
print(manager.default_height)
dataset = FlowerDataset("../datasets/17flowers/jpg/",
                        "../datasets/trimaps/",
                        transforms.Compose([Resize((manager.default_width, manager.default_height)), transforms.ToTensor()]))

train_dataset_size = int(train_dataset_ratio * len(dataset))
test_dataset_size = int(test_dataset_ratio * len(dataset))
validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size

torch.manual_seed(seed)  # to ensure creating same sets
train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])
print(f"Dataset size: {len(dataset)}")
print(f"Train_dataset size: {len(train_dataset)}")
print(f"Test_dataset size: {len(test_dataset)}")
print(f"Validation_dataset size: {len(validation_dataset)}")
# print("dataset --getitem-- return type: ", type(train_dataset[0][0]))

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size)

show_batch(test_dataloader)

model = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=number_of_classes, aux_loss=None).eval()
# model = FlowerNet()
# train(model, epochs, learning_rate, train_dataloader, validation_dataloader)
# evaluate(model, test_dataloader)
# torch.save(model.state_dict(), model_path + "flower1")
output = predict(model, train_dataset[3][0], train_dataset[3][1])
rgb = decode_segmap(output, number_of_classes)
print("RGB shape: ", rgb.shape)
manager.displayTensor(dataset[3][0])
manager.displayTensor(dataset[3][1])
print("Trimap: ", dataset[3][1])
manager.displayImage(rgb)

print("End")

#outs = torch.tensor([[0.0001, 0.0002, 0.0003], [0.0004, 0.0006, 0.0007], [0.00010, 0.0005, 0.0004]])
# outs = torch.tensor([[1, 2, 3], [4, 6, 7], [10, 5, 4]])
# _, predicted = torch.max(outs, 1)
# print(f"Predicted: {predicted}")
# train(model, num_epochs, learning_rate, train_dl, val_dl)
# torch.save(model.state_dict(), model_path + "cifar10")
#
# show_img(*test_set[1000], test_set)
# img, label = test_set[1000]
# print(f"Pred: {predict(model, img, classes)}, Label: {classes[label]}")

#model = SkyNet()
#model.load_state_dict(torch.load(model_path + 'cifar10'))

# evaluate(model, test_dl)
'''
print("End")

# data_dir = "../datasets/17flowers/jpg"
# # split data_dir into 3 sets: train, validate, test
# files = os.listdir(data_dir)
# print(f"Files: {files[:5]}")
# del files[:2]
# print(f"Files: {files[:5]}")
# # split_into_sets(files, seed, images_per_class)
# trimaps_images = loadmat("../datasets/trimaps/imlist.mat")
# print(f"trimaps keys: {trimaps_images.keys()}")
# # print(type(annots["imlist"]), annots["imlist"].shape)
# print([trimaps_images["imlist"][0][-1]]) #print 3 last images
# print([trimaps_images["imlist"][0][-2]])
# print([trimaps_images["imlist"][0][-3]])
# print(trimaps_images["imlist"])  # list containing the images with ground truth
#
# print(type(trimaps_images["imlist"][0][0]))
# indexes = list(map(int, trimaps_images["imlist"][0]))
# print(type(indexes[0]))
# indexes = [x - 1 for x in indexes]
# print(f"Indexes: {indexes}")
# dataset = [files[i] for i in indexes]
# print(f"Dataset: {dataset}")
# count_flower_types(dataset)
# tris = os.listdir("../datasets/trimaps")
# # difference in size, because some classes of flowers aren't considered
# print(f"Trimaps length: {len(tris)}")
# print(f"Trimaps actual length: {len(dataset)}")

'''
# test_set = ImageFolder(data_dir + '/test', transform=ToTensor())
# dataset = ImageFolder(data_dir + '/train',
#                       transform=ToTensor())  # returns a 2-element tuple, tensor(image data,:rgb, width, height) and int (in which class, airplane, ship, ... it belongs)
#
# val_size = int(set_ratio * len(dataset))
# train_size = len(dataset) - val_size
#
# torch.manual_seed(seed)  # to ensure creating same sets
# train_set, val_set = random_split(dataset, [train_size, val_size])
# print(f"Train_set size: {len(train_set)}")
# print(f"Val_set size: {len(val_set)}")
# # show_img(*train_set[0], dataset)
#
# train_dl = DataLoader(train_set, batch_size, shuffle = True)
# val_dl = DataLoader(val_set, batch_size, shuffle = True)
#
# test_dl = DataLoader(test_set, batch_size)
# #show_batch(train_dl)
#
# model = SkyNet()
# #outs = torch.tensor([[0.0001, 0.0002, 0.0003], [0.0004, 0.0006, 0.0007], [0.00010, 0.0005, 0.0004]])
# outs = torch.tensor([[1, 2, 3], [4, 6, 7], [10, 5, 4]])
# _, predicted = torch.max(outs, 1)
# print(f"Predicted: {predicted}")
# train(model, num_epochs, learning_rate, train_dl, val_dl)
# torch.save(model.state_dict(), model_path + "cifar10")
#
# show_img(*test_set[1000], test_set)
# img, label = test_set[1000]
# print(f"Pred: {predict(model, img, classes)}, Label: {classes[label]}")
#
# #model = SkyNet()
# #model.load_state_dict(torch.load(model_path + 'cifar10'))
#
# evaluate(model, test_dl)
