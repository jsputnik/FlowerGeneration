import torch

import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from nn.FlowerNet import FlowerNet
from utils.ImageManager import ImageManager
from nn.FlowerDataset import FlowerDataset
from nn.transforms import Resize
# import torch.utils.data
# from torch.utils.data import random_split


def show_img(img, label, ds):
    print(f"Label: {ds.classes[label]}")
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def show_batch(dl):
    for images, labels in dl:
        print(f"labels: {labels}")
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(make_grid(images, nrow=10).permute(1, 2, 0))
        plt.show()
        break
    print("ahaah")


def evaluate(model, val_dl, loss_fun=nn.CrossEntropyLoss()):
    model.eval()  # turn off regularisation
    print("Evaluating...")
    with torch.no_grad():
        total = 0
        correct = 0
        for batch in val_dl:
            inputs, labels = batch
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)  # calculate loss on the batch
            _, predicted = torch.max(outputs, 1)
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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


def train(model, num_epochs, learning_rate, train_dl, val_dl, loss_fun=nn.CrossEntropyLoss(), opt_fun=torch.optim.SGD):
    opt = opt_fun(model.parameters(), learning_rate)
    for epoch in range(num_epochs):
        model.train()  # turn on regularisation, to prevent overfitting
        print(f"Training... Epoch {epoch}...")
        i = 1
        for batch in train_dl:
            # print(f"batch: {batch}")
            inputs, labels = batch
            opt.zero_grad()  # reset gradients
            outputs = model(inputs)  # put input batch through the model
            loss = loss_fun(outputs, labels)  # calculate loss on the batch
            loss.backward()  # calculate gradients
            opt.step()  # update parameters?
            # print(f"Loss: {loss.item()}")
            # print(f"Outputs: {outputs}")
            # print(f"Labels: {labels}")
            if (i % 10 == 0):
                print(f"Batch {i}")
            i += 1
        evaluate(model, val_dl)


def predict(model, image, classes):
    input = image.unsqueeze(0)
    output = model(input)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]


def split_into_sets(data, seed, images_per_class):
    print("hey")

print("Start")

# hyperparameters
seed = 42
set_ratio = 0.2
train_dataset_ratio = 0.8
test_dataset_ratio = 0.1
validation_dataset_ratio = 0.1

batch_size = 200
num_epochs = 30
learning_rate = 0.05
model_path = "./models/"
images_per_class = 80

manager = ImageManager("../datasets/17flowers/jpg", "../datasets/trimaps", "../datasets/trimaps/imlist.mat")
manager.load()
manager.set_image_dimensions()
print(manager.default_width)
print(manager.default_height)
dataset = FlowerDataset("../datasets/17flowers/jpg/", "../datasets/trimaps/", Resize((manager.default_width, manager.default_height)))
# manager.display(dataset[0])
# cv2.waitKey(0)

train_dataset_size = int(train_dataset_ratio * len(dataset))
test_dataset_size = int(test_dataset_ratio * len(dataset))
validation_dataset_size = len(dataset) - train_dataset_size - test_dataset_size

torch.manual_seed(seed)  # to ensure creating same sets
train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, test_dataset_size, validation_dataset_size])
print(f"Dataset size: {len(dataset)}")
print(f"Train_dataset size: {len(train_dataset)}")
print(f"Test_dataset size: {len(test_dataset)}")
print(f"Validation_dataset size: {len(validation_dataset)}")
manager.display(train_dataset[0][1])

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size)
# show_batch(train_dataloader)

model = FlowerNet()

print("End")
# train(model, num_epochs, learning_rate, train_dataloader, validation_dataloader)

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
