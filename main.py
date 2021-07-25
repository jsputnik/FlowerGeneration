import torch
import cv2

import torch.nn as nn

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from utils.ImageManager import ImageManager


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


# class SkyNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             #nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             #nn.ReLU(),
#             #nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             #nn.ReLU(),
#             #nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
#
#             #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             #nn.ReLU(),
#             #nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             #nn.ReLU(),
#             #nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
#
#             #nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             #nn.ReLU(),
#             #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             #nn.ReLU(),
#             #nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
#
#             #nn.Flatten(),
#             #nn.Linear(256*4*4, 1024),
#             #nn.ReLU(),
#             #nn.Linear(1024, 512),
#             #nn.ReLU(),
#             #nn.Linear(512, 10)
#             nn.Conv2d(3, 16, 3, padding = 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(16, 16, 3, padding = 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(16, 16, 3, padding = 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(16, 16, 3, padding = 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(16, 16, 3, padding = 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Flatten(),
#             nn.Linear(16, 10)
#             )
#
#     def forward(self, xb):
#         return self.network(xb)

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
# image1 = cv2.imread("../datasets/17flowers/jpg/image_0003.jpg")
# # print(type(image1[0][0][0]))
# print(image1)
# # print(f"Image1 type: {type(image1)}")
# cv2.imshow("Image1", image1)
# cv2.waitKey(0)
# resized = cv2.resize(image1, (400, 400), )
# cv2.imshow("Image1 resized", resized)
# cv2.waitKey(0)

manager = ImageManager("../datasets/17flowers/jpg", "../datasets/trimaps", "../datasets/trimaps/imlist.mat")
manager.load()
manager.set_image_dimensions()
print("Statistics: ", manager.get_statistics())
print(f"Wide and tall: {manager.get_wide_and_tall()}")
print(f"Flower trimaps count for each class: {manager.count_flower_types()}")

# manager.display(manager.data["image_0390.jpg"])
manager.resize_all()
# manager.resize(manager.data["image_0001.jpg"])
cv2.destroyAllWindows()


# hyperparameters
seed = 42
set_ratio = 0.2

batch_size = 200
num_epochs = 30
learning_rate = 0.05
model_path = "./models/"
images_per_class = 80

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

# ds = FlowerDataset("mydataset")
# print(ds.name)
# ds.print()
# dataset = ImageFolder(data_dir, transform = ToTensor())

'''
test_set = ImageFolder(data_dir + '/test', transform=ToTensor())
dataset = ImageFolder(data_dir + '/train',
                      transform=ToTensor())  # returns a 2-element tuple, tensor(image data,:rgb, width, height) and int (in which class, airplane, ship, ... it belongs)

val_size = int(set_ratio * len(dataset))
train_size = len(dataset) - val_size

torch.manual_seed(seed)  # to ensure creating same sets
train_set, val_set = random_split(dataset, [train_size, val_size])
print(f"Train_set size: {len(train_set)}")
print(f"Val_set size: {len(val_set)}")
# show_img(*train_set[0], dataset)

train_dl = DataLoader(train_set, batch_size, shuffle = True)
val_dl = DataLoader(val_set, batch_size, shuffle = True)

test_dl = DataLoader(test_set, batch_size)
#show_batch(train_dl)

model = SkyNet()
#outs = torch.tensor([[0.0001, 0.0002, 0.0003], [0.0004, 0.0006, 0.0007], [0.00010, 0.0005, 0.0004]])
outs = torch.tensor([[1, 2, 3], [4, 6, 7], [10, 5, 4]])
_, predicted = torch.max(outs, 1)
print(f"Predicted: {predicted}")
train(model, num_epochs, learning_rate, train_dl, val_dl)
torch.save(model.state_dict(), model_path + "cifar10")

show_img(*test_set[1000], test_set)
img, label = test_set[1000]
print(f"Pred: {predict(model, img, classes)}, Label: {classes[label]}")

#model = SkyNet()
#model.load_state_dict(torch.load(model_path + 'cifar10'))

evaluate(model, test_dl)
'''
print("End")
