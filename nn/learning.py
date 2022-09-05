import numpy as np
import torch
import cv2
import utils.Helpers as Helpers
import utils.Device as Device
import torch.nn as nn
import nn.transforms as Transforms
from torchvision.transforms import transforms
import segmentation_models_pytorch as smp
import utils.image_operations as imops
import torchmetrics


def train(model, epochs, learning_rate, train_dataloader, validation_dataloader, loss_fun=nn.CrossEntropyLoss(),
          opt_fun=torch.optim.SGD):
    opt = opt_fun(model.parameters(), learning_rate)
    for epoch in range(epochs):
        model.train()  # turn on regularisation, to prevent overfitting
        print(f"Training... Epoch {epoch}...")
        i = 1
        for batch in train_dataloader:
            inputs, trimaps = batch[0].to(Device.get_default_device()), batch[1].to(Device.get_default_device())
            opt.zero_grad()  # reset gradients
            outputs = model(inputs)  # put input batch through the model
            loss = loss_fun(outputs, torch.squeeze(trimaps))  # calculate loss on the batch
            loss.backward()  # calculate gradients
            opt.step()  # update parameters
            if i % 10 == 0:
                print(f"Batch {i}")
            i += 1
        evaluate(model, validation_dataloader)


def evaluate(model, validation_dataloader, metric=torchmetrics.Accuracy(mdmc_average="samplewise")):
    model.eval()  # turn off regularisation
    print("Evaluating...")
    avg_accuracy = 0
    label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])
    with torch.no_grad():
        for batch in validation_dataloader:
            inputs, trimaps = batch[0].to(Device.get_default_device()), batch[1]  # both are tensors
            outputs = model(inputs)
            for output, trimap in zip(outputs, trimaps):
                segmap_image = Helpers.decode_segmap(output, label_colors)
                transform = transforms.Compose([Transforms.ToMask(), transforms.ToTensor()])
                segmap_mask = transform(segmap_image)
                accuracy = metric(segmap_mask, trimap)
                avg_accuracy += accuracy
    avg_accuracy = avg_accuracy / len(validation_dataloader.dataset) * 100
    return avg_accuracy


def segment(image_path, model, number_of_classes):
    model.to(Device.get_default_device())
    test_image = cv2.imread(image_path)
    output = Helpers.predict(model, test_image)
    label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 0)])
    # label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])
    if number_of_classes == 3:
        label_colors = np.array([(0, 0, 0), (255, 255, 255), (128, 128, 128)])
    segmap = Helpers.decode_segmap(output, label_colors=label_colors)
    return segmap


def segment_image(image, model_path="C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/static/models/95.38flower", number_of_classes=4):
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=number_of_classes,  # model output channels (number of classes in your dataset)
    ).to(Device.get_default_device())
    model.load_state_dict(torch.load(model_path))
    output = Helpers.predict(model, image)
    label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])
    if number_of_classes == 3:
        label_colors = np.array([(0, 0, 0), (255, 255, 255), (128, 128, 128)])
    segmap = Helpers.decode_segmap(output, label_colors=label_colors)
    return segmap
