import numpy as np
import torch
import utils.Helpers as Helpers
import utils.Device as Device
import torch.nn as nn
import nn.transforms as Transforms
from torchvision.transforms import transforms


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
            opt.step()  # update parameters?
            print(f"Loss: {loss.item()}")
            if (i % 10 == 0):
                print(f"Batch {i}")
            i += 1
        evaluate(model, validation_dataloader)


def evaluate(model, validation_dataloader):
    model.eval()  # turn off regularisation
    print("Evaluating...")
    number_of_classes = 4
    avg_accuracy = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            print("Evaluating batch")
            inputs, trimaps = batch[0].to(Device.get_default_device()), batch[1]  # both are tensors
            outputs = model(inputs)
            for output, trimap in zip(outputs, trimaps):
                segmap_image = Helpers.decode_segmap(output, number_of_classes)
                transform = transforms.Compose([Transforms.ToMask(), transforms.ToTensor()])
                segmap_mask = transform(segmap_image).numpy()
                correctly_assigned_pixels = np.sum(segmap_mask == trimap.numpy())
                accuracy: float = float(correctly_assigned_pixels) / segmap_mask.size
                # # TODO(?): only calculate not cropped area
                print(f"accuracy: {accuracy}")
                avg_accuracy += accuracy
                # image.displayImage(segmap_image)
    avg_accuracy = avg_accuracy / len(validation_dataloader.dataset) * 100
    return avg_accuracy
