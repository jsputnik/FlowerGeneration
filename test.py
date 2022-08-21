import torch
import cv2
import numpy as np
import utils.Helpers as Helpers
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops

number_of_classes = 4

model = smp.MAnet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,  # model output channels (number of classes in your dataset)
).to(Device.get_default_device())

model_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/models/94.88Manet"
# image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/browneyedsusan.jpg"
image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/testFlower.png"
# image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/rose.jpg"
# image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/17flower_dataset/17flowers/jpg/image_1265.jpg"

model.load_state_dict(torch.load(model_path))
image = cv2.imread(image_path)
output = Helpers.predict(model, image)
# rgb = output.squeeze().detach().numpy().transpose((1, 2, 0))
rgb = Helpers.decode_segmap(output, number_of_classes)
# imops.displayImagePair(image, rgb)
cv2.imwrite("C:/Users/iwo/Documents/PW/PrInz/FlowerGen/testFlowerNoneManet.png", rgb)
