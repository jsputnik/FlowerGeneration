import torch
import sys
import cv2
import os

import utils.Helpers as Helpers
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops

number_of_classes = 4

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,  # model output channels (number of classes in your dataset)
).to(Device.get_default_device())

# if len(sys.argv) != 3:
#     raise Exception("Invalid number of parameters (need 2)")

model_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/FlowerGeneration/models/90.18flower"
image_path = "C:/Users/iwo/Documents/PW/PrInz/FlowerGen/testFLower.png"#"C:/Users/iwo/Documents/PW/PrInz/FlowerGen/17flower_dataset/17flowers/jpg/image_0008.jpg"
# model_path = sys.argv[1]
# image_path = sys.argv[2]

model.load_state_dict(torch.load(model_path))
image = cv2.imread(image_path)
output = Helpers.predict(model, image)
# rgb = output.squeeze().detach().numpy().transpose((1, 2, 0))
rgb = Helpers.decode_segmap(output, number_of_classes)
imops.displayImagePair(image, rgb)
