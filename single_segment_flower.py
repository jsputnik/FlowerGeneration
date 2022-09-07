import torch
import cv2
import numpy as np
import utils.Helpers as Helpers
import utils.Device as Device
import segmentation_models_pytorch as smp
import utils.image_operations as imops

number_of_classes = 4

# user specific parameters
model_path = "./models/95.30UnetRotate"
image_path = "../browneyedsusan.jpg"
model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,  # model output channels (number of classes in your dataset)
).to(Device.get_default_device())

model.load_state_dict(torch.load(model_path))
image = cv2.imread(image_path)
output = Helpers.predict(model, image)
label_colors = np.array([(0, 0, 0), (128, 128, 0), (128, 0, 0), (128, 128, 128)])
rgb = Helpers.decode_segmap(output, label_colors)
imops.displayImage(rgb)
