import os
import numpy as np
from scipy.io import loadmat
import cv2


class ImageManager:
    # trimaps and data images have same names
    def __init__(self, data_dir: str, trimaps_dir: str, trimaps_list_file: str):
        self.data_dir = data_dir  # path to dir with data
        self.trimaps_dir = trimaps_dir  # not used
        self.trimaps_list_file = trimaps_list_file  # path to matlab file
        self.trimaps = {}  # trimap images not names
        self.data = {}  # data images not names
        self.resized = {}  # resized images
        self.default_width = 0
        self.default_height = 0

    # loads data and trimaps for 17flowers dataset, only the data that have corresponding trimaps
    def load(self):
        trimap_indexes = loadmat(self.trimaps_list_file)  # to assign data to trimaps
        # print("Trimaps type: ", type(trimap_indexes["imlist"]))  #uint16
        trimap_indexes = list(map(int, trimap_indexes["imlist"][0]))
        trimap_indexes = [x - 1 for x in trimap_indexes]
        # print("Indexes type: ", type(indexes[0]))  # int
        image_names = os.listdir(self.data_dir)  # only difference is format jpg and png
        del image_names[:2]
        image_names = [image_names[i] for i in trimap_indexes]
        print("Data image names: ", image_names)
        for i in range(0, len(image_names)):
            self.data.update({image_names[i]: cv2.imread(self.data_dir + "/" + image_names[i])})
            self.trimaps.update({image_names[i]: cv2.imread(self.trimaps_dir + "/" + image_names[i])})
            # self.data.append(cv2.imread(self.data_dir + "/" + self.image_names[i]))
            # self.trimaps.append(cv2.imread(self.trimaps_dir + "/" + self.image_names[i]))
        # files = os.listdir(self.data_dir)  # data files names
        # del files[:2]  # remove first 2 files not containing any images
        # self.image_names = [files[i] for i in indexes]

    def displayTensor(self, image):
        cv2.imshow("Image", image.numpy().transpose((1, 2, 0)))
        cv2.waitKey(0)

    def displayImage(self, image):
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    def resize_all(self):
        for name, data in self.data.items():
            print("Current img: ", name)
            self.resized.update({name: self.resize(data)})
        print("Resized images keys: ", self.resized.keys())

    def resize(self, img):
        height, width = img.shape[:2]
        ratio = height / width
        if height > self.default_height:
            height = self.default_height
            width = int(self.default_height // ratio)
            img = cv2.resize(img, (width, height))
        result = np.zeros((self.default_height, self.default_width, 3), np.uint8)
        start_x = (self.default_width - width) // 2
        start_y = (self.default_height - height) // 2
        result[start_y:start_y + height, start_x:start_x + width] = img
        return result

    def set_image_dimensions(self):
        width = 0
        height = 0
        for img in self.data.values():
            new_height, new_width = img.shape[:2]
            if new_width > width:
                width = new_width
                height = new_height
        self.default_width = width
        self.default_height = height
        print("Default width: ", self.default_width)
        print("Default height: ", self.default_height)

    # must be caled after load()
    def get_statistics(self):
        widths_names = {}
        heights_names = {}
        # image_sizes = {}
        max_width = 0
        max_width_img = ""
        max_height = 0
        max_height_img = ""
        for name, img in self.data.items():
            img_height, img_width = img.shape[:2]
            widths_names.update({name: img_width})
            heights_names.update({name: img_height})
            if img_width > max_width:
                max_width = img_width
                max_width_img = name
            if img_height > max_height:
                max_height = img_height
                max_height_img = name
        widths = list(widths_names.values())
        heights = list(heights_names.values())
        print(widths)
        print(heights)
        widths.sort()
        heights.sort()
        return max_width_img, max_width, max_height_img, max_height, widths[len(widths)//2], heights[len(heights)//2]

    def get_wide_and_tall(self):
        wide = 0
        tall = 0
        for img in self.data.values():
            img_height, img_width = img.shape[:2]
            if img_width > img_height:
                wide += 1
            else:
                tall += 1
        return wide, tall

    # imlist.mat outdated, all categories have trimaps except cowslips
    def count_flower_types(self):
        daffodils = 0
        snowdrops = 0
        lily_valleys = 0
        bluebells = 0
        crocuses = 0
        irises = 0
        tigerlilies = 0
        tulips = 0
        fritillaries = 0
        sunflowers = 0
        daisies = 0
        colts_foots = 0
        dandellions = 0
        cowslips = 0
        buttercups = 0
        windflowers = 0
        pansies = 0
        for im in self.data.keys():
            if "image_0000.jpg" <= im <= "image_0080.jpg":
                daffodils = daffodils + 1
            elif "image_0081.jpg" <= im <= "image_0160.jpg":
                snowdrops = snowdrops + 1
            elif "image_0161.jpg" <= im <= "image_0240.jpg":
                lily_valleys = lily_valleys + 1
            elif "image_0241.jpg" <= im <= "image_0320.jpg":
                bluebells = bluebells + 1
            elif "image_0321.jpg" <= im <= "image_0400.jpg":
                crocuses = crocuses + 1
            elif "image_0401.jpg" <= im <= "image_0480.jpg":
                irises = irises + 1
            elif "image_0481.jpg" <= im <= "image_0560.jpg":
                tigerlilies = tigerlilies + 1
            elif "image_0561.jpg" <= im <= "image_0640.jpg":
                tulips = tulips + 1
            elif "image_0641.jpg" <= im <= "image_0720.jpg":
                fritillaries = fritillaries + 1
            elif "image_0721.jpg" <= im <= "image_0800.jpg":
                sunflowers = sunflowers + 1
            elif "image_0801.jpg" <= im <= "image_0880.jpg":
                daisies = daisies + 1
            elif "image_0881.jpg" <= im <= "image_0960.jpg":
                colts_foots = colts_foots + 1
            elif "image_0961.jpg" <= im <= "image_1040.jpg":
                dandellions = dandellions + 1
            elif "image_1041.jpg" <= im <= "image_1120.jpg":
                cowslips = cowslips + 1
            elif "image_1121.jpg" <= im <= "image_1200.jpg":
                buttercups = buttercups + 1
            elif "image_1201.jpg" <= im <= "image_1280.jpg":
                windflowers = windflowers + 1
            elif "image_1281.jpg" <= im <= "image_1360.jpg":
                pansies = pansies + 1
        return daffodils, snowdrops, lily_valleys, bluebells, crocuses, irises, tigerlilies, tulips, \
            fritillaries, sunflowers, daisies, colts_foots, dandellions, cowslips, buttercups, windflowers, \
            pansies
