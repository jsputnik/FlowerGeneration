from torchvision.transforms import transforms
import utils.Helpers as Helpers
import nn.transforms as Transforms
import numpy as np
import cv2
import utils.image_operations as imops
from skimage.draw import line
from skimage.measure import regionprops
import math
from utils.Color import Color


# takes black and white image
def threshold_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, result_image = cv2.threshold(image, 127, 255, 0)
    return result_image


def get_center_point(image):
    mask = np.all(image == np.array([0, 128, 0]), axis=-1).astype(int)
    properties = regionprops(mask, image)
    center_of_mass = properties[0].centroid
    return math.floor(center_of_mass[0]), math.floor(center_of_mass[1])


def get_big_contours(contours):
    big_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # add to parameters
            print("Contour len: ", len(contour))
            big_contours.append(contour)
    return big_contours


def detect_intersection_points(image, middle_point, contours, worm_length, min_distance):
    for contour in contours:
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

            image[middle[1]][middle[0]] = np.array([255, 255, 255])
            rr, cc = line(head[1], head[0], tail[1], tail[0])
            if (image[rr[1]][cc[1]] == np.array([255, 255, 255])).all() or \
                    (image[rr[-2]][cc[-2]] == np.array([255, 255, 255])).all():
                continue
            distance = np.linalg.norm(np.cross(head - tail, tail - middle)) / np.linalg.norm(head - tail)
            if potential_intersection and distance <= previous_distance:
                intersection_counter += 1
                intersections.append(previous_pixel)
                potential_intersection = False
                image[previous_pixel[1]][previous_pixel[0]] = np.array([255, 255, 0])
            if distance > min_distance and distance > previous_distance:  # adjust
                potential_intersection = True
            # print("Distance: ", distance)
            previous_pixel = middle
            previous_distance = distance
        # print("Counter: ", intersection_counter)
        result_image = image.copy()
        for point in intersections:
            x, y = point.ravel()
            cv2.line(result_image, (x, y), (middle_point[1], middle_point[0]), (0, 0, 255))
    imops.displayImage(result_image)
    return result_image, intersections


# intersection points are sorted in the same direction contour is being iterated
def divide_petals(image, intersection_points, middle, contour, center_pixels):
    petals = []
    first_point = intersection_points[0]
    previous_connecting_line_pixels = []
    current_connecting_line_pixels = []
    petal_pixels = []

    # modify contour
    ipoint_contour_indexes = np.all(contour == intersection_points[0], axis=1).nonzero()
    assert(len(ipoint_contour_indexes) == 1)
    first_contour_point_index = ipoint_contour_indexes[0][0]
    print("first ipoint: ", intersection_points[0])
    print("first index: ", first_contour_point_index)
    contour = contour.tolist()
    contour = contour[first_contour_point_index + 1:] + contour[:first_contour_point_index + 1]
    print("Contour: ", contour)

    # find intersection_points[0] in contour
    x_coords, y_coords = line(first_point[1], first_point[0], middle[1], middle[0])
    assert(len(x_coords) == len(y_coords))
    for index in range(len(x_coords)):
        previous_connecting_line_pixels.append(np.array([x_coords[index], y_coords[index]]))
    current_contour_index = 0
    # find pixels belonging to middle <-> current intersection point line
    for index in range(1, len(intersection_points) + 1):
        petal_contour_pixels = []
        current_ipoint = intersection_points[0] if index == len(intersection_points) else intersection_points[index]
        x_coords, y_coords = line(current_ipoint[1], current_ipoint[0], middle[1], middle[0])
        # print("rr: ", x_coords)
        # print("cc: ", y_coords)
        assert (len(x_coords) == len(y_coords))
        for pixel_index in range(len(x_coords)):
            current_connecting_line_pixels.append(np.array([x_coords[pixel_index], y_coords[pixel_index]]))
        while np.any(contour[current_contour_index] != current_ipoint):
            normalized_pixel = np.array([contour[current_contour_index][1], contour[current_contour_index][0]])
            petal_contour_pixels.append(normalized_pixel)
            current_contour_index += 1
        closed_contour_pixels = previous_connecting_line_pixels + current_connecting_line_pixels + petal_contour_pixels
        petal_image = fill_petal_contour(image.copy(), closed_contour_pixels)
        rows, columns, depth = petal_image.shape
        for row in range(rows):
            for column in range(columns):
                if (petal_image[row][column][:] == np.array([255, 255, 255])).all():
                    petal_pixels.append(np.array([row, column]))
        for center_pixel in center_pixels:
            petal_pixels = np.delete(petal_pixels, np.where(np.all(petal_pixels == center_pixel, axis=-1)), axis=0)
        petals.append(petal_pixels)
        previous_connecting_line_pixels = current_connecting_line_pixels
        current_connecting_line_pixels = []
        petal_pixels = []
    return petals


def fill_petal_contour(image, contour):
    # TODO: fix connecting line pixels in different petals overlapping
    for pixel in contour:
        image[pixel[0]][pixel[1]] = np.array([255, 0, 0])
    # imops.displayImage(image)
    black_white_transform = transforms.Compose(
        [Transforms.ChangeColor(np.array([255, 255, 255]),
                                np.array([0, 0, 0])),
         Transforms.ChangeColor(np.array([0, 0, 255]),
                                np.array([0, 0, 0])),
         Transforms.ChangeColor(np.array([255, 0, 0]), np.array([255, 255, 255]))])
    result_image = black_white_transform(image)
    thresholded_image = threshold_image(result_image)
    contours, hierarchy = cv2.findContours(thresholded_image,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(result_image, contours, 0, (255, 255, 255), thickness=-1)
    # imops.displayImage(result_image)
    return result_image


def fill_petals_with_color(image, petals):
    # TODO: handle situation when too many petals and not enough colors
    current_color = Color.first_petal_color.copy()
    # color_difference = 20
    for petal in petals:
        for pixel in petal:
            image[pixel[0]][pixel[1]] = current_color
        current_color -= Color.color_difference
    return image


def decomposition_algorithm(image):
    center_mask = np.all(image == np.array([0, 128, 0]), axis=-1)
    raw_center_pixels = np.where(np.all(image == np.array([0, 128, 0]), axis=-1))
    if np.size(raw_center_pixels) == 0:
        return image, 0
    center_pixels = []
    # from y and x coords in 2 separate arrays to 1
    for index in range(len(raw_center_pixels[0])):
        y = raw_center_pixels[0][index]
        x = raw_center_pixels[1][index]
        center_pixels.append(np.array([y, x]))
    # imops.displayImage(image)
    center_point = get_center_point(image)
    black_white_transform = transforms.Compose([Transforms.ChangeColor(np.array([0, 128, 128]), np.array([0, 0, 0])),
                                                Transforms.ChangeColor(np.array([0, 0, 128]),
                                                                       np.array([255, 255, 255])),
                                                Transforms.ChangeColor(np.array([128, 128, 128]),
                                                                       np.array([0, 0, 0])),
                                                Transforms.ChangeColor(np.array([0, 128, 0]),
                                                                       np.array([255, 255, 255]))
                                                ])
    black_white_image = black_white_transform(image)
    thresholded_image = threshold_image(black_white_image)
    # imops.displayImagePair(image_gray, segmap)
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
    big_contours = get_big_contours(contours)
    cv2.drawContours(black_white_image, big_contours, 0, (0, 255, 0))
    result_image, intersection_points = detect_intersection_points(black_white_image, center_point, big_contours, worm_length=21, min_distance=4.5)
    number_of_parts = len(intersection_points)
    middle = np.array([center_point[1], center_point[0]]).astype(int)
    petals = divide_petals(result_image.copy(), intersection_points, middle, big_contours[0].squeeze(), center_pixels)
    result_image = fill_petals_with_color(result_image, petals)
    result_image = Helpers.apply_boolean_mask(result_image, center_mask, new_color=Color.center_color)
    imops.displayImage(result_image)
    return result_image, number_of_parts



