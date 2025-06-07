import tifffile
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import sqlite3
import csv
import pandas as pd
import torchstain
import torch


def data_visualization_test(data_path, results_path):
    """
    This script is used to visualize the data from the MIDOG++ dataset.
    It will load the data from the json file and visualize the annotations on the images.

    :param data_path: path to the data
    :param results_path: path to the results

    :return: None(outputs images with annotations, only one for testing)
    """
    with open(database, 'r') as f:
        data = json.load(f)
        print(data.keys())
        print(data['images'][0].keys())
        print(data['annotations'][0].keys())

        print(data['annotations'][0])

        for image in data['images']:
            print(image['file_name'])
            slide = tifffile.imread(os.path.join(data_path, image['file_name']))

            for annotation in data['annotations']:
                if annotation['image_id'] == image['id']:
                    print(annotation['category_id'])
                    print(annotation['bbox'])
                    x, y, w, h = annotation['bbox']
                    cv2.rectangle(slide, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 10)
                    cv2.putText(slide, str(annotation['category_id']), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0), 30)
                    cv2.imwrite(os.path.join(results_path, image['file_name']), slide)
            plt.figure(figsize=(10, 10))
            plt.imshow(slide)
            plt.axis('on')  # or 'off' to hide axes
            plt.title("Scroll & Zoom Enabled")
            plt.show()
            break


def data_info(database):
    """
    :param database: path to the database
    :return: None(outputs information about the label distribution)
    """
    with open(database, 'r') as f:
        data = json.load(f)
        print(data.keys())
        print(data['images'][0].keys())
        print(data['annotations'][0].keys())

        label_dict = {}

        for label in data['annotations']:
            category_id = label['category_id']
            if category_id not in label_dict:
                label_dict[category_id] = 1
            else:
                label_dict[category_id] += 1

        print(f"Number of mitotic figure: {label_dict[1]}, and number of non-mitotic figure: {label_dict[2]}")

        percents = [100 * label_dict[key] / sum(label_dict.values()) for key in label_dict]
        print(f"Percent of mitotic figure: {percents[1]}, and percent of non-mitotic figure: {percents[0]}")


def Patch_extraction(image, patch_size, overlap):
    """
    :param image: image to be extracted
    :param patch_size: size of the patch
    :param overlap: overlap between patches
    :return: patches
    """
    patches = []
    for i in range(0, image.shape[0], patch_size - overlap):
        for j in range(0, image.shape[1], patch_size - overlap):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

# Skipping rotation and flipping for now and only adding jitter
def Data_augmentation(image):
    """
    :param image: image to be augmented
    :return: augmented image
    """

    # adding jitter
    jitter = np.random.randint(-10, 10, size=image.shape)
    image = image + jitter
    image = np.clip(image, 0, 255)
    return image


def Labling_and_patching_test(data_path, results_path):
    """
    The problem I am facing is that I don't know how to label and have an accurate bounding box if I segment them into 
    patches because the bounding box can overlap into multiple patches and if I just label all the patches that it is in,
    then I will have a lot of false positives.

    The code works backwards and makes everything that doesn't overlap red and everything that does overlap blue. :D Science
    """

    with open(database, 'r') as f:
        data = json.load(f)
        print(data.keys())
        print(data['images'][0].keys())
        print(data['annotations'][0].keys())

        print(data['annotations'][0])
        first = True

        for image in data['images']:
            if first:
                first = False
                continue
            print(image['file_name'])
            slide = tifffile.imread(os.path.join(data_path, image['file_name']))

            for annotation in data['annotations']:
                if annotation['image_id'] == image['id']:
                    print(annotation['category_id'])
                    print(annotation['bbox'])
                    x, y, w, h = annotation['bbox']
                    cv2.rectangle(slide, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 20)
                    cv2.putText(slide, str(annotation['category_id']), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0), 30)
                    cv2.imwrite(os.path.join(results_path, image['file_name']), slide)
            # draw boxes of 256x256 size on the image with 50% overlap simulating the patches but make the ones that have the bounding box in them red
            print(slide.shape)
            print(slide.shape[0])
            print(slide.shape[1])

            for i in range(0, slide.shape[0] - 256, 128):
                for j in range(0, slide.shape[1] - 256, 128):
                    patch_has_overlap = False  # assume patch has no overlap

                    for annotation in data['annotations']:
                        if annotation['image_id'] == image['id']:
                            x, y, w, h = annotation['bbox']

                            # bounding box corners
                            box_x1 = x
                            box_y1 = y
                            box_x2 = x + w
                            box_y2 = y + h

                            # patch corners
                            patch_x1 = j
                            patch_y1 = i
                            patch_x2 = j + 256
                            patch_y2 = i + 256

                            # check for any overlap
                            if not (box_x2 <= patch_x1 or box_x1 >= patch_x2 or
                                    box_y2 <= patch_y1 or box_y1 >= patch_y2):
                                patch_has_overlap = True
                                break  # one overlap is enough to make it red

                    color = (0, 0, 255) if patch_has_overlap else (255, 0, 0)  # red if overlaps, blue otherwise
                    cv2.rectangle(slide, (j, i), (j + 256, i + 256), color, 10)

            plt.figure(figsize=(10, 10))
            plt.imshow(slide)
            plt.axis('off')  # or 'off' to hide axes
            plt.title("Segmented and labeled image")
            plt.show()
            break


if __name__ == '__main__':
    data_path = 'Data/Images/'
    results_path = 'databases'

    database = results_path + '/MIDOG++.json'

    # data_visualization_test(data_path, results_path)

    # data_info(database)

    # test_image = tifffile.imread(os.path.join(data_path, '001.tiff'))
    # patches = Patch_extraction_with_annotation(test_image, 256, 0)
    # for patch in patches:
    #     plt.imshow(patch)
    #     plt.show()

    # test_image = tifffile.imread(os.path.join(data_path, '001.tiff'))
    # new_image = Data_augmentation(test_image)
    # plt.imshow(new_image)
    # plt.show()

    Labling_and_patching_test(data_path, results_path)