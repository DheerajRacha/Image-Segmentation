import os
import cv2
import glob
import numpy as np
from natsort import natsorted

path_rgb = "inria_dataset/train/JPEGImages"
path_annotations = "inria_dataset/train/SegmentationClass"

folders = ["austin", "chicago", "kitsap", "tyrol", "vienna"]

for folder in folders:

    images_names_rgb = glob.glob(path_rgb + "/" + folder + "/*.jpeg")
    images_names_rgb = natsorted(images_names_rgb)

    images_names_annotation = glob.glob(path_annotations + "/" + folder + "/*.jpeg")
    images_names_annotation = natsorted(images_names_annotation)

    for m in range(len(images_names_annotation)):

        image_rgb = cv2.cvtColor(cv2.imread(images_names_rgb[m], -1), cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

        image_annotation = cv2.cvtColor(cv2.imread(images_names_annotation[m], -1), cv2.COLOR_BGR2RGB)
        image_annotation = cv2.resize(image_annotation, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

        rgb_patches = np.array(
            [image_rgb[i:i + 500, j:j + 500] for j in range(0, 5000, 500) for i in range(0, 5000, 500)])

        annotation_patches = np.array(
            [image_annotation[i:i + 500, j:j + 500] for j in range(0, 5000, 500) for i in range(0, 5000, 500)])

        new_filename = images_names_rgb[m].split("/")[-1].split(".")[0]

        j = 1
        for rgb_patch, annotation_patch in zip(rgb_patches, annotation_patches):
            rgb_save = os.path.join("processed_inria/train/JPEGImages", folder,
                                    str(new_filename) + "_" + str(j) + ".jpeg")
            annotaion_save = os.path.join("processed_inria/train/SegmentationClass", folder,
                                          str(new_filename) + "_" + str(j) + ".jpeg")

            cv2.imwrite(rgb_save, rgb_patch)
            cv2.imwrite(annotaion_save, annotation_patch)
            j = j + 1
    print(folder)
