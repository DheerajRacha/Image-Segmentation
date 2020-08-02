import os
import cv2
import glob
import numpy as np
from natsort import natsorted


class DataLoader:
    def __init__(self, dataset_dir, mode):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.images_names, self.annotation_names = self.prepare_data()
        self.dataset_size = len(self.images_names)

    def prepare_data(self):
        if self.mode == "inference":
            images = os.path.join(self.dataset_dir, "JPEGImages")
            images_names = glob.glob(images + "/**/*.jpeg")
            images_names = natsorted(images_names)

            annotation_names = None
        else:
            images = os.path.join(self.dataset_dir, "JPEGImages")
            images_names = glob.glob(images + "/**/*.jpeg")
            images_names = natsorted(images_names)

            annotations = os.path.join(self.dataset_dir, "SegmentationClass")
            annotation_names = glob.glob(annotations + "/**/*.jpeg")
            annotation_names = natsorted(annotation_names)

        self.dataset_size = len(images_names)
        return images_names, annotation_names

    @staticmethod
    def load_img(image_path):
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    def load_data(self, batch_size):
        permutation = np.random.permutation(self.dataset_size)
        num_batches = int(self.dataset_size / batch_size)

        if self.mode == "inference":
            for curr_batch in range(num_batches):
                x_batch = []
                for i in range(curr_batch * batch_size, curr_batch * batch_size + batch_size):
                    x_batch.append(self.load_img(self.images_names[permutation[i]]) / 255.0)
                yield np.array(x_batch), self.images_names[permutation[i]]
        else:
            for curr_batch in range(num_batches):
                x_batch, y_batch = [], []
                for i in range(curr_batch * batch_size, curr_batch * batch_size + batch_size):
                    x_batch.append(self.load_img(self.images_names[permutation[i]]) / 255.0)
                    y_batch.append(self.load_img(self.annotation_names[permutation[i]]))
                yield np.array(x_batch), np.array(y_batch)
