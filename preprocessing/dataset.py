import os

import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory

from files import is_low_distortion_file, is_icon, is_high_distortion_file


class Dataset:
    def __init__(self, path, name, height, width, batch_size=None):
        # Path to where this data lives. This should be in a pre-processed form of images/testing/A, etc.
        self.path = path

        # Training or Testing
        self.name = name

        self.height = height
        self.width = width

        self.batch_size = batch_size

        self.size = 0
        self.get_size_of_dataset()

        # Images
        self.images = np.zeros((self.size, self.height, self.width, 3), dtype=np.float64)

        # Labels for each image
        self.optotype = np.zeros(self.size, dtype=object)     # cake, C, D, duck, etc.
        self.angle = np.zeros(self.size, dtype=np.int)        # 0, 45, 90, or 135
        self.size = np.zeros(self.size, dtype=object)         # S, M, or L
        self.distortion = np.zeros(self.size, dtype=object)   # Low or High

    def get_size_of_dataset(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):
                    self.size += 1

    def make_labels(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):
                    if is_low_distortion_file(file_name, rel_path):
                        train += 1
                        print("training image #", train)

                    elif is_icon(file):
                        pass
                        # copy_to_training_dir(rel_path, abs_path)

                    elif is_med_or_high_distortion_file(file_name, rel_path):
                        test += 1
                        print("testing image #", test)
                        copy_to_testing_dir(rel_path, abs_path, file.title(), test)

                    else:
                        print("Unable to determine what {x} is".format(x=file))

    def get_tf_dataset(self, shuffle=True):
        """
        Utility method, simply calls the image_dataset_from_directory method from tensorflow.
        """
        if self.batch_size is None:
            raise ValueError("Must set batch.size before getting the Tensorflow dataset")
        return image_dataset_from_directory(self.path, shuffle=shuffle, batch_size=self.batch_size,
                                            image_size=(self.height, self.width))
