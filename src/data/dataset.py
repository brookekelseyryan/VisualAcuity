import os

import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from preprocessing.files import is_low_distortion_file, is_icon, is_high_distortion_file, extract_angle


class Dataset:
    def __init__(self, path, name, height, width, batch_size=None):
        # Path to where this multiplechoice_processed_data lives. This should be in a pre-processed form of images/testing/, etc.
        self.path = path

        # Training or Testing
        self.name = name

        self.height = height
        self.width = width

        self.batch_size = batch_size

        self.size = 0
        self.get_size_of_dataset()

        self.distortions = np.zeros(shape=(self.size,), dtype=object)  # Low or High
        self.sizes = np.zeros(shape=(self.size,), dtype=object)  # S, M, or L
        self.angles = np.zeros(shape=(self.size,), dtype=np.int)  # 0, 45, 90, or 135
        self.optotypes = np.zeros(shape=(self.size,), dtype=object)  # cake, C, D, duck, etc.
        self.images = np.zeros((self.size, self.height, self.width, 3), dtype=np.float64)

    def get_size_of_dataset(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):
                    self.size += 1
        print(self.name, "Dataset contains", self.size, "images")

    def process_labels(self):

        print("####################")
        print("Processing labels...")
        print("####################")

        i = 0
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):
                    abs_path = os.path.join(root, file)
                    file_title = file.title()
                    self.images[i] = img_to_array(load_img(abs_path))
                    self.optotypes[i] = extract_optotype(abs_path)
                    self.angles[i] = extract_angle(file_title.split("_")[0])
                    self.sizes[i] = extract_size(file_title)
                    self.distortions[i] = extract_distortion(file_title)
                    i += 1

    def get_tf_dataset(self, shuffle=True):
        """
        Utility method, simply calls the image_dataset_from_directory method from tensorflow.
        """
        return image_dataset_from_directory(self.path, shuffle=shuffle, batch_size=self.batch_size,
                                            image_size=(self.height, self.width))


def extract_size(title):
    if "S" in title:
        return "S"
    elif "M" in title:
        return "M"
    elif "L" in title:
        return "L"
    else:
        raise Exception("Could not extract size from", title)


def extract_distortion(file_title):
    if is_low_distortion_file(file_title):
        return "low"
    elif is_high_distortion_file(file_title):
        return "high"
    else:
        raise Exception("Could not extract distortion from", file_title)

def extract_optotype(path):
    """
    Extracts the optotype from the path of the image.
    :param path: Path of the image, must contain up to at least the grandparent.
    :return: Optotype label: C, D, H, cake, etc.
    """
    optotype = path.split("/")[-2]
    return optotype