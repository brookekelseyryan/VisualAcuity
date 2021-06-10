import os

import numpy as np
from tensorflow.keras import preprocessing as p

from preprocessing import files as f
from preprocessing import constants as const


class Dataset:
    def __init__(self, path, name, height, width, batch_size, include_augmented=False):
        # Path to where this multiplechoice_processed_data lives. This should be in a pre-processed form of images/testing/, etc.
        self.path = path

        self.name = name  # Training or Testing
        self.__include_augmented = include_augmented  # Only relevant for Training.

        self.height = height
        self.width = width

        self.batch_size = batch_size

        self.num_total_images = 0
        self.num_augmented_images = 0
        self.__get_size_of_dataset()

        self.images = np.zeros((self.num_total_images, self.height, self.width, 3), dtype=np.float64)

        self.acuities = np.zeros(shape=(self.num_total_images,), dtype=object)  # SSa, SSl, HOTV, etc.
        self.angles = np.zeros(shape=(self.num_total_images,), dtype=float)  # this is for optotypes like C-0
        self.augmented = np.zeros(shape=(self.num_total_images,), dtype=object)  # Name or 'None'
        self.character = np.zeros(shape=(self.num_total_images,),
                                  dtype=object)  # alpha, num, dingbat, teller (used for determining literacy stuff)
        self.distortions = np.zeros(shape=(self.num_total_images,), dtype=object)  # low or high
        self.optotypes = np.zeros(shape=(self.num_total_images,), dtype=object)  # cake, C, D, duck, etc.
        self.sizes = np.zeros(shape=(self.num_total_images,), dtype=object)  # S, M, or L

        self.__process_labels()

    def __get_size_of_dataset(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):

                    if is_augmented_image(file.title()):
                        if self.__include_augmented:
                            self.num_total_images += 1
                            self.num_augmented_images += 1
                    else:
                        self.num_total_images += 1

        print(self.name, "Dataset contains", self.num_total_images, "total images.")
        print(self.name, "Dataset contains", self.num_augmented_images, "augmented images.")

    def __process_labels(self):

        print("####################")
        print("Processing labels...")
        print("####################")

        if self.__include_augmented:
            print("\nincluding augmented images...")

        i = 0

        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):
                    abs_path = os.path.join(root, file)

                    self.images[i] = p.image.img_to_array(p.image.load_img(abs_path))

                    acuity, optotype, angle, character = extract_acuity_optotype_angle_character(abs_path)
                    self.acuities[i] = acuity
                    self.angles[i] = angle
                    self.optotypes[i] = optotype
                    self.character[i] = character

                    distortion, size, augmentation = extract_distortion_size_augmentation(file.title())

                    self.augmented[i] = augmentation
                    self.distortions[i] = distortion
                    self.sizes[i] = size

                    i += 1

        print("Done processing labels.")

    def get_tf_dataset(self, shuffle=True):
        """
        Utility method, simply calls the image_dataset_from_directory method from tensorflow.
        """
        return p.image_dataset_from_directory(self.path, shuffle=shuffle, batch_size=self.batch_size,
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


def extract_acuity_optotype_angle_character(path):
    """
    Extracts the directory acuityNAME_optotypeNAME
    :param path: Path of the image, must contain up to at least the grandparent.
    :return: <acuity_rawoptotype>
    """
    acuity_raw_optotype = path.split("/")[-2]
    acuity, optotype = separate_acuity_from_optotype(acuity_raw_optotype)

    if is_angle_in_optotype(optotype):
        angle = float(strip_optotype_from_angle(optotype))
        optotype = strip_angle_from_optotype(optotype)
    else:
        angle = 0

    character = extract_character(acuity, optotype)

    return acuity, optotype, angle, character


def extract_distortion_size_augmentation(file_title):
    distortion = f.extract_distortion_level(file_title.split("_")[0])
    size = extract_size(file_title)
    augmentation = extract_augmentation(file_title)

    return distortion, size, augmentation


def extract_optotype(path):
    """
    Extracts the optotype from the path of the image.
    :param path: Path of the image, must contain up to at least the grandparent.
    :return: Optotype label: C, D, H, cake, etc.
    """
    optotype = path.split("/")[-2]
    return strip_angle_from_optotype(optotype)


def is_angle_in_optotype(optotype):
    return any(angle in optotype for angle in const.angles_str)


def strip_angle_from_optotype(optotype):
    if is_angle_in_optotype(optotype):
        return optotype.split("-")[0]


def strip_optotype_from_angle(optotype):
    if is_angle_in_optotype(optotype):
        return optotype.split("-")[1]


def separate_acuity_from_optotype(acuity_optotype):
    assert len(acuity_optotype.split("_")) == 2

    acuity, optotype = acuity_optotype.split("_")[0], acuity_optotype.split("_")[1]
    return acuity, optotype


def is_augmented_image(file_title):
    """
    An augmented image has more than 1 underscore in the name
    Ex: augmented image title - 0-0-0_L_80_bright.png
    non-augmented image title - 0-0-0_L.png
    :param path:
    :return:
    """
    if file_title.count("_") == 1:
        return False
    elif file_title.count("_") >= 2:
        return True
    else:
        raise Exception("Could not determine if image was augmented", file_title)


def extract_augmentation(file_title):
    if is_augmented_image(file_title):
        return file_title.split("_")[-1]
    else:
        return "None"


def extract_character(acuity, optotype):
    if acuity == "Teller":
        return "teller"
    elif optotype in ["2", "3", "5", "6", "8", "9"]:
        return "numeric"
    elif optotype in ['C', 'D', 'E', 'F', 'H', 'K', 'L', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'Z']:
        return "alpha"
    elif optotype in ['apple', 'bird', 'cake', 'car', 'circle', 'cow', 'cup', 'duck', 'flat-line', 'flat-square',
                      'frown-line', 'frown-square', 'hand', 'horse', 'house', 'panda', 'phone', 'smile-line',
                      'smile-square', 'square', 'star', 'train', 'tree', 'x-blank', 'x-circle', 'x-diamond', 'x-square',
                      '+blank', '+circle', '+diamond', '+square']:
        return "wingding"
    else:
        raise Exception("Could not determine character type of", acuity, optotype)
