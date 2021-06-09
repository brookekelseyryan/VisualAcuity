import os
import numpy as np

from constants import low_distortion_filenames, high_distortion_filenames, TRAINING_IMAGE_SIZES, TESTING_IMAGE_SIZES


def is_low_distortion_file(file_name, path):
    """
    Must be from a L folder and also be in the low distortion filenames
    :param file_name: File name without extension
    :param path: Used to get the size of the image from the parent directory, S, M, L
    :return: Boolean, True if it is a low distortion file
    """
    size = path.split("/")[-2]
    return file_name in low_distortion_filenames and "L" in size


def is_high_distortion_file(file_name, path):
    """
    Must be from a M or S folder, or L folder that isn't already a low distortion filename
    :param file_name: File name without extension
    :param path: Used to get the size of the image from the parent directory, S, M, L
    :return: Boolean, True if it is a medium or high distortion file
    """
    size = extract_size(path)
    if "M" in size or "S" in size:
        return True
    elif file_name in high_distortion_filenames:
        return True
    else:
        return False


def is_icon(file):
    """
    Determines if the given file is an icon (instead of a training/test image).
    :param file: Name of the file with extension
    :return: Boolean
    """
    return file.count("-") < 2 and file.endswith(".png")


def extract_optotype(path):
    """
    Extracts the optotype from the path of the image.
    :param path: Path of the image, must contain up to at least the grandparent.
    :return: Optotype label: C, D, H, cake, etc.
    """
    optotype = path.split("/")[-3]
    return optotype


def extract_angle(file_title):
    """
    Extracts the angle of rotation from the file name.
    :param file_name: Name of file with extension.
    :return: Angle of rotation, will be either 0, 45, 90 or 135.
    """
    file_name = extract_file_name(file_title)
    angle = np.int(file_name.split("-")[-1])
    return angle


def extract_size(path):
    """
    Extracts the size of the optotype from the path of the image.
    :param path:  Path of the image, must contain up to at least the parent.
    :return: S, M, or L
    """
    size = path.split("/")[-2]
    # Sanity check
    assert size in TRAINING_IMAGE_SIZES or size in TESTING_IMAGE_SIZES

    return size


def extract_distortion_level(path):
    filename = extract_file_name(path)

    if filename in low_distortion_filenames:
        return "low"
    elif filename in high_distortion_filenames:
        return "high"
    else:
        raise Exception("Image cannot be classified as high or low distortion:", path)


def extract_file_name(file):
    file_title = os.path.splitext(file)[0]

    # accounts for some images which were named x-y-z-image000 out of the standard x-y-z
    if "image" in file_title.lower():
        return file_title.split("-")[0] + "-" + file_title.split("-")[1] + "-" + file_title.split("-")[2]
    if is_icon(file_title):
        return file_title
    else:
        return file_title
