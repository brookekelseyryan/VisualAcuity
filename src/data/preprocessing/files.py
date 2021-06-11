import os
import numpy as np
from data.preprocessing import constants as const


def is_low_distortion_file(file_title):
    """
    Must be from a L folder and also be in the low distortion filenames
    :param file_name: File title
    :param path: Used to get the size of the image from the parent directory, S, M, L
    :return: Boolean, True if it is a low distortion file
    """
    file_name = extract_file_name(file_title).split("_")[0]
    return file_name in const.low_distortion_filenames


def is_high_distortion_file(file_title):
    """
    Must be from a M or S folder, or L folder that isn't already a low distortion filename
    :param file_name: File title
    :param path: Used to get the size of the image from the parent directory, S, M, L
    :return: Boolean, True if it is a medium or high distortion file
    """
    file_name = extract_file_name(file_title).split("_")[0]
    return file_name in const.high_distortion_filenames


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


def extract_acuity(path):
    """
    Extracts the acuity from the path of the image
    :param path: Path of the image, must contain up to at least the great-grandparent
    :return: Acuity: ETDRS, HOTV, SSa, etc.
    """
    acuity = path.split("/")[-4]
    assert acuity in const.acuities, "Unknown acuity {a}".format(a=acuity)
    return acuity


def extract_axis(file_title):
    """
    Extracts the axis from the file name.
    :param file_name: Name of file with extension.
    :return: Angle of rotation, will be either 0, 45, 90 or 135.
    """
    file_name = extract_file_name(file_title)
    axis = np.int(file_name.split("-")[-1])
    return axis


def extract_size(path):
    """
    Extracts the size of the optotype from the path of the image.
    :param path:  Path of the image, must contain up to at least the parent.
    :return: S, M, or L
    """
    size = path.split("/")[-2]
    # Sanity check
    assert size in const.TRAINING_IMAGE_SIZES or size in const.TESTING_IMAGE_SIZES

    return size


def extract_distortion_level(path):
    filename = extract_file_name(path)

    if filename in const.low_distortion_filenames:
        return "low"
    elif filename in const.high_distortion_filenames:
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
