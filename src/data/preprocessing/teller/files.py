import os
import numpy as np
from preprocessing.teller import constants as const
from preprocessing import files as f


def is_low_distortion_file(file_title):
    f.is_low_distortion_file(file_title)


def is_high_distortion_file(file_title):
    f.is_high_distortion_file(file_title)


def is_icon(path):
    parent_file = path.split("/")[-2]
    if "Teller" in parent_file:
        return True
    else:
        return False


def extract_angle(path):
    parent_file = path.split("/")[-2]

    a = parent_file.split("-")[-1]

    if any(angle in a for angle in const.angles_str):
        return a
    elif "gray" in a:
        return None
    else:
        raise Exception("Angle", a, "not found in const.angles_str, orig path is", path)


def make_file_name(optotype, angle):
    if angle is None:
        return optotype
    else:
        return optotype + "-" + angle


def extract_optotype(path):
    parent_file = path.split("/")[-2]

    if "grad" in parent_file:
        shape = "gradientCircle"
    elif "reg" in parent_file:
        shape = "regularCircle"
    elif "gray" in parent_file:
        return "grayCircle"
    else:
        raise Exception("Could not determine optotype from", parent_file, "\n Original path is", path)

    if "L" in parent_file:
        size = "L"
    elif "M" in parent_file:
        size = "M"
    elif "S" in parent_file:
        size = "S"
    else:
        raise Exception("Could not determine size from", parent_file, "\n Original path is", path)

    return size + shape


def extract_acuity():
    return "Teller"


def extract_axis(file_title):
    return f.extract_axis(file_title)


def extract_size(path):
    """
    Teller sizes work a bit different, they seem to be small so that's what I will use for now.
    """
    return "S"


def extract_distortion_level(path):
    return f.extract_distortion_level(path)


def extract_file_name(file):
    return f.extract_file_name(file)
