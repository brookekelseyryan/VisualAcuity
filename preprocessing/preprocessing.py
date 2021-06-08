import os
from pathlib import Path
from os.path import dirname, abspath
from shutil import copy2
import numpy as np
from PIL import Image
import cv2

from tensorflow.keras.preprocessing import image

low_distortion_filenames = ["0-0-0",
                            "0-1-0",
                            "0-1-45",
                            "0-1-90",
                            "0-1-135",
                            "0-3-0",
                            "0-3-45",
                            "0-3-90",
                            "0-3-135",
                            "0-6-0",
                            "0-6-45",
                            "0-6-90",
                            "0-6-135",
                            "2-0-0",
                            "2-1-0",
                            "2-1-45",
                            "2-1-90",
                            "2-1-135",
                            "2-3-0",
                            "2-3-45",
                            "2-3-90",
                            "2-3-135",
                            "4-0-0",
                            "4-1-0",
                            "4-1-45",
                            "4-1-90",
                            "4-1-135"]

local_root = "/Users/brookeryan/Developer/Baldi Lab/Visual-Acuity/"
remote_root = "/baldig/bioprojects2/VisualAcuity/"

training_path = "/Users/brookeryan/Developer/Baldi Lab/Visual-Acuity/data/training/"
testing_path = "/Users/brookeryan/Developer/Baldi Lab/Visual-Acuity/data/testing/"
data_root = "/Users/brookeryan/Developer/Baldi Lab/Visual-Acuity/MultipleChoiceTest2/"

x_train = np.zeros((2862, 400, 400, 3))
x_test = np.zeros((13876, 400, 400, 3))

y_train = np.zeros(2862, dtype=str)
y_test = np.zeros(13876, dtype=str)


def make_directories(dest_path):
    if not os.path.exists(dest_path):
        os.chmod(dest_path, 0o666)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)


# Must be from a L folder and also be in the low distortion filenames
def is_low_distortion_file(file_name, path):
    size = path.split("/")[-2]
    return file_name in low_distortion_filenames and "L" in size


def is_med_or_high_distortion_file(file_name, path):
    size = path.split("/")[-2]
    if "M" in size or "S" in size:
        return True
    elif file_name not in low_distortion_filenames:
        return True
    else:
        return False


def is_icon(file):
    return file.count("-") < 2 and file.endswith(".png")


def extract_label(path):
    # we need the third element of the dashes
    # angle = file_name.split("-")[-1]

    # everythingelse/label/size/filename
    return path.split("/")[-3]


def copy_to_testing_dir(rel_path, abs_path, index):
    dest_path = os.path.join(testing_path, rel_path)
    make_directories(dest_path)
    copy2(abs_path, dest_path)

    print(rel_path)

    x_test[index] = cv2.resize(cv2.imread(abs_path), (400, 400))

    y_test[index] = extract_label(rel_path)


def copy_to_training_dir(rel_path, abs_path, index):
    dest_path = os.path.join(training_path, rel_path)
    make_directories(dest_path)
    copy2(abs_path, dest_path)

    x_train[index] = np.asarray(Image.open(abs_path))
    y_train[index] = extract_label(rel_path)


def sort():
    num = 0
    train = 0
    test = 0
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".png") or file.endswith(".tif"):

                file_name = file.title().split(".")[0]
                abs_path = os.path.join(root, file)
                rel_path = abs_path.replace(data_root, "")

                if is_low_distortion_file(file_name, rel_path):
                    train += 1
                    copy_to_training_dir(rel_path, abs_path, train)

                elif is_icon(file):
                    pass
                    copy_to_training_dir(rel_path, abs_path, train)

                elif is_med_or_high_distortion_file(file_name, rel_path):
                    test += 1
                    copy_to_testing_dir(rel_path, abs_path, test)

                else:
                    print("Unable to determine what {x} is".format(x=file))
    print(num)
    print("train = {x}".format(x=train))
    print("test={x}".format(x=test))


# Main
if __name__ == '__main__':
    sort()
