import os
import shutil
from pathlib import Path
from os.path import dirname, abspath
from shutil import copy2
import numpy as np
import PIL
from PIL import ImageEnhance
from PIL import Image, ImageOps
import cv2
import logging
import tensorflow as tf
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

imbalanced_classes = ["+blank", "+circle", "+diamond", "+square", "2", "3", "5", "6", "8", "9", "apple", "bird", "cake",
                      "car", "circle", "cow", "cup", "D",
                      "duck", "F", "flat-line", "flat-square",
                      "frown-line", "frown-square", "H", "hand", "horse", "house", "K", "L", "N", "O", "P",
                      "panda", "phone", "R", "S", "smile-line", "smile-square", "square", "star", "T", "train",
                      "tree", "V", "x-blank", "x-circle", "x-diamond", "x-square", "Z"]

# PROJECT ROOT
local_project_root = "/Users/brookeryan/Developer/BaldiLab/Visual-Acuity/"
remote_project_root = "/home/brooker/VisualAcuity/"

# DATA ROOT
remote_data_root = "/baldig/bioprojects2/VisualAcuity/"
local_data_root = "/Users/brookeryan/Developer/BaldiLab/Visual-Acuity/MultipleChoiceTest2/"

# OPTIONS
data_root = local_data_root
project_root = local_project_root

IMAGES_PATH = project_root + "images/"
TRAINING_PATH = project_root + "images/training/"
TESTING_PATH = project_root + "images/testing/"

training_size = 2431
testing_size = 11610
h = w = 400

x_train = np.zeros((training_size, h, w, 3), dtype=np.float64)
x_test = np.zeros((testing_size, h, w, 3), dtype=np.float64)

y_train_optotype = np.zeros(training_size, dtype=object)
y_test_optotype = np.zeros(testing_size, dtype=object)

y_train_angle = np.zeros(training_size, dtype=np.int)
y_test_angle = np.zeros(testing_size, dtype=np.int)


def make_directories(dest_path):
    """
    Ensures that the directories for placing the training/test images have appropriate permissions
    :param dest_path: directory
    """
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # os.chmod(dest_path, 0o666)


def is_low_distortion_file(file_name, path):
    """
    Must be from a L folder and also be in the low distortion filenames
    :param file_name: File name without extension
    :param path: Used to get the size of the image from the parent directory, S, M, L
    :return: Boolean, True if it is a low distortion file
    """
    size = path.split("/")[-2]
    return file_name in low_distortion_filenames and "L" in size


def is_med_or_high_distortion_file(file_name, path):
    """
    Must be from a M or S folder, or L folder that isn't already a low distortion filename
    :param file_name: File name without extension
    :param path: Used to get the size of the image from the parent directory, S, M, L
    :return: Boolean, True if it is a medium or high distortion file
    """
    size = path.split("/")[-2]
    if "M" in size or "S" in size:
        return True
    elif file_name not in low_distortion_filenames:
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


def extract_file_name(file):
    file_title = os.path.splitext(file)[0]

    # accounts for some images which were named x-y-z-image000 out of the standard x-y-z
    if "image" in file_title.lower():
        return file_title.split("-")[0] + "-" + file_title.split("-")[1] + "-" + file_title.split("-")[2]
    if is_icon(file_title):
        return file_title
    else:
        return file_title


def copy_to_testing_dir(rel_path, abs_path, file_title, index, make_dataset=True):
    """
    Copies given image to testing directory
    :param make_dataset: for preprocessing Keras-style
    :param rel_path: Relative path (up to the level above the optotype) to be copied to the new directory
    :param abs_path: Absolute path of the image in the initial data directory
    :param index: Integer value indicating the i_th testing image encountered
    """
    image = cv2.cvtColor(cv2.resize(cv2.imread(abs_path), (h, w)), cv2.COLOR_BGR2RGB)

    x_test[index] = np.asarray(image, dtype=np.float64) / 255
    y_test_optotype[index] = extract_optotype(rel_path)
    y_test_angle[index] = extract_angle(file_title)

    dest_path = os.path.join(TESTING_PATH, rel_path)

    # for preprocessing Keras-style
    if make_dataset:
        dest_path = os.path.join(TESTING_PATH, y_test_optotype[index] + "/" + extract_file_name(file_title) + ".png")

    make_directories(dest_path)
    copy2(abs_path, dest_path)


def augment(image, index, file_title_prefix):
    """
    Augments a given image.
    :param image: Original image
    :param index: Index of optotype
    :param file_title_prefix: The name of the original file, e.g. "0-0-0.png"
    """
    augmented_images = {"gray": PIL.ImageOps.grayscale(image),
                        "120_bright": PIL.ImageEnhance.Brightness(image).enhance(1.2),
                        "120_contrast": PIL.ImageEnhance.Contrast(image).enhance(1.2),
                        "80_bright": PIL.ImageEnhance.Brightness(image).enhance(0.8),
                        "80_contrast": PIL.ImageEnhance.Contrast(image).enhance(0.8)}

    for title, img in augmented_images.items():
        # Converts a PIL Image instance to a Numpy array. Needed for processing later on.
        array = tf.keras.preprocessing.image.img_to_array(img)

        # Saves an image stored as a Numpy array to a path or file object.
        tf.keras.preprocessing.image.save_img(
            os.path.join(TRAINING_PATH,
                         y_train_optotype[index] + "/" + extract_file_name(file_title_prefix) + "_" + title + ".png"),
            array,
            file_format='png')


def copy_to_training_dir(rel_path, abs_path, file_title, index, make_dataset=True):
    """
    Copies given image to training directory
    :param make_dataset: for preprocessing Keras-style
    :param rel_path: Relative path (up to the level above the optotype) to be copied to the new directory
    :param abs_path: Absolute path of the image in the initial data directory
    :param index: Integer value indicating the i_th training image encountered
    """
    image = cv2.cvtColor(cv2.resize(cv2.imread(abs_path), (h, w)), cv2.COLOR_BGR2RGB)
    image_array = np.asarray(image, dtype=np.float64) / 255

    x_train[index] = image_array
    y_train_optotype[index] = extract_optotype(rel_path)
    y_train_angle[index] = extract_angle(file_title)

    dest_path = os.path.join(TRAINING_PATH, rel_path)

    # for preprocessing Keras-style
    if make_dataset:
        dest_path = os.path.join(TRAINING_PATH, y_train_optotype[index] + "/" + extract_file_name(file_title) + ".png")

    make_directories(dest_path)

    # Saves an image stored as a Numpy array to a path or file object.
    tf.keras.preprocessing.image.save_img(dest_path, image_array, file_format='png')

    # Data Augmentation
    if y_train_optotype[index] in imbalanced_classes:
        augment(PIL.Image.open(abs_path), index, file_title)


def clear_previous_dirs():
    print("#################################")
    print("Clearing previous directories...")
    print("#################################\n")
    shutil.rmtree(IMAGES_PATH)

    for path in [IMAGES_PATH, TRAINING_PATH, TESTING_PATH]:
        make_directories(path)

    print("#################################")
    print("Assert directories exist...")
    print("#################################\n")
    for path in [IMAGES_PATH, TRAINING_PATH, TESTING_PATH]:
        assert os.path.exists(path)
        print(path, "exists.")

    assert len(os.listdir(IMAGES_PATH)) == 2
    assert len(os.listdir(TRAINING_PATH)) == 0
    assert len(os.listdir(TESTING_PATH)) == 0
    print("\nDirectories empty.")


def size_of_dirs():
    print("#################################")
    print("Size of directories")
    print("#################################\n")
    train_size = sum([len(files) for r, d, files in os.walk(TRAINING_PATH)])
    test_size = sum([len(files) for r, d, files in os.walk(TESTING_PATH)])
    print("Training size = ", train_size)
    print("Testing size = ", test_size)


def sort():
    """
    Sorts the images from the original data folder to their appropriate testing/training directories
    :return:
    """

    clear_previous_dirs()

    train = 0
    test = 0

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    print("#################################")
    print("Beginning data preprocessing...")
    print("#################################\n")

    print("Current directory: ", os.getcwd())
    print("Data root: ", data_root)
    print("Project root: ", project_root, "\n")

    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".png") or file.endswith(".tif"):
                abs_path = os.path.join(root, file)
                rel_path = (abs_path.replace(data_root, ""))
                # removing this from rel_path since doesn't work .replace(file.title(), "")
                file_name = extract_file_name(file.title())

                if is_low_distortion_file(file_name, rel_path):
                    train += 1
                    print("training image #", train)
                    copy_to_training_dir(rel_path, abs_path, file.title(), train)

                elif is_icon(file):
                    pass
                    # copy_to_training_dir(rel_path, abs_path)

                elif is_med_or_high_distortion_file(file_name, rel_path):
                    test += 1
                    print("testing image #", test)
                    copy_to_testing_dir(rel_path, abs_path, file.title(), test)

                else:
                    print("Unable to determine what {x} is".format(x=file))

    print("Data processing complete.")

    size_of_dirs()


########
# Main #
########
if __name__ == '__main__':
    sort()
