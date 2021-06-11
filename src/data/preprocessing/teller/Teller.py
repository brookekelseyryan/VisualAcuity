import os
import shutil

import PIL
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import save_img

from data.preprocessing.teller import constants as const
from data.preprocessing.teller import files as f
from data.preprocessing import images as imgs
from data.preprocessing import utils

def is_training_image(path, title):
    """
    Large images with low distortion levels are training images.
    """
    filename = f.extract_file_name(title)

    if filename in const.low_distortion_filenames:
        return True
    elif filename in const.high_distortion_filenames:
        return False
    else:
        raise Exception("Image cannot be classified as testing or training image:", path, "filename:", filename)


def is_testing_image(path, title):
    """
    Small or Medium image, or a Large image with high distortion level.
    """
    filename = f.extract_file_name(title)

    if filename in const.low_distortion_filenames:
        return False
    elif filename in const.high_distortion_filenames:
        return True
    else:
        raise Exception("Image cannot be classified as testing or training image:", path, "filename:", filename)

def copy_to_dir(abs_path, file_title, training, file_format='.png', data_augmentation=False):
    if training:
        path = const.TRAINING_PATH
    else:
        path = const.TESTING_PATH

    image = cv2.cvtColor(cv2.resize(cv2.imread(abs_path), (const.H, const.W)), cv2.COLOR_BGR2RGB)
    image_array = np.asarray(image, dtype=np.float64) / 255

    optotype = f.extract_optotype(abs_path)
    acuity = f.extract_acuity()
    size = f.extract_size(abs_path)
    angle = f.extract_angle(abs_path)

    dest_dir = os.path.join(path, acuity + "_" + f.make_file_name(optotype, angle) + "/")
    utils.make_dir(dest_dir)

    file_name = f.extract_file_name(file_title) + "_" + size
    images = {file_name: image_array}

    # Data Augmentation
    if data_augmentation:
        if training and optotype in const.imbalanced_classes:
            images.update(imgs.augment(PIL.Image.open(abs_path), file_name))

    for title, img in images.items():
        # Saves an image stored as a Numpy array to a path or file object.
        path = os.path.join(dest_dir, title + file_format)
        save_img(path, img)
        print("Saved", path)


def check_previous_dirs():
    print("#################################")
    print("Clearing previous directories...")
    print("#################################\n")
    shutil.rmtree(const.IMAGES_PATH)

    for path in [const.IMAGES_PATH, const.TRAINING_PATH, const.TESTING_PATH]:
        utils.make_dir(path)

    print("#################################")
    print("Assert directories exist...")
    print("#################################\n")
    for path in [const.IMAGES_PATH, const.TRAINING_PATH, const.TESTING_PATH]:
        assert os.path.exists(path)
        print(path, "exists.")

    assert len(os.listdir(const.IMAGES_PATH)) == 2
    assert len(os.listdir(const.TRAINING_PATH)) == 0
    assert len(os.listdir(const.TESTING_PATH)) == 0
    print("\nDirectories empty.")


def log_size_of_dirs():
    print("#################################")
    print("Size of directories")
    print("#################################\n")
    train_size = sum([len(files) for r, d, files in os.walk(const.TRAINING_PATH)])
    test_size = sum([len(files) for r, d, files in os.walk(const.TESTING_PATH)])
    print("Training size = ", train_size)
    print("Testing size = ", test_size)


def process():
    """
    Processes images from Teller dir to images/ where they will be used as a data.
    """
    for root, dirs, files in os.walk(const.DATA_ROOT):
        for file in files:
            if file.endswith(".png") or file.endswith(".tif"):
                abs_path = os.path.join(root, file)

                if f.is_icon(abs_path):
                    # Right now, just don't do anything for icons
                    pass

                elif is_training_image(abs_path, file.title()):
                    copy_to_dir(abs_path, file.title(), training=True)

                elif is_testing_image(abs_path, file.title()):
                    copy_to_dir(abs_path, file.title(), training=False)

                else:
                    print("Unable to determine what {x} is".format(x=file))


########
# Main #
########
if __name__ == '__main__':
    print("Current directory: ", os.getcwd())
    print("Data root: ", const.DATA_ROOT)
    print("Project root: ", const.PROJECT_ROOT, "\n")

    check_previous_dirs()

    print("####################################")
    print("Beginning Teller data processing...")
    print("####################################\n")

    process()

    print("Data processing complete.")

    log_size_of_dirs()
