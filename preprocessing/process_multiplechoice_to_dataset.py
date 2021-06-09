import os
import shutil

import PIL
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import save_img

from constants import *
from files import extract_file_name, extract_size, is_icon, extract_optotype
from images import augment
from utils import make_dir


def is_training_image(path, title):
    """
    Large images with low distortion levels are training images.
    """
    size = extract_size(path)
    filename = extract_file_name(title)

    if size in TESTING_IMAGE_SIZES:
        return False
    elif size in TRAINING_IMAGE_SIZES and filename in low_distortion_filenames:
        return True
    elif size in TRAINING_IMAGE_SIZES and filename in high_distortion_filenames:
        return False
    else:
        raise Exception("Image cannot be classified as testing or training image:", path, "filename:", filename,
                        "size:", size)


def is_testing_image(path, title):
    """
    Small or Medium image, or a Large image with high distortion level.
    """
    size = extract_size(path)
    filename = extract_file_name(title)

    if size in TESTING_IMAGE_SIZES:
        return True
    elif size in TRAINING_IMAGE_SIZES and filename in low_distortion_filenames:
        return False
    elif size in TRAINING_IMAGE_SIZES and filename in high_distortion_filenames:
        return True
    else:
        raise Exception("Image cannot be classified as testing or training image:", path, "filename:", filename,
                        "size:", size)


def copy_to_dir(abs_path, file_title, training, file_format='.png', data_augmentation=True):
    if training:
        path = TRAINING_PATH
    else:
        path = TESTING_PATH

    image = cv2.cvtColor(cv2.resize(cv2.imread(abs_path), (H, W)), cv2.COLOR_BGR2RGB)
    image_array = np.asarray(image, dtype=np.float64) / 255

    optotype = extract_optotype(abs_path)
    size = extract_size(abs_path)

    dest_dir = os.path.join(path, optotype + "/")
    make_dir(dest_dir)

    file_name = extract_file_name(file_title) + "_" + size
    images = {file_name: image_array}

    # Data Augmentation
    if data_augmentation:
        if training and optotype in imbalanced_classes:
            images.update(augment(PIL.Image.open(abs_path), file_name))

    for title, img in images.items():
        # Saves an image stored as a Numpy array to a path or file object.
        path = os.path.join(dest_dir, title + file_format)
        save_img(path, img)
        print("Saved", path)


def clear_previous_dirs():
    print("#################################")
    print("Clearing previous directories...")
    print("#################################\n")
    shutil.rmtree(IMAGES_PATH)

    for path in [IMAGES_PATH, TRAINING_PATH, TESTING_PATH]:
        make_dir(path)

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


def log_size_of_dirs():
    print("#################################")
    print("Size of directories")
    print("#################################\n")
    train_size = sum([len(files) for r, d, files in os.walk(TRAINING_PATH)])
    test_size = sum([len(files) for r, d, files in os.walk(TESTING_PATH)])
    print("Training size = ", train_size)
    print("Testing size = ", test_size)


def process():
    """
    Processes images from MultipleChoice file to images/ where they will be used as a dataset.
    """
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(".png") or file.endswith(".tif"):
                abs_path = os.path.join(root, file)

                if is_icon(file):
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
    print("Data root: ", DATA_ROOT)
    print("Project root: ", PROJECT_ROOT, "\n")

    clear_previous_dirs()

    print("#################################")
    print("Beginning data processing...")
    print("#################################\n")

    process()

    print("Data processing complete.")

    log_size_of_dirs()
