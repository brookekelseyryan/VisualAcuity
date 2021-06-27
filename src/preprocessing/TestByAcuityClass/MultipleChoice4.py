import os
import shutil

import PIL
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import save_img

from preprocessing.TestByAcuityClass import constants as const
import images as imgs
from preprocessing import utils

def clear_previous_dirs():
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


########
# Main #
########
if __name__ == '__main__':
    print("Current directory: ", os.getcwd())
    print("Data root: ", const.DATA_ROOT)
    print("Project root: ", const.PROJECT_ROOT, "\n")

    clear_previous_dirs()