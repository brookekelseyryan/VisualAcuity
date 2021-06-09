import os
import shutil

from constants import IMAGES_PATH, TRAINING_PATH, TESTING_PATH


def make_dir(dest_path):
    """
    Ensures that the directories for placing the training/test images have appropriate permissions
    :param dest_path: directory
    """
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # os.chmod(dest_path, 0o666)
