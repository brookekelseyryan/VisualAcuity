from preprocessing import constants as const

low_distortion_filenames = const.low_distortion_filenames

high_distortion_filenames = const.high_distortion_filenames

axes = const.axes

imbalanced_classes = []

# image sizes
TRAINING_IMAGE_SIZES = const.TRAINING_IMAGE_SIZES
TESTING_IMAGE_SIZES = const.TESTING_IMAGE_SIZES

# default height and width
H = const.H
W = const.W

# PROJECT ROOT
local_project_root = const.local_project_root
remote_project_root = const.remote_project_root

# DATA ROOT
local_data_root = "/Users/brookeryan/Developer/BaldiLab/Visual-Acuity/Teller/"

# OPTIONS
DATA_ROOT = local_data_root
PROJECT_ROOT = local_project_root

IMAGES_PATH = PROJECT_ROOT + "tellerImages/"
TRAINING_PATH = IMAGES_PATH + "training/"
TESTING_PATH = IMAGES_PATH + "testing/"

angles_str = ["0", "22.5", "45", "67.5", "90", "112.5", "135", "157.5"]
