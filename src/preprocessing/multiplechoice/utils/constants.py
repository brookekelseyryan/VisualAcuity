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

high_distortion_filenames = ["2-6-0",
                             "2-6-45",
                             "2-6-90",
                             "2-6-135",
                             "4-3-0",
                             "4-3-45",
                             "4-3-90",
                             "4-3-135",
                             "4-6-0",
                             "4-6-45",
                             "4-6-90",
                             "4-6-135",
                             "6-0-0",
                             "6-1-0",
                             "6-1-45",
                             "6-1-90",
                             "6-1-135",
                             "6-3-0",
                             "6-3-45",
                             "6-3-90",
                             "6-3-135",
                             "6-6-0",
                             "6-6-45",
                             "6-6-90",
                             "6-6-135"]

acuity_to_optotypes = {
    'A': ['cake', 'car', 'duck', 'hand', 'horse', 'house', 'panda', 'phone', 'tree'],
    'C': ['C-0', 'C-135', 'C-180', 'C-225', 'C-270', 'C-315', 'C-45', 'C-90'],
    'E': ['E-0', 'E-180', 'E-270', 'E-90'],
    'ETDRS': ['C', 'D', 'H', 'K', 'N', 'O', 'R', 'S', 'V', 'Z'],
    'ETL-face': ['flat-line', 'flat-square', 'frown-line', 'frown-square', 'smile-line', 'smile-square'],
    'ETL-x': ['+blank', '+circle', '+diamond', '+square', 'x-blank', 'x-circle', 'x-diamond', 'x-square'],
    'HOTV': ['H', 'O', 'T', 'V'],
    'L': ['apple', 'circle', 'house', 'square'],
    'NL': ['5', '6', '8', '9'], # TODO 5,6,9
    'NPV': ['2', '3', '5', '6', '9'],
    'P': ['apple', 'circle', 'house', 'square', 'star'], # TODO L
    'SSa': ['C', 'D', 'E', 'F', 'L', 'O', 'P', 'T', 'Z'],
    'SSl': ['C', 'D', 'E', 'F', 'L', 'O', 'P', 'T', 'Z'],
    'W': ['bird', 'cow', 'cup', 'house', 'train']
}

acuities = ['A', 'C', 'E', 'ETDRS', 'ETL-face', 'ETL-x', 'HOTV', 'L', 'NL', 'NPV', 'P', 'SSa', 'SSl', 'Teller', 'W']

# default height and width
H = 400
W = 400

# PROJECT ROOT
local_project_root = "/Users/brookeryan/Developer/BaldiLab/Visual-Acuity/"
remote_project_root = "/home/brooker/VisualAcuity/"

# DATA ROOT
remote_data_root = "/baldig/bioprojects2/VisualAcuity/"
local_data_root = "/Users/brookeryan/Developer/BaldiLab/Visual-Acuity/MultipleChoiceTest4/"

# OPTIONS
DATA_ROOT = local_data_root
PROJECT_ROOT = local_project_root

IMAGES_PATH = PROJECT_ROOT + "images4/"
TRAINING_PATH = PROJECT_ROOT + "images4/training/"
TESTING_PATH = PROJECT_ROOT + "images4/testing/"

TRAINING_SIZE = 8480
TESTING_SIZE = 11610

label_path = "/Users/brookeryan/Developer/BaldiLab/Visual-Acuity/numpyData/"
