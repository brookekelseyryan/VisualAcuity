import os

from preprocessing.constants import TRAINING_PATH, TESTING_PATH
from data.dataset import Dataset

########
# Main #
########


if __name__ == '__main__':
    print("Current directory: ", os.getcwd())
    print("HI")

    training_dataset = Dataset(TRAINING_PATH, name="Training", height=400, width=400, batch_size=30)
    testing_dataset = Dataset(TESTING_PATH, name="Training", height=400, width=400, batch_size=30)

    training_dataset.process_labels()

    print(training_dataset.angles)

