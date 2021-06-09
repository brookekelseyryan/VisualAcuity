import numpy as np
from preprocessing import constants


def test_labels(dataset, name):
    print("~~~~~~~~~~~~")
    print("   ", name)
    print("~~~~~~~~~~~~")
    print("optotypes: ", np.unique(dataset.optotypes))
    print("angles: ", np.unique(dataset.angles))
    print("image sizes: ", np.unique(dataset.sizes))
    print("distortions: ", np.unique(dataset.distortions))


def test_training_labels(dataset):
    test_label(np.unique(dataset.optotypes), np.array(constants.optotypes), size=64)
    test_label(np.unique(dataset.angles), np.array(constants.angles))
    test_label(np.unique(dataset.sizes), np.array(constants.TRAINING_IMAGE_SIZES))
    test_label(np.unique(dataset.distortions), np.array(["low"]))

    test_labels(dataset, "Train")


def test_testing_labels(dataset):
    test_label(np.unique(dataset.optotypes), np.array(constants.optotypes), size=64)
    test_label(np.unique(dataset.angles), np.array(constants.angles))
    test_label(np.unique(dataset.sizes), np.array(["S", "M", "L"]))
    test_label(np.unique(dataset.distortions), np.array(["low", "high"]))

    test_labels(dataset, "Test")


def test_label(dataset_attribute, constant_attribute, size=None):
    assert np.all(dataset_attribute == constant_attribute)
    if size:
        assert np.size(dataset_attribute) == size
