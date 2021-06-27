import numpy as np
from PIL import Image

from preprocessing import constants as const
from preprocessing import files as f


def test_labels(dataset, name):
    print("~~~~~~~~~~~~")
    print("   ", name)
    print("~~~~~~~~~~~~")
    print("optotypes: ", np.unique(dataset.optotypes))
    print("angles: ", np.unique(dataset.angles))
    print("image sizes: ", np.unique(dataset.sizes))
    print("distortions: ", np.unique(dataset.distortions))

    if name.lower() == "train":
        test_training_labels(dataset)
    if name.lower() == "test":
        test_testing_labels(dataset)


def test_training_labels(dataset):
    test_same_size(dataset.images.shape[0], dataset.num_total_images)
    test_same_size(np.count_nonzero(dataset.augmented != "None"), dataset.num_augmented_images)

    test_label(np.unique(dataset.angles), np.array(const.angles_int))
    test_label(np.unique(dataset.distortions), np.array(["low"]))
    test_label(np.unique(dataset.optotypes), np.array(const.optotypes))
    test_label(np.unique(dataset.sizes), np.array(const.TRAINING_IMAGE_SIZES))


def test_testing_labels(dataset):
    test_same_size(dataset.images.shape[0], dataset.num_total_images)
    test_same_size(dataset.num_augmented_images, 0)

    test_label(np.unique(dataset.angles), np.array(const.angles_int))
    test_label(np.unique(dataset.distortions), np.array(["high", "low"]))
    test_label(np.unique(dataset.optotypes), np.array(const.optotypes))
    test_label(np.unique(dataset.sizes), np.array(["S", "M", "L"]))


def test_same_size(len_array, size):
    assert len_array == size, "Length of array {l} neq Size {s}".format(l=len_array, s=size)


def test_label(dataset_attribute, constant_attribute, size=None):
    assert np.all(dataset_attribute == constant_attribute)
    if size:
        assert np.size(dataset_attribute) == size


def test_random_index(dataset, ax, index=None):
    if index is None:
        index = np.random.randint(0, dataset.images.shape[0])

    ax.axis('off')
    ax.imshow(dataset.images[index]/255)
    ax.set_title("Optotype={o}\nAngle={a}\nSize={s}\nDistortion={d}\nisAugmented={ia}".format(o=dataset.optotypes[index],
                                                                                              a=dataset.angles[index],
                                                                                              s=dataset.sizes[index],
                                                                                              d=dataset.distortions[index],
                                                                                              ia=dataset.augmented[index]))


def test_angles_accurate():
    """
    TODO
    """
    y_train_angle = np.zeros(27, dtype=np.int)
    for i, sample_name in enumerate(const.low_distortion_filenames):
        thing = f.extract_axis(sample_name)
        y_train_angle[i] = thing
    print(np.unique(y_train_angle))





