import numpy as np

from sklearn import model_selection

import wandb


class Dataset:

    def __init__(self, name):
        self.name = name


    def log_values(self):
        print("################")
        print(self.name)
        print("###############")
        print("size =", self.images.shape[0])
        print("images dtype = ", self.images.dtype)
        print("acuities = ", self.acuities)
        print("angles = ", self.angles)
        print("augmented = ", self.augmented)
        print("character = ", self.character)
        print("distortions =", self.distortions)
        print("optotypes =", self.optotypes)
        print("sizes = ", self.sizes)

        wandb.log({self.name + " examples": [wandb.Image(image) for image in self.images[:50]]})

    @staticmethod
    def from_path(name, path):
        print("Path:", path)

        dataset = Dataset(name=name)

        dataset.acuities = np.load(path + "/acuities.npy", allow_pickle=True)
        dataset.acuities_numeric = np.load(path + "/acuities_numeric.npy")

        dataset.angles = np.load(path + "/angles.npy")

        dataset.augmented = np.load(path + "/augmented.npy", allow_pickle=True)
        dataset.augmented_numeric = np.load(path + "/augmented_numeric.npy")

        dataset.character = np.load(path + "/character.npy", allow_pickle=True)
        dataset.character_numeric = np.load(path + "/character_numeric.npy")

        dataset.distortions = np.load(path + "/distortions.npy", allow_pickle=True)
        dataset.distortions_numeric = np.load(path + "/distortions_numeric.npy")

        dataset.images = np.load(path + "/images.npy")

        dataset.optotypes = np.load(path + "/optotypes.npy", allow_pickle=True)
        dataset.optotypes_numeric = np.load(path + "/optotypes_numeric.npy")

        dataset.sizes = np.load(path + "/sizes.npy", allow_pickle=True)
        dataset.sizes_numeric = np.load(path + "/sizes_numeric.npy")

        return dataset

    @staticmethod
    def from_attributes(name, images, acuities=None, acuities_numeric=None, angles=None, augmented=None,
                        augmented_numeric=None, character=None, character_numeric=None, distortions=None,
                        distortions_numeric=None, optotypes=None, optotypes_numeric=None, sizes=None,
                        sizes_numeric=None):

        dataset = Dataset(name=name)

        dataset.images = images

        dataset.acuities = acuities
        dataset.acuities_numeric = acuities_numeric

        dataset.angles = angles

        dataset.augmented = augmented
        dataset.augmented_numeric = augmented_numeric

        dataset.character = character
        dataset.character_numeric = character_numeric

        dataset.distortions = distortions
        dataset.distortions_numeric = distortions_numeric

        dataset.optotypes = optotypes
        dataset.optotypes_numeric = optotypes_numeric

        dataset.sizes = sizes
        dataset.sizes_numeric = sizes_numeric

        return dataset


def train_validate_split(training_dataset, split):
    training_images, validation_images, \
    training_acuities, validation_acuities, \
    training_acuities_numeric, validation_acuities_numeric, \
    training_angles, validation_angles, \
    training_augmented, validation_augmented, \
    training_augmented_numeric, validation_augmented_numeric, \
    training_character, validation_character, \
    training_character_numeric, validation_character_numeric, \
    training_distortions, validation_distortions, \
    training_distortions_numeric, validation_distortions_numeric, \
    training_optotypes, validation_optotypes, \
    training_optotypes_numeric, validation_optotypes_numeric, \
    training_sizes, validation_sizes, \
    training_sizes_numeric, validation_sizes_numeric = model_selection.train_test_split(
                                                                        training_dataset.images,
                                                                        training_dataset.acuities,
                                                                        training_dataset.acuities_numeric,
                                                                        training_dataset.angles,
                                                                        training_dataset.augmented,
                                                                        training_dataset.augmented_numeric,
                                                                        training_dataset.character,
                                                                        training_dataset.character_numeric,
                                                                        training_dataset.distortions,
                                                                        training_dataset.distortions_numeric,
                                                                        training_dataset.optotypes,
                                                                        training_dataset.optotypes_numeric,
                                                                        training_dataset.sizes,
                                                                        training_dataset.sizes_numeric,
                                                                        test_size=split,
                                                                        stratify=training_dataset.optotypes)

    training_dataset = Dataset.from_attributes("Training", training_images, training_acuities, training_acuities_numeric,
                               training_angles,
                               training_augmented, training_augmented_numeric,
                               training_character, training_character_numeric, training_distortions,
                               training_distortions_numeric, training_optotypes,
                               training_optotypes_numeric, training_sizes, training_sizes_numeric)

    validation_dataset = Dataset.from_attributes("Validation", validation_images, validation_acuities, validation_acuities_numeric,
                                 validation_angles, validation_augmented,
                                 validation_augmented_numeric,
                                 validation_character, validation_character_numeric,
                                 validation_distortions, validation_distortions_numeric,
                                 validation_optotypes,
                                 validation_optotypes_numeric, validation_sizes,
                                 validation_sizes_numeric)

    return training_dataset, validation_dataset


def load_train_validate_test_datasets(config, split=0.20):
    testing_dataset = Dataset.from_path("Testing", config['np_testing'])
    training_dataset = Dataset.from_path("Training", config['np_training'])

    training_dataset, validation_dataset = train_validate_split(training_dataset, split)

    training_dataset.log_values()
    validation_dataset.log_values()
    testing_dataset.log_values()

    return training_dataset, validation_dataset, testing_dataset
