import numpy as np


class Dataset:
    def __init__(self, path):
        print("Path:", path)
        self.acuities = np.load(path + "/acuities.npy", allow_pickle=True)
        self.acuities_numeric = np.load(path + "/acuities_numeric.npy")

        self.angles = np.load(path + "/angles.npy")

        self.augmented = np.load(path + "/augmented.npy", allow_pickle=True)
        self.augmented_numeric = np.load(path + "/augmented_numeric.npy")

        self.character = np.load(path + "/character.npy", allow_pickle=True)
        self.character_numeric = np.load(path + "/character_numeric.npy")

        self.distortions = np.load(path + "/distortions.npy", allow_pickle=True)
        self.distortions_numeric = np.load(path + "/distortions_numeric.npy")

        self.images = np.load(path + "/images.npy")

        self.optotypes = np.load(path + "/optotypes.npy", allow_pickle=True)
        self.optotypes_numeric = np.load(path + "/optotypes_numeric.npy")

        self.sizes = np.load(path + "/sizes.npy", allow_pickle=True)
        self.sizes_numeric = np.load(path + "/sizes_numeric.npy")
