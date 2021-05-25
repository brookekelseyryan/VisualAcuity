from unittest import TestCase
from main import create_datasets


class Test(TestCase):
    def test_create_datasets(self):
        training_set, validation_set, testing_set = create_datasets()
        print("training set: ", training_set)
        print("testing set: ", testing_set)
