import os
import sys

import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
# Imports
from tensorflow.keras.callbacks import TensorBoard

import util.wb
import util.wb
import wandb
from data import dataset as ds
from model import model as m

########
# Main #
########
if __name__ == '__main__':
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = ""

    CONFIG_PATH = "./config/arcus.yaml"

    models = ["xception_frozen_weights",
              "resnet152_frozen_weights", "inceptionresnetv2_frozen_weights"]

    print("Initiating...\n")

    with open(CONFIG_PATH, 'r') as y:
        print("Opening config...\n")
        config = yaml.load(y)

    if config['cuda'] is True:
        print("Setting CUDA device environment variables...")
        os.environ['CUDA_DEVICE_ORDER'] = config['CUDA_DEVICE_ORDER']
        os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']

    os.environ["WANDB_RUN_GROUP"] = "experiment " + experiment_name
    util.wb.init_wandb(yaml=config['wandb_config_path'], run_name="init")

    training_dataset, validation_dataset, testing_dataset = ds.load_train_validate_test_datasets(config)

    model = m.Model(training_dataset, validation_dataset, testing_dataset)

    for m in models:
        model.run(m)
        model.evaluate()
        model.clear()
