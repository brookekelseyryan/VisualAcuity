import os
import sys
from datetime import datetime

import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory
from wandb.keras import WandbCallback
import numpy as np

import wandb

# 64 classes of optotypes. Should correspond to the folder names under images/testing and images/training
from confusion_wanb import WandbClassificationCallback

labels = ["+blank", "+circle", "+diamond", "+square", "2", "3", "5", "6", "8", "9", "apple", "bird", "C", "C-0", "C-45",
          "C-90", "C-135", "C-180", "C-225", "C-270", "C-315", "cake", "car", "circle", "cow", "cup", "D",
          "duck", "E", "E-0", "E-90", "E-180", "E-270", "F", "flat-line", "flat-square",
          "frown-line", "frown-square", "H", "hand", "horse", "house", "K", "L", "N", "O", "P",
          "panda", "phone", "R", "S", "smile-line", "smile-square", "square", "star", "T", "train",
          "tree", "V", "x-blank", "x-circle", "x-diamond", "x-square", "Z"]


def init_wandb(argv, sync_tensorboard=True):
    """
    Initializes the wandb config yaml file and run name variables.
    :param argv: One command-line argument that is just the run name. Optional.
    :param sync_tensorboard: WandB parameter that enables Tensorboard to be tracked.
    :return:
    """
    wandb.init(config="config-defaults.yaml", project="Visual_Acuity", sync_tensorboard=sync_tensorboard)

    run_name = ""

    if len(argv) > 1:
        run_name = argv[1]

    wandb.run.name = run_name + datetime.now().strftime(" %H:%M:%S, %m/%d/%Y, id= ") + wandb.run.id


def log_model_params(model, wandb_config, args):
    """
    NOT USED CURRENTLY
    Extract params of interest about the model (e.g. number of different layer types).
    Log these and any experiment-level settings to wandb.
    :param model:
    :param wandb_config:
    :param args:
    :return:
    """
    num_conv_layers = 0
    num_fc_layers = 0
    for l in model.layers:
        layer_type = l.get_config()["name"].split("_")[0]
        if layer_type == "conv2d":
            num_conv_layers += 1
        elif layer_type == "dense":
            num_fc_layers += 1


def create_datasets_test_train():
    """
    Generates two `tf.data.Dataset` from image files in the project.
    NOTE: images must be put under project root in directory images/testing and images/training
    :return: Two `tf.data.Dataset` objects, one for testing and one for training.
    """
    training_set = image_dataset_from_directory("./images/training/",
                                                shuffle=True,
                                                batch_size=30,
                                                image_size=(wandb.config.image_dms, wandb.config.image_dms))
    testing_set = image_dataset_from_directory("./images/testing/",
                                               shuffle=True,
                                               batch_size=30,
                                               image_size=(wandb.config.image_dms, wandb.config.image_dms))
    return training_set, testing_set


def create_datasets(validation_split=0.2):
    """
    Generates three `tf.data.Dataset` from image files in the project.
    @:param validation_split: float between 0 and 1, fraction of data to reserve for validation.
    NOTE: images must be put under project root in directory images/testing and images/training.
    :return: Three `tf.data.Dataset` objects, testing, training, and validation.
    """
    training_set = image_dataset_from_directory("./images/training/",
                                                shuffle=True,
                                                batch_size=30,
                                                validation_split=validation_split,
                                                subset="training",
                                                seed=0,
                                                image_size=(wandb.config.image_dms, wandb.config.image_dms))
    validation_set = image_dataset_from_directory("./images/training/",
                                                  shuffle=False,
                                                  batch_size=30,
                                                  validation_split=validation_split,
                                                  subset="validation",
                                                  seed=0,
                                                  image_size=(wandb.config.image_dms, wandb.config.image_dms))
    testing_set = image_dataset_from_directory("./images/testing/",
                                               shuffle=True,
                                               batch_size=30,
                                               image_size=(wandb.config.image_dms, wandb.config.image_dms))
    return training_set, validation_set, testing_set


# Just trying to replicate the same thing that I had last time
def create_model():
    vgg = VGG16(include_top=False, weights='imagenet',
                input_shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))
    vgg.trainable = False
    inputs = tf.keras.Input(shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))
    x = preprocess_input(inputs)
    x = vgg(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    return model


def extract_features_2(input_shape=(400, 400, 3),
                       n_classes=64,
                       optimizer='adam',
                       fine_tune=0):
    """
    TODO: This currently doesn't work :( Need help on getting transfer learning working.

    Using this tutorial blog: https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
    Look at this section: Using Pre-trained Layers for Feature Extraction

    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """

    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    vgg = VGG16(include_top=False, weights='imagenet',
                input_shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))
    vgg.trainable = False
    inputs = tf.keras.Input(shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))
    x = preprocess_input(inputs)
    x = vgg(x, training=False)

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    # if fine_tune > 0:
    #     for layer in conv_base.layers[:-fine_tune]:
    #         layer.trainable = False
    # else:
    #     for layer in conv_base.layers:
    #         layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    # top_model = conv_base.output
    top_model = tf.keras.layers.GlobalAveragePooling2D()(x)
    top_model = tf.keras.layers.Dense(4096, activation='relu')(top_model)
    top_model = tf.keras.layers.Dense(1072, activation='relu')(top_model)
    # top_model = tf.keras.layers.Dropout(0.2)(top_model)
    output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = tf.keras.Model(inputs, output_layer)

    model.summary()

    # Compiles the model for training.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_set, epochs=25, verbose=2, validation_data=validation_set,
              callbacks=[WandbCallback(data_type="image", labels=labels),
                         TensorBoard(log_dir=wandb.run.dir)])
    model.evaluate(testing_set, verbose=1)


def extract_features():
    """
    TODO: This also currently doesn't work :( Need help on getting transfer learning working.

    Using this tutorial blog: https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/
    Trying to use VGG16 to extract features
    """
    training_set, testing_set = create_datasets_test_train()

    vgg = VGG16(include_top=False, weights='imagenet',
                input_shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))

    # Extracting features from the training dataset
    features_train = vgg.predict(training_set)

    # Extracting features from the testing dataset
    features_test = vgg.predict(testing_set)

    # Flattening the layers to conform to input
    print("shape of features_train: ", features_train.shape)
    print("shape of features_test: ", features_test.shape)
    num_features = 12 * 12 * 512
    # this worked !!
    train = features_train.reshape(1500, num_features)
    test = features_test.reshape(3223, num_features)

    # Converting target variable to array
    # TODO: not doing this part from tutorial want to see what happens

    # Creating an MLP model (https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/)
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(1000, input_dim=num_features, activation='relu', kernel_initializer='uniform'))
    tf.keras.layers.Dropout(0.3)

    model.add(tf.keras.layers.Dense(500, input_dim=1000, activation='sigmoid'))
    tf.keras.layers.Dropout(0.3)

    model.add(tf.keras.layers.Dense(150, input_dim=500, activation='sigmoid'))
    tf.keras.layers.Dropout(0.2)

    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(training_set)
    model.fit(train, epochs=25, verbose=2, validation_split=0.2,
              callbacks=[WandbCallback(data_type="image", labels=labels), TensorBoard(log_dir=wandb.run.dir)])
    model.evaluate(test, verbose=1)


def train():
    """
    TODO: Currently, this gives a really bad accuracy rating, like 1% or something.

    Creates base of VGG16 transfer learning model with imagenet weights and input shape of the same size in config.
    Then trains and evaluates the model.

    """
    training_set, validation_set, testing_set = create_datasets()
    vgg = VGG16(include_top=False, weights='imagenet',
                input_shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))
    vgg.trainable = False
    inputs = tf.keras.Input(shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))
    x = preprocess_input(inputs)
    x = vgg(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    model.fit(training_set, epochs=25, verbose=2, validation_data=validation_set,
              callbacks=[WandbCallback(data_type="image", labels=labels), TensorBoard(log_dir=wandb.run.dir)])
    model.evaluate(testing_set, verbose=1)


def create_base_model():
    """
    Creates base of VGG16 transfer learning model with imagenet weights and input shape of the same size in config.
    :return: VGG16 model
    """
    vgg = VGG16(include_top=False, weights="imagenet",
                input_shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))
    vgg.trainable = False
    return vgg


def runner():
    """
    Essentially the main function. Feel free to change as you see fit.
    """

    # Uncomment if on arcus servers
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    training_set, validation_set, testing_set = create_datasets()

    base_model = create_base_model()

    inputs = tf.keras.Input(shape=(wandb.config.image_dms, wandb.config.image_dms, wandb.config.color_channels))

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(64, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_set, epochs=25, verbose=2, validation_data=validation_set,
              callbacks=[
                  WandbClassificationCallback(input_type="image", log_confusion_matrix=True,
                                              confusion_examples=3, confusion_classes=5,
                                              validation_data=validation_set, labels=labels),
                  WandbCallback(data_type="image", labels=labels),
                  TensorBoard(log_dir=wandb.run.dir)])

    model.evaluate(testing_set, verbose=1)

    # Confusion matrix
    predictions = np.array([])
    L = np.array([])
    for x, y in testing_set:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
        L = np.concatenate([L, np.argmax(y.numpy(), axis=-1)])

    print(predictions.shape)
    print(L.shape)
    print(predictions)
    print(L)

    wandb.log({"my_conf_mat_id": wandb.plot.confusion_matrix(
        preds=predictions, y_true=L,
        class_names=labels)})


if __name__ == '__main__':
    init_wandb(sys.argv)
    runner()
