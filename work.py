from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import wandb
from datetime import datetime


def init_wandb():
    """
    Initializes the wandb config yaml file and run name variables.
    """
    wandb.init(config="config-defaults.yaml", project="Visual_Acuity")
    wandb.run.name = datetime.now().strftime("%H:%M:%S, %m/%d/%Y, id= ") + wandb.run.id


def log_model_params(model, wandb_config, args):
    """ Extract params of interest about the model (e.g. number of different layer types).
      Log these and any experiment-level settings to wandb """
    num_conv_layers = 0
    num_fc_layers = 0
    for l in model.layers:
        layer_type = l.get_config()["name"].split("_")[0]
        if layer_type == "conv2d":
            num_conv_layers += 1
        elif layer_type == "dense":
            num_fc_layers += 1


def create_datasets():
    """
    Generates a `tf.data.Dataset` from image files in the project.
    :return: Two `tf.data.Dataset` objects, one for testing and one for training.
      - If `label_mode` is None, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
      - Otherwise, it yields a tuple `(images, labels)`, where `images`
        has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.
    """
    training_set = image_dataset_from_directory("./images/training/",
                                                shuffle=True,
                                                batch_size=32,
                                                image_size=(400, 400))
    testing_dataset = image_dataset_from_directory("./images/testing/",
                                                   shuffle=True,
                                                   batch_size=32,
                                                   image_size=(400, 400))
    return training_set, testing_dataset


# Just trying to replicate the same thing that I had last time
def create_model():
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(400,400,3))
    vgg.trainable = False
    inputs = tf.keras.Input(shape=(400, 400, 3))
    x = preprocess_input(inputs)
    x = vgg(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    return model


def train():
    training_set, testing_dataset = create_datasets()
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(400, 400, 3))
    vgg.trainable = False
    inputs = tf.keras.Input(shape=(400, 400, 3))
    x = preprocess_input(inputs)
    x = vgg(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    model.fit(training_set, epochs=20)


def create_base_model():
    vgg = VGG16(include_top=False, weights="imagenet", input_shape=(150, 150, 3))
    vgg.trainable = False
    return vgg


import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory


def pleasework():
    import os, sys
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    training_set = image_dataset_from_directory("/home/brooker/VisualAcuity/images/training/",
                                                shuffle=True,
                                                batch_size=32,
                                                image_size=(150, 150))
    val_dataset = image_dataset_from_directory("/home/brooker/VisualAcuity/images/testing/",
                                               shuffle=True,
                                               batch_size=32,
                                               image_size=(150, 150))
    base_model = create_base_model()
    inputs = tf.keras.Input(shape=(150, 150, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_set, epochs=3000, verbose=2)
    model.evaluate(val_dataset, verbose=1)


if __name__ == '__main__':
    init_wandb()
    pleasework()

#
# def load_config():
#
#
#
#
#
# def create_base_model():
#     vgg = VGG16(include_top=False, weights="imagenet", input_shape=(h, w, 3))
#     vgg.trainable = False
#     return vgg
#

# init_wandb()
#
# base_model = create_base_model()
# x = base_model(x_train, training=False)
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.2)(x)
# inputs = keras.Input(shape=(h, w, 3))
# outputs = keras.layers.Dense(1, activation=tf.keras.activations.softmax)(x)
# model = keras.Model(inputs, outputs)
# model.summary()
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(training_set, epochs=20, validation_data=val_dataset)
#
# # %%
#
# ssl._create_default_https_context = ssl._create_unverified_context
# # !pip install pydot
# # !pip install graphviz
# # !pip install pydotplus
#
# model = tf.keras.applications.VGG16()
# # plot_model(model)
#
# model.summary()
#
# # %%
#
#
# base_model = VGG16(weights='imagenet')
# model_VGG16 = models.Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
#
# model_VGG16.summary()
#
# # %%
#
# model = VGG16(include_top=False, weights="imagenet", input_shape=(h, w, 3))
# model.summary()
#
# # %%
#
# for layer in model.layers[:45]:
#     layer.trainable = False
#     print(layer)
#
#
# # %%
#
#

#
# # %%
