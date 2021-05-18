from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory
import wandb
from datetime import datetime


def init_wandb():
    """
    Initializes the wandb config yaml file and run name variables.
    """
    wandb.init(config="config-defaults.yaml", project="Visual_Acuity")
    wandb.run.name = datetime.now().strftime("%H:%M:%S, %m/%d/%Y, id= ") + wandb.run.id
    print(wandb.config.epochs)
    print(wandb.config.optimizer)


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
    training_set = image_dataset_from_directory("/images/training/",
                                                shuffle=True,
                                                batch_size=wandb.config.batch_size,
                                                image_size=(wandb.config.image_height, wandb.config.image_width))
    testing_dataset = image_dataset_from_directory("/images/testing/",
                                                   shuffle=True,
                                                   batch_size=wandb.config.batch_size,
                                                   image_size=(wandb.config.image_height, wandb.config.image_width))
    return training_set, testing_dataset


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

init_wandb()
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
# def create_transfer_model_2():
#     input_t = K.Input(shape=(h, w, 3))
#     res_model = K.applications.ResNet50(include_top=False,
#                                         weights="imagenet",
#                                         input_tensor=input_t)
#
#     for layer in res_model.layers[:45]:
#         layer.trainable = False
#     # to_res = (224, 224)
#     model = K.models.Sequential()
#     # model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
#     model.add(res_model)
#     model.add(K.layers.Flatten())
#     model.add(K.layers.BatchNormalization())
#     model.add(K.layers.Dense(256, activation='relu'))
#     model.add(K.layers.Dropout(0.5))
#     model.add(K.layers.BatchNormalization())
#     model.add(K.layers.Dense(128, activation='relu'))
#     model.add(K.layers.Dropout(0.5))
#     model.add(K.layers.BatchNormalization())
#     model.add(K.layers.Dense(64, activation='relu'))
#     model.add(K.layers.Dropout(0.5))
#     model.add(K.layers.BatchNormalization())
#     model.add(K.layers.Dense(4, activation='softmax'))  # this
#
#     model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
#
#     return model
#
#
# model = create_transfer_model_2()
# model.summary()
#
# # %%
