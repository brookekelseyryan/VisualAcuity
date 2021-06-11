import os
import sys

import tensorflow as tf
import wandb
from keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import TensorBoard

from data import constants as const
from visualization import wandb_utils as wb


def create_base_model():
    """
    Creates base of VGG16 transfer learning model with imagenet weights and input shape of the same size in config.
    :return: VGG16 model
    """
    vgg = VGG16(include_top=False, weights=wandb.config.weights,
                input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels))
    vgg.trainable = False
    return vgg


def runner():

    training_dataset = ds.Dataset(const.TRAINING_PATH, name="Training", include_augmented=True, height=wandb.config.height, width=wandb.config.width, channels=wandb.config.channels)
    testing_dataset = ds.Dataset(const.TESTING_PATH, name="Testing", height=wandb.config.height, width=wandb.config.width, channels=wandb.config.channels)

    base_model = create_base_model()

    inputs = tf.keras.Input(shape=(wandb.config.height, wandb.config.width, wandb.config.channels))

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(len(const.optotypes), activation=wandb.config.activation)(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    model.compile(loss=wandb.config.loss, optimizer=wandb.config.optimizer, metrics=['accuracy'])
    model.fit(x=training_dataset.images, y=training_dataset.optotypes, epochs=25, verbose=2, validation_split=0.20,
              callbacks=[
                  wb.WandbClassificationCallback(input_type="image",
                                                 log_confusion_matrix=True,
                                                 confusion_examples=3,
                                                 confusion_classes=5,
                                                 training_data=(training_dataset.images, training_dataset.optotypes)),
                  TensorBoard(log_dir=wandb.run.dir)])

    score = model.evaluate(x=testing_dataset.images, y=testing_dataset.optotypes, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    # Uncomment if on arcus servers
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    wb.init_wandb(sys.argv)

    runner()
