import tensorflow as tf
import wandb
from keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import TensorBoard

import util.wb
from data import dataset as ds
from sklearn.model_selection import train_test_split


# Imports
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard

import wandb
from wandb.keras import WandbCallback
from sklearn.metrics import confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
import io


def run():
    print("Loading training dataset...\n")
    training_dataset = ds.Dataset(config['np_training'])

    print("Loading testing dataset...\n")
    testing_dataset = ds.Dataset(config['np_testing'])

    training_images, validation_images, training_optotypes, validation_optotypes = train_test_split(training_dataset.images, training_dataset.optotypes_numeric, test_size=wandb.config.validation_split)

    base_model = create_base_model()

    inputs = tf.keras.Input(shape=(wandb.config.height, wandb.config.width, wandb.config.channels))

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(wandb.config.num_optotypes, activation=wandb.config.activation)(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    # Define the per-epoch callback
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    model.compile(loss=wandb.config.loss, optimizer=wandb.config.optimizer, metrics=wandb.config.metrics)
    model.fit(x=training_images, y=training_optotypes, epochs=wandb.config.epochs, verbose=2,
              validation_data=(validation_images, validation_optotypes),
              callbacks=[TensorBoard(log_dir=wandb.run.dir), cm_callback])

    score = model.evaluate(x=testing_dataset.images, y=testing_dataset.optotypes_numeric, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset
    test_pred_raw = model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=1)

    # Calculate the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), test_pred)
    # Log the confusion matrix as an image summary
    figure = plot_confusion_matrix(cm, class_names=labels)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def create_base_model():
    """
    Creates base of VGG16 transfer learning model with imagenet weights and input shape of the same size in config.
    :return: VGG16 model
    """
    vgg = VGG16(include_top=False, weights=wandb.config.weights,
                input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels))
    vgg.trainable = False
    return vgg