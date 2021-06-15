import tensorflow as tf
import wandb
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet152
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge

from tensorflow.keras.callbacks import TensorBoard
from wandb.keras import WandbCallback
import util.wb


class Model:

    # This is the same used throughout many models
    def __init__(self, training_dataset, validation_dataset, testing_dataset, application, model, config={}):
        """
        Allows re-use of datasets and callbacks throughout runs.
        """

        self.name = application.__name__
        self.application = application
        self.wandb_run = util.wb.init_wandb(params=config, run_name=self.name)
        self.model = frozen_weights(model(include_top=False, weights=wandb.config.weights,
                                          input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels),
                                          classifier_activation=wandb.config.activation))

        self.training = training_dataset
        self.validation = validation_dataset
        self.testing = testing_dataset

        # Preprocessing images data
        self.training.preprocess_input(application)
        self.validation.preprocess_input(application)
        self.testing.preprocess_input(application)

        print("self.validation.optotypes_labels", self.validation.optotypes_labels)

        self.callbacks = [TensorBoard(log_dir=wandb.run.dir),
                          util.wb.WWandbCallback(log_weights=True,
                                                 log_gradients=True,
                                                 data_type="image",
                                                 testing_data=(self.testing.images_processed, self.testing.optotypes_numeric),
                                                 testing_labels=self.testing.optotypes,
                                                 validation_labels=self.validation.optotypes,
                                                 predictions=10,
                                                 input_type="image",
                                                 output_type="label",
                                                 log_evaluation=False,
                                                 labels=self.validation.optotypes_labels,
                                                 training_data=(self.training.images_processed, self.training.optotypes_numeric),
                                                 validation_data=(self.validation.images_processed, self.validation.optotypes_numeric))]

    def finish(self):
        self.wandb_run.finish()

    def reinit(self, config={}, run_name=""):
        self.wandb_run.finish()
        self.wandb_run = util.wb.init_wandb(params=config, run_name=self.name)

    def train(self, config={}):
        print("Compiling model", self.model, "...")
        self.model.compile(loss=wandb.config.loss, optimizer=wandb.config.optimizer, metrics=wandb.config.metrics)

        print("Training model", self.model, "...")
        self.model.fit(x=self.training.images_processed, y=self.training.optotypes_numeric, epochs=wandb.config.epochs,
                       verbose=2,
                       validation_data=(self.validation.images_processed, self.validation.optotypes_numeric),
                       callbacks=self.callbacks)

        print("Training complete.")

        return self

    def evaluate(self):
        print("Evaluating model...")

        score = self.model.evaluate(x=self.testing.images_processed, y=self.testing.optotypes_numeric, verbose=1)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        self.wandb_run.log({"Test Loss": score[0], "Test Accuracy": score[1]})

        return self

    def predict(self):
        predictions = self.model.predict(x=self.testing.images_processed, callbacks=self.callbacks)
        self.wandb_run.log({"Model Predictions": predictions})


def frozen_weights(model):
    """
    Creates base of given transfer learning model with imagenet weights and input shape of the same size in config.
    :return: model
    """

    model.trainable = False
    inputs = tf.keras.Input(shape=(wandb.config.height, wandb.config.width, wandb.config.channels))
    x = model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(wandb.config.num_optotypes, activation=wandb.config.activation)(x)

    model = tf.keras.Model(inputs, outputs)

    model.summary()

    return model
