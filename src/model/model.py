import tensorflow as tf
import wandb
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from keras.applications.resnet import ResNet152

from keras.applications.xception import Xception

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.nasnet import NASNetLarge

from tensorflow.keras.callbacks import TensorBoard
import util.wb


class Model:

    # This is the same used throughout many models
    def __init__(self, training_dataset, validation_dataset, testing_dataset):
        """
        Allows re-use of datasets and callbacks throughout runs.
        """
        self.training = training_dataset
        self.validation = validation_dataset
        self.testing = testing_dataset

        self.model = None
        self.wandb_run = None

    def clear(self):
        """
        If doing more than one run, call this before model.run()
        """
        self.wandb_run.finish()

        self.model = None
        self.wandb_run = None

    def run(self, model, config={}):
        """
        This is what makes it unique. Which model and which config to use.
        """

        self.wandb_run = util.wb.init_wandb(params=config, run_name=model)

        if model.lower() == "vgg_16_frozen_weights":
            self.model = frozen_weights(VGG16(include_top=False, weights=wandb.config.weights, input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels)))
        elif model.lower() == "vgg_19_frozen_weights":
            self.model = frozen_weights(VGG19(include_top=False, weights=wandb.config.weights, input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels)))
        elif model.lower() == "nasnetlarge_frozen_weights":
            self.model = frozen_weights(NASNetLarge(include_top=False, weights=wandb.config.weights, input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels)))
        elif model.lower() == "xception_frozen_weights":
            self.model = frozen_weights(Xception(include_top=False, weights=wandb.config.weights, input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels)))
        elif model.lower() == "resnet152_frozen_weights":
            self.model = frozen_weights(ResNet152(include_top=False, weights=wandb.config.weights, input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels)))
        elif model.lower() == "inceptionresnetv2_frozen_weights":
            self.model = frozen_weights(InceptionResNetV2(include_top=False, weights=wandb.config.weights, input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels)))
        else:
            raise Exception("Model", model, "is undefined")

        print("Compiling model", model, "...")
        self.model.compile(loss=wandb.config.loss, optimizer=wandb.config.optimizer, metrics=wandb.config.metrics)

        print("Training model", model, "...")
        self.model.fit(x=self.training.images, y=self.training.optotypes_numeric, epochs=wandb.config.epochs, verbose=2,
                       validation_data=(self.validation.images, self.validation.optotypes_numeric),
                       callbacks=[TensorBoard(log_dir=wandb.run.dir)])

        print("Training complete.")

        return self

    def evaluate(self):
        print("Evaluating model...")

        score = self.model.evaluate(x=self.testing.images, y=self.testing.optotypes_numeric, verbose=1)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        self.wandb_run.log({"Test Loss": score[0], "Test Accuracy": score[1]})

        return self


def vgg_16_frozen_weights():
    """
    Creates base of VGG16 transfer learning model with imagenet weights and input shape of the same size in config.
    :return: VGG16 model
    """
    vgg16 = VGG16(include_top=False, weights=wandb.config.weights,
                  input_shape=(wandb.config.height, wandb.config.width, wandb.config.channels))
    vgg16.trainable = False
    inputs = tf.keras.Input(shape=(wandb.config.height, wandb.config.width, wandb.config.channels))
    x = vgg16(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(wandb.config.num_optotypes, activation=wandb.config.activation)(x)

    model = tf.keras.Model(inputs, outputs)

    model.summary()

    return model


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
