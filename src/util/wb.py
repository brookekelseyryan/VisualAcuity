# @title Double click to see the code

import os
from itertools import chain

import numpy as np
import wandb
from wandb.keras import WandbCallback
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger


def init_wandb(run_name, params={}, yaml="/home/brooker/VisualAcuity/src/config/config-defaults.yaml", reinit=True,
               sync_tensorboard=True):
    """
    Initializes the config config yaml file and run name variables.
    :param config_path: path to config-defaults.yaml
    :param run_name:
    :param argv: One command-line argument that is just the run name. Optional.
    :param sync_tensorboard: WandB parameter that enables Tensorboard to be tracked.
    :return:
    """
    print("Initializing wandb run", run_name, "with config_path", yaml, "and additional params:", params)

    run = wandb.init(allow_val_change=True, reinit=reinit, config=yaml, project="Visual_Acuity",
                     sync_tensorboard=sync_tensorboard, group=os.environ["WANDB_RUN_GROUP"])
    wandb.config.update(params, allow_val_change=True)

    wandb.run.name = run_name + wandb.run.id

    return run


class WWandbCallback(WandbCallback):
    def __init__(
            self,
            testing_data=None,
            testing_labels=None,
            validation_labels=None,
            monitor="val_loss",
            verbose=0,
            mode="auto",
            save_weights_only=False,
            log_weights=False,
            log_gradients=False,
            save_model=True,
            training_data=None,
            validation_data=None,
            labels=[],
            data_type=None,
            predictions=10,
            generator=None,
            input_type=None,
            output_type=None,
            log_evaluation=False,
            validation_steps=None,
            class_colors=None,
            log_batch_frequency=None,
            log_best_prefix="best_",
            save_graph=True,
            validation_indexes=None,
            validation_row_processor=None,
            prediction_row_processor=None,
            infer_missing_processors=True, ):

        self.testing_data = testing_data
        self.testing_labels = testing_labels
        self.validation_labels = validation_labels

        super().__init__(monitor=monitor,
                         verbose=verbose,
                         mode=mode,
                         save_weights_only=save_weights_only,
                         log_weights=log_weights,
                         log_gradients=log_gradients,
                         save_model=save_model,
                         training_data=training_data,
                         validation_data=validation_data,
                         labels=labels,
                         data_type=data_type,
                         generator=generator,
                         predictions=predictions,
                         input_type=input_type,
                         output_type=output_type,
                         log_evaluation=log_evaluation,
                         validation_steps=validation_steps,
                         class_colors=class_colors,
                         log_batch_frequency=log_batch_frequency,
                         log_best_prefix=log_best_prefix,
                         save_graph=save_graph,
                         validation_indexes=validation_indexes,
                         validation_row_processor=validation_row_processor,
                         prediction_row_processor=prediction_row_processor,
                         infer_missing_processors=infer_missing_processors)

    def on_train_end(self, logs=None):
        wandb.log(
            {"Low Distortion Final Predictions": self.log_images(num_images=self.predictions,
                                                                 name="validation")},
            commit=False)

        wandb.log(
            {"High Distortion Final Predictions": self.log_images(num_images=self.predictions, name="test")},
            commit=False)

        self._init_validation_gen()
        self._log_validation_table()

        self._init_testing_gen()
        self._log_testing_table()

    def on_epoch_end(self, epoch, logs={}):
        """
        Had to change the base method here as well
        """
        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary[
                    "%s%s" % (self.log_best_prefix, self.monitor)
                    ] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f"
                        % (epoch, self.monitor, self.best, self.current)
                    )
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    ###########################
    #        HELPERS          #
    ###########################

    def _logits_to_captions(self, logits):
        if logits[0].shape[-1] == 1:
            # Scalar output from the model
            # TODO: handle validation_y
            if len(self.labels) == 2:
                # User has named true and false
                captions = [
                    self.labels[1] if logits[0] > 0.5 else self.labels[0]
                    for logit in logits
                ]
            else:
                if len(self.labels) != 0:
                    wandb.termwarn(
                        'keras model is producing a single output, so labels should be a length two array: ["False label", "True label"].'
                    )
                captions = [logit[0] for logit in logits]
        else:
            # Vector output from the model
            # TODO: handle validation_y
            predicted_classes = np.argmax(np.stack(logits), axis=1)
            print("predicted_classes", predicted_classes)

            if len(self.labels) > 0:
                # User has named the categories in self.labels
                captions = []
                print("self.labels=", self.labels)
                print("length=", len(self.labels))
                for p in predicted_classes:
                    try:
                        captions.append(self.labels[p])
                        print("self.labels[{p}]={pr}".format(p=p, pr=self.labels[p]))
                    except IndexError:
                        captions.append(self.labels)
            else:
                captions = self.labels
        return captions

    def _log_testing_table(self):
        if self.log_evaluation and self.testing_data_logger:
            try:
                if not self.model:
                    wandb.termwarn("WandbCallback unable to read model from trainer")
                else:
                    self.testing_data_logger.log_predictions(
                        predictions=self.testing_data_logger.make_predictions(
                            self.model.predict
                        )
                    )
            except Exception as e:
                wandb.termwarn("Error durring prediction logging for epoch: " + str(e))

    def _log_validation_table(self):
        if self.log_evaluation and self.validation_data_logger:
            try:
                if not self.model:
                    wandb.termwarn("WandbCallback unable to read model from trainer")
                else:
                    self.validation_data_logger.log_predictions(
                        predictions=self.validation_data_logger.make_predictions(
                            self.model.predict
                        )
                    )
            except Exception as e:
                wandb.termwarn("Error durring prediction logging for epoch: " + str(e))

    def _init_testing_gen(self):
        """
        Helper method for initializing Testing data table
        """
        if self.log_evaluation:
            try:
                testing_data = None
                if self.testing_data:
                    testing_data = self.testing_data
                    self.testing_data_logger = ValidationDataLogger(
                        inputs=testing_data[0],
                        targets=testing_data[1],
                        indexes=None,
                        validation_row_processor=None,
                        prediction_row_processor=None,
                        class_labels=self.labels,
                        infer_missing_processors=self.infer_missing_processors)
            except Exception as e:
                wandb.termwarn(
                    "Error initializing ValidationDataLogger in WandbCallback. Skipping logging validation data. Error: " + str(
                        e))

    def _init_validation_gen(self):
        """
        Helper method for initializing Validation data table
        """
        if self.log_evaluation:
            try:
                validation_data = None
                if self.validation_data:
                    validation_data = self.validation_data
                    self.validation_data_logger = ValidationDataLogger(
                        inputs=validation_data[0],
                        targets=validation_data[1],
                        indexes=None,
                        validation_row_processor=None,
                        prediction_row_processor=None,
                        class_labels=self.labels,
                        infer_missing_processors=self.infer_missing_processors)
            except Exception as e:
                wandb.termwarn(
                    "Error initializing ValidationDataLogger in WandbCallback. Skipping logging validation data. Error: " + str(
                        e))

    def log_images(self, name, num_images=36):
        """
        Utility, have to override the one from the super class because only does on Validation images
        """
        if name == "validation":
            X = self.validation_data[0]
            y = self.validation_data[1]
            labels = self.validation_labels
        elif name == "test":
            X = self.testing_data[0]
            y = self.testing_data[1]
            labels = self.testing_labels
        else:
            raise Exception("No name set for logging image type", name, self)

        validation_length = len(X)

        if validation_length > num_images:
            # pick some data at random
            indices = np.random.choice(validation_length, num_images, replace=False)
            print("Random indicies", indices)
        else:
            indices = range(validation_length)
            print("non random indices", indices)

        test_data = np.take(X, indices, axis=0)
        test_output = np.take(labels, indices)

        print("test data")
        print(test_data.shape)
        print(test_data)

        print("test output")
        print(test_output.shape)
        print(test_output)

        test_data = test_data
        test_output = test_output.tolist()


        if self.model.stateful:
            print("stateful model")
            predictions = self.model.predict(np.stack(test_data), batch_size=1)
            self.model.reset_states()
        else:
            print("non stateful model")
            predictions = self.model.predict(
                np.stack(test_data), batch_size=self._prediction_batch_size)
            if len(predictions) != len(test_data):
                print("predictions batch size =1")
                self._prediction_batch_size = 1
                predictions = self.model.predict(
                    np.stack(test_data), batch_size=self._prediction_batch_size
                )

        if self.input_type == "label":
            print("input type is label")
            if self.output_type in ("image", "images", "segmentation_mask"):
                print("output type is image")
                captions = self._logits_to_captions(test_data)
                output_image_data = (
                    self._masks_to_pixels(predictions)
                    if self.output_type == "segmentation_mask"
                    else predictions
                )
                reference_image_data = (
                    self._masks_to_pixels(test_output)
                    if self.output_type == "segmentation_mask"
                    else test_output
                )
                output_images = [
                    wandb.Image(data, caption=captions[i], grouping=2)
                    for i, data in enumerate(output_image_data)
                ]
                reference_images = [
                    wandb.Image(data, caption=captions[i])
                    for i, data in enumerate(reference_image_data)
                ]
                return list(chain.from_iterable(zip(output_images, reference_images)))
        elif self.input_type in ("image", "images", "segmentation_mask"):
            print("input type is image")
            input_image_data = (
                self._masks_to_pixels(test_data)
                if self.input_type == "segmentation_mask"
                else test_data
            )
            if self.output_type == "label":
                print("output type is label")
                # we just use the predicted label as the caption for now
                captions = self._logits_to_captions(predictions)
                for i, data in enumerate(test_data):
                    print("caption[{i}]={c}".format(
                        i=i, c=captions[i]
                    ))
                return [
                    wandb.Image(data, caption=captions[i])
                    for i, data in enumerate(test_data)
                ]
            elif self.output_type in ("image", "images", "segmentation_mask"):
                print("output type is image")
                output_image_data = (
                    self._masks_to_pixels(predictions)
                    if self.output_type == "segmentation_mask"
                    else predictions
                )
                reference_image_data = (
                    self._masks_to_pixels(test_output)
                    if self.output_type == "segmentation_mask"
                    else test_output
                )
                input_images = [
                    wandb.Image(data, grouping=3)
                    for i, data in enumerate(input_image_data)
                ]
                output_images = [
                    wandb.Image(data) for i, data in enumerate(output_image_data)
                ]
                reference_images = [
                    wandb.Image(data) for i, data in enumerate(reference_image_data)
                ]
                return list(
                    chain.from_iterable(
                        zip(input_images, output_images, reference_images)
                    )
                )
            else:
                print("unknown output, just log the input images")
                # unknown output, just log the input images
                return [wandb.Image(img) for img in test_data]
        elif self.output_type in ("image", "images", "segmentation_mask"):
            print("unknown input, just log the predicted and reference outputs without captions")
            # unknown input, just log the predicted and reference outputs without captions
            output_image_data = (
                self._masks_to_pixels(predictions)
                if self.output_type == "segmentation_mask"
                else predictions
            )
            reference_image_data = (
                self._masks_to_pixels(test_output)
                if self.output_type == "segmentation_mask"
                else test_output
            )
            output_images = [
                wandb.Image(data, grouping=2)
                for i, data in enumerate(output_image_data)
            ]
            reference_images = [
                wandb.Image(data) for i, data in enumerate(reference_image_data)
            ]
            return list(chain.from_iterable(zip(output_images, reference_images)))
