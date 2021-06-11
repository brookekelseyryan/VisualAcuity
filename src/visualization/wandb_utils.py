# @title Double click to see the code
import numpy as np
import wandb
from wandb.keras import WandbCallback
from datetime import datetime

from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def log_model_params(model, wandb_config, args):
    """
    NOT USED CURRENTLY
    Extract params of interest about the model (e.g. number of different layer types).
    Log these and any experiment-level settings to visualization.
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


def init_wandb(argv, sync_tensorboard=True):
    """
    Initializes the visualization config yaml file and run name variables.
    :param argv: One command-line argument that is just the run name. Optional.
    :param sync_tensorboard: WandB parameter that enables Tensorboard to be tracked.
    :return:
    """
    wandb.init(config="/home/brooker/VisualAcuity/src/visualization/config-defaults.yaml", project="Visual_Acuity", sync_tensorboard=sync_tensorboard)

    run_name = ""

    if len(argv) > 1:
        run_name = argv[1]

    wandb.run.name = run_name + datetime.now().strftime(" %H:%M:%S, %m/%d/%Y, id= ") + wandb.run.id


class WandbClassificationCallback(WandbCallback):
    """
    Arguments:
        monitor (str): name of metric to monitor.  Defaults to val_loss.
        mode (str): one of {"auto", "min", "max"}.
            "min" - save model when monitor is minimized
            "max" - save model when monitor is maximized
            "auto" - try to guess when to save the model (default).
        save_model:
            True - save a model when monitor beats all previous epochs
            False - don't save models
        save_graph: (boolean): if True save model graph to wandb (default: True).
        save_weights_only (boolean): if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        log_weights: (boolean) if True save histograms of the model's layer's weights.
        log_gradients: (boolean) if True log histograms of the training gradients
        training_data: (tuple) Same format (X,y) as passed to model.fit.  This is needed
            for calculating gradients - this is mandatory if `log_gradients` is `True`.
        validate_data: (tuple) Same format (X,y) as passed to model.fit.  A set of data
            for wandb to visualize.  If this is set, every epoch, wandb will
            make a small number of predictions and save the results for later visualization.
        generator (generator): a generator that returns validation data for wandb to visualize.  This
            generator should return tuples (X,y).  Either validate_data or generator should
            be set for wandb to visualize specific data examples.
        validation_steps (int): if `validation_data` is a generator, how many
            steps to run the generator for the full validation set.
        labels (list): If you are visualizing your data with wandb this list of labels
            will convert numeric output to understandable string if you are building a
            multiclass classifier.  If you are making a binary classifier you can pass in
            a list of two labels ["label for false", "label for true"].  If validate_data
            and generator are both false, this won't do anything.
        predictions (int): the number of predictions to make for visualization each epoch, max
            is 100.
        input_type (string): type of the model input to help visualization. can be one of:
            ("image", "images", "segmentation_mask").
        output_type (string): type of the model output to help visualziation. can be one of:
            ("image", "images", "segmentation_mask").
        log_evaluation (boolean): if True, save a Table containing validation data and the
            model's preditions at each epoch. See `validation_indexes`,
            `validation_row_processor`, and `output_row_processor` for additional details.
        class_colors ([float, float, float]): if the input or output is a segmentation mask,
            an array containing an rgb tuple (range 0-1) for each class.
        log_batch_frequency (integer): if None, callback will log every epoch.
            If set to integer, callback will log training metrics every log_batch_frequency
            batches.
        log_best_prefix (string): if None, no extra summary metrics will be saved.
            If set to a string, the monitored metric and epoch will be prepended with this value
            and stored as summary metrics.
        validation_indexes ([wandb.data_types._TableLinkMixin]): an ordered list of index keys to associate
            with each validation example.  If log_evaluation is True and `validation_indexes` is provided,
            then a Table of validation data will not be created and instead each prediction will
            be associated with the row represented by the TableLinkMixin. The most common way to obtain
            such keys are is use Table.get_index() which will return a list of row keys.
        validation_row_processor (Callable): a function to apply to the validation data, commonly used to visualize the data.
            The function will receive an ndx (int) and a row (dict). If your model has a single input,
            then row["input"] will be the input data for the row. Else, it will be keyed based on the name of the
            input slot. If your fit function takes a single target, then row["target"] will be the target data for the row. Else,
            it will be keyed based on the name of the output slots. For example, if your input data is a single ndarray,
            but you wish to visualize the data as an Image, then you can provide `lambda ndx, row: {"img": wandb.Image(row["input"])}`
            as the processor. Ignored if log_evaluation is False or `validation_indexes` are present.
        output_row_processor (Callable): same as validation_row_processor, but applied to the model's output. `row["output"]` will contain
            the results of the model output.
        infer_missing_processors (bool): Determines if validation_row_processor and output_row_processor
            should be inferred if missing. Defaults to True. If `labels` are provided, we will attempt to infer classification-type
            processors where appropriate.
    """

    def __init__(self,
                 monitor='val_loss',
                 verbose=0,
                 mode='auto',
                 save_weights_only=False,
                 log_weights=False,
                 log_gradients=False,
                 save_model=True,
                 training_data=None,
                 validation_data=None,
                 labels=[],
                 data_type=None,
                 predictions=1,
                 generator=None,
                 input_type=None,
                 output_type=None,
                 log_evaluation=False,
                 validation_steps=None,
                 class_colors=None,
                 log_batch_frequency=None,
                 log_best_prefix="best_",
                 log_confusion_matrix=False,
                 confusion_examples=0,
                 confusion_classes=5):

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
                         predictions=predictions,
                         generator=generator,
                         input_type=input_type,
                         output_type=output_type,
                         log_evaluation=log_evaluation,
                         validation_steps=validation_steps,
                         class_colors=class_colors,
                         log_batch_frequency=log_batch_frequency,
                         log_best_prefix=log_best_prefix)

        self.log_confusion_matrix = log_confusion_matrix
        self.confusion_examples = confusion_examples
        self.confusion_classes = confusion_classes

    def on_epoch_end(self, epoch, logs={}):
        if self.generator:
            self.validation_data = next(self.generator)

        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if self.log_confusion_matrix:
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log(self._log_confusion_matrix(), commit=False)

        if self.input_type in ("image", "images", "segmentation_mask") or self.output_type in (
        "image", "images", "segmentation_mask"):
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                if self.confusion_examples > 0:
                    wandb.log({'confusion_examples': self._log_confusion_examples(
                        confusion_classes=self.confusion_classes,
                        max_confused_examples=self.confusion_examples)}, commit=False)
                if self.predictions > 0:
                    wandb.log({"examples": self._log_images(
                        num_images=self.predictions)}, commit=False)

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary["%s%s" % (self.log_best_prefix, self.monitor)] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f' % (
                        epoch, self.monitor, self.best, self.current))
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    def _log_confusion_matrix(self):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
        confdiag = np.eye(len(confmatrix)) * confmatrix
        np.fill_diagonal(confmatrix, 0)

        confmatrix = confmatrix.astype('float')
        n_confused = np.sum(confmatrix)
        confmatrix[confmatrix == 0] = np.nan
        confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': self.labels, 'y': self.labels, 'z': confmatrix,
                                 'hoverongaps': False,
                                 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

        confdiag = confdiag.astype('float')
        n_right = np.sum(confdiag)
        confdiag[confdiag == 0] = np.nan
        confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': self.labels, 'y': self.labels, 'z': confdiag,
                               'hoverongaps': False,
                               'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})

        fig = go.Figure((confdiag, confmatrix))
        transparent = 'rgba(0, 0, 0, 0)'
        n_total = n_right + n_confused
        fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1,
                                                                                                          f'rgba(180, 0, 0, {max(0.2, (n_confused / n_total) ** 0.5)})']],
                                          'showscale': False}})
        fig.update_layout({'coloraxis2': {
            'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right / n_total) ** 2)})'],
                           [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

        xaxis = {'title': {'text': 'y_true'}, 'showticklabels': False}
        yaxis = {'title': {'text': 'y_pred'}, 'showticklabels': False}

        fig.update_layout(title={'text': 'Confusion matrix', 'x': 0.5}, paper_bgcolor=transparent,
                          plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

        return {'confusion_matrix': wandb.data_types.Plotly(fig)}

    def _log_confusion_examples(self, rescale=255, confusion_classes=5, max_confused_examples=3):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        # Grayscale to rgb
        if x_val.shape[-1] == 1:
            x_val = np.concatenate((x_val, x_val, x_val), axis=-1)

        confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
        np.fill_diagonal(confmatrix, 0)

        def example_image(class_index, x_val=x_val, y_pred=y_pred, y_val=y_val, labels=self.labels, rescale=rescale):
            image = None
            title_text = 'No example found'
            color = 'red'

            right_predicted_images = x_val[np.logical_and(y_pred == class_index, y_val == class_index)]
            if len(right_predicted_images) > 0:
                image = rescale * right_predicted_images[0]
                title_text = 'Predicted right'
                color = 'rgb(46, 184, 46)'
            else:
                ground_truth_images = x_val[y_val == class_index]
                if len(ground_truth_images) > 0:
                    image = rescale * ground_truth_images[0]
                    title_text = 'Example'
                    color = 'rgb(255, 204, 0)'

            return image, title_text, color

        n_cols = max_confused_examples + 2
        subplot_titles = [""] * n_cols
        subplot_titles[-2:] = ["y_true", "y_pred"]
        subplot_titles[max_confused_examples // 2] = "confused_predictions"

        n_rows = min(len(confmatrix[confmatrix > 0]), confusion_classes)
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
        for class_rank in range(1, n_rows + 1):
            indx = np.argmax(confmatrix)
            indx = np.unravel_index(indx, shape=confmatrix.shape)
            if confmatrix[indx] == 0:
                break
            confmatrix[indx] = 0

            class_pred, class_true = indx[0], indx[1]
            mask = np.logical_and(y_pred == class_pred, y_val == class_true)
            confused_images = x_val[mask]

            # Confused images
            n_images_confused = min(max_confused_examples, len(confused_images))
            for j in range(n_images_confused):
                fig.add_trace(go.Image(z=rescale * confused_images[j],
                                       name=f'Predicted: {self.labels[class_pred]} | Instead of: {self.labels[class_true]}',
                                       hoverinfo='name', hoverlabel={'namelength': -1}),
                              row=class_rank, col=j + 1)
                fig.update_xaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j + 1, mirror=True)
                fig.update_yaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j + 1, mirror=True)

            # Comparison images
            for i, class_index in enumerate((class_true, class_pred)):
                col = n_images_confused + i + 1
                image, title_text, color = example_image(class_index)
                fig.add_trace(
                    go.Image(z=image, name=self.labels[class_index], hoverinfo='name', hoverlabel={'namelength': -1}),
                    row=class_rank, col=col)
                fig.update_xaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True,
                                 title_text=title_text)
                fig.update_yaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True,
                                 title_text=self.labels[class_index])

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return wandb.data_types.Plotly(fig)
