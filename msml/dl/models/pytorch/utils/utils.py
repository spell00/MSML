import io
import itertools
import math
import os
from itertools import cycle

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
from matplotlib import pyplot as plt, cm
from sklearn import metrics
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay


def get_optimizer(model, learning_rate, weight_decay, optimizer_type, momentum=0.9):
    """

    Args:
        model: The PyTorch model to optimize
        learning_rate: The optimizer's learning rate
        weight_decay: The optimizer's weight decay
        optimizer_type: The optimizer's type [adam or sgd]
        momentum:

    Returns:

    """
    # TODO Add more optimizers
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay,
                                     )
    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum
                                    )
    return optimizer


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


def get_labels(fname):
    meta = pd.read_excel(fname, header=0)
    toremove = pd.isnull(meta.values[:, 0])
    tokeep = [i for i, x in enumerate(toremove) if x == 0]

    meta = meta.iloc[tokeep, :]
    samples_classes = meta['Pathological type']
    classes = np.unique(samples_classes)

    return classes, samples_classes


def to_categorical(y, num_classes, dtype=torch.int):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes, dtype=dtype)[y]


def split_labels_indices(labels, train_inds):
    train_indices = []
    test_indices = []
    for j, sample in enumerate(list(labels)):
        if sample in train_inds:
            train_indices += [j]
        else:
            test_indices += [j]

    assert len(test_indices) != 0
    assert len(train_indices) != 0

    return train_indices, test_indices


def split_train_test(labels):
    from sklearn.model_selection import StratifiedKFold
    # First, get all unique samples and their category
    unique_samples = []
    unique_cats = []
    for sample, cat in zip(labels['sample'], labels['category']):
        if sample not in unique_samples:
            unique_samples += [sample]
            unique_cats += [cat]

    # StratifiedKFold with n_splits of 5 to ranmdomly split 80/20.
    # Used only once for train/test split.
    # The train split needs to be split again into train/valid sets later
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    train_inds, test_inds = next(skf.split(unique_samples, unique_cats))

    # After the samples are split, we get the duplicates of all samples.
    train_samples, test_samples = [unique_samples[s] for s in train_inds], [unique_samples[s] for s in test_inds]
    train_cats = [unique_cats[ind] for ind in train_inds]

    assert len(unique_samples) == len(train_inds) + len(test_inds)
    assert len([x for x in test_inds if x in train_inds]) == 0
    assert len([x for x in test_samples if x in train_samples]) == 0
    assert len(np.unique([unique_cats[ind] for ind in test_samples])) > 1

    return train_samples, test_samples, train_cats


def get_scaler_from_string(scaler_str):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
    if str(scaler_str) == 'normalizer':
        scaler = Normalizer
    elif str(scaler_str) == 'standard':
        scaler = StandardScaler
    elif str(scaler_str) == 'minmax':
        scaler = MinMaxScaler
    elif str(scaler_str) == "robust":
        scaler = RobustScaler
    else:
        exit(f"Invalid scaler {scaler_str}")
    return scaler


def plot_confusion_matrix(cm, class_names, acc):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (Acc: {np.mean(acc)})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(logger, epoch, lists, unique_labels, traces):
    for values in lists:
        # Calculate the confusion matrix.
        preds = np.concatenate(lists[values]['preds']).argmax(1)
        classes = np.concatenate(lists[values]['classes'])
        cm = sklearn.metrics.confusion_matrix(classes, preds)
        figure = plot_confusion_matrix(cm, class_names=unique_labels, acc=traces[values]['acc'])
        logger.add_figure(f"CM_{values}_all", figure, epoch)
        for conc in lists[values]['concs'].keys():
            inds = torch.concat(lists[values]['concs'][conc]).detach().cpu().numpy()
            inds = np.array([i for i, x in enumerate(inds) if x > -1])
            cm = sklearn.metrics.confusion_matrix(classes[inds], preds[inds])
            figure = plot_confusion_matrix(cm, class_names=unique_labels, acc=traces[values][f'acc_{conc}'])
            logger.add_figure(f"CM_{values}_{conc}", figure, epoch)
        del cm, figure


def save_roc_curve(model, x_test, y_test, unique_labels, name, binary, acc, epoch=None, logger=None):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        roc_score = roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC.png')
                stuck = False
            except:
                print('stuck...')
    else:
        # Compute ROC curve and ROC area for each class
        from sklearn.preprocessing import label_binarize
        y_pred_proba = model.predict_proba(x_test)
        y_preds = model.predict(x_test)
        n_classes = len(unique_labels)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes = np.arange(len(unique_labels))

        bin_label = label_binarize(y_test, classes=classes)
        roc_score = roc_auc_score(y_true=label_binarize(y_test, classes=classes[bin_label.sum(0) != 0]),
                                  y_score=label_binarize(y_preds, classes=classes[bin_label.sum(0) != 0]),
                                  multi_class='ovr')
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(label_binarize(y_test, classes=classes)[:, i], y_pred_proba[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # roc for each class
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title(f'ROC curve (AUC={np.round(roc_score, 3)}, acc={np.round(acc, 3)})')
        # ax.plot(fpr[0], tpr[0], label=f'AUC = {np.round(roc_score, 3)} (All)', color='k')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'AUC = {np.round(roc_auc[i], 3)} ({unique_labels[i]})')
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        # sns.despine()
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC.png')
                stuck = False
            except:
                print('stuck...')

        if logger is not None:
            logger.add_figure(name, fig, epoch)

    plt.close()

    return roc_score


def save_precision_recall_curve(model, x_test, y_test, unique_labels, name, binary, acc, epoch=None, logger=None):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
        aps = metrics.average_precision_score(fpr, tpr)
        # aps = metrics.roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='Precision-Recall curve (average precision score = %0.2f)' % aps)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (True Positive Rate) (TP/P)')
        plt.ylabel('Precision (TP/PP)')
        plt.title(f'Precision-Recall curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/Precision-Recall.png')
                stuck = False
            except:
                print('stuck...')
    else:
        # Compute Precision-Recall curve and Precision-Recall area for each class
        from sklearn.preprocessing import label_binarize
        y_pred_proba = model.predict_proba(x_test)
        y_preds = model.predict(x_test)
        n_classes = len(unique_labels)
        precisions = dict()
        recalls = dict()
        average_precision = dict()
        classes = np.arange(len(unique_labels))
        y_test_bin = label_binarize(y_test, classes=classes)
        for i in range(n_classes):
            y_true = y_test_bin[:, i]
            precisions[i], recalls[i], _ = metrics.precision_recall_curve(y_true, y_pred_proba[:, i], pos_label=1)
            average_precision[i] = metrics.average_precision_score(y_true=y_true, y_score=y_pred_proba[:, i])
        # roc for each class
        fig, ax = plt.subplots()
        # A "micro-average": quantifying score on all classes jointly
        precisions["micro"], recalls["micro"], _ = metrics.precision_recall_curve(
            y_test_bin.ravel(), y_pred_proba.ravel()
        )
        average_precision["micro"] = metrics.average_precision_score(y_test_bin, y_pred_proba,
                                                                     average="micro")  # sns.despine()
        display = PrecisionRecallDisplay(
            recall=recalls["micro"],
            precision=precisions["micro"],
            average_precision=average_precision["micro"],
        )
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/PRC.png')
                stuck = False
            except:
                print('stuck...')
        display.plot()
        fig = display.figure_
        if logger is not None:
            logger.add_figure(name, fig, epoch)

        # setup plot details
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
        viridis = cm.get_cmap('viridis', 256)
        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recalls["micro"],
            precision=precisions["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(n_classes), viridis.colors):
            display = PrecisionRecallDisplay(
                recall=recalls[i],
                precision=precisions[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        fig = display.figure_
        plt.savefig(f'{dirs}/{name}_multiclass.png')
        if logger is not None:
            logger.add_figure(f'{name}_multiclass', fig, epoch)

    plt.close()

    # return pr_auc


def get_empty_dicts():
    values = {
        "rec_loss": [],
        "dom_loss": [],
        "dom_acc": [],
        "set_batch_metrics": {
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
        },
        "train": {
            "closs": [],
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "kld": [],
            "acc": [],
            'acc_l': [],
            'acc_h': [],
            'acc_v': [],
            "mcc": [],
            'mcc_l': [],
            'mcc_h': [],
            'mcc_v': []
        },
        "valid": {
            "closs": [],
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "kld": [],
            "acc": [],
            'acc_l': [],
            'acc_h': [],
            'acc_v': [],
            "mcc": [],
            'mcc_l': [],
            'mcc_h': [],
            'mcc_v': []
        },
        "test": {
            "closs": [],
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "kld": [],
            "acc": [],
            'acc_l': [],
            'acc_h': [],
            'acc_v': [],
            "mcc": [],
            'mcc_l': [],
            'mcc_h': [],
            'mcc_v': []
        },
    }
    best_values = {
        'rec_loss': 100,
        'dom_loss': 100,
        'dom_acc': 0,
        'train_loss': 100,
        'valid_loss': 100,
        'test_loss': 100,
        'train_acc': 0,
        'valid_acc': 0,
        'test_acc': 0,
        'train_acc_l': 0,
        'train_acc_h': 0,
        'train_acc_v': 0,
        'valid_acc_l': 0,
        'valid_acc_h': 0,
        'valid_acc_v': 0,
        'test_acc_l': 0,
        'test_acc_h': 0,
        'test_acc_v': 0,
        'train_mcc': 0,
        'valid_mcc': 0,
        'test_mcc': 0,
        'train_mcc_l': 0,
        'train_mcc_h': 0,
        'train_mcc_v': 0,
        'valid_mcc_l': 0,
        'valid_mcc_h': 0,
        'valid_mcc_v': 0,
        'test_mcc_l': 0,
        'test_mcc_h': 0,
        'test_mcc_v': 0,
    }
    best_lists = {
        'train': {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'classes': [],
            'domains': [],
            'labels': [],
            'encoded_values': [],
            'inputs': [],
            'rec_values': [],
            'concs': {'l': [], 'h': [], 'v': []},
        },
        'valid': {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'classes': [],
            'domains': [],
            'labels': [],
            'encoded_values': [],
            'rec_values': [],
            'inputs': [],
            'concs': {'l': [], 'h': [], 'v': []},
        },
        'test': {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'classes': [],
            'labels': [],
            'domains': [],
            'encoded_values': [],
            'rec_values': [],
            'inputs': [],
            'concs': {'l': [], 'h': [], 'v': []},
        }
    }
    best_traces = {
        'rec_loss': [1000],
        'dom_loss': [1000],
        'dom_acc': [0],
        'train': {
            'closs': [100],
            'acc': [0],
            'acc_l': [0],
            'acc_h': [0],
            'acc_v': [0],
            'mcc': [0],
            'mcc_l': [0],
            'mcc_h': [0],
            'mcc_v': [0],
        },
        'valid': {
            'closs': [100],
            'acc': [0],
            'acc_l': [0],
            'acc_h': [0],
            'acc_v': [0],
            'mcc': [0],
            'mcc_l': [0],
            'mcc_h': [0],
            'mcc_v': [0],
        },
        'test': {
            'closs': [100],
            'acc': [0],
            'acc_l': [0],
            'acc_h': [0],
            'acc_v': [0],
            'mcc': [0],
            'mcc_l': [0],
            'mcc_h': [0],
            'mcc_v': [0],
        },
    }

    return values, best_values, best_lists, best_traces


def get_empty_traces(subsets):
    traces = {
        'rec_loss': [],
        'dom_loss': [0],
        'dom_acc': [0],
    }
    lists = {}
    lists = {
        'train': {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'classes': [],
            'domains': [],
            'labels': [],
            'encoded_values': [],
            'inputs': [],
            'rec_values': [],
            'concs': {'l': [], 'h': [], 'v': []},
        },
        'valid': {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'classes': [],
            'domains': [],
            'labels': [],
            'encoded_values': [],
            'inputs': [],
            'rec_values': [],
            'concs': {'l': [], 'h': [], 'v': []},
        },
        'test': {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'classes': [],
            'labels': [],
            'domains': [],
            'encoded_values': [],
            'inputs': [],
            'rec_values': [],
            'concs': {'l': [], 'h': [], 'v': []},
        }
    }
    for g in ['train', 'valid', 'test']:
        lists[g] = {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'classes': [],
            'domains': [],
            'labels': [],
            'encoded_values': [],
            'inputs': [],
            'rec_values': [],
            'concs': {s: [] for s in subsets},
        }
        traces[g] = {
            'closs': [],
            'acc': [],
            'mcc': None,
        }
        for s in subsets:
            traces[g][f'acc_{s}'] = []
            traces[g][f'mcc_{s}'] = []

    return lists, traces


def log_traces(traces, values, subsets):
    values['dom_loss'] += [np.mean(traces['dom_loss'])]
    values['dom_acc'] += [np.mean(traces['dom_acc'])]
    values['rec_loss'] += [np.mean(traces['rec_loss'])]

    for g in list(traces.keys())[3:]:
        values[g]['closs'] += [np.mean(traces[g]['closs'])]
        values[g]['acc'] += [np.mean(traces[g]['acc'])]
        values[g]['mcc'] += [traces[g]['mcc']]
        for s in subsets:
            values[g][f'acc_{s}'] += [np.mean(traces[g][f'acc_{s}'])]
            values[g][f'mcc_{s}'] += [traces[g][f'mcc_{s}']]

    return values


def get_best_values_from_tb(event_acc):
    values = {}
    # plugin_data = event_acc.summary_metadata['_hparams_/session_start_info'].plugin_data.content
    # plugin_data = plugin_data_pb2.HParamsPluginData.FromString(plugin_data)
    # for name in event_acc.summary_metadata.keys():
    #     if name not in ['_hparams_/experiment', '_hparams_/session_start_info']:
    best_closs = event_acc.Tensors('valid/loss')
    best_closs = tf.make_ndarray(best_closs[0].tensor_proto).item()

    return best_closs


def get_best_values(values, ae_only, subs):
    if ae_only:
        best_values = {
            'rec_loss': np.mean(values['rec_loss']),
            'dom_loss': np.mean(values['dom_loss']),
            'dom_acc': np.mean(values['dom_acc']),
            'train_loss': math.nan,
            'valid_loss': math.nan,
            'test_loss': math.nan,
            'train_acc': math.nan,
            'valid_acc': math.nan,
            'test_acc': math.nan,
            'train_acc_l': math.nan,
            'train_acc_h': math.nan,
            'train_acc_v': math.nan,
            'valid_acc_l': math.nan,
            'valid_acc_h': math.nan,
            'valid_acc_v': math.nan,
            'test_acc_l': math.nan,
            'test_acc_h': math.nan,
            'test_acc_v': math.nan,
            'train_mcc': math.nan,
            'valid_mcc': math.nan,
            'test_mcc': math.nan,
            'train_mcc_l': math.nan,
            'train_mcc_h': math.nan,
            'train_mcc_v': math.nan,
            'valid_mcc_l': math.nan,
            'valid_mcc_h': math.nan,
            'valid_mcc_v': math.nan,
            'test_mcc_l': math.nan,
            'test_mcc_h': math.nan,
            'test_mcc_v': math.nan,
        }

    else:
        best_values = {
            'dom_loss': values['dom_loss'][-1],
            'dom_acc': values['dom_acc'][-1],
            'rec_loss': values['rec_loss'][-1],
        }
        for g in ['train', 'valid', 'test']:
            for k in ['acc', 'mcc']:
                best_values[f'{g}_{k}'] = values[g][k][-1]
                for sub in subs:
                    best_values[f'{g}_{k}_{sub}'] = values[g][f'{k}_{sub}'][-1]
                best_values[f'{g}_loss'] = values[g]['closs'][-1]

    return best_values


def add_to_logger(values, logger, epoch):
    if not np.isnan(values['rec_loss'][-1]):
        logger.add_scalar(f'rec_loss', values['rec_loss'][-1], epoch)
        logger.add_scalar(f'dom_loss', values['dom_loss'][-1], epoch)
        logger.add_scalar(f'dom_acc', values['dom_acc'][-1], epoch)
    for group in list(values.keys())[4:]:
        logger.add_scalar(f'/closs/{group}', values[group]['closs'][-1], epoch)
        logger.add_scalar(f'/acc/{group}/all_concentrations', values[group]['acc'][-1], epoch)
        logger.add_scalar(f'/acc/{group}/lows', values[group]['acc_l'][-1], epoch)
        logger.add_scalar(f'/acc/{group}/vhighs', values[group]['acc_v'][-1], epoch)
        logger.add_scalar(f'/acc/{group}/high', values[group]['acc_h'][-1], epoch)


def count_labels(arr):
    """
    Counts elements in array

    :param arr:
    :return:
    """
    elements_count = {}
    for element in arr:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1
    to_remove = []
    for key, value in elements_count.items():
        print(f"{key}: {value}")
        if value <= 2:
            to_remove += [key]

    return to_remove