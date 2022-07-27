import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer


def scale_data(scale, ncols, data):
    if scale == 'binarize':
        all_data = data['all'].iloc[:, :ncols]
        train_data = data['train'].iloc[:, :ncols]
        valid_data = data['valid'].iloc[:, :ncols]
        test_data = data['test'].iloc[:, :ncols]
        all_data[all_data > 0.] = 1
        train_data[train_data > 0.] = 1
        valid_data[valid_data > 0.] = 1
        test_data[test_data > 0.] = 1
    elif scale == 'robust':
        scaler = RobustScaler()

        all_data = scaler.fit_transform(data['all'].iloc[:, :ncols])
        train_data = scaler.transform(data['train'].iloc[:, :ncols])
        valid_data = scaler.transform(data['valid'].iloc[:, :ncols])
        test_data = scaler.transform(data['test'].iloc[:, :ncols])
    elif scale == 'standard':
        scaler = StandardScaler()

        all_data = scaler.fit_transform(data['all'].iloc[:, :ncols])
        train_data = scaler.transform(data['train'].iloc[:, :ncols])
        valid_data = scaler.transform(data['valid'].iloc[:, :ncols])
        test_data = scaler.transform(data['test'].iloc[:, :ncols])
    elif scale == 'l1':
        scaler = Normalizer(norm='l1')

        all_data = scaler.fit_transform(data['all'].iloc[:, :ncols])
        train_data = scaler.transform(data['train'].iloc[:, :ncols])
        valid_data = scaler.transform(data['valid'].iloc[:, :ncols])
        test_data = scaler.transform(data['test'].iloc[:, :ncols])
    elif scale == 'l2':
        scaler = Normalizer(norm='l2')

        all_data = scaler.fit_transform(data['all'].iloc[:, :ncols])
        train_data = scaler.transform(data['train'].iloc[:, :ncols])
        valid_data = scaler.transform(data['valid'].iloc[:, :ncols])
        test_data = scaler.transform(data['test'].iloc[:, :ncols])
    else:
        all_data = data['all'].iloc[:, :ncols]
        train_data = data['train'].iloc[:, :ncols]
        valid_data = data['valid'].iloc[:, :ncols]
        test_data = data['test'].iloc[:, :ncols]
    # if scale != 'none':
    scaler = MinMaxScaler()

    all_data = scaler.fit_transform(all_data)
    train_data = scaler.transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)

    return {'all': all_data, 'train': train_data, 'valid': valid_data, 'test': test_data}


def plot_confusion_matrix(cm, class_names, acc):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    cm_normal = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm_normal[np.isnan(cm_normal)] = 0
    plt.imshow(cm_normal, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (acc: {acc})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = 0.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_normal[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def get_unique_labels(labels):
    """
    Get unique labels for a set of labels
    :param labels:
    :return:
    """
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels += [label]
    return np.array(unique_labels)