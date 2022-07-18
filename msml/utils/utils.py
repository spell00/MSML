import matplotlib.pyplot as plt
import numpy as np
import itertools


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


