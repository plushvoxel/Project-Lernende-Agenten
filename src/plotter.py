from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import newaxis

def plot_log_loss(training_errors, validation_errors, filename, show=False):
    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    if filename:
        plt.savefig(filename)
        plt.close(filename)
    if show:
        plt.show()
    plt.close()

def plot_confusion_matrix(test_targets, final_predictions, filename, show=False):
    con_mat = confusion_matrix(test_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class).
    con_mat_normalized = con_mat.astype("float") / con_mat.sum(axis=1)[:, newaxis]
    ax = sns.heatmap(con_mat_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if filename:
        plt.savefig(filename)
        plt.close(filename)
    if show:
        plt.show()
    plt.close()
