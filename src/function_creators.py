from tensorflow.python.data import Dataset
from numpy import array
from numpy.random import permutation
from data_loader import MODKEY

def parse_labels_and_features(dataset):
    """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
    dataset: A Pandas `Dataframe`, containing the label on the first column and
      monochrome pixel values on the remaining columns, in row major order.
    Returns:
    A `tuple` `(labels, features)`:
      labels: A Pandas `Series`.
      features: A Pandas `DataFrame`.
    """
    labels = dataset[MODKEY]
    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset.iloc[:,1:]
    return labels, features

def create_predict_input_fn(features, labels, batch_size, repeat_count = 1):
    """
    Args:
    features: The features to base predictions on.
    labels: The labels of the prediction examples.

    Returns:
    A function that returns features and labels for predictions.
    """
    def _input_fn():
        raw_features = {"features": features.values}
        raw_targets = array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets)) # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(repeat_count)


        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn

def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True, repeat_count=1):
    """A custom input_fn for sending MNIST data to the estimator for training.

    Args:
    features: The training features.
    labels: The training labels.
    batch_size: Batch size to use during training.

    Returns:
    A function that returns batches of training features and labels during
    training.
    """
    def _input_fn(num_epochs=None, shuffle=True):
        # Input pipelines are reset with each call to .train(). To ensure model
        # gets a good sampling of data, even when number of steps is small, we
        # shuffle all the data before creating the Dataset object
        idx = permutation(features.index)
        raw_features = {"features":features.reindex(idx)}
        raw_targets = array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features,raw_targets)) # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(repeat_count)

        if shuffle:
          ds = ds.shuffle(10000)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn
