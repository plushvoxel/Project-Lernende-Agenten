#!/usr/bin/env python3

from data_loader import load_data
from function_creators import *
from plotter import *

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
import shutil
import os
import time
import glob


CLASSES = ["data/fm.tar", "data/pager.tar", "data/smartwares.tar"]
num_classes = len(CLASSES)
num_features = 0

def train_nn_classification_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    test_examples,
    test_targets,
    line,
    time_string):
    """Trains a neural network classification model

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.

    Args:
        learning_rate: An `int`, the learning rate to use.
        steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        hidden_units: A `list` of int values, specifying the number of neurons in each layer.
        training_examples: A `DataFrame` containing the training features.
        training_targets: A `DataFrame` containing the training labels.
        validation_examples: A `DataFrame` containing the validation features.
        validation_targets: A `DataFrame` containing the validation labels.

    Returns:
        The trained `DNNClassifier` object.
    """

    periods = 10
    # Caution: input pipelines are reset with each call to train.
    # If the number of steps is small, your model may never see most of the data.
    # So with multiple `.train` calls like this you may want to control the length
    # of training with num_epochs passed to the input_fn. Or, you can do a really-big shuffle,
    # or since it's in-memory data, shuffle all the data in the `input_fn`.
    steps_per_period = steps / periods
    # Create the input functions.
    predict_training_input_fn     = create_predict_input_fn(training_examples     , training_targets  , batch_size)
    predict_validation_input_fn   = create_predict_input_fn(validation_examples   , validation_targets, batch_size)
    predict_test_input_fn         = create_predict_input_fn(test_examples         , test_targets      , batch_size)
    training_input_fn             = create_training_input_fn(training_examples    , training_targets  , batch_size)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column('features', shape=num_features)]

    # Create a DNNClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=num_classes,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=10, model_dir="models")
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []

    for period in range (0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        name= str(classifier.latest_checkpoint())
        oldfilemeta = name + ".meta"
        oldfiledata1 = name + ".data-00000-of-00002"
        oldfiledata2 = name + ".data-00001-of-00002"
        newfile = "saves/ckp"+str(period)+time_string
        shutil.copy(oldfilemeta, newfile+".meta")
        shutil.copy(oldfiledata1, newfile+".data1")
        shutil.copy(oldfiledata2, newfile+".data2")
        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,num_classes)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,num_classes)

        test_predictions = list(classifier.predict(input_fn=predict_test_input_fn))
        test_probabilities = np.array([item['probabilities'] for item in test_predictions])
        test_pred_class_id = np.array([item['class_ids'][0] for item in test_predictions])
        test_pred_one_hot = tf.keras.utils.to_categorical(test_pred_class_id,num_classes)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    model_general= []
    val_log_len = len(validation_errors)
    model_general.append(validation_errors[0]+ 2*abs(validation_errors[0]-validation_errors[1]))
    for index in range(1, val_log_len-1):
        model_general.append(validation_errors[index]+ abs(validation_errors[index]-validation_errors[index-1]) +  abs(validation_errors[index]-validation_errors[index+1]))
    model_general.append(validation_errors[val_log_len-1]+ 2*abs(validation_errors[val_log_len-1]-validation_errors[val_log_len-2]))
    index = np.argmin(model_general)
    savemodel_name = "ckp"+ str(index) + time_string
    shutil.copy("saves/"+savemodel_name+".meta", savemodel_name+".meta")
    shutil.copy("saves/"+savemodel_name+ ".data1", savemodel_name+".data1")
    shutil.copy("saves/"+savemodel_name+".data2", savemodel_name+".data2")
    shutil.rmtree('saves')
    os.makedirs("saves")
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_test_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])


    accuracy = metrics.accuracy_score(test_targets, final_predictions)
    print("Final accuracy (validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plot_log_loss(training_errors, validation_errors, "{}_logloss.png".format(line), show=False)
    # Output a plot of the confusion matrix.
    plot_confusion_matrix(test_targets, final_predictions, "{}_conmat.png".format(line), show=False)

    model = ""+str(hidden_units).strip('[]')+"]"
    print(model)
    print(learning_rate)
    print(training_examples.shape[0])
    print(validation_examples.shape[0])
    print(batch_size)
    print(accuracy)
    
    return classifier, accuracy

def train_automated(
    signal_dataframe,
    training_set_size,
    validating_set_size,
    test_set_size,
    learning_rate,
    steps,
    batch_size,
    model,
    line,
    time_string):
    """ Function used for automate the process of trying new network configurations

    Args:
    training_set_size: An 'int', number of samples used for training
    validating_set_size: An 'int', number of samples used for validation
    test_set_size: An 'int', number of samples used for test
    learning_rate: An `int`, the learning rate to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    model: A list of 'int', define the number of neurons in each hidden layer

    Returns:
    The trained `DNNClassifier` object.
    """
    activation_function = "RELU" #@param ["RELU", "Sigmoid", "Tanh"]
    regression = "None" #@param ["None", "L1", "L2"]
    regression_rate = 3 #@param ["3", "1", "0.3", "0.1", "0.03", "0.01", "0.003", "0.001"] {type:"raw"}
    training_targets, training_examples     = parse_labels_and_features(signal_dataframe[0:training_set_size])
    validation_targets, validation_examples = parse_labels_and_features(signal_dataframe[training_set_size:(training_set_size+validating_set_size)])
    test_targets, test_examples             = parse_labels_and_features(signal_dataframe[(training_set_size+validating_set_size):(training_set_size+validating_set_size+test_set_size)])
    nn_classification, accuracy = train_nn_classification_model(
        learning_rate=learning_rate,
        steps=steps,
        batch_size=batch_size,
        hidden_units=model,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets,
        test_examples=test_examples,
        test_targets=test_targets,
        line=line,
        time_string=time_string)
    return accuracy

def main():
    global num_features
    signal_dataframe = load_data(CLASSES)
    num_features = signal_dataframe.shape[1]-1
    num_samples = signal_dataframe.shape[0]
    learning_rate_steps = [0.1]
    data_set_distribution= [[60, 20, 20]]
    models = [[num_features, num_features//2] , [num_features, num_features//2, num_features//4]]
    batch_sizes=[1]
    time_string = time.strftime("%Y%m%d-%H%M%S")
    line=1
    shutil.rmtree("saves", ignore_errors=True)
    shutil.rmtree("models", ignore_errors=True)
    os.makedirs("saves")
    for model in models: #number of neurons per layer
        for learning_rate in learning_rate_steps: # learning_rate used for training
            for v in data_set_distribution: # try several dataset_distributions
                for batch_size in batch_sizes:
                    last_accuracy=max_accuracy=0
                    ckp_best_model=""
                    training_set_size = int(num_samples * (v[0]/100))
                    validating_set_size = int(num_samples * (v[1]/100))
                    test_set_size = int(num_samples * (v[2]/100))
                    for x in range(0,3):
                        last_accuracy=train_automated(
                            signal_dataframe,
                            training_set_size,
                            validating_set_size,
                            test_set_size,
                            learning_rate,
                            training_set_size//batch_size,
                            batch_size,
                            model,
                            line,
                            time_string)
                        line = line + 1
                        if max_accuracy < last_accuracy:
                            max_accuracy = last_accuracy
                            ckpname= str(model)+"-"+time_string + "-" + str(x)
                            os.rename("./models/checkpoint", ckpname)
                            if ckp_best_model != "":
                                os.remove(ckp_best_model)
                            ckp_best_model = ckpname
                        else:
                            os.remove("models/checkpoint")
                    line = line + 1

if __name__ == "__main__":
    main()
