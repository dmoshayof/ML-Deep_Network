import loglinear as ll
import utils as ut
import numpy as np
import random

STUDENT = {'name': 'Ariel Vetzler_Daniel Moshayof',
           'ID': '207458688_311126668'}
EPOCHS = 30
ETA = 0.4


# TODO
def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    # We will create a vector - Feature Extraction from Text - https://www.youtube.com/watch?v=7YacOe4XwhY
    feats_vec = np.zeros(len(ut.vocab))
    indices = [ut.F2I[f] if f in ut.vocab else -1 for f in features]
    for feature_key in indices:
        feats_vec[feature_key] += 1.

    feats_vec /= len(features)
    # Should return a numpy vector of features.
    return feats_vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)  # convert features to a vector.
        y_hat = ll.predict(x, params)
        if y_hat == ut.L2I[label]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = ut.L2I[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            # Teata = Theata  - learnning rate * gradient.
            params[0] = params[0] - learning_rate * grads[0]
            params[1] = params[1] - learning_rate * grads[1]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)

        print (I, train_loss,train_accuracy,dev_accuracy)
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    firstLayerNeuronsLength = len(ut.F2I)
    outputLayerNeuronsLength = len(ut.L2I)
    params = ll.create_classifier(firstLayerNeuronsLength, outputLayerNeuronsLength)
    trained_params = train_classifier(ut.TRAIN, ut.DEV, EPOCHS, ETA, params)
