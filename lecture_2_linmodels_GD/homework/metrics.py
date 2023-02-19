import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = (y_pred + y_true == 2).sum()
    fp = (y_pred - y_true == 1).sum()
    fn = (y_pred - y_true == -1).sum()
    tn = (y_pred + y_true == 0).sum()

    precision = tp/(fp + tp)
    recall = tp/(tp+fn)
    f1 = 2/(precision**(-1) + recall**(-1))
    accuracy = (tp + tn)/(tp+tn+fp+fn)

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    return (y_pred == y_true).sum()/len(y_pred)


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - ((y_pred - y_true)**2).sum()/((y_true - y_true.mean())**2).sum()
    return r2

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = ((y_true - y_pred)**2).mean()
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = (abs(y_true - y_pred)).mean()
    return mae
    