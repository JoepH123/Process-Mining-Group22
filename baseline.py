import copy
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix as cm, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from predictors_columns import pipeline
import splitter, constants
from performance_measures import classification_performance, regression_performance, time_execution

def clean_mode(x):
    """Returns the first mode in the list of modes, or None if it's empty.

    :param x: the list of modes
    :type x: list or type convertable with tolist()
    :return: The first element of x, if it exists, None otherwise
    :rtype: None or type of the first element in x
    """
    if x.tolist() == []:
        return None
    else:
        return x[0]
    
    
def train_baseline_model(train_data_in, timer):
    """Trains the baseline prediction model. It uses the current case position
    as input and predicts the next case position and time until next position.
    For a given case position, the most common next position in the dataset
    is predicted as "next position". The time until next position is predicted as
    the simple average of the time for all cases in the dataset with the given
    case position to move to a new case position. Also print the accuracy and
    mean absolute error of the prediction.

    :param train_data_in: The dataframe containing the training set,
        with the actual values inside 'next event' and 'time until next event'
    :type train_data_in: DataFrame
    :return: A tuple containing a DataFrame predicting the next event, and a
        DataFrame predicting the time until next event
    :rtype: Tuple of 2 DataFrames
    """
    # copy so we don't modify the original training set
    train_data = copy.deepcopy(train_data_in)

    timer.send("Time to deepcopy (in seconds): ")

    # get the most frequent event for each case position
    next_event_df = pd.DataFrame(train_data.groupby(constants.CASE_STEP_NUMBER_COLUMN)[
        constants.NEXT_EVENT].agg(lambda x: clean_mode(pd.Series.mode(x))))
    next_event_df.rename(
        columns={constants.NEXT_EVENT: constants.NEXT_EVENT_PREDICTION}, inplace=True)
    train_data = train_data.merge(
        next_event_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)

    timer.send("Time to train baseline model classification (in seconds): ")

    # get the average of the time elapsed between the events at case positions i and i+1
    time_elapsed_df = pd.DataFrame(train_data.groupby(constants.CASE_STEP_NUMBER_COLUMN)[
        constants.TIME_DIFFERENCE].agg('mean'))
    time_elapsed_df.rename(
        columns={constants.TIME_DIFFERENCE: constants.TIME_DIFFERENCE_PREDICTION}, inplace=True)
    train_data = train_data.merge(
        time_elapsed_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)

    # this datafarame stores all the rows where time until next event and next event are null
    # we should decide what to do with it, for now i remove missing values
    exp_df = train_data[train_data[constants.TIME_DIFFERENCE].isnull()]
    train_data.dropna(inplace=True)

    # calculate performance for train set
    print("-------------------------------------------------------------------")
    print('Train set:')
    classification_performance(train_data, 'Confusion_Matrices/conf_matrix_baseline_train.png')
    regression_performance(train_data)
    print("-------------------------------------------------------------------")
    timer.send("Time to train baseline model time (in seconds): ")
    return next_event_df, time_elapsed_df


    
def evaluate_baseline_model(baseline_next_event_df, baseline_time_elapsed_df, test_data, timer):
    # make predictions for test set
    test_data = test_data.merge(
        baseline_next_event_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)
    test_data = test_data.merge(
        baseline_time_elapsed_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)
    
    # also remove missing values:
    test_data.dropna(inplace=True)

    # calculate performance for test set
    print("-------------------------------------------------------------------")
    print('Test set:')
    timer.send("Time to prepare for evaluation baseline model (in seconds): ")

    classification_performance(test_data, 'Confusion_Matrices/conf_matrix_baseline_test.png')
    timer.send("Time to evaluate baseline model classification (in seconds): ")
    regression_performance(test_data)
    timer.send("Time to evaluate baseline model (in seconds): ")
    print("-------------------------------------------------------------------")

    


