
import copy
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
import splitter, constants

def clean_mode(x):
    """If there is no mode, return None. Otherwise, return the first mode in the list of modes.

    Args:
        x (object): List of modes

    Returns:
        object: The first mode in the list of modes, or None if there is no mode
    """
    if x.tolist() == []:
        return None
    else:
        return x[0]

def train_baseline_model(train_data_in):
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

    # get the most frequent event for each case position
    next_event_df = pd.DataFrame(train_data.groupby(constants.CASE_STEP_NUMBER_COLUMN)[
        'next event'].agg(lambda x: clean_mode(pd.Series.mode(x))))
    next_event_df.rename(
        columns={'next event': 'predicted next event'}, inplace=True)
    train_data = train_data.merge(
        next_event_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)

    # get the average of the time elapsed between the events at case positions i and i+1
    time_elapsed_df = pd.DataFrame(train_data.groupby(constants.CASE_STEP_NUMBER_COLUMN)[
        'time until next event'].agg('mean'))
    time_elapsed_df.rename(
        columns={'time until next event': 'predicted time until next event'}, inplace=True)
    train_data = train_data.merge(
        time_elapsed_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)

    # this datafarame stores all the rows where time until next event and next event are null
    # we should decide what to do with it, for now i remove missing values
    exp_df = train_data[train_data['time until next event'].isnull()]
    train_data.dropna(inplace=True)

    print("Accuracy score of baseline for train set (next event): ", accuracy_score(
        train_data['next event'], train_data['predicted next event']))
    print("Number of misclassifications: ", accuracy_score(
        train_data['next event'], train_data['predicted next event'], normalize=False))
    print("Mean absolute error of baseline for train set (time until next event): ", mean_absolute_error(
        train_data['time until next event'], train_data['predicted time until next event']))
    print("-------------------------------------------------------------------")
    return next_event_df, time_elapsed_df


def main():
    # do this if the files are not split already
    # splitter.split_dataset(0.2)
    
    train_data = pd.read_csv(constants.TRAINING_DATA_PATH)
    test_data = pd.read_csv(constants.TEST_DATA_PATH)

    baseline_next_event_df, baseline_time_elapsed_df = train_baseline_model(
        train_data)

    # make predictions for test set
    test_data = test_data.merge(
        baseline_next_event_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)
    test_data = test_data.merge(
        baseline_time_elapsed_df, how='left', on=constants.CASE_STEP_NUMBER_COLUMN)
    # also remove missing values:
    test_data.dropna(inplace=True)

    # calculate accuracy of the classification for test set
    print("Accuracy score of baseline for test set (next event): ", accuracy_score(
        test_data['next event'], test_data['predicted next event']))
    print("Number of misclassifications: ", accuracy_score(
        test_data['next event'], test_data['predicted next event'], normalize=False))
    print("Mean absolute error of baseline for test set (time until next event): ", mean_absolute_error(
        test_data['time until next event'], test_data['predicted time until next event']))


if __name__ == "__main__":
    main()
