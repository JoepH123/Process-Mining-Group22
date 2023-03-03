
import copy
import time
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
import splitter, constants

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

def time_execution():
    """A couroutine that prints a message it recieves through .send()
    and the the seconds passed since the last time it was called.
    """
    start_time = time.process_time()
    while True:
        string = (yield)
        new_time = time.process_time()
        print(string, new_time - start_time)
        start_time = new_time



def main():
    # set up the timer
    timer = time_execution()
    timer.__next__()

    # do this if the files are not split already
    # splitter.convert_raw_dataset(constants.RAW_DATASET_PATH, constants.CONVERTED_DATASET_PATH)
    # timer.send("Time to convert dataset (in seconds): ")

    full_data = pd.read_csv(constants.CONVERTED_DATASET_PATH)

    train_data, test_data = splitter.split_dataset(full_data, 0.2)

    timer.send("Time to split dataset (in seconds): ")

    baseline_next_event_df, baseline_time_elapsed_df = train_baseline_model(
        train_data)

    timer.send("Time to train baseline model (in seconds): ")

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

    timer.send("Time to predict baseline model (in seconds): ")

if __name__ == "__main__":
    main()
