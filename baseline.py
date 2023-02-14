
import copy
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

# ----------------------CONSTANTS----------------------------------
"""The path to the training set, as outputted by 'xes2csv.jar'."""
TRAINING_DATA_PATH = './Datasets/BPI_Challenge_2012-training.csv'
"""The path to the testing set, as outputted by 'xes2csv.jar'."""
TEST_DATA_PATH = './Datasets/BPI_Challenge_2012-test.csv'
"""The identifier of dataset column holding the case ID."""
CASE_ID_COLUMN = 'case concept:name'
"""The identifier of dataset column holding the case position/state."""
CASE_POSITION_COLUMN = 'event concept:name'
"""The identifier of dataset column holding the timestamp."""
CASE_TIMESTAMP_COLUMN = 'event time:timestamp'


def load_dataset():
    """Loads the dataset, fixes the timestamps and adds columns 
    for actual results.

    The expected input is the output of the "xes2csv.jar" file.

    :return: A tuple containing the training, validation and test data.
    :rtype: Tuple of 3 DataFrames
    """

    # load the data
    train_data = pd.read_csv(TRAINING_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    
    # TODO: figure out the right split so that the test set isn't empty
    # remove cases that overlap
    # use a list to modify the test frame after iteration
    # overlapping_cases = []
    # for case_id in test_data[CASE_ID_COLUMN].unique():
    #     if case_id in train_data[CASE_ID_COLUMN] :
    #         # print(case_id)
    #         overlapping_cases.append(case_id)
            
    # for case_id in overlapping_cases:
    #     train_data = train_data[train_data[CASE_ID_COLUMN] != case_id]
    #     test_data = test_data[test_data[CASE_ID_COLUMN] != case_id]

    # Ideally this should not be zero
    # print(test_data.size)

    # change the event time:timestamp to datetime
    train_data[CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        train_data[CASE_TIMESTAMP_COLUMN])
    test_data[CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        test_data[CASE_TIMESTAMP_COLUMN])

    # get the event that follows the current one for each case (row), and the time elapsed
    train_data['time until next event'] = - train_data.groupby(
        CASE_ID_COLUMN)[CASE_TIMESTAMP_COLUMN].diff(-1).dt.total_seconds()
    train_data['next event'] = train_data.groupby(
        CASE_ID_COLUMN)[CASE_POSITION_COLUMN].shift(-1)
    test_data['time until next event'] = - test_data.groupby(
        CASE_ID_COLUMN)[CASE_TIMESTAMP_COLUMN].diff(-1).dt.total_seconds()
    test_data['next event'] = test_data.groupby(
        CASE_ID_COLUMN)[CASE_POSITION_COLUMN].shift(-1)
    # TODO: add validation data set
    return train_data, None, test_data


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

    # get the most frequent event that follows a certain event
    next_event_df = pd.DataFrame(train_data.groupby(CASE_POSITION_COLUMN)[
        'next event'].agg(pd.Series.mode))
    next_event_df.rename(
        columns={'next event': 'predicted next event'}, inplace=True)
    train_data = train_data.merge(
        next_event_df, how='left', on=CASE_POSITION_COLUMN)

    # get the average of the time elapsed after a certain event
    time_elapsed_df = pd.DataFrame(train_data.groupby(CASE_POSITION_COLUMN)[
        'time until next event'].agg('mean'))
    time_elapsed_df.rename(
        columns={'time until next event': 'predicted time until next event'}, inplace=True)
    train_data = train_data.merge(
        time_elapsed_df, how='left', on=CASE_POSITION_COLUMN)

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
    train_data, validation_data, test_data = load_dataset()

    baseline_next_event_df, baseline_time_elapsed_df = train_baseline_model(
        train_data)

    # make predictions for test set
    test_data = test_data.merge(
        baseline_next_event_df, how='left', on=CASE_POSITION_COLUMN)
    test_data = test_data.merge(
        baseline_time_elapsed_df, how='left', on=CASE_POSITION_COLUMN)
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
