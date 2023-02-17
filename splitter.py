import pm4py
import constants
import copy
import numpy as np
import pandas as pd

def split_dataset(split_ratio: float):
    """Converts the raw dataset into .csv files, so that the number of cases
    in each is as close as possible to the parametrized value.
    The csv files are saved in the same directory as the raw dataset.

    """
    dataframe = pm4py.convert_to_dataframe(pm4py.read_xes(constants.RAW_DATASET_PATH))
    pd.set_option('display.max_columns', None)

    # change the event time:timestamp to datetime
    dataframe[constants.CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        dataframe[constants.CASE_TIMESTAMP_COLUMN])
    dataframe.sort_values(by=[constants.CASE_TIMESTAMP_COLUMN], inplace=True)
    # get the event that follows the current one for each case (row), and the time elapsed
    dataframe['time until next event'] = - dataframe.groupby(
        constants.CASE_ID_COLUMN)[constants.CASE_TIMESTAMP_COLUMN].diff(-1).dt.total_seconds()
    dataframe['next event'] = dataframe.groupby(
        constants.CASE_ID_COLUMN)[constants.CASE_POSITION_COLUMN].shift(-1)
    dataframe[constants.CASE_STEP_NUMBER_COLUMN] = dataframe.groupby(
        constants.CASE_ID_COLUMN).cumcount()

    countframe = copy.deepcopy(dataframe)

    # constants are local because we will drop the column by the end
    CASE_STEP_NUMBER_INVERSE_COLUMN = 'number of activity in case inverse'
    CASE_START_COUNT = 'case start count'
    CASE_END_COUNT = 'case end count'
    countframe[constants.CASE_STEP_NUMBER_COLUMN] = countframe.groupby(
        constants.CASE_ID_COLUMN).cumcount()
    countframe[CASE_STEP_NUMBER_INVERSE_COLUMN] = countframe.groupby(
        constants.CASE_ID_COLUMN).cumcount(ascending=False)
    countframe = countframe[(countframe[constants.CASE_STEP_NUMBER_COLUMN] == 0) |
        (countframe[CASE_STEP_NUMBER_INVERSE_COLUMN] == 0)]
    countframe[CASE_START_COUNT] = countframe[
        countframe[constants.CASE_STEP_NUMBER_COLUMN] == 0].groupby(
            constants.CASE_STEP_NUMBER_COLUMN).cumcount(ascending=False) + 1
    countframe[CASE_END_COUNT] = countframe[
        countframe[CASE_STEP_NUMBER_INVERSE_COLUMN] == 0].groupby(
            CASE_STEP_NUMBER_INVERSE_COLUMN).cumcount() + 1

    countframe[CASE_START_COUNT].fillna(method='ffill', inplace = True)
    countframe[CASE_END_COUNT].fillna(method='ffill', inplace = True)
    
    # filter out only the points that yield a test set of at least the split ratio
    countframe = countframe[countframe[CASE_START_COUNT]/(countframe[CASE_START_COUNT] + 
        countframe[CASE_END_COUNT]) > split_ratio]

    # get the timestamp of the last such point, as it yields the smallest test set
    split_timestamp = countframe.iloc[-1][constants.CASE_TIMESTAMP_COLUMN]
    train_data = copy.deepcopy(dataframe[dataframe[constants.CASE_TIMESTAMP_COLUMN] < split_timestamp])
    test_data = copy.deepcopy(dataframe[dataframe[constants.CASE_TIMESTAMP_COLUMN] >= split_timestamp])
    
    # remove cases that overlap
    # use a list to modify the test frame after iteration
    overlapping_cases = []
    for case_id in test_data[constants.CASE_ID_COLUMN].unique():
        if case_id in train_data[constants.CASE_ID_COLUMN] :
            print(case_id)
            overlapping_cases.append(case_id)
            
    for case_id in overlapping_cases:
        train_data = train_data[train_data[constants.CASE_ID_COLUMN] != case_id]
        test_data = test_data[test_data[constants.CASE_ID_COLUMN] != case_id]

    # Ideally this should not be zero
    print(test_data.size)

    train_data.to_csv(constants.TRAINING_DATA_PATH)
    test_data.to_csv(constants.TEST_DATA_PATH)

if __name__ == "__main__":
    split_dataset(0.5)