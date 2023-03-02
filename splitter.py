import pm4py
import constants
import copy
import numpy as np
import pandas as pd

def convert_raw_dataset(raw_path, converted_path):
    """Converts the raw dataset (.xes.gz) into a .csv file. Adds some
    preprocessing artifacts.

    :param raw_path: the raw dataset path
    :type raw_path: string
    :param converted_path: converted dataset path to save the result
    :type converted_path: string
    """
    dataframe = pm4py.convert_to_dataframe(pm4py.read_xes(raw_path))

    # set when printing head to see all columns
    # pd.set_option('display.max_columns', None)

    # change the event time:timestamp to datetime
    dataframe[constants.CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        dataframe[constants.CASE_TIMESTAMP_COLUMN])

    dataframe.sort_values(by=[constants.CASE_TIMESTAMP_COLUMN], inplace=True)

    # get the event that follows the current one for each case (row), and the time elapsed
    dataframe['time until next event'] = - dataframe.groupby(
        constants.CASE_ID_COLUMN)[constants.CASE_TIMESTAMP_COLUMN].diff(-1).dt.total_seconds()
    dataframe['next event'] = dataframe.groupby(
        constants.CASE_ID_COLUMN)[constants.CASE_POSITION_COLUMN].shift(-1)

    # get the number of the current event in the case (starts at 0)
    dataframe[constants.CASE_STEP_NUMBER_COLUMN] = dataframe.groupby(
        constants.CASE_ID_COLUMN).cumcount()

    # how many events are left until the end of the case
    dataframe[constants.CASE_STEP_NUMBER_INVERSE_COLUMN] = dataframe.groupby(
        constants.CASE_ID_COLUMN).cumcount(ascending=False)

    # # get only the events that start or end a case
    # countframe = dataframe[(dataframe[constants.CASE_STEP_NUMBER_COLUMN] == 0) |
    #     (dataframe[constants.CASE_STEP_NUMBER_INVERSE_COLUMN] == 0)]

    # how many cases start after the current event, inclusive
    dataframe[constants.CASE_START_COUNT] = dataframe[
        dataframe[constants.CASE_STEP_NUMBER_COLUMN] == 0].groupby(
            constants.CASE_STEP_NUMBER_COLUMN).cumcount(ascending=False) + 1

    # how many cases end before the current event, inclusive
    dataframe[constants.CASE_END_COUNT] = dataframe[
        dataframe[constants.CASE_STEP_NUMBER_INVERSE_COLUMN] == 0].groupby(
            constants.CASE_STEP_NUMBER_INVERSE_COLUMN).cumcount() + 1

    dataframe[constants.CASE_START_COUNT].fillna(method='ffill', inplace = True)
    dataframe[constants.CASE_END_COUNT].fillna(method='ffill', inplace = True)

    print("shape of converted dataset: ", dataframe.shape)

    dataframe.to_csv(converted_path)

def split_dataset(dataframe: pd.DataFrame, test_split_ratio: float):
    """Splits the given data frame into two parts,
    so that the relation of cases in the second part to the whole
    is as close as possible to the parametrized value.
    Overlapping cases are removed.

    :param dataframe: the data frame to be split. It is not altered in the process.
    :type dataframe: pd.DataFrame
    :param test_split_ratio: target portion of cases not dropped
        in the second half
    :type split_ratio: float
    :return: a tuple of two data frames, corresponding to each part
    :rtype: tuple(pd.DataFrame, pd.DataFrame)
    """

    # NOTE: this way of filtering possibly throws away 1 case more than neeeded
    # but rewriting it to not do so would add like 20 lines of code

    # print("shape of initial data set: ", dataframe.shape)

    # filter out only the points that yield a test set of at least the split ratio
    countframe = dataframe[dataframe[constants.CASE_START_COUNT]/(dataframe[constants.CASE_START_COUNT] +
        dataframe[constants.CASE_END_COUNT]) > test_split_ratio]

    # get the timestamp of the last such point, as it yields the smallest test set
    split_timestamp = countframe.iloc[-1][constants.CASE_TIMESTAMP_COLUMN]

    # split the data
    first_part = dataframe[dataframe[constants.CASE_TIMESTAMP_COLUMN] < split_timestamp]
    second_part = dataframe[dataframe[constants.CASE_TIMESTAMP_COLUMN] >= split_timestamp]

    # print("shape of initial train set: ", first_part.shape)
    # print("shape of initial test set: ", second_part.shape)

    # remove cases that overlap
    # use a list to modify the test frame after iteration
    overlapping_cases = []
    for case_id in second_part[constants.CASE_ID_COLUMN].unique():
        if case_id in first_part[constants.CASE_ID_COLUMN].values :
            overlapping_cases.append(case_id)

    print("cases dropped: ", len(overlapping_cases))

    first_part = first_part[~first_part[constants.CASE_ID_COLUMN].isin(overlapping_cases)]
    second_part = second_part[~second_part[constants.CASE_ID_COLUMN].isin(overlapping_cases)]

    # Ideally this should not be zero
    print("shape of train set: ", first_part.shape);
    print("shape of test set: ", second_part.shape)
    return first_part, second_part

if __name__ == "__main__":
    split_dataset(0.2)
