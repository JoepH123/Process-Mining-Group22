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

    :return: The converted dataframe
    :rtype: dataframe
    """
    dataframe = pm4py.convert_to_dataframe(pm4py.read_xes(raw_path))

    # set when printing head to see all columns
    # pd.set_option('display.max_columns', None)

    # change the event time:timestamp to datetime
    dataframe[constants.CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        dataframe[constants.CASE_TIMESTAMP_COLUMN])

    dataframe.sort_values(by=[constants.CASE_TIMESTAMP_COLUMN], inplace=True)
    dataframe.to_csv(converted_path)

    print("shape of converted dataset: ", dataframe.shape)

    return dataframe

def split_dataset(dataframe: pd.DataFrame, split_ratio: float):
    """Splits the given data frame into two parts,
    so that the relation of cases in the second part to the whole
    is as close as possible to the parametrized value.
    Overlapping cases are removed.

    :param dataframe: the data frame to be split, with artifacts showing the percentile
    of each case. It is not altered in the process.
    :type dataframe: pd.DataFrame
    :param split_ratio: out of the initial dataset, cases of which last percentile
    should be in the second part
    :type split_ratio: float
    :return: a tuple of two data frames, corresponding to each part
    :rtype: tuple(pd.DataFrame, pd.DataFrame)
    """

    # NOTE: this way of filtering possibly throws away 1 case more than neeeded
    # but rewriting it to not do so would add like 20 lines of code

    # print("shape of initial data set: ", dataframe.shape)

    # filter out only the points that yield a second part of at least the split ratio
    countframe = dataframe[dataframe[constants.CASE_START_COUNT]/(dataframe[constants.CASE_START_COUNT] +
        dataframe[constants.CASE_END_COUNT]) >= split_ratio]

    # get the timestamp of the last such point, as it yields the smallest set
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

def get_percentile(entry: pd.Series):
    print(entry[constants.CASE_START_COUNT])
    print((entry[constants.CASE_START_COUNT] +
        entry[constants.CASE_END_COUNT]))
    return entry[constants.CASE_START_COUNT]/(entry[constants.CASE_START_COUNT] +
        entry[constants.CASE_END_COUNT])

def time_series_split(dataframe: pd.DataFrame, k: int):
    begin_percentile = get_percentile(dataframe.iloc[0])
    end_percentile = get_percentile(dataframe.iloc[-1])
    # print("Begin percentile ", begin_percentile)
    # print("End percentile ", end_percentile)

    partition_list = []
    for i in range(1, k):
        current_percentile = (begin_percentile - end_percentile) * i/k + end_percentile
        # print("Current percentile ", current_percentile)
        first_part, new_part = split_dataset(dataframe, current_percentile)
        # remove the rows that are already in a partition
        for existing_partition in partition_list:
            new_part = new_part[~new_part.index.isin(existing_partition.index)]
        partition_list.insert(0, new_part)
    # get the rest as well
    partition_list.insert(0, first_part)

    training_data = []
    test_data = []
    df_list_cumulative = [] # for efficient multiple appending
    for i in range(0, k-1):
        df_list_cumulative.append(partition_list[i])
        training_data.append(pd.concat(df_list_cumulative))
        test_data.append(partition_list[i+1])

    # for frame in training_data:
    #     print("Shape of train dataframe ", frame.shape)

    # for frame in test_data:
    #     print("Shape of test dataframe ", frame.shape)


if __name__ == "__main__":
    split_dataset(pd.read_csv("Datasets/BPI_Challenge_2017.csv"), 0.2)
