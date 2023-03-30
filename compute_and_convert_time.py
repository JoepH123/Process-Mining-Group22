"""
This module contains function that compute the average duration of a case in the input dataset. This is information that we include on the poster
so therefore this contains the code to compute it. Furthermore it contains two functions which can be used to convert seconds to a suitable time 
unit. There are two different approaches. 

You can choose either approach, note that both the functions of both approaches use the display_time(seconds) function as a dependency.
"""

# Imports

import pandas as pd
import constants
from datetime import datetime, timedelta

# Computing average time of a case

def compute_case_relative_time(data):
    """
    This takes a dataframe as input and computes the case_relative_time column. 

    :param data: The dataframe for which we want to compute the case_relative_time column
    :type data: pd.DataFrame
    """
    data['case_relative_time'] = 0.0
    all_cases = data['case:concept:name'].unique().tolist()
    for case in all_cases:
        case_data = data[data['case:concept:name'] == case]
        first_case_counter = 0
        for i, row in case_data.iterrows():
            if first_case_counter < 1: # first event of case
                start_time = data.loc[i, 'time:timestamp']
                first_case_counter += 1
                continue # Because the case_relative_time column was initialized with zeros
            else:
                current_time = data.loc[i, 'time:timestamp']
                time_diff = current_time - start_time
                time_diff = time_diff.total_seconds()
                data.loc[i, 'case_relative_time'] = time_diff
    return data


def load_data_and_compute_case_relative_time(path):
    """
    This function loads the data and computes the case_relative_time column. 

    :param path: The path to the location of the dataset that we want to analyze
    :type path: string
    """
    #training_data_2012 = pd.read_csv(path, parse_dates=['case:REG_DATE', 'time:timestamp']) 
    training_data_2012 = pd.read_csv(path, parse_dates=['time:timestamp']) 
    training_data_2012 = training_data_2012.sort_values(by=["time:timestamp"])
    data = compute_case_relative_time(training_data_2012)
    return data


def compute_avg_case_duration(data_with_time_column):
    """
    This function computes the average duration of a case, based on the input dataframe. For this computation it uses the case_relative_time column
    which is created in the compute_case_relative_time function. 

    :param data_with_time_column: A dataframe with the case_relative_time column
    :type data_with_time_column: pd.DataFrame
    """
    all_cases = data_with_time_column['concept:name'].unique().tolist()
    nr_cases = len(all_cases)
    total_duration = 0
    for case in all_cases:
        case_data = data_with_time_column[data_with_time_column['concept:name'] == case]
        case_duration = case_data.case_relative_time.tolist()[-1]
        total_duration += case_duration
    avg_duration = total_duration / nr_cases
    return int(round(avg_duration, 0))

# Time display functions

def display_time(seconds):
    """
    This function converts the number of seconds to the years, weeks, days, hours, minutes, seconds. It fills these
    units up from highest to lowest. 

    :param seconds: Number of seconds that are is given as input. This could be the MAE of a model for example.
    :type seconds: Interger
    """
    intervals = (
        ('years', 31557600), # 60 * 60 * 24 * 365.25
        ('weeks', 604800),   # 60 * 60 * 24 * 7
        ('days', 86400),     # 60 * 60 * 24
        ('hours', 3600),     # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )
    if seconds < 1:
        return [0, 0, 0, 0, 0, seconds]
    result = []
    for name, count in intervals:
        value = seconds // count
        seconds -= value * count
        result.append(value)
    return result


def max_four_digit_time_rep_accurate(seconds):
    """ 
    Prints any number of seconds given as input into a suitable time unit with at most 4 digits. This is a more complex approach. 
    It prefers smaller values, so if year, week, day, hour, minute, has a value smaller than zero it converts it to the smaller unit.
    This results in 9 days being converted to hours. The benefit of this approach is that the approximation of time is more precise. 
    No large time quantities will be left out when approximating, however, it might not be most interpretive. The function ... gives a 
    more naive approach. It is less exact, but it might be better interpretable.  

    :param seconds: Number of seconds that are is given as input. This could be the MAE of a model for example.
    :type seconds: Interger
    """
    units = display_time(seconds)  # fixed_length of 6
    multiplier = [365.25/7, 7, 24, 60, 60]
    names = ['years', 'weeks', 'days', 'hours', 'minutes', 'seconds']
    for i in range(len(units)):
        if units[i] >= 10:
            if i == 5:
                return f"{round(units[i], 2)} seconds"
            first_unit = (units[i], i)
            second_unit = (units[i+1], i+1)
            break # break # stop the for loop
            # Convert these units to a good time indicator (at most 4 digits)
        elif units[i] == 0:
            continue
        else:  # if units[i] < 10:
            if i < 5:
                value = units[i]
                units[i+1] += value * multiplier[i]   
                units[i] = 0
            if i == 5:
                value_name = names[i]
                if units[i] <= 1:
                    value_name = value_name.rstrip('s')
                return f"{round(units[i], 2)} {value_name}"
    first_value_name = names[first_unit[1]]
    second_value_name = names[second_unit[1]]
    if first_unit[0] == 1:
        first_value_name = first_value_name.rstrip('s')
    if second_unit[0] == 1:
        second_value_name = second_value_name.rstrip('s')
    if second_unit[0]:
        string_result = f"{first_unit[0]} {first_value_name}, {second_unit[0]} {second_value_name}"
    else:
        string_result = f"{first_unit[0]} {first_value_name}"
    return string_result


def max_four_digit_time_rep_naive(seconds):
    """ 
    This function computes a suitable time unit, by finding the largest time unit that does not equal zero and 
    than approximating the time to this unit and the time unit that is one rank lower. This approach does not convert
    between time units. It is less exact, but more consistent with its time units.

    :param seconds: Number of seconds that are is given as input. This could be the MAE of a model for example.
    :type seconds: Interger
    """
    units = display_time(seconds)  # fixed_length of 6
    names = ['years', 'weeks', 'days', 'hours', 'minutes', 'seconds']
    for i in range(len(units)):
        if units[i] == 0:
            continue
        else:  # There is a value for this time unit
            first_unit = (units[i], i)
            second_unit = (units[i+1], i+1)
            break
    if second_unit[0] == 0:
        return f"{first_unit[0]} {names[first_unit[1]]}"
    else:  # The second largest time unit is not equal to 0
        return f"{first_unit[0]} {names[first_unit[1]]}, {second_unit[0]} {names[second_unit[1]]}"


# data = load_data_and_compute_case_relative_time(constants.GLOBAL_DATASET_PATH)
# avg_time = compute_avg_case_duration(data)
# print(avg_time)

# exact_rounded_time = max_four_digit_time_rep_accurate(avg_time)
# naive_rounded_time = max_four_digit_time_rep_naive(avg_time)
# print(exact_rounded_time)
# print(naive_rounded_time)
