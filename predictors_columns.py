# Imports 
import pandas as pd
import time
from datetime import datetime, timedelta
import constants
import datetime
import workalendar
import country_converter as cc
import numpy as np
from typing import Union

datapath = '../Data/Prepare&Load Data/'

def load_data(datapath):
    """
    Load data from some datapath, and sort the data based on the event timestamp
    """ 
    start = time.time() ###
    print("Load in the CSV file data") ###
    print("-------------------------") ###
    
    data = pd.read_csv(datapath+'BPI_Challenge_2012/BPI_Challenge_2012-training.csv', parse_dates=['case REG_DATE', constants.CASE_TIMESTAMP_COLUMN])
    
    print("Sort values based on event timestamp") ###
    
    data = data.sort_values(by=["event time:timestamp"])
    data.reset_index(inplace=True, drop=True)
    
    end = time.time() ###
    print(f"Function time taken: {end-start}") ###
    print("------------------------------------") ###
    
    return data


def compute_case_relative_time(data):
    """
    Compute case relative time based event timestamp columns. e.g. 0 seconds at the first event, 
    then for each event, the number of seconds since the start of the case.
    """
    start = time.time() ###
    
    data['case_relative_time'] = 0.0
    all_cases = data[constants.CASE_ID_COLUMN].unique().tolist()
    
    print("Compute case relative time") ###
    nr_cases = len(all_cases) ###
    count = 0 ###
    
    for case in all_cases:
        
        count += 1 ###
        if count % 1000 == 0: ###
            print(f"{(count / nr_cases) * 100}% Done") ###
        
        case_data = data[data[constants.CASE_ID_COLUMN] == case]
        number_of_rows = len(case_data.index)-1
        first_case_counter = 0
        for i, row in case_data.iterrows():
            if first_case_counter < 1: # first event of case
                start_time = data.loc[i, constants.CASE_TIMESTAMP_COLUMN]
                first_case_counter += 1
                # print(start_time)
                continue # Because the case_relative_time column was initialized with zeros
            else:
                current_time = data.loc[i, constants.CASE_TIMESTAMP_COLUMN]
                # print(current_time)
                time_diff = current_time - start_time
                # print(time_diff)
                time_diff = time_diff.total_seconds()
                # print(time_diff)
                data.loc[i, 'case_relative_time'] = time_diff
                
    end = time.time() ###
    print(f"Function time taken: {end-start}") ###          
    print("--------------------------") ###

    return data


def compute_case_lag_event_column(data, lag, column_name):
    """
    This function adds a column for a specified event lag to the dataframe
    
    lag (int) : Integer that indicates the number of what number of lags we want 
                to add as a column. e.g. lag=1, adds the column for the first lag 
                of the constants.CASE_POSITION_COLUMN column
    column_name (str) : The name of the column for the lagged events. 
    """
    start = time.time() ###
    print(f"Create {lag}lag column for case events") ###
    
    data[column_name] = ''
    list_cases_with_lagged_column = []
    all_cases = data[constants.CASE_ID_COLUMN].unique().tolist()
    for case in all_cases:
        case_data = data[data[constants.CASE_ID_COLUMN] == case].copy()
        case_data[column_name] = case_data[constants.CASE_POSITION_COLUMN].shift(lag, fill_value='no_lagged_events')
        list_cases_with_lagged_column.append(case_data)
    total_data_with_added_column = pd.concat(list_cases_with_lagged_column)
    data_with_new_column = total_data_with_added_column.sort_values(by=[constants.CASE_TIMESTAMP_COLUMN])
    
    end = time.time() ###
    print(f"Function time taken: {end-start}") ###
    print("------------------------------------") ###
    
    return data_with_new_column


def compute_workrate_employees(data):
    """
    This function computes the workrate of each employee (resource). This means, a dictionary with all the different
    employees (resources) as keys, and the average time until next event for this employee as the value. This dictionary
    is then used to create the workrate column in the dataframe. For each event, the workrate that corresponds to the 
    employee of that event in inputed.
    """
    start = time.time() ###
    
    data['workrate'] = 0
    dict_workrate = {}
    data = data.fillna(0)
    all_cases = data[constants.CASE_ID_COLUMN].unique().tolist()
    
    print("Compute workrate of employees") ###
    nr_cases = len(all_cases) ###
    count = 0 ###
    
    for case in all_cases:
        
        count += 1 ###
        if count % 1000 == 0: ###
            print(f"{(count / nr_cases) * 100}% Done") ###
        
        case_data = data[data[constants.CASE_ID_COLUMN] == case]
        case_data.reset_index(inplace=True, drop=True)
        for i, row in case_data.iterrows():
            if i+1 < len(case_data.index):  # last event of a case can be ignored
                emp_of_event = str(row['org:resource'])
                time_of_event = case_data.loc[i + 1, 'case_relative_time'] - case_data.loc[i, 'case_relative_time']
                
                if time_of_event < 0: ###
                    print("NegativeValueERROR: There are negative responsetimes by employees. This is impossible. \
                          The dataframe is likely not sorted by event timestamp") ###
                    return "ERROR: NOT SORTED ON EVENT TIMESTAMP", (case_data, emp_of_event, row['event_id']) ###
                
                if emp_of_event not in dict_workrate:
                    dict_workrate[emp_of_event] = [time_of_event]
                else:
                    dict_workrate[emp_of_event].append(time_of_event)
    
    print("Finishing up computation of workrate") ###
    
    for emp in dict_workrate:
        dict_workrate[emp] = sum(dict_workrate[emp]) / len(dict_workrate[emp])
        
    print('------------------------------------') ###
    print("Adding the workload as a column") ###
    
    for i in range(len(data.index)):
        employee = data.loc[i, 'org:resource']
        data.loc[i, 'workrate'] = dict_workrate[str(employee)]
        
    end = time.time() ###
    print(f"Function time taken: {end-start}") ###
    print("-------------------------------") ###
        
    return data

def compute_active_cases(data):
    total_cases = data[constants.CASE_START_COUNT].loc[0]
    data[constants.ACTIVE_CASES] = total_cases - data[constants.CASE_START_COUNT].shift(
        periods = -1, fill_value=0) - data[constants.CASE_END_COUNT]
    return data


def obtain_holidays(country: str, years: Union[list, np.ndarray]):
    """
    Creates a list of all holidays using a country of choice and the year(s) of choice

    :param country: Country of which the holidays need to be computed
    :type country: string
    :param years: List with all the years of interest
    :type years: Union[list, np.ndarray]
    :return: A list with datetime.date entries for all holidays in that country in those years
    :rtype: list
    """

    holidays = []

    for year in years:
        holidays += [i[0] for i in eval('{}().holidays({})'.format(country, str(year)))]
        if country == 'Netherlands':
            holidays.append(datetime.date(year, 12, 24))
            holidays.append(datetime.date(year, 12, 31))

    return holidays


def compute_time_based_columns(dataframe, date_column: str, start_hours= 9, stop_hours= 17, country='Netherlands'):
    """
    Adds the extra columns regarding to the dates on which events happen. These are things like holidays, weekends & working hours

    :param dataframe: original imported pandas dataframe
    :type dataframe: pd.DataFrame
    :param date_column: which column the dates of the event are stored
    :type date_column: string
    :param start_hours: specification of when the work day starts
    :type start_hours: integer
    :param stop_hours: specification of when the work day ends
    :type stop_hours: integer
    :param country: Country of which the holidays need to be computed
    :type country: string
    :return: The new complete dataframe with the following columns
            seconds_since_week_start: seconds (float) since Monday 00:00
            is_weekend: Boolean value if its a weekend or not
            is_work_time: Boolean value if its work time or not
            seconds_to_work_hours: seconds (float) until the next work day starts (or 0 if its during working hours)
            days_until_holiday: Days (int) until the next holiday starts
            is_holiday: Boolean value if its a national holiday or not
    :rtype: pd.DataFrame
    """
    exec('from workalendar.' + cc.convert(country, to='Continent').lower() + ' import ' + country, globals())

    dataframe[date_column] = pd.to_datetime(dataframe[date_column])

    years = np.unique(dataframe[date_column].dropna().dt.year)
    holidays = obtain_holidays(country=country, years=years)
    dataframe[date_column] = dataframe[date_column].apply(lambda x: x.replace(tzinfo=None))

    dataframe['days_until_next_holiday'] = dataframe[date_column].apply(lambda x: 0 if x.date() in holidays else
    min((holiday - x.date()).days for holiday in holidays if (holiday - x.date()).days > 0))
    dataframe['is_holiday'] = dataframe[date_column].apply(lambda x: True if x.date() in holidays else False)

    dataframe['seconds_since_week_start'] = dataframe[date_column].apply(
        lambda x: pd.Timedelta('{} days {}'.format(x.weekday(), x.time())).total_seconds())
    dataframe['is_work_time'] = dataframe[date_column].apply(
        lambda x: False if 4 < x.weekday() < 7 else True if start_hours <= x.hour < stop_hours else False)
    dataframe['seconds_to_work_hours'] = dataframe[date_column].apply(
        lambda x: pd.Timedelta('0h0m').total_seconds() if x.weekday() < 5 and start_hours <= x.hour < stop_hours else
        (datetime.datetime.combine(x.date(), datetime.time(start_hours, 0, 0,
                                                           0)) - x).total_seconds() if x.weekday() < 5 and start_hours > x.hour else
        (datetime.datetime.combine(x.date() + datetime.timedelta(days=1), datetime.time(start_hours, 0, 0,
                                                                                        0)) - x).total_seconds() if x.weekday() < 4 and stop_hours <= x.hour else
        (datetime.datetime.combine((x.date() + datetime.timedelta(days=7 - x.weekday())),
                                   datetime.time(start_hours, 0, 0, 0)) - x).total_seconds())

    dataframe['is_weekend'] = dataframe[date_column].apply(lambda x: True if 4 < x.weekday() < 7 else True
    if x.weekday() == 4 and x.hour > stop_hours - 1 else True
    if x.weekday() == 0 and x.hour < start_hours else False)
    return dataframe

def add_liquidity(df, date_column):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    
    df['total_amount'] = df.groupby(['year', 'month', 'day'], as_index=False).agg(total_amount=('case:AMOUNT_REQ', 'sum'))['total_amount']

    return df

def pipeline(data):
    """
    This pipeline combines all the functions defined above. It makes sure that all steps are executed, and it returns
    the total time that the pipeline took to execute. 
    """
    total_start = time.time()
    data = compute_case_relative_time(data)
    data = compute_case_lag_event_column(data, lag=1, column_name='first_lag_event')
    data = compute_case_lag_event_column(data, lag=2, column_name='second_lag_event')
    data = compute_workrate_employees(data)
    data = compute_active_cases(data)
    data = add_liquidity(data, "time:timestamp")

    total_end = time.time()
    print(f"Total pipeline time taken: {total_end-total_start}") ###
    return data
