# Imports 
import pandas as pd
import time
import constants


def compute_basic_features(dataframe):
    """Computes the basic features for the dataset such as next evet, case number, case count.

    :param dataframe: the dataframe to compute the features on

    :return: The input dataframe with the computed features
    :rtype: dataframe
    """
    dataframe[constants.CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        dataframe[constants.CASE_TIMESTAMP_COLUMN])

    # get the event that follows the current one for each case (row), and the time elapsed
    dataframe[constants.TIME_DIFFERENCE] = - dataframe.groupby(
        constants.CASE_ID_COLUMN)[constants.CASE_TIMESTAMP_COLUMN].diff(-1).dt.total_seconds()
    dataframe[constants.NEXT_EVENT] = dataframe.groupby(
        constants.CASE_ID_COLUMN)[constants.CURRENT_EVENT].shift(-1)
    dataframe = dataframe.astype({constants.NEXT_EVENT: 'str'})

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

    dataframe.at[-1, constants.CASE_START_COUNT] = 0
    dataframe.at[0, constants.CASE_END_COUNT] = 0
    dataframe[constants.CASE_START_COUNT].fillna(method='bfill', inplace = True)
    dataframe[constants.CASE_END_COUNT].fillna(method='ffill', inplace = True)

    # Change column names if dataset is from 2017
    if "case:AMOUNT_REQ" not in dataframe.columns and "case:RequestedAmount" in dataframe.columns:
        dataframe.rename(columns={"case:RequestedAmount": "case:AMOUNT_REQ"}, inplace=True)

    return dataframe

def compute_case_relative_time(data):
    """
    Compute case relative time based event timestamp columns. e.g. 0 seconds at the first event, 
    then for each event, the number of seconds since the start of the case.
    """
    start = time.time() ###
    
    data[constants.TIME_SINCE_START_OF_CASE] = 0.0
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
                data.loc[i, constants.TIME_SINCE_START_OF_CASE] = time_diff
                
    end = time.time() ###
    print(f"Function time taken: {end-start}") ###          
    print("--------------------------") ###

    return data


def compute_case_lag_event_column(data, lag, column_name):
    """
    This function adds a column for a specified event lag to the dataframe
    
    lag (int) : Integer that indicates the number of what number of lags we want 
                to add as a column. e.g. lag=1, adds the column for the first lag 
                of the constants.CURRENT_EVENT column
    column_name (str) : The name of the column for the lagged events. 
    """
    start = time.time() ###
    print(f"Create {lag}lag column for case events") ###
    
    data[column_name] = ''
    list_cases_with_lagged_column = []
    all_cases = data[constants.CASE_ID_COLUMN].unique().tolist()
    for case in all_cases:
        case_data = data[data[constants.CASE_ID_COLUMN] == case].copy()
        case_data[column_name] = case_data[constants.CURRENT_EVENT].shift(lag, fill_value='no_lagged_events')
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
                time_of_event = case_data.loc[i + 1, constants.TIME_SINCE_START_OF_CASE] - case_data.loc[i, constants.TIME_SINCE_START_OF_CASE]
                
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


def pipeline(data, timer):
    """
    This pipeline combines all the functions defined above. It makes sure that all steps are executed, and it returns
    the total time that the pipeline took to execute. 
    """

    data = compute_basic_features(data)
    data = compute_case_relative_time(data)
    data = compute_case_lag_event_column(data, lag=1, column_name='first_lag_event')
    data = compute_case_lag_event_column(data, lag=2, column_name='second_lag_event')
    data = compute_workrate_employees(data)
    data = compute_active_cases(data)

    timer.send("Time to add local variables (in seconds): ")
    return data
