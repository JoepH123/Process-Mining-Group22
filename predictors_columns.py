# Imports 
import pandas as pd
import time
from datetime import datetime, timedelta

datapath = '../Data/Prepare&Load Data/'

def load_data(datapath):
    """
    Load data from some datapath, and sort the data based on the event timestamp
    """ 
    start = time.time() ###
    print("Load in the CSV file data") ###
    print("-------------------------") ###
    
    data = pd.read_csv(datapath+'BPI_Challenge_2012/BPI_Challenge_2012-training.csv', parse_dates=['case REG_DATE', 'event time:timestamp'])
    
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
    all_cases = data['case concept:name'].unique().tolist()
    
    print("Compute case relative time") ###
    nr_cases = len(all_cases) ###
    count = 0 ###
    
    for case in all_cases:
        
        count += 1 ###
        if count % 1000 == 0: ###
            print(f"{(count / nr_cases) * 100}% Done") ###
        
        case_data = data[data['case concept:name'] == case]
        number_of_rows = len(case_data.index)-1
        first_case_counter = 0
        for i, row in case_data.iterrows():
            if first_case_counter < 1: # first event of case
                start_time = data.loc[i, 'event time:timestamp']
                first_case_counter += 1
                # print(start_time)
                continue # Because the case_relative_time column was initialized with zeros
            else:
                current_time = data.loc[i, 'event time:timestamp']
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
                of the 'event concept:name' column
    column_name (str) : The name of the column for the lagged events. 
    """
    start = time.time() ###
    print(f"Create {lag}lag column for case events") ###
    
    data[column_name] = ''
    list_cases_with_lagged_column = []
    all_cases = data['case concept:name'].unique().tolist()
    for case in all_cases:
        case_data = data[data['case concept:name'] == case].copy()
        case_data[column_name] = case_data['event concept:name'].shift(lag, fill_value='no_lagged_events')
        list_cases_with_lagged_column.append(case_data)
    total_data_with_added_column = pd.concat(list_cases_with_lagged_column)
    data_with_new_column = total_data_with_added_column.sort_values(by=['event time:timestamp'])
    
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
    all_cases = data['case concept:name'].unique().tolist()
    
    print("Compute workrate of employees") ###
    nr_cases = len(all_cases) ###
    count = 0 ###
    
    for case in all_cases:
        
        count += 1 ###
        if count % 1000 == 0: ###
            print(f"{(count / nr_cases) * 100}% Done") ###
        
        case_data = data[data['case concept:name'] == case]
        case_data.reset_index(inplace=True, drop=True)
        for i, row in case_data.iterrows():
            if i+1 < len(case_data.index):  # last event of a case can be ignored
                emp_of_event = str(row['event org:resource'])
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
        employee = data.loc[i, 'event org:resource']
        data.loc[i, 'workrate'] = dict_workrate[str(employee)]
        
    end = time.time() ###
    print(f"Function time taken: {end-start}") ###
    print("-------------------------------") ###
        
    return data


def pipeline(path):
    """
    This pipeline combines all the functions defined above. It makes sure that all steps are executed, and it returns
    the total time that the pipeline took to execute. 
    """
    total_start = time.time()
    data = load_data(path)
    data = compute_case_relative_time(data)
    data = compute_case_lag_event_column(data, lag=1, column_name='first_lag_event')
    data = compute_case_lag_event_column(data, lag=2, column_name='second_lag_event')
    data = compute_workrate_employees(data)
    total_end = time.time()
    print(f"Total pipeline time taken: {total_end-total_start}") ###
    return data
