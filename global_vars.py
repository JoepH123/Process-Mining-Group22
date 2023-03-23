import pandas as pd
import datetime
import country_converter as cc
import numpy as np
from typing import Union
import warnings
from tqdm import tqdm
import constants

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


def time_based_columns(dataframe, date_column: str, start_hours= 9, stop_hours= 17, country='Netherlands'):
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
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    
    df['total_amount'] = df.groupby(['year', 'month', 'day'], as_index=False).agg(total_amount=(constants.AMOUNT_REQUESTED_COLUMN, 'sum'))['total_amount']
    
    return df


def resource_active_cases(dataframe, time_col=constants.CASE_TIMESTAMP_COLUMN, event_col=constants.CASE_POSITION_COLUMN,
                          status_col='lifecycle:transition', resource_col='org:resource',
                          time_between_events_col=constants.TIME_DIFFERENCE, case_id_col=constants.CASE_ID_COLUMN,
                          act_nr_col=constants.CASE_STEP_NUMBER_COLUMN, next_event_col=constants.NEXT_EVENT,
                          activity_method='w_activities', start_variable='START'):
    '''
    Function to obtain all the active cases for a resource at the time of starting a new event.
    Two different approaches can be used to calculate this number; based on only the work activities,
    as they have a start and end point and based on the average duration for all entries

    :param dataframe: Original imported dataframe
    :type dataframe: pd.DataFrame
    :param time_col: Column name of the provided dataframe which contains the time when an event started
    :type time_col: string
    :param event_col: Column name of the provided dataframe which contains the type of event
    :type event_col: string
    :param status_col: Column name of the provided dataframe which contains the status of an event at the point of database entry
    :type status_col: string
    :param resource_col: Column name of the provided dataframe which contains the recourse which works on the event
    :type resource_col: string
    :param time_between_events_col: Column name of the provided dataframe which contains the time between the current and the next event
    :type time_between_events_col: string
    :param case_id_col: Column name of the provided dataframe which contains the identifier of the case
    :type case_id_col: string
    :param act_nr_col: Column name of the provided dataframe which contains the number of preceding events when looking at the total sequence of activities for the current case
    :type act_nr_col: string
    :param next_event_col: Column name of the provided dataframe which contains the type of event for the next event in the sequence for the case
    :type next_event_col: string
    :param activity_method: Entry which is used to determine which method is chosen to determine all the possible active cases. The following methods can be chosen: \n
        w_activities: In this case only the activities which have a start and end point within the input dataset are used to obtain the active cases. \n
        all_activities: In this case all the activities in the dataset are used to determine the active cases, however as not all entries might have a determined start and end point, the median time is used to determine the duration of the events. \n
        both: In this case, both of the aforementioned methods are used to return both outcomes. \n
    :type time_col: string
    :param start_variable: How the start of a case is indicated in the status column of the given dataset
    :return: The new dataframe with the number of active activites for each resource at the start of the event for all event entries
    :rtype: pd.Dataframe
    '''
    warnings.warn('Function is very costly, so it might take a while to produce the results')
    if activity_method == 'w_activities':
        for index in tqdm(dataframe[dataframe[status_col] == start_variable].index):
            broken = False
            or_index = index
            case_id = dataframe.at[index, case_id_col]
            act_nr = dataframe.at[index, act_nr_col]
            target = act_nr + 1

            while dataframe.at[or_index, event_col] != dataframe.at[index, next_event_col]:
                try:
                    index = dataframe[(dataframe[case_id_col] == case_id) & (dataframe[act_nr_col] == target)].index[0]
                    target += 1

                except:
                    dataframe.at[or_index, 'w_end_time'] = pd.NaT
                    broken = True
                    break

            if broken:
                continue
            else:
                dataframe.at[or_index, 'w_end_time'] = dataframe.at[
                    dataframe[(dataframe[case_id_col] == case_id) & (dataframe[act_nr_col] == target)].index[0], time_col]

        dataframe[resource_col] = dataframe[resource_col].fillna(0)
        work_frame = dataframe[['w_end_time', time_col, resource_col]].copy()
        for index in tqdm(work_frame.index):
            resource = work_frame.at[index, resource_col]
            time_event = work_frame.at[index, time_col]
            dataframe.at[index, 'active_resource_cases'] = len(set.intersection(set(
                np.where((work_frame['w_end_time'] > time_event) & (work_frame[resource_col] == resource))[0].tolist()),
                                                                                set(np.where((work_frame[
                                                                                                  time_col] < time_event) & (
                                                                                                         work_frame[
                                                                                                             resource_col] == resource))[
                                                                                        0].tolist())))

        return dataframe.drop('w_end_time', axis=1)

    elif activity_method == 'all_activities':
        median_duration = dataframe.groupby(event_col).median()[time_between_events_col].to_dict()
        for index in dataframe.index:
            dataframe.at[index, 'end_time'] = dataframe.at[index, time_col] + datetime.timedelta(
                seconds=median_duration[dataframe.at[index, event_col]])

        dataframe[resource_col] = dataframe[resource_col].fillna(0)
        work_frame = dataframe[['end_time', time_col, resource_col]].copy()
        for index in tqdm(work_frame.index):
            resource = work_frame.at[index, resource_col]
            time_event = work_frame.at[index, time_col]
            dataframe.at[index, 'active_resource_cases'] = len(set.intersection(set(
                np.where((work_frame['end_time'] > time_event) & (work_frame[resource_col] == resource))[0].tolist()),
                                                                                set(np.where((work_frame[
                                                                                                  time_col] < time_event) & (
                                                                                                         work_frame[
                                                                                                             resource_col] == resource))[
                                                                                        0].tolist())))

        return dataframe.drop('end_time', axis=1)

    elif activity_method == 'both':
        median_duration = dataframe.groupby(event_col).median()[time_between_events_col].to_dict()
        for index in dataframe.index:
            dataframe.at[index, 'end_time'] = dataframe.at[index, time_col] + datetime.timedelta(
                seconds=median_duration[dataframe.at[index, event_col]])

        for index in tqdm(dataframe[dataframe[status_col] == start_variable].index):
            broken = False
            or_index = index
            case_id = dataframe.at[index, case_id_col]
            act_nr = dataframe.at[index, act_nr_col]
            target = act_nr + 1

            while dataframe.at[or_index, event_col] != dataframe.at[index, next_event_col]:
                try:
                    index = dataframe[(dataframe[case_id_col] == case_id) & (dataframe[act_nr_col] == target)].index[0]
                    target += 1

                except:
                    dataframe.at[or_index, 'w_end_time'] = pd.NaT
                    broken = True
                    break

            if broken:
                continue
            else:
                dataframe.at[or_index, 'w_end_time'] = dataframe.at[
                    dataframe[(dataframe[case_id_col] == case_id) & (dataframe[act_nr_col] == target)].index[0], time_col]

        dataframe[resource_col] = dataframe[resource_col].fillna(0)
        work_frame = dataframe[['end_time', 'w_end_time', time_col, resource_col]].copy()
        for index in tqdm(work_frame.index):
            resource = work_frame.at[index, resource_col]
            time_event = work_frame.at[index, time_col]
            dataframe.at[index, 'active_resource_cases(all)'] = len(set.intersection(set(
                np.where((work_frame['end_time'] > time_event) & (work_frame[resource_col] == resource))[0].tolist()),
                                                                                     set(np.where((work_frame[
                                                                                                       time_col] < time_event) & (
                                                                                                              work_frame[
                                                                                                                  resource_col] == resource))[
                                                                                             0].tolist())))
            dataframe.at[index, 'active_resource_cases(w)'] = len(set.intersection(set(
                np.where((work_frame['w_end_time'] > time_event) & (work_frame[resource_col] == resource))[0].tolist()),
                                                                                   set(np.where((work_frame[
                                                                                                     time_col] < time_event) & (
                                                                                                            work_frame[
                                                                                                                resource_col] == resource))[
                                                                                           0].tolist())))

        return dataframe.drop(['end_time', 'w_end_time'], axis=1)

    else:
        raise ValueError('Unknown method used, try one of the following methods: [w_activities, all_activities, both]')

def pipeline(data, timer):
    """
    This pipeline combines all the functions defined above. It makes sure that all steps are executed, and it returns
    the total time that the pipeline took to execute. 
    """
    #resource_active_cases(data)
    time_based_columns(data, "time:timestamp")

    timer.send("Time to add global variables (in seconds): ")
    return data
