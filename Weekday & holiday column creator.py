import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

def weekend_and_work_hours(dataframe, date_column:str, start_hours:int, stop_hours:int):
    """
    Adds the extra columns regarding to the weekend and working hours
    
    :param dataframe: original imported pandas dataframe
    :type dataframe: pd.DataFrame
    :param date_column: which column the dates of the event are stored
    :type date_column: string
    :param start_hours: specification of when the work day starts
    :type start_hours: integer
    :param stop_hours: specification of when the work day ends
    :type stop_hours: integer
    :return: The new complete dataframe with the following columns 
            time_since_week_start: Time (string) since Monday 00:00
            weekend: Boolean value if its a weekend or not
            work_time: Boolean value if its work time or not
            time_to_work_hours: Time (string) until the next work day starts (or 0 if its during working hours)
    :rtype: pd.DataFrame
    """
    dataframe['time_since_week_start'] = dataframe[date_column].apply(lambda x: pd.Timedelta('{} days {}'.format(x.weekday(), x.time())))
    dataframe['weekend'] = dataframe[date_column].apply(lambda x: True if 4<x.weekday()<7 else True
                                                        if x.weekday() == 4 and x.hour>stop_hours-1 else True 
                                                        if x.weekday() == 0 and x.hour<start_hours else False)
    dataframe['work_time'] = dataframe[date_column].apply(lambda x: False if 4<x.weekday()<7 else True if start_hours<=x.hour<stop_hours else False)
    dataframe['time_to_work_hours'] = dataframe[date_column].apply(lambda x: pd.Timedelta('0h0m') if x.weekday()<5 and start_hours<=x.hour<stop_hours else
                                                                       datetime.datetime.combine(x.date(), datetime.time(start_hours,0,0,0)) - x if x.weekday()<5 and start_hours>x.hour else
                                                                       datetime.datetime.combine(x.date()+datetime.timedelta(days = 1), datetime.time(stop_hours,0,0,0)) - x if x.weekday()<4 and stop_hours<=x.hour else
                                                                        datetime.datetime.combine((x.date()+datetime.timedelta(days = 7-x.weekday())), datetime.time(start_hours,0,0,0)) - x) 
    return dataframe

def national_holidays(dataframe, date_column:str, holiday_dates:list):
    """
    Adds the national holidays to a dataframe by using a pre-provided list of holidays
    
    :param dataframe: original imported pandas dataframe
    :type dataframe: pd.DataFrame
    :param date_column: which column the dates of the event are stored
    :type date_column: string
    :param holiday_dates: List with all the holiday dates which happen
    :type holiday_dates: list
    :return: New complete dataframe with the following columns added
            time_until_holiday: Day (string) until the next holiday starts
            is_holiday: Boolean value if its a national holiday or not
    :rtype: pd.DataFrame
    """
    dataframe['time_until_next_holiday'] = dataframe[date_column].apply(lambda x: pd.Timedelta('0h0m') if x.date() in holiday_dates else 
                                                                            min(holiday - x.date() for holiday in holiday_dates if (holiday - x.date())>pd.Timedelta('0 days')))
    dataframe['is_holiday'] = dataframe[date_column].apply(lambda x: True if x.date() in holiday_dates else False)
    return dataframe

