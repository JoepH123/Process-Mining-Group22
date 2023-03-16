import pandas as pd
import datetime
import workalendar
import country_converter as cc
import numpy as np
from typing import Union


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
    
    df['total_amount'] = df.groupby(['year', 'month', 'day'], as_index=False).agg(total_amount=('case:AMOUNT_REQ', 'sum'))['total_amount']
    
    return df


from constants import GLOBAL_DATASET_PATH

df = pd.read_csv(GLOBAL_DATASET_PATH)
add_liquidity(df, "time:timestamp")
