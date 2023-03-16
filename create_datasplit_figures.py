import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import constants
import numpy as np


def load_data(train_path=constants.TRAINING_DATA_PATH, test_path=constants.TEST_DATA_PATH, total_path=constants.CONVERTED_DATASET_PATH):
    """
    This function downloads the datasets that are necessary to plot the two visualizations for the data split.

    :param train_path: path to the training dataset
    :type train_path: string
    :param test_path: path to the test dataset
    :type test_path: string
    :param total_path: path to the converted dataset with all data (as before the train-test split)
    :type total_path: string
    """
    training_data_2012 = pd.read_csv(train_path, parse_dates=['case:REG_DATE', 'time:timestamp']) 
    training_data_2012 = training_data_2012.sort_values(by=["time:timestamp"])
    test_data_2012 = pd.read_csv(test_path, parse_dates=['case:REG_DATE', 'time:timestamp']) 
    test_data_2012 = test_data_2012.sort_values(by=["time:timestamp"])
    total_data_2012 = pd.read_csv(total_path, parse_dates=['case:REG_DATE', 'time:timestamp'])
    total_data_2012 = total_data_2012.sort_values(by=["time:timestamp"])
    return training_data_2012, test_data_2012, total_data_2012


def create_plot_without_removed_cases_with_split_line(training_data_2012, test_data_2012):
    """
    This function creates the plot that shows the correctness of the train-test split. It also has a red line indicating that no data
    from the future is used. This red line, shows that no cases in the training set are still going on during the test set's timeline.

    :param training_data_2012: Dataframe containing the training dataset
    :type training_data_2012: pd.DataFrame
    :param test_data_2012: Dataframe containing the test dataset
    :type test_data_2012: pd.DataFrame
    """
    fig = go.Figure(data=go.Scatter(x=training_data_2012["time:timestamp"], y=training_data_2012["case:concept:name"], mode='markers', marker={'color': '#aecf9e', 'size': 2}, hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=test_data_2012["time:timestamp"], y=test_data_2012["case:concept:name"], mode='markers', marker={'color': '#2ab4ea', 'size': 2}, hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[datetime.datetime(2012, 2, 6, 15, 35, 0), datetime.datetime(2012, 2, 6, 15, 35, 0)], y=[170000,215000], mode='lines', line={'color': 'red'}, hoverinfo='skip', showlegend=False))
    return fig.show()


def create_plot_with_removed_cases(train, test, total):
    """
    This function creates the plot that shows the correctness of the train-test split. It also shows the activities of the cases that were
    removed, to make this valid train-test split. 

    :param train: Dataframe containing the training dataset
    :type train: pd.DataFrame
    :param test: Dataframe containing the test dataset
    :type test: pd.DataFrame
    :param total: Dataframe containing the entire dataset, before the train-test split. 
    :type total: pd.DataFrame
    """
    test_cases = np.unique(train['case:concept:name'])
    train_cases = np.unique(test['case:concept:name'])
    total['in_test'] = total['case:concept:name'].isin(test_cases)
    total['in_train'] = total['case:concept:name'].isin(train_cases)

    total['color'] = total.apply(lambda x: 'blue' if x['in_train'] else 'red' if x['in_test'] else 'grey_2' if 
                     total[total['case:concept:name'] == x['case:concept:name']]['time:timestamp'].min()+ datetime.timedelta(days = 30) < x['time:timestamp']
                     else 'grey_4' if pd.Timestamp('2011-12-18 15:35:28.600000+0000', tz='UTC') < total[total['case:concept:name'] == x['case:concept:name']]['time:timestamp'].min() < pd.Timestamp('2012-01-01 15:35:28.600000+0000', tz='UTC')
                     else 'grey_3' if total[total['case:concept:name'] == x['case:concept:name']]['time:timestamp'].min() < pd.Timestamp('2012-01-01 15:35:28.600000+0000', tz='UTC')
                     else 'grey_1' if x['time:timestamp'] < pd.Timestamp('2012-02-06 15:35:28.600000+0000', tz='UTC') else 'grey_2', axis = 1)
    
    fig = px.scatter(total, x="time:timestamp", y="case:concept:name", color='color',
                     hover_data=['case:concept:name', 'time:timestamp', 'concept:name'],
                     color_discrete_sequence= ['#bce6ba', 'rgb(0, 0, 0)', 'rgb(0, 0, 0)', 'rgb(0,0,0)', 'rgb(0,0,0)', '#a9e2fb'])

    opacity = {'blue': 1, 'grey_1': 0.02, 'red': 1, 'grey_2': 0.50, 'grey_3': 0.2, 'grey_4': 0.1}
    fig.for_each_trace(lambda trace: trace.update(opacity = opacity[trace.name]) if trace.name in opacity.keys() else (),)

    fig.update_traces(marker={'size': 3})

    for trace in fig['data']: 
        if trace['name'] == 'grey_1' or trace['name'] == 'grey_3' or trace['name'] == 'grey_4': 
            trace['showlegend'] = False
            
    newnames = {'blue':'In training set', 'grey_2': 'Removed from both sets', 'red':'In test set', 'grey_1': 'dc',
                'grey_3': 'dc', 'grey_4': 'dc'}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    return fig.show()


# Running the functions:

train, test, total = load_data()
create_plot_without_removed_cases_with_split_line(train, test)
create_plot_with_removed_cases(train, test, total)
