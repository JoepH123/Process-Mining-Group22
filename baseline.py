
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error


# IMPORTANT: change the filepaths


# load the data
train_data = pd.read_csv('C:/Users/USER/Documents/University/Year 3/Process mining/BPI_Challenge_2012-training.csv')
test_data = pd.read_csv('C:/Users/USER/Documents/University/Year 3/Process mining/BPI_Challenge_2012-test.csv')

# change the event time:timestamp to datetime
train_data['event time:timestamp'] = pd.to_datetime(train_data['event time:timestamp'])
test_data['event time:timestamp'] = pd.to_datetime(test_data['event time:timestamp'])

# NOTES:
# case concept:name = id
# event concept:name = position
# event time:timestamp = time
# no validation set due to no hyperparameters
# for now uses given split, new split needed

# get the event that follows the current one for each case (row), and the time elapsed
train_data['time until next event'] = - train_data.groupby('case concept:name')['event time:timestamp'].diff(-1).dt.total_seconds()
train_data['next event'] = train_data.groupby('case concept:name')['event concept:name'].shift(-1)
test_data['time until next event'] = - test_data.groupby('case concept:name')['event time:timestamp'].diff(-1).dt.total_seconds()
test_data['next event'] = test_data.groupby('case concept:name')['event concept:name'].shift(-1)
 
print('here')
# ACTUAL MODEL
# get the most frequent event that follows a certain event
next_event_df = pd.DataFrame(train_data.groupby('event concept:name')['next event'].agg(pd.Series.mode))
next_event_df.rename(columns={'next event': 'predicted next event'}, inplace=True)
train_data = train_data.merge(next_event_df, how='left', on='event concept:name')

# get the average of the time elapsed after a certain event
time_elapsed_df = pd.DataFrame(train_data.groupby('event concept:name')['time until next event'].agg('mean'))
time_elapsed_df.rename(columns={'time until next event': 'predicted time until next event'}, inplace=True)
train_data = train_data.merge(time_elapsed_df, how='left', on='event concept:name')

# this datafarame stores all the rows where time until next event and next event are null
# we should decide what to do with it, for now i remove missing values
exp_df = train_data[train_data['time until next event'].isnull()]
train_data.dropna(inplace=True)

# make predictions for test set
test_data = test_data.merge(next_event_df, how='left', on='event concept:name')
test_data = test_data.merge(time_elapsed_df, how='left', on='event concept:name')
# also remove missing values:
test_data.dropna(inplace=True)

# calculate accuracy of the classification for predicted next event
print("Accuracy score of baseline for train set (next event): ", accuracy_score(
    train_data['next event'], train_data['predicted next event']))
print("Number of misclassifications: ", accuracy_score(
    train_data['next event'], train_data['predicted next event'], normalize=False))
print("-------------------------------------------------------------------")
print("Accuracy score of baseline for test set (next event): ", accuracy_score(
    test_data['next event'], test_data['predicted next event']))
print("Number of misclassifications: ", accuracy_score(
    test_data['next event'], test_data['predicted next event'], normalize=False))
print("-------------------------------------------------------------------")
print("Mean absolute error of baseline for train set (time until next event): ", mean_absolute_error(
    train_data['time until next event'], train_data['predicted time until next event']))
print("Mean absolute error of baseline for test set (time until next event): ", mean_absolute_error(
    test_data['time until next event'], test_data['predicted time until next event']))



