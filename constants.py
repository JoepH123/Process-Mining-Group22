# ----------------------CONSTANTS----------------------------------
"""The path to the raw dataset, in .xes.gz"""
RAW_DATASET_PATH = './Datasets/BPI_Challenge_2012.xes.gz'
"""The path to the converted dataset, in .csv"""
CONVERTED_DATASET_PATH = './Datasets/BPI_Challenge_2012.csv'
"""The path to the dataset with additional columns, in .csv"""
GLOBAL_DATASET_PATH = './Datasets/dataset_holiday_weekday.csv'
"""The path to the training set, as outputted by 'xes2csv.jar'."""
TRAINING_DATA_PATH = './Datasets/BPI_Challenge_2012-training.csv'
"""The path to the testing set, as outputted by 'xes2csv.jar'."""
TEST_DATA_PATH = './Datasets/BPI_Challenge_2012-test.csv'

"""The identifier of dataset column holding the case ID."""
CASE_ID_COLUMN = 'case:concept:name'
"""The identifier of dataset column holding the case position/state."""
CASE_POSITION_COLUMN = 'concept:name'
"""The identifier of dataset column holding the timestamp."""
CASE_TIMESTAMP_COLUMN = 'time:timestamp'
"""The identifier of dataset column counting the number of the current event in case."""
CASE_STEP_NUMBER_COLUMN = 'activity number in case'
"""The identifier of dataset column counting the number of events until end of case."""
CASE_STEP_NUMBER_INVERSE_COLUMN = 'number of activity in case inverse'
"""The identifier of dataset column counting the number of cases that start
after the current event, inclusive."""
CASE_START_COUNT = 'case start count'
"""The identifier of dataset column counting the number of cases that end
before the current event, inclusive."""
CASE_END_COUNT = 'case end count'
