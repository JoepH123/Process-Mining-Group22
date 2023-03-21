# ----------------------DATASETS----------------------------------
"""The path to the raw dataset, in .xes.gz"""
RAW_DATASET_PATH = './Datasets/BPI_Challenge_2012.xes.gz'
"""The path to the converted dataset, in .csv"""
CONVERTED_DATASET_PATH = './Datasets/BPI_Challenge_2012.csv'
"""The path to the dataset with added predictors, in .csv"""
PIPELINED_DATASET_PATH = './Datasets/BPI_Challenge_2012_pipeline.csv'

# 2017 DATASETS #
"""The path to the raw dataset, in .xes.gz"""
RAW_2017_DATASET_PATH = './Datasets/BPI_Challenge_2012.xes.gz'
"""The path to the converted dataset, in .csv"""
CONVERTED_2017_DATASET_PATH = './Datasets/BPI_Challenge_2017.csv'
"""The path to the dataset with added predictors, in .csv"""
PIPELINED_2017_DATASET_PATH = './Datasets/BPI_Challenge_2017_pipeline.csv'

# ----------------------ATTRIBUTES----------------------------------
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
"""The identifier of dataset column counting the number of cases that are open
after the current event happens."""
ACTIVE_CASES = 'active cases'
"""The identifier of dataset column holding the amount requested in the case."""
AMOUNT_REQUESTED_COLUMN = 'case:AMOUNT_REQ'

# ----------------------OUTPUTS----------------------------------
"""The identifier of dataset column holding the next event."""
NEXT_EVENT = 'next event'
"""The identifier of dataset column holding the time until next event."""
TIME_DIFFERENCE = 'time until next event'
"""The identifier of dataset column holding the next event prediction."""
NEXT_EVENT_PREDICTION = 'predicted next event'
"""The identifier of dataset column holding the time until next event prediction."""
TIME_DIFFERENCE_PREDICTION = 'predicted time until next event'
