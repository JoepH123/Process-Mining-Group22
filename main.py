import copy
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix as cm, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from predictors_columns import pipeline
import splitter, constants, baseline
from performance_measures import classification_performance, regression_performance, time_execution

def prepare_data(timer):

    
    # do this if the files are not split already
    # splitter.convert_raw_dataset(constants.RAW_DATASET_PATH, constants.CONVERTED_DATASET_PATH)
    # timer.send("Time to convert dataset (in seconds): ")

    full_data = pd.read_csv(constants.CONVERTED_DATASET_PATH)
    full_data[constants.CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        full_data[constants.CASE_TIMESTAMP_COLUMN])

    data = pipeline(full_data)
    data.to_csv(constants.PIPELINED_DATASET_PATH)

    train_data, test_data = splitter.split_dataset(data, 0.2)
    # splitter.time_series_split(train_data, 5)

    timer.send("Time to split dataset (in seconds): ")

    return train_data, test_data

    

def main():
    # set up the timer
    timer = time_execution()
    timer.__next__()

    train_data, test_data = prepare_data(timer)

    baseline_next_event_df, baseline_time_elapsed_df = baseline.train_baseline_model(
        train_data, timer)
    
    baseline.evaluate_baseline_model(baseline_next_event_df, baseline_time_elapsed_df, test_data, timer)


if __name__ == "__main__":
    main()