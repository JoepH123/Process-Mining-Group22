import copy
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix as cm, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import predictors_columns
import global_vars
import splitter, constants, baseline, create_figures
from performance_measures import time_execution
import decision_tree
import argparse
from compute_and_convert_time import *
import warnings

def prepare_data(unprocessed_dataset, pipeline_dataset, timer):
    # do this if the files are not split already
    # splitter.convert_raw_dataset(constants.RAW_DATASET_PATH, constants.CONVERTED_DATASET_PATH)
    # timer.send("Time to convert dataset (in seconds): ")

    full_data = pd.read_csv(unprocessed_dataset)

    # ------- ONLY FOR TESTING ----------

    # full_data = full_data[-100000:].reset_index(drop=True)

    # ------- END -----------------------

    # Add calculated predictors
    data = predictors_columns.pipeline(full_data, timer)
    data = global_vars.pipeline(data, timer)

    # Save the data to a file so we don't have to do this again
    data.to_csv(pipeline_dataset)

    train_data, test_data = splitter.split_dataset(data, 0.2)
    # splitter.time_series_split(train_data, 5)

    timer.send("Time to split dataset (in seconds): ")
    
    # return train_data, test_data

def read_data(pipeline_dataset, timer):
    data = pd.read_csv(pipeline_dataset)
    
    print(f"Average case duration: {max_four_digit_time_rep_accurate(compute_avg_case_duration(data))}")
    
    data[constants.CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        data[constants.CASE_TIMESTAMP_COLUMN])
    data = data.astype({constants.NEXT_EVENT: 'str'})
    train_data, test_data = splitter.split_dataset(data, 0.2)
    # splitter.time_series_split(train_data, 5)
    timer.send("Time to read dataset (in seconds): ")
    return train_data, test_data

def main(args):
    # Parsed arguments
    dataset = args.dataset
    generate = args.generate
    plots = args.plots

    # set up the timer
    timer = time_execution()
    timer.__next__()
    
    if plots:
        warnings.warn('Plots are currently generated, This might take a while')


    # Condition in the dataset version
    if(dataset==2017):
        # 2017 Dataset
        unprocessed_dataset = constants.CONVERTED_2017_DATASET_PATH
        pipeline_dataset = constants.PIPELINED_2017_DATASET_PATH
    else:
        # 2012 Dataset
        unprocessed_dataset = constants.CONVERTED_DATASET_PATH
        pipeline_dataset = constants.PIPELINED_DATASET_PATH


    # Condition on whether to re-run the data preprocessing
    if(generate):
        # Includes calculation of predictors
        #train_data, test_data = prepare_data(unprocessed_dataset, pipeline_dataset, timer)
        print('generating dataset....')
        prepare_data(unprocessed_dataset, pipeline_dataset, timer)
    print('reading dataset....')
    train_data, test_data = read_data(pipeline_dataset, timer)

    # FORWARD CHAINING EXAMPLE
    # fc_train, fc_test = splitter.time_series_split(train_data, 5)
    # for i in range(0, 4):
    #     print("Tail of train of",i+1,":\n", fc_train[i].tail())
    #     print("Tail of test of",i+1,":\n", fc_test[i].tail())

    # BASELINE MODEL
    baseline_next_event_df, baseline_time_elapsed_df = baseline.train_baseline_model(
        train_data, timer)

    # Evaluating the model
    baseline.evaluate_baseline_model(baseline_next_event_df, baseline_time_elapsed_df, test_data, timer)

    # DECISION TREE AND RANDOM FOREST
    decision_tree.compare_all_models(train_data, test_data, timer)

    if plots:
        create_figures.generate_plots()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="year of dataset", default=2012, type=int
    )
    parser.add_argument(
        "--generate", help="0 if the data should be read, 1 if it should be generated", default=0, type=int
    )
    parser.add_argument(
        "--plots", help="0 if plots should not be generated, 1 if plots should be generated", default=0, type=int
    )

    args = parser.parse_args()
    main(args)
