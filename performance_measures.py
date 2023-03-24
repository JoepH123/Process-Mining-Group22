import copy
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix as cm, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from predictors_columns import pipeline
import splitter, constants
from compute_and_convert_time import max_four_digit_time_rep_accurate

def classification_performance(data, conf_matrix_name):
    """Prints out performance metrics for the classification task.
    
    :param data: dataframe containing these columns: next event, predicted next event
    :type data: dataframe
    :param conf_matrix_name: name of file to save the confusion matrix in
    :type conf_matrix_name: string
    :return: None
    """
    print("Accuracy score (next event): ", accuracy_score(
        data[constants.NEXT_EVENT].astype('str'), data[constants.NEXT_EVENT_PREDICTION].astype('str')))
    print("Number of misclassifications: ", accuracy_score(
        data[constants.NEXT_EVENT].astype('str'), data[constants.NEXT_EVENT_PREDICTION].astype('str'), normalize=False))
    #print('Confusion matrix:')
    
    # create confusion matrix
    cf_matrix = cm(data[constants.NEXT_EVENT].astype('str'), data[constants.NEXT_EVENT_PREDICTION].astype('str'))

    # normalize the confusion matrix
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    indices = list(set(data[constants.NEXT_EVENT].unique()) | set(data[constants.NEXT_EVENT_PREDICTION].unique()))
    cf_df = pd.DataFrame(cf_matrix, index = indices,
                         columns = indices)

    # plot the confusion matrix, add xticklabels=cf_df.columns, yticklabels=cf_df.index
    # as arguments to see the event names on the axes
    ax = sns.heatmap(cf_df, annot=False, fmt='g', xticklabels=False, yticklabels=False, annot_kws={"fontsize":10})
    ax.set_title('Confusion Matrix \n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    
    # display the visualization of the confusion matrix, save it as png
    fig = ax.get_figure()
    fig.savefig(conf_matrix_name)
    plt.show()
    plt.clf()
    
    print("Classification report: ", classification_report(data[constants.NEXT_EVENT].astype('str'), 
       data[constants.NEXT_EVENT_PREDICTION].astype('str'), zero_division=1))
    
    # Experimental - use only if decided
    # print("Precision, recall, fscore, support with possible aggregation: ",
    #       precision_recall_fscore_support(data['next event'], data['predicted next event'], average=None))
    
    
def regression_performance(data):
    """Prints out performance metrics for the regression task.
    
    :param data: dataframe containing these columns: time until next event, 
        predicted time until next event
    :type data: dataframe
    :return: None
    """
    
    MAE = max_four_digit_time_rep_accurate(
            mean_absolute_error(data[constants.TIME_DIFFERENCE],
            data[constants.TIME_DIFFERENCE_PREDICTION]))
    RMSE = max_four_digit_time_rep_accurate(
            mean_squared_error(data[constants.TIME_DIFFERENCE], 
            data[constants.TIME_DIFFERENCE_PREDICTION], squared=False))
    data_no_zero = data[data[constants.TIME_DIFFERENCE] != 0]
    RMSPE = max_four_digit_time_rep_accurate(np.sqrt(np.mean(
        np.square(((data_no_zero[constants.TIME_DIFFERENCE] 
                    - data_no_zero[constants.TIME_DIFFERENCE_PREDICTION]) / 
                   data_no_zero[constants.TIME_DIFFERENCE])), axis=0)))

    print("Mean absolute error (time until next event): ", MAE)
    print("Root mean squared error: ", RMSE)
    print("Root mean squared percentage error: ", RMSPE)

def time_execution():
    """A couroutine that prints a message it recieves through .send()
    and the the seconds passed since the last time it was called.
    """
    start_time = time.process_time()
    while True:
        string = (yield)
        new_time = time.process_time()
        print('\n' + string, new_time - start_time)
        start_time = new_time
