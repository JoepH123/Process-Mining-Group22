import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import constants
import numpy as np
from tqdm import tqdm
import predictors_columns
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import copy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import splitter


def load_data(total_path=constants.PIPELINED_DATASET_PATH):
    """
    This function downloads the datasets that are necessary to plot the two visualizations for the data split.

    :param train_path: path to the training dataset
    :type train_path: string
    :param test_path: path to the test dataset
    :type test_path: string
    :param total_path: path to the converted dataset with all data (as before the train-test split)
    :type total_path: string
    """
    total_data_2012 = pd.read_csv(total_path, parse_dates=['case:REG_DATE', constants.CASE_TIMESTAMP_COLUMN])
    total_data_2012 = total_data_2012.sort_values(by=[constants.CASE_TIMESTAMP_COLUMN])
    training_data_2012, test_data_2012 = splitter.split_dataset(total_data_2012, 0.2)
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
    fig = go.Figure(data=go.Scatter(x=training_data_2012[constants.CASE_TIMESTAMP_COLUMN], y=training_data_2012["case:concept:name"], mode='markers', marker={'color': '#aecf9e', 'size': 2}, hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=test_data_2012[constants.CASE_TIMESTAMP_COLUMN], y=test_data_2012["case:concept:name"], mode='markers', marker={'color': '#2ab4ea', 'size': 2}, hoverinfo='skip', showlegend=False))
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
    test_cases = np.unique(train[constants.CASE_ID_COLUMN])
    train_cases = np.unique(test[constants.CASE_ID_COLUMN])
    total['in_test'] = total[constants.CASE_ID_COLUMN].isin(test_cases)
    total['in_train'] = total[constants.CASE_ID_COLUMN].isin(train_cases)

    total['color'] = total.apply(lambda x: 'blue' if x['in_train'] else 'red' if x['in_test'] else 'grey_2' if
                     total[total[constants.CASE_ID_COLUMN] == x[constants.CASE_ID_COLUMN]][constants.CASE_TIMESTAMP_COLUMN].min()+ datetime.timedelta(days = 30) < x[constants.CASE_TIMESTAMP_COLUMN]
                     else 'grey_4' if pd.Timestamp('2011-12-18 15:35:28.600000+0000', tz='UTC') < total[total[constants.CASE_ID_COLUMN] == x[constants.CASE_ID_COLUMN]][constants.CASE_TIMESTAMP_COLUMN].min() < pd.Timestamp('2012-01-01 15:35:28.600000+0000', tz='UTC')
                     else 'grey_3' if total[total[constants.CASE_ID_COLUMN] == x[constants.CASE_ID_COLUMN]][constants.CASE_TIMESTAMP_COLUMN].min() < pd.Timestamp('2012-01-01 15:35:28.600000+0000', tz='UTC')
                     else 'grey_1' if x[constants.CASE_TIMESTAMP_COLUMN] < pd.Timestamp('2012-02-06 15:35:28.600000+0000', tz='UTC') else 'grey_2', axis = 1)

    fig = px.scatter(total, x=constants.CASE_TIMESTAMP_COLUMN, y="case:concept:name", color='color',
                     hover_data=[constants.CASE_ID_COLUMN, constants.CASE_TIMESTAMP_COLUMN, constants.CURRENT_EVENT],
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

def retrieve_training_activity_performances(train_data, clf, columns, normal = True):
    """
    This is an adapted function of the train_activity_model in decision_tree.py, to be able to return the metric scores

    :param train_data: the training data for the model
    :type train_data: pd.DataFrame
    :param clf: scikitlearn model input with the wanted parameters
    :param columns: columns to use for training
    :type columns: list
    :return: the trained model accompanied by the training scores on the following parameters; precision, recall, f1-score & accuracy
    """
    X = train_data[columns]

    if not normal:
        y = train_data['next_activity_id']
    else:
        y = train_data[constants.NEXT_EVENT]


    clf.fit(X, y)
    preds = clf.predict(X)

    train_data[constants.NEXT_EVENT_PREDICTION] = preds

    rep = classification_report(train_data[constants.NEXT_EVENT],
                                train_data[constants.NEXT_EVENT_PREDICTION], zero_division=1, output_dict=True)

    prec, recall, f1, acc = rep['weighted avg']['precision'], rep['weighted avg']['recall'], rep['weighted avg']['f1-score'], rep['accuracy']


    return clf, prec, recall, f1, acc

def retrieve_testing_activity_performances(test_data, clf, columns, normal=True):
    """
    This is an adapted function of the test_activity_model in decision_tree.py, to be able to return the metric scores

    :param test_data: the testing data for the model
    :type test_data: pd.DataFrame
    :param clf: trained scikitlearn model
    :param columns: columns to used to make predictions (should be the same as the ones used for testing)
    :type columns: list
    :return: the testing scores on the following parameters; precision, recall, f1-score & accuracy
    """
    X = test_data[columns]

    if not normal:
        y = test_data['next_activity_id']
    else:
        y = test_data[constants.NEXT_EVENT]

    preds = clf.predict(X)

    test_data[constants.NEXT_EVENT_PREDICTION] = preds

    rep = classification_report(test_data[constants.NEXT_EVENT],
                                test_data[constants.NEXT_EVENT_PREDICTION], zero_division=1, output_dict=True)

    prec, recall, f1, acc = rep['weighted avg']['precision'], rep['weighted avg']['recall'], rep['weighted avg']['f1-score'], rep['accuracy']

    return prec, recall, f1, acc

def retrieve_training_time_performances(train_data, clf, columns):
    """
    This is an adapted function of the train_time_model in decision_tree.py, to be able to return the metric scores

    :param train_data: the training data for the model
    :type train_data: pd.DataFrame
    :param clf: scikitlearn model input with the wanted parameters
    :param columns: columns to use for training
    :type columns: list
    :return: the trained model accompanied by the training scores on the following parameters; Root Mean Squared Percentage Error & Mean Absolute Error
    """
    X = train_data[columns]

    y = train_data[constants.TIME_DIFFERENCE]

    clf.fit(X, y)
    preds = clf.predict(X)

    train_data[constants.TIME_DIFFERENCE_PREDICTION] = preds

    data_no_zero = train_data[train_data[constants.TIME_DIFFERENCE] != 0]

    rmspe, mae = np.sqrt(np.mean(np.square(((data_no_zero[constants.TIME_DIFFERENCE] - data_no_zero[constants.TIME_DIFFERENCE_PREDICTION]) / data_no_zero[constants.TIME_DIFFERENCE])), axis=0)), mean_absolute_error(
        train_data[constants.TIME_DIFFERENCE], train_data[constants.TIME_DIFFERENCE_PREDICTION])

    return clf, rmspe, mae

def retrieve_testing_time_performances(test_data, clf, columns):
    """
    This is an adapted function of the test_time_model in decision_tree.py, to be able to return the metric scores

    :param test_data: the testing data for the model
    :type test_data: pd.DataFrame
    :param clf: trained scikitlearn model
    :param columns: columns to used to make predictions (should be the same as the ones used for testing)
    :type columns: list
    :return: the testing scores on the following parameters; Root Mean Squared Percentage Error & Mean Absolute Error
    """
    X = test_data[columns]

    y = test_data[constants.TIME_DIFFERENCE]
    preds = clf.predict(X)

    test_data[constants.TIME_DIFFERENCE_PREDICTION] = preds

    data_no_zero = test_data[test_data[constants.TIME_DIFFERENCE] != 0]

    rmspe, mae = np.sqrt(np.mean(np.square(((data_no_zero[constants.TIME_DIFFERENCE] - data_no_zero[constants.TIME_DIFFERENCE_PREDICTION]) / data_no_zero[constants.TIME_DIFFERENCE])),axis=0)), mean_absolute_error(
        test_data[constants.TIME_DIFFERENCE], test_data[constants.TIME_DIFFERENCE_PREDICTION])

    return rmspe, mae

def compute_model_performances(train_data, test_data):
    """
    An adapted version of the compare_all_models function in decision_tree.py to be able to return the metric scores for all the applicable models

    :param train_data: The training data
    :type train_data: pd.DataFrame
    :param test_data: The testing data
    :type test_data: pd.DataFrame
    :return: A tuple with the metric scores for all the models in the order: Decision tree (precision, recall, f1-score, accuracy),
        Random forest on the test set (precision, recall, f1-score, accuracy), Linear regression (RMSPE, MAE), random forest regression on the test set
        (RMSPE, MAE), Random Forest (also Regression) on the train set (precision, recall, f1-score, accuracy, RMSPE, MAE)
    :rtype: tuple
    """
    cols = [column for column in train_data.columns if column not in [constants.NEXT_EVENT, constants.TIME_DIFFERENCE]]

    train_data = copy.deepcopy(train_data)
    train_data['next_activity_id'] = pd.factorize(train_data[constants.NEXT_EVENT])[0]

    test_data = copy.deepcopy(test_data)
    test_data['next_activity_id'] = pd.factorize(test_data[constants.NEXT_EVENT])[0]

    train_data_dummies = train_data[[constants.NEXT_EVENT, constants.TIME_DIFFERENCE, 'next_activity_id']]
    for column in cols:
        train_data_dummies = train_data_dummies.join(pd.get_dummies(train_data[column]).add_prefix(column + '_'))

    test_data_dummies = test_data[[constants.NEXT_EVENT, constants.TIME_DIFFERENCE, 'next_activity_id']]
    for column in cols:
        test_data_dummies = test_data_dummies.join(pd.get_dummies(test_data[column]).add_prefix(column + '_'))

    for col in [column for column in train_data_dummies.columns if column not in test_data_dummies.columns]:
        test_data_dummies[col] = 0

    cols = [column for column in train_data_dummies.columns if
            column not in [constants.NEXT_EVENT, constants.TIME_DIFFERENCE]]

    clf = DecisionTreeClassifier()
    dec_tree_clas, DT_train_prec, DT_train_recall, DT_train_f1, DT_train_acc = retrieve_training_activity_performances(train_data_dummies,clf, cols)
    DT_prec, DT_recall, DT_f1, DT_acc = retrieve_testing_activity_performances(test_data_dummies, dec_tree_clas, cols)

    clf = RandomForestClassifier()
    rand_forest_class, RF_train_prec, RF_train_recall, RF_train_f1, RF_train_acc = retrieve_training_activity_performances(train_data_dummies, clf, cols)
    RF_prec, RF_recall, RF_f1, RF_acc = retrieve_testing_activity_performances(test_data_dummies, rand_forest_class, cols)

    reg = LinearRegression()
    lin_regr, LR_train_rmspe, LR_train_mae = retrieve_training_time_performances(train_data_dummies, reg, cols)
    LR_rmspe, LR_mae = retrieve_testing_time_performances(test_data_dummies, lin_regr, cols)

    reg = RandomForestRegressor()
    rand_forest_regr, RF_train_rmspe, RF_train_mae = retrieve_training_time_performances(train_data_dummies, reg, cols)
    RF_rmspe, RF_mae = retrieve_testing_time_performances(test_data_dummies, rand_forest_regr, cols)

    return DT_prec, DT_recall, DT_f1, DT_acc, RF_prec, RF_recall, RF_f1, RF_acc, LR_rmspe, LR_mae, RF_rmspe, RF_mae, RF_train_prec, RF_train_recall, RF_train_f1, RF_train_acc, RF_train_rmspe, RF_train_mae




def isolated_lags_plots(folder, n_lags = 10):
    """
    Creates the plots using only the lags as a prediction. Uses the load_data function above to load in the data

    :param folder: name and path to the wanted folder to put the plots in
    :type folder: str
    :param n_lags: number of lags included in the plots
    :type n_lags: int
    """
    DT_precisions = []
    DT_recalls = []
    DT_f1s = []
    DT_accuracies = []
    LR_rmspes = []
    LR_maes = []
    RF_precisions = []
    RF_recalls = []
    RF_f1s = []
    RF_accuracies = []
    RF_rmspes = []
    RF_maes = []
    RF_train_precisions = []
    RF_train_recalls = []
    RF_train_f1s = []
    RF_train_accuracies = []
    RF_train_rmspes = []
    RF_train_maes = []
    lags = []
    train_data, test_data, total_data = load_data()
    isolated_train = train_data[[constants.NEXT_EVENT, constants.TIME_DIFFERENCE, constants.CASE_ID_COLUMN, constants.CURRENT_EVENT, constants.CASE_TIMESTAMP_COLUMN]].copy()
    isolated_test = test_data[[constants.NEXT_EVENT, constants.TIME_DIFFERENCE, constants.CASE_ID_COLUMN, constants.CURRENT_EVENT, constants.CASE_TIMESTAMP_COLUMN]].copy()

    for i in tqdm(range(1, n_lags+1)):
        isolated_train = predictors_columns.compute_case_lag_event_column(isolated_train, lag=i,column_name='lag_' + str(lags))
        isolated_test = predictors_columns.compute_case_lag_event_column(isolated_test, lag=i,column_name='lag_' + str(lags))



        DT_prec, DT_recall, DT_f1, DT_acc, RF_prec, RF_recall, RF_f1, RF_acc, LR_rmspe, LR_mae, RF_rmspe, RF_mae, RF_train_prec, RF_train_recall, RF_train_f1, RF_train_acc, RF_train_rmspe, RF_train_mae = compute_model_performances(isolated_train.drop(
            [constants.CASE_ID_COLUMN, constants.CURRENT_EVENT, constants.CASE_TIMESTAMP_COLUMN], axis = 1), isolated_test.drop([constants.CASE_ID_COLUMN, constants.CURRENT_EVENT, constants.CASE_TIMESTAMP_COLUMN], axis = 1))




        DT_precisions.append(round(DT_prec, 2))
        DT_recalls.append(round(DT_recall, 2))
        DT_f1s.append(round(DT_f1, 2))
        DT_accuracies.append(round(DT_acc, 2))
        LR_rmspes.append(round(LR_rmspe, 0))
        LR_maes.append(round(LR_mae, 0))
        RF_precisions.append(round(RF_prec, 2))
        RF_recalls.append(round(RF_recall, 2))
        RF_f1s.append(round(RF_f1, 2))
        RF_accuracies.append(round(RF_acc, 2))
        RF_rmspes.append(round(RF_rmspe, 0))
        RF_maes.append(round(RF_mae, 0))
        RF_train_precisions.append(round(RF_train_prec, 2))
        RF_train_recalls.append(round(RF_train_recall, 2))
        RF_train_f1s.append(round(RF_train_f1, 2))
        RF_train_accuracies.append(round(RF_train_acc, 2))
        RF_train_rmspes.append(round(RF_train_rmspe, 0))
        RF_train_maes.append(round(RF_train_mae, 0))
        lags.append(i)

    # precision plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_precisions, marker='P')
    plt.plot(lags, RF_precisions, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_precisions[i - 1]),
                     xy=(float(i) - 0.04, DT_precisions[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_precisions[i - 1]),
                     xy=(float(i) - 0.04, RF_precisions[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test precision performance against number of lags', fontsize=16)
    plt.ylabel('Precision')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/precision_plot.png')

    # Accuracy plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_accuracies, marker='P')
    plt.plot(lags, RF_accuracies, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_accuracies[i - 1]),
                     xy=(float(i) - 0.04, DT_accuracies[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_accuracies[i - 1]),
                     xy=(float(i) - 0.04, RF_accuracies[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test accuracy performance against number of lags', fontsize=16)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/accuracy_plot.png')

    # Recall plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_recalls, marker='P')
    plt.plot(lags, RF_recalls, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_recalls[i - 1]), xy=(float(i) - 0.04, DT_recalls[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_recalls[i - 1]), xy=(float(i) - 0.04, RF_recalls[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test recall performance against number of lags', fontsize=16)
    plt.ylabel('Recall')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/recall_plot.png')

    # f1_plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_f1s, marker='P')
    plt.plot(lags, RF_f1s, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_f1s[i - 1]), xy=(float(i) - 0.04, DT_f1s[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_f1s[i - 1]), xy=(float(i) - 0.04, RF_f1s[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test f1-score performance against number of lags', fontsize=16)
    plt.ylabel('F1-score')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/F1_score_plot.png')

    # RMSPE plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, LR_rmspes, marker='P')
    plt.plot(lags, RF_rmspes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(LR_rmspes[i - 1]), xy=(float(i) - 0.08, LR_rmspes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_rmspes[i - 1]), xy=(float(i) - 0.08, RF_rmspes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Linear Regression', 'Random Forest Regression'])
    fig.suptitle('Model test RMSPE performance against number of lags', fontsize=16)
    plt.ylabel('Root Mean Squared Percentage Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/RMSPE_plot.png')

    # MAE plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, LR_maes, marker='P')
    plt.plot(lags, RF_maes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(LR_maes[i - 1]), xy=(float(i) - 0.08, LR_maes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_maes[i - 1]), xy=(float(i) - 0.08, RF_maes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Linear Regression', 'Random Forest Regression'])
    fig.suptitle('Model test MAE performance against number of lags', fontsize=16)
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/MAE_plot.png')

    # train presicion plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_precisions, marker='s')
    plt.plot(lags, RF_precisions, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_precisions[i - 1]),
                     xy=(float(i) - 0.04, RF_train_precisions[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_precisions[i - 1]),
                     xy=(float(i) - 0.04, RF_precisions[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest precision performance against number of lags', fontsize=16)
    plt.ylabel('Precision')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_precision_plot.png')

    # train accuracy plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_accuracies, marker='s')
    plt.plot(lags, RF_accuracies, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_accuracies[i - 1]),
                     xy=(float(i) - 0.04, RF_train_accuracies[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_accuracies[i - 1]),
                     xy=(float(i) - 0.04, RF_accuracies[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest accuracy performance against number of lags', fontsize=16)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_accuracy_plot.png')

    # train recall plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_recalls, marker='s')
    plt.plot(lags, RF_recalls, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_recalls[i - 1]),
                     xy=(float(i) - 0.04, RF_train_recalls[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_recalls[i - 1]), xy=(float(i) - 0.04, RF_recalls[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest recall performance against number of lags', fontsize=16)
    plt.ylabel('Recall')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_recall_plot.png')

    # train f1 plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_f1s, marker='s')
    plt.plot(lags, RF_f1s, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_f1s[i - 1]),
                     xy=(float(i) - 0.04, RF_train_f1s[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_f1s[i - 1]), xy=(float(i) - 0.04, RF_f1s[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest F1-score performance against number of lags', fontsize=16)
    plt.ylabel('F1-score')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_F1_plot.png')

    # train rmspe plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_rmspes, marker='s')
    plt.plot(lags, RF_rmspes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_rmspes[i - 1]),
                     xy=(float(i) - 0.08, RF_train_rmspes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_rmspes[i - 1]), xy=(float(i) - 0.08, RF_rmspes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest RMSPE performance against number of lags', fontsize=16)
    plt.ylabel('Root Mean Squared Percentage Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_RMSPE_plot.png')

    # train mae plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_maes, marker='s')
    plt.plot(lags, RF_maes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_maes[i - 1]),
                     xy=(float(i) - 0.08, RF_train_maes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_maes[i - 1]), xy=(float(i) - 0.08, RF_maes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest MAE performance against number of lags', fontsize=16)
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_MAE_plot.png')

def non_isolated_lags_plots(folder, n_lags = 10):
    """
    Creates the plots using the lags and the other data as a prediction. Uses the load_data function above to load in the data

    :param folder: name and path to the wanted folder to put the plots in
    :type folder: str
    :param n_lags: number of lags included in the plots
    :type n_lags: int
    """

    DT_precisions = []
    DT_recalls = []
    DT_f1s = []
    DT_accuracies = []
    LR_rmspes = []
    LR_maes = []
    RF_precisions = []
    RF_recalls = []
    RF_f1s = []
    RF_accuracies = []
    RF_rmspes = []
    RF_maes = []
    RF_train_precisions = []
    RF_train_recalls = []
    RF_train_f1s = []
    RF_train_accuracies = []
    RF_train_rmspes = []
    RF_train_maes = []
    lags = []
    train_data, test_data, total_data = load_data()
    try:
        train_data.drop(['first_lag_event', 'second_lag_event'], axis = 1, inplace = True)
        test_data.drop(['first_lag_event', 'second_lag_event'], axis=1, inplace=True)

    except:
        print('No lags detected in the input data')


    for i in tqdm(range(1, n_lags+1)):
        non_isolated_train = predictors_columns.compute_case_lag_event_column(train_data, lag=i,column_name='lag_' + str(lags))
        non_isolated_test = predictors_columns.compute_case_lag_event_column(test_data, lag=i,column_name='lag_' + str(lags))



        DT_prec, DT_recall, DT_f1, DT_acc, RF_prec, RF_recall, RF_f1, RF_acc, LR_rmspe, LR_mae, RF_rmspe, RF_mae, RF_train_prec, RF_train_recall, RF_train_f1, RF_train_acc, RF_train_rmspe, RF_train_mae = compute_model_performances(non_isolated_train.drop(
            [constants.CASE_ID_COLUMN, constants.CURRENT_EVENT, constants.CASE_TIMESTAMP_COLUMN], axis = 1), non_isolated_test.drop([constants.CASE_ID_COLUMN, constants.CURRENT_EVENT, constants.CASE_TIMESTAMP_COLUMN], axis = 1))




        DT_precisions.append(round(DT_prec, 2))
        DT_recalls.append(round(DT_recall, 2))
        DT_f1s.append(round(DT_f1, 2))
        DT_accuracies.append(round(DT_acc, 2))
        LR_rmspes.append(round(LR_rmspe, 0))
        LR_maes.append(round(LR_mae, 0))
        RF_precisions.append(round(RF_prec, 2))
        RF_recalls.append(round(RF_recall, 2))
        RF_f1s.append(round(RF_f1, 2))
        RF_accuracies.append(round(RF_acc, 2))
        RF_rmspes.append(round(RF_rmspe, 0))
        RF_maes.append(round(RF_mae, 0))
        RF_train_precisions.append(round(RF_train_prec, 2))
        RF_train_recalls.append(round(RF_train_recall, 2))
        RF_train_f1s.append(round(RF_train_f1, 2))
        RF_train_accuracies.append(round(RF_train_acc, 2))
        RF_train_rmspes.append(round(RF_train_rmspe, 0))
        RF_train_maes.append(round(RF_train_mae, 0))
        lags.append(i)

    # precision plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_precisions, marker='P')
    plt.plot(lags, RF_precisions, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_precisions[i - 1]),
                     xy=(float(i) - 0.04, DT_precisions[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_precisions[i - 1]),
                     xy=(float(i) - 0.04, RF_precisions[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test precision performance against number of lags', fontsize=16)
    plt.ylabel('Precision')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/precision_plot.png')

    # Accuracy plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_accuracies, marker='P')
    plt.plot(lags, RF_accuracies, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_accuracies[i - 1]),
                     xy=(float(i) - 0.04, DT_accuracies[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_accuracies[i - 1]),
                     xy=(float(i) - 0.04, RF_accuracies[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test accuracy performance against number of lags', fontsize=16)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/accuracy_plot.png')

    # Recall plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_recalls, marker='P')
    plt.plot(lags, RF_recalls, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_recalls[i - 1]), xy=(float(i) - 0.04, DT_recalls[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_recalls[i - 1]), xy=(float(i) - 0.04, RF_recalls[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test recall performance against number of lags', fontsize=16)
    plt.ylabel('Recall')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/recall_plot.png')

    # f1_plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, DT_f1s, marker='P')
    plt.plot(lags, RF_f1s, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(DT_f1s[i - 1]), xy=(float(i) - 0.04, DT_f1s[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_f1s[i - 1]), xy=(float(i) - 0.04, RF_f1s[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Desicion Tree', 'Random Forest'])
    fig.suptitle('Model test f1-score performance against number of lags', fontsize=16)
    plt.ylabel('F1-score')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/F1_score_plot.png')

    # RMSPE plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, LR_rmspes, marker='P')
    plt.plot(lags, RF_rmspes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(LR_rmspes[i - 1]), xy=(float(i) - 0.08, LR_rmspes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_rmspes[i - 1]), xy=(float(i) - 0.08, RF_rmspes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Linear Regression', 'Random Forest Regression'])
    fig.suptitle('Model test RMSPE performance against number of lags', fontsize=16)
    plt.ylabel('Root Mean Squared Percentage Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/RMSPE_plot.png')

    # MAE plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, LR_maes, marker='P')
    plt.plot(lags, RF_maes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(LR_maes[i - 1]), xy=(float(i) - 0.08, LR_maes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_maes[i - 1]), xy=(float(i) - 0.08, RF_maes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Linear Regression', 'Random Forest Regression'])
    fig.suptitle('Model test MAE performance against number of lags', fontsize=16)
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/MAE_plot.png')

    # train presicion plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_precisions, marker='s')
    plt.plot(lags, RF_precisions, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_precisions[i - 1]),
                     xy=(float(i) - 0.04, RF_train_precisions[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_precisions[i - 1]),
                     xy=(float(i) - 0.04, RF_precisions[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest precision performance against number of lags', fontsize=16)
    plt.ylabel('Precision')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_precision_plot.png')

    # train accuracy plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_accuracies, marker='s')
    plt.plot(lags, RF_accuracies, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_accuracies[i - 1]),
                     xy=(float(i) - 0.04, RF_train_accuracies[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_accuracies[i - 1]),
                     xy=(float(i) - 0.04, RF_accuracies[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest accuracy performance against number of lags', fontsize=16)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_accuracy_plot.png')

    # train recall plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_recalls, marker='s')
    plt.plot(lags, RF_recalls, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_recalls[i - 1]),
                     xy=(float(i) - 0.04, RF_train_recalls[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_recalls[i - 1]), xy=(float(i) - 0.04, RF_recalls[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest recall performance against number of lags', fontsize=16)
    plt.ylabel('Recall')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_recall_plot.png')

    # train f1 plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_f1s, marker='s')
    plt.plot(lags, RF_f1s, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_f1s[i - 1]),
                     xy=(float(i) - 0.04, RF_train_f1s[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_f1s[i - 1]), xy=(float(i) - 0.04, RF_f1s[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest F1-score performance against number of lags', fontsize=16)
    plt.ylabel('F1-score')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_F1_plot.png')

    # train rmspe plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_rmspes, marker='s')
    plt.plot(lags, RF_rmspes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_rmspes[i - 1]),
                     xy=(float(i) - 0.08, RF_train_rmspes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_rmspes[i - 1]), xy=(float(i) - 0.08, RF_rmspes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest RMSPE performance against number of lags', fontsize=16)
    plt.ylabel('Root Mean Squared Percentage Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_RMSPE_plot.png')

    # train mae plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, RF_train_maes, marker='s')
    plt.plot(lags, RF_maes, marker='D')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + abs(y_max - y_min) * 0.1)

    for i in lags:
        plt.annotate(str(RF_train_maes[i - 1]),
                     xy=(float(i) - 0.08, RF_train_maes[i - 1] + abs(y_max - y_min) * 0.03))
        plt.annotate(str(RF_maes[i - 1]), xy=(float(i) - 0.08, RF_maes[i - 1] + abs(y_max - y_min) * 0.03))

    plt.legend(['Train set', 'Test set'])
    fig.suptitle('Random Forest MAE performance against number of lags', fontsize=16)
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of lags included')
    plt.show()
    fig.savefig(folder + '/train_MAE_plot.png')

# Running the functions:
if __name__ == '__main__':
    train, test, total = load_data()
    create_plot_without_removed_cases_with_split_line(train, test)
    create_plot_with_removed_cases(train, test, total)
    isolated_lags_plots('lag_plots/Isolated_lags')
    isolated_lags_plots('lag_plots/general_lags')
