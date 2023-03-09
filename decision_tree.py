import copy
import time
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
import splitter, constants
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from datetime import datetime
from baseline import classification_performance, regression_performance
from sklearn.linear_model import LinearRegression

def importance(model, X_val, y_val):
    r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X_val.columns[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

def tune_rf_activity(train_data, columns):
    X = train_data[columns]
    y = train_data['next event']

    
    ######## Parameters #######
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [1, 2, 5, 10, 15, 50]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False] # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, 
                                   param_distributions = random_grid, 
                                   n_iter = 10, 
                                   cv = 3, 
                                   verbose=0, 
                                   random_state=42, 
                                   n_jobs = -1)
    rf_random.fit(X, y)

    preds = rf_random.best_estimator_.predict(X)
    
    
    train_data['predicted next event'] = preds

    print(rf_random.best_estimator_)
    print(accuracy_score(preds,y))

    return rf_random.best_estimator_

def train_activity_model(train_data, clf, columns, normal=True):
    X = train_data[columns]

    if not normal:
        y = train_data['next_activity_id']
    else:
        y = train_data['next event']

    clf.fit(X,y)
    preds = clf.predict(X)
    
    train_data['predicted next event'] = preds
    

    classification_performance(train_data, "whatever.png")
    
    print(accuracy_score(preds,y))
    return clf

def train_time_model(train_data, clf, columns):
    X = train_data[columns]

    y = train_data['time until next event']

    clf.fit(X,y)
    preds = clf.predict(X)

    train_data["predicted time until next event"] = preds

    regression_performance(train_data)

    print(mean_absolute_error(preds,y))
    return clf

def test_activity_model(test_data, clf, columns, normal=True):
    X = test_data[columns]

    if not normal:
        y = test_data['next_activity_id']
    else:
        y = test_data['next event']
    
    preds = clf.predict(X)

    test_data['predicted next event'] = preds
    
    classification_performance(test_data, "whatever.png")

    acc = accuracy_score(preds, y)
    print(acc)
    return acc

def test_time_model(test_data, clf, columns):
    X = test_data[columns]

    y = test_data['time until next event']
    preds = clf.predict(X)

    test_data["predicted time until next event"] = preds

    regression_performance(test_data)
    
    return 1

def compare_all_models(train_data, test_data, timer):
    cols = ['activity number in case', 'case end count',
            'A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED',
       'A_CANCELLED', 'A_DECLINED', 'A_FINALIZED', 'A_PARTLYSUBMITTED',
       'A_PREACCEPTED', 'A_REGISTERED', 'A_SUBMITTED', 'O_ACCEPTED',
       'O_CANCELLED', 'O_CREATED', 'O_DECLINED', 'O_SELECTED', 'O_SENT',
       'O_SENT_BACK', 'W_Afhandelen leads', 'W_Beoordelen fraude',
       'W_Completeren aanvraag', 'W_Nabellen incomplete dossiers',
       'W_Nabellen offertes', 'W_Valideren aanvraag', 'time_until_next_holiday',
       'weekend', 'week_start', 'work_time', 'work_hours', 'is_holiday']

    print(train_data.columns)
    
    # copy so we don't modify the original training set
    train_data = copy.deepcopy(train_data).rename(columns={'concept:name': 'name'})
    names_ohe = pd.get_dummies(train_data['name'])
    train_data = train_data.drop('name', axis=1).join(names_ohe).dropna()
    train_data['next_activity_id'] = pd.factorize(train_data['next event'])[0]
    

    # copy test dataset
    test_data = copy.deepcopy(test_data).rename(columns={'concept:name': 'name'})
    names_ohe = pd.get_dummies(test_data['name'])
    test_data = test_data.drop('name', axis=1).join(names_ohe).dropna()
    test_data['next_activity_id'] = pd.factorize(test_data['next event'])[0]

    timer.send("Time to prepare columns (in seconds): ")
    
    print("Decision Tree:")
    print("-----------------------------")
    print("Next activity:")
    clf = DecisionTreeClassifier()
    dec_tree_clas = train_activity_model(train_data, clf, cols)

    timer.send("Time to train decision tree classifier (in seconds): ")

    test_activity_model(test_data, dec_tree_clas, cols) 
    
    timer.send("Time to evaluate decision tree classifier (in seconds): ")

    print("Random Forest:")
    print("-----------------------------")
    print("Next activity:")
    clf = RandomForestClassifier()
    rand_forest_class = train_activity_model(train_data, clf, cols)

    timer.send("Time to train random forest classifier (in seconds): ")

    test_activity_model(test_data, rand_forest_class, cols)
    
    timer.send("Time to evaluation random forest classifier (in seconds): ")

    print("Linear Regression:")
    print("-----------------------------")
    print("Time to next activity:")
    reg = LinearRegression()
    lin_regr = train_time_model(train_data, reg, cols)

    timer.send("Time to train linear regression (in seconds): ")

    test_time_model(test_data, lin_regr, cols)

    timer.send("Time to evaluate linear regression (in seconds): ")

    print("Random Forest Regression:")
    print("-----------------------------")
    print("Time to next activity:")
    reg = RandomForestRegressor()
    rand_forest_regr = train_time_model(train_data, reg, cols)

    timer.send("Time to train random forest regression (in seconds): ")

    test_time_model(test_data, rand_forest_regr, cols)

    timer.send("Time to evaluate random forest regression (in seconds): ")

    #print("XGBoost:")
    #print("-----------------------------")
    #print("Next activity:")
    #clf = XGBClassifier()
    #test_activity_model(test_data, train_activity_model(train_data, clf, cols, False), cols, False)
    #print("Time to next activity:")
    #reg = XGBRegressor()
    #test_time_model(test_data, train_time_model(train_data, reg, cols), cols)


    print("Random Forest Hyperparameter Tuning:")
    print("-----------------------------")
    print("Next activity:")
    test_activity_model(test_data, tune_rf_activity(train_data, cols), cols)


    print("Permutation importance (Decision Tree):")
    importance(dec_tree_clas, *train_data)
    

def time_execution():
    """A couroutine that prints a message it recieves through .send()
    and the the seconds passed since the last time it was called.
    """
    start_time = time.process_time()
    while True:
        string = (yield)
        new_time = time.process_time()
        print(string, new_time - start_time)
        start_time = new_time

def correct_data(data):
    li = data["time_until_next_holiday"]
    data["time_until_next_holiday"] = [int(el.split(" ")[0]) for el in li]
    # time_since_week_start
    # time_to_work_hours

    li = data["time_since_week_start"]
    x = [int(el.split(" ")[0])*86400 + sum(np.array([3600, 60, 1]) * np.array([int(i) for i in el.split(" ")[2].split(".")[0].split(":")])) for el in li]
    data["week_start"] = x
    
    li = data["time_to_work_hours"]
    x = [int(el.split(" ")[0])*86400 + sum(np.array([3600, 60, 1]) * np.array([int(i) for i in el.split(" ")[2].split(".")[0].split(":")])) for el in li]
    data["work_hours"] = x
    return data


def main():
    # set up the timer
    timer = time_execution()
    timer.__next__()

    # do this if the files are not split already
    # splitter.convert_raw_dataset(constants.RAW_DATASET_PATH, constants.CONVERTED_DATASET_PATH)
    # timer.send("Time to convert dataset (in seconds): ")

    full_data = pd.read_csv(constants.GLOBAL_DATASET_PATH)
    full_data = correct_data(full_data)
    splitter.time_series_split(full_data, 5)
    data = splitter.split_dataset(full_data, 0.2)
    
    timer.send("Time to split dataset (in seconds): ")

    compare_all_models(*data, timer)

if __name__ == "__main__":
    main()
