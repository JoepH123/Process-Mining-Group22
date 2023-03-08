
import copy
import time
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
import splitter, constants
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

def train_activity_model(train_data_in, clf):
    # copy so we don't modify the original training set
    train_data = copy.deepcopy(train_data_in).rename(columns={'concept:name': 'name'})
    names_ohe = pd.get_dummies(train_data['name'])
    train_data = train_data.drop('name', axis=1).join(names_ohe).dropna()

    X = train_data[['activity number in case',
       'number of activity in case inverse', 'case start count',
       'case end count', 'A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED',
       'A_CANCELLED', 'A_DECLINED', 'A_FINALIZED', 'A_PARTLYSUBMITTED',
       'A_PREACCEPTED', 'A_REGISTERED', 'A_SUBMITTED', 'O_ACCEPTED',
       'O_CANCELLED', 'O_CREATED', 'O_DECLINED', 'O_SELECTED', 'O_SENT',
       'O_SENT_BACK', 'W_Afhandelen leads', 'W_Beoordelen fraude',
       'W_Completeren aanvraag', 'W_Nabellen incomplete dossiers',
       'W_Nabellen offertes', 'W_Valideren aanvraag']]

    y = train_data['next event']

    clf = DecisionTreeClassifier()
    clf.fit(X,y)
    preds = clf.predict(X)

    print(accuracy_score(preds,y))
    return clf

def train_time_model(train_data_in, clf):
    # copy so we don't modify the original training set
    train_data = copy.deepcopy(train_data_in).rename(columns={'concept:name': 'name'})
    names_ohe = pd.get_dummies(train_data['name'])
    train_data = train_data.drop('name', axis=1).join(names_ohe).dropna()

    X = train_data[['activity number in case',
       'number of activity in case inverse', 'case start count',
       'case end count', 'A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED',
       'A_CANCELLED', 'A_DECLINED', 'A_FINALIZED', 'A_PARTLYSUBMITTED',
       'A_PREACCEPTED', 'A_REGISTERED', 'A_SUBMITTED', 'O_ACCEPTED',
       'O_CANCELLED', 'O_CREATED', 'O_DECLINED', 'O_SELECTED', 'O_SENT',
       'O_SENT_BACK', 'W_Afhandelen leads', 'W_Beoordelen fraude',
       'W_Completeren aanvraag', 'W_Nabellen incomplete dossiers',
       'W_Nabellen offertes', 'W_Valideren aanvraag']]

    y = train_data['time until next event']

    clf = DecisionTreeRegressor()
    clf.fit(X,y)
    preds = clf.predict(X)

    print(mean_absolute_error(preds,y))
    return clf

def test_activity_model(test_data_in, clf):
    test_data = copy.deepcopy(test_data_in).rename(columns={'concept:name': 'name'})
    names_ohe = pd.get_dummies(test_data['name'])
    test_data = test_data.drop('name', axis=1).join(names_ohe).dropna()

    X = test_data[['activity number in case',
       'number of activity in case inverse', 'case start count',
       'case end count', 'A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED',
       'A_CANCELLED', 'A_DECLINED', 'A_FINALIZED', 'A_PARTLYSUBMITTED',
       'A_PREACCEPTED', 'A_REGISTERED', 'A_SUBMITTED', 'O_ACCEPTED',
       'O_CANCELLED', 'O_CREATED', 'O_DECLINED', 'O_SELECTED', 'O_SENT',
       'O_SENT_BACK', 'W_Afhandelen leads', 'W_Beoordelen fraude',
       'W_Completeren aanvraag', 'W_Nabellen incomplete dossiers',
       'W_Nabellen offertes', 'W_Valideren aanvraag']]

    y = test_data['next event']
    preds = clf.predict(X)

    acc = accuracy_score(preds, y)
    print(acc)
    return acc

def test_time_model(test_data_in, clf):
    test_data = copy.deepcopy(test_data_in).rename(columns={'concept:name': 'name'})
    names_ohe = pd.get_dummies(test_data['name'])
    test_data = test_data.drop('name', axis=1).join(names_ohe).dropna()

    X = test_data[['activity number in case',
       'number of activity in case inverse', 'case start count',
       'case end count', 'A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED',
       'A_CANCELLED', 'A_DECLINED', 'A_FINALIZED', 'A_PARTLYSUBMITTED',
       'A_PREACCEPTED', 'A_REGISTERED', 'A_SUBMITTED', 'O_ACCEPTED',
       'O_CANCELLED', 'O_CREATED', 'O_DECLINED', 'O_SELECTED', 'O_SENT',
       'O_SENT_BACK', 'W_Afhandelen leads', 'W_Beoordelen fraude',
       'W_Completeren aanvraag', 'W_Nabellen incomplete dossiers',
       'W_Nabellen offertes', 'W_Valideren aanvraag']]

    y = test_data['time until next event']
    preds = clf.predict(X)

    mae = mean_absolute_error(preds, y)
    print(mae)
    return mae

def test_all_models(train_data, test_data):
    print("Decision Tree:")
    print("-----------------------------")
    print("Next activity:")
    clf = DecisionTreeClassifier()
    test_activity_model(test_data, train_activity_model(train_data, clf))
    print("Time to next activity:")
    reg = DecisionTreeRegressor()
    test_time_model(test_data, train_time_model(train_data, reg))

    print("Random Forest:")
    print("-----------------------------")
    print("Next activity:")
    clf = RandomForestClassifier()
    test_activity_model(test_data, train_activity_model(train_data, clf))
    print("Time to next activity:")
    reg = RandomForestRegressor()
    test_time_model(test_data, train_time_model(train_data, reg))

    print("XGBoost:")
    print("-----------------------------")
    print("Next activity:")
    clf = XGBClassifier()
    test_activity_model(test_data, train_activity_model(train_data, clf))
    print("Time to next activity:")
    reg = XGBRegressor()
    test_time_model(test_data, train_time_model(train_data, reg))


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



def main():
    # set up the timer
    timer = time_execution()
    timer.__next__()

    # do this if the files are not split already
    # splitter.convert_raw_dataset(constants.RAW_DATASET_PATH, constants.CONVERTED_DATASET_PATH)
    # timer.send("Time to convert dataset (in seconds): ")

    full_data = pd.read_csv(constants.CONVERTED_DATASET_PATH)
    splitter.time_series_split(full_data, 5)
    data = splitter.split_dataset(full_data, 0.2)

    test_all_models(*data)

if __name__ == "__main__":
    main()
