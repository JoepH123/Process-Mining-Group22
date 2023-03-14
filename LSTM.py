import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import keras.layers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import splitter
import constants
from sklearn.metrics import accuracy_score
from decision_tree import correct_data
from baseline import classification_performance, regression_performance

plt.style.use('fivethirtyeight')

# Encodes string data as integers
def get_label_encoder(data):
    le = LabelEncoder()
    le.fit(data)
    return le

# Encodes string data as one-hot vectors
def get_one_hot_encoder(data):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data)
    return enc

# Normalizes numerical data
def normalizer(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler

def preprocess_event_X(train_data, test_data, enc):

    vector_case_position_train = enc.transform(train_data[[constants.CASE_POSITION_COLUMN]]).toarray()
    # number_data_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN, 'case end count', 'time_until_next_holiday', 'weekend', 'week_start', 'work_time', 'work_hours', 'is_holiday']]

    number_data_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN]]
    case_step_number_normalizer = normalizer(number_data_train[[constants.CASE_STEP_NUMBER_COLUMN]])
    # case_end_count_normalizer = normalizer(number_data_train[['case end count']])
    # time_until_next_holiday_normalizer = normalizer(number_data_train[['time_until_next_holiday']])
    # week_start_normalizer = normalizer(number_data_train[['week_start']])
    # work_hours_normalizer = normalizer(number_data_train[['work_hours']])

    number_data_train[constants.CASE_STEP_NUMBER_COLUMN] = case_step_number_normalizer.transform(number_data_train[[constants.CASE_STEP_NUMBER_COLUMN]])
    # number_data_train['case end count'] = case_end_count_normalizer.transform(number_data_train[['case end count']])
    # number_data_train['time_until_next_holiday'] = time_until_next_holiday_normalizer.transform(number_data_train[['time_until_next_holiday']])
    # number_data_train['week_start'] = week_start_normalizer.transform(number_data_train[['week_start']])
    # number_data_train['work_hours'] = work_hours_normalizer.transform(number_data_train[['work_hours']]) 

    X_train = keras.layers.concatenate([vector_case_position_train, number_data_train])

    vector_case_position = enc.transform(test_data[[constants.CASE_POSITION_COLUMN]]).toarray()
    # number_data = test_data[[constants.CASE_STEP_NUMBER_COLUMN, 'case end count', 'time_until_next_holiday', 'weekend', 'week_start', 'work_time', 'work_hours', 'is_holiday']]
    number_data = test_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    number_data[constants.CASE_STEP_NUMBER_COLUMN] = case_step_number_normalizer.transform(number_data[[constants.CASE_STEP_NUMBER_COLUMN]])
    # number_data['case end count'] = case_end_count_normalizer.transform(number_data[['case end count']])
    # number_data['time_until_next_holiday'] = time_until_next_holiday_normalizer.transform(number_data[['time_until_next_holiday']])
    # number_data['week_start'] = week_start_normalizer.transform(number_data[['week_start']])
    # number_data['work_hours'] = work_hours_normalizer.transform(number_data[['work_hours']]) 

    print(number_data_train)
    print(number_data)

    X_test = keras.layers.concatenate([vector_case_position, number_data])

    X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test

def preprocess_event_y(train_data, test_data, enc_next):
    y_train = enc_next.transform(train_data[['next event']]).toarray()
    y_test = enc_next.transform(test_data[['next event']]).toarray()
    return y_train, y_test

def preprocess_event_y_labels(train_data, test_data, label_encoder):
    y_train = label_encoder.transform(train_data[['next event']])
    y_test = label_encoder.transform(test_data[['next event']])

    y_train = np.array(y_train).reshape(y_train.shape[0],1)
    y_test = np.array(y_test).reshape(y_test.shape[0],1)

    return y_train, y_test

def preprocess_event():
    full_data = pd.read_csv(constants.GLOBAL_DATASET_PATH)
    full_data = correct_data(full_data)
    full_data.weekend = full_data.weekend.replace({True: 1, False: 0})
    full_data.work_time = full_data.work_time.replace({True: 1, False: 0})
    full_data.is_holiday = full_data.is_holiday.replace({True: 1, False: 0})

    train_data, test_data = splitter.split_dataset(full_data, 0.2)

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    enc = get_one_hot_encoder(full_data[[constants.CASE_POSITION_COLUMN]]) 
    enc_next = get_one_hot_encoder(full_data[['next event']])
    # label_encoder = get_label_encoder(full_data['next event'])


    X_train, X_test = preprocess_event_X(train_data, test_data, enc)
    y_train, y_test = preprocess_event_y(train_data, test_data, enc_next)

    return X_train, y_train, X_test, y_test, enc_next

def train_event(X_train, y_train):
    model = Sequential()
    model.add(keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.LSTM(units = 50, return_sequences = True))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.LSTM(units = 50, return_sequences = True))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.LSTM(units = 50))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(units = y_train.shape[1]))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    # FOR LABEL ENCODER
    # model.add(keras.layers.Dense(units = 1))
    # model.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy'])

    model.fit(X_train, y_train, epochs = 15, batch_size = 100)
    return model


def test(X_test, y_test, model, enc):
    predictions = model.predict(X_test)
    # score = model.evaluate(X_test, y_test, verbose=0)
    predictions = enc.inverse_transform(predictions)
    y_test = enc.inverse_transform(y_test)
    
    y_test = pd.DataFrame(y_test,columns=['next event']).join(pd.DataFrame(predictions, columns=['predicted next event']))

    classification_performance(y_test, "LSTM.png")

def load_model(file):
    model = keras.models.load_model(file)
    return model

def train_model(X_train, y_train, file=None):
    model = train_event(X_train, y_train)
    if file:
        model.save(file)
    return model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, enc = preprocess_event()
    model = train_model(X_train, y_train,'LSTM_models/class_model_more_inputs2.h5')
    # model = load_model('LSTM_models/class_model_more_inputs.h5')
    test(X_test, y_test, model, enc)