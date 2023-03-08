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
plt.style.use('fivethirtyeight')

# def scale_dataset(df):
#     """Scales the dataset using MinMaxScaler. It returns the scaled dataset

#     :param df: The dataset to be scaled
#     :type df: DataFrame
#     :return: The scaled dataset
#     :rtype: DataFrame
#     """
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df)
#     print(scaled_data)
#     return pd.DataFrame(scaled_data)

# def string_converter(X_train, X_test):
#     ohe = OneHotEncoder()
#     ohe.fit(X_train)
#     X_train_enc = ohe.transform(X_train)
#     X_test_enc = ohe.transform(X_test)
#     return X_train_enc, X_test_enc



def get_label_encoder(data):
    le = LabelEncoder()
    le.fit(data)
    return le

def get_string_vectorizer(data):
    data = np.asarray(data).astype('str')
    layer = tf.keras.layers.StringLookup(output_mode='one_hot')
    layer.adapt(data)
    return layer

# def normalizer(data):
#     layer = tf.keras.layers.Normalization(axis=None)
#     layer.adapt(data)
#     return layer(data)

def preprocess_event_labels():
    full_data = pd.read_csv(constants.CONVERTED_DATASET_PATH)
    train_data, test_data = splitter.split_dataset(full_data, 0.2)

    string_vectorizer = get_string_vectorizer(full_data[constants.CASE_POSITION_COLUMN])
    label_encoder = get_label_encoder(full_data['next event'])

    vector_case_position_train = string_vectorizer(train_data[constants.CASE_POSITION_COLUMN])
    case_step_number_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_train = keras.layers.concatenate([vector_case_position_train, case_step_number_train])
    y_train = label_encoder.transform(train_data[['next event']])

    vector_case_position = string_vectorizer(test_data[constants.CASE_POSITION_COLUMN])
    case_step_number = test_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_test = keras.layers.concatenate([vector_case_position, case_step_number])
    y_test = label_encoder.transform(test_data[['next event']])

    X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = np.array(y_train).reshape(y_train.shape[0],1)
    y_test = np.array(y_test).reshape(y_test.shape[0],1)

    return X_train, y_train, X_test, y_test

def preprocess_event_dummies():
    full_data = pd.read_csv(constants.CONVERTED_DATASET_PATH)

    names_ohe = pd.get_dummies(full_data[constants.CASE_POSITION_COLUMN])
    full_data = full_data.join(names_ohe).dropna()

    print(full_data.head())

    train_data, test_data = splitter.split_dataset(full_data, 0.2)

    string_vectorizer = get_string_vectorizer(full_data[constants.CASE_POSITION_COLUMN])
    label_encoder = get_label_encoder(full_data['next event'])

    vector_case_position_train = string_vectorizer(train_data[constants.CASE_POSITION_COLUMN])
    case_step_number_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_train = keras.layers.concatenate([vector_case_position_train, case_step_number_train])
    y_train = label_encoder.transform(train_data[['next event']])

    vector_case_position = string_vectorizer(test_data[constants.CASE_POSITION_COLUMN])
    case_step_number = test_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_test = keras.layers.concatenate([vector_case_position, case_step_number])
    y_test = label_encoder.transform(test_data[['next event']])

    X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = np.array(y_train).reshape(y_train.shape[0],1)
    y_test = np.array(y_test).reshape(y_test.shape[0],1)

    return X_train, y_train, X_test, y_test

def preprocess_time():
    full_data = pd.read_csv(constants.CONVERTED_DATASET_PATH)
    train_data, test_data = splitter.split_dataset(full_data, 0.2)
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    string_vectorizer = get_string_vectorizer(full_data[[constants.CASE_POSITION_COLUMN]])
    label_encoder = get_label_encoder(full_data[[constants.CASE_POSITION_COLUMN]])
    
    vector_case_position_train = string_vectorizer(train_data[[constants.CASE_POSITION_COLUMN]])
    # vector_case_position_train = string_vectorizer(train_data[['next event']])
    case_step_number_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_train = keras.layers.concatenate([vector_case_position_train, case_step_number_train])
    y_train = train_data[['time until next event']]

    

    vector_case_position = string_vectorizer(test_data[[constants.CASE_POSITION_COLUMN]])
    case_step_number = test_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_test = keras.layers.concatenate([vector_case_position, case_step_number])
    print(X_test)
    y_test = test_data[['time until next event']]
    
    return X_train, y_train, X_test, y_test

def preprocess_event():
    full_data = pd.read_csv(constants.CONVERTED_DATASET_PATH)
    train_data, test_data = splitter.split_dataset(full_data, 0.2)

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    string_vectorizer = get_string_vectorizer(full_data[constants.CASE_POSITION_COLUMN])
    string_vectorizer_next = get_string_vectorizer(full_data['next event'])
    # label_encoder = get_label_encoder(full_data['next event'])

    vector_case_position_train = string_vectorizer(train_data[constants.CASE_POSITION_COLUMN])
    case_step_number_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_train = keras.layers.concatenate([vector_case_position_train, case_step_number_train])
    y_train = string_vectorizer_next(train_data['next event'])

    vector_case_position = string_vectorizer(test_data[constants.CASE_POSITION_COLUMN])
    case_step_number = test_data[[constants.CASE_STEP_NUMBER_COLUMN]]

    X_test = keras.layers.concatenate([vector_case_position, case_step_number])
    y_test = string_vectorizer_next(test_data['next event'])

    X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    print(y_train, y_test)
    # y_train = np.array(y_train).reshape(y_train.shape[0],1)
    # y_test = np.array(y_test).reshape(y_test.shape[0],1)

    return X_train, y_train, X_test, y_test

def train_labels(X_train, y_train):
    # Initialising the RNN
    model = Sequential()
    model.add(keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))
    # Adding a second LSTM layer and Dropout layer
    # model.add(keras.layers.LSTM(units = 50, return_sequences = True))
    # model.add(keras.layers.Dropout(0.2))
    # # Adding a third LSTM layer and Dropout layer
    # model.add(keras.layers.LSTM(units = 50, return_sequences = True))
    # model.add(keras.layers.Dropout(0.2))
    # Adding a fourth LSTM layer and and Dropout layer
    model.add(keras.layers.LSTM(units = 50))
    model.add(keras.layers.Dropout(0.2))
    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(keras.layers.Dense(units = 1))
    # rmsprop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy'])
    model.fit(X_train, y_train, epochs = 5, batch_size = 50)
    return model

def train(X_train, y_train):
    # Initialising the RNN
    model = Sequential()
    model.add(keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))
    # Adding a second LSTM layer and Dropout layer
    model.add(keras.layers.LSTM(units = 50, return_sequences = True))
    model.add(keras.layers.Dropout(0.2))
    # Adding a third LSTM layer and Dropout layer
    model.add(keras.layers.LSTM(units = 50, return_sequences = True))
    model.add(keras.layers.Dropout(0.2))
    # Adding a fourth LSTM layer and and Dropout layer
    model.add(keras.layers.LSTM(units = 50))
    model.add(keras.layers.Dropout(0.2))
    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(keras.layers.Dense(units = 25))
    # rmsprop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
    model.fit(X_train, y_train, epochs = 5, batch_size = 50)
    return model


def test(X_test, y_test, model):
    predictions = model.predict(X_test)
    print(predictions)
    print(y_test)
    print(model.evaluate(X_test, y_test))

def test_with_ready_model(X_test, y_test, file):
    model = keras.models.load_model(file)
    predictions = model.predict(X_test)

    test(X_test, y_test, model)

def train_and_test(X_train, y_train, X_test, y_test, file=None):
    model = train(X_train, y_train)
    if file:
        model.save(file)
    test(X_test, y_test, model)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess_event()

    train_and_test(X_train, y_train, X_test, y_test, 'class_model.h5')
    # test_with_ready_model(X_test, y_test, 'class_model.h5')