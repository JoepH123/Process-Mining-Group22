import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
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
    scaler = Normalizer()
    scaler.fit(data)
    return scaler

def preprocess_event_X(train_data, test_data, enc, max_case_len):

    num_of_predictors = 2    # CHANGE THIS AS NEEDED, rn its time and amount requested

    # number of features is the total number of distinct events plus the
    # number of other columns we use as predictors
    num_of_features = len(train_data[constants.CASE_POSITION_COLUMN].unique()) + num_of_predictors

    train_grouped = train_data.groupby([constants.CASE_ID_COLUMN])
    X_train = np.zeros((len(train_data.index), max_case_len, num_of_features), dtype=np.float32)

    i = 0
    row_counter = 0
    for case_id, group in train_grouped:
        event_sequence = []
        for index, row in group.iterrows():
            current_event = list(enc.transform(row[[constants.CASE_POSITION_COLUMN]].to_numpy().reshape(1, -1)).toarray()[0])
            current_event.append(row[constants.CASE_TIMESTAMP_COLUMN])
            current_event.append(row[constants.AMOUNT_REQUESTED_COLUMN])

            event_sequence.append(current_event_encoded)
            print(event_sequence)
            event_index = num_of_features - num_of_predictors - 1
            for event in event_sequence.reverse():
                np.copyto(X_train[row_counter, event_index], np.array(event))
                event_index -= 1

            i += 1
            row_counter += 1
            if i>2:
                break
        break

        # for row in group:
        #     #event_sequence.append(enc.transform(row[constants.CASE_POSITION_COLUMN]))
        #     print(row)
        #     print(enc.transform(row[constants.CASE_POSITION_COLUMN]))
        #     i +=1
        #     if i>5:
        #         break

            # event_sequence.append()
            # X_train[] = event_sequence...
    vector_case_position_train = enc.transform(train_data[[constants.CASE_POSITION_COLUMN]]).toarray()


    # number_data_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN, 'case end count', 'time_until_next_holiday', 'weekend', 'week_start', 'work_time', 'work_hours', 'is_holiday']]

    number_data_train = train_data[[constants.CASE_STEP_NUMBER_COLUMN, constants.AMOUNT_REQUESTED_COLUMN, constants.ACTIVE_CASES]]
    # case_step_number_normalizer = normalizer(number_data_train[[constants.CASE_STEP_NUMBER_COLUMN]])
    amount_requested_normalizer = normalizer(number_data_train[[constants.AMOUNT_REQUESTED_COLUMN]])
    # case_end_count_normalizer = normalizer(number_data_train[['case end count']])
    # time_until_next_holiday_normalizer = normalizer(number_data_train[['time_until_next_holiday']])
    # week_start_normalizer = normalizer(number_data_train[['week_start']])
    # work_hours_normalizer = normalizer(number_data_train[['work_hours']])
    workrate_normalizer = normalizer(number_data_train[[constants.ACTIVE_CASES]])

    # number_data_train[constants.CASE_STEP_NUMBER_COLUMN] = case_step_number_normalizer.transform(number_data_train[[constants.CASE_STEP_NUMBER_COLUMN]])
    number_data_train[constants.AMOUNT_REQUESTED_COLUMN] = amount_requested_normalizer.transform(number_data_train[[constants.AMOUNT_REQUESTED_COLUMN]])
    # number_data_train['case end count'] = case_end_count_normalizer.transform(number_data_train[['case end count']])
    # number_data_train['time_until_next_holiday'] = time_until_next_holiday_normalizer.transform(number_data_train[['time_until_next_holiday']])
    # number_data_train['week_start'] = week_start_normalizer.transform(number_data_train[['week_start']])
    # number_data_train['work_hours'] = work_hours_normalizer.transform(number_data_train[['work_hours']])
    number_data_train[constants.ACTIVE_CASES] = workrate_normalizer.transform(number_data_train[[constants.ACTIVE_CASES]])

    X_train = keras.layers.concatenate([vector_case_position_train, number_data_train, vector_event_lag_train, vector_event_second_lag_train])

    vector_case_position = enc.transform(test_data[[constants.CASE_POSITION_COLUMN]]).toarray()
    vector_event_lag_test = enc.transform(test_data[['first_lag_event']]).toarray()
    vector_event_second_lag_test = enc.transform(test_data[['second_lag_event']]).toarray()
    # number_data = test_data[[constants.CASE_STEP_NUMBER_COLUMN, 'case end count', 'time_until_next_holiday', 'weekend', 'week_start', 'work_time', 'work_hours', 'is_holiday']]
    number_data = test_data[[constants.CASE_STEP_NUMBER_COLUMN, constants.AMOUNT_REQUESTED_COLUMN, constants.ACTIVE_CASES]]

    # number_data[constants.CASE_STEP_NUMBER_COLUMN] = case_step_number_normalizer.transform(number_data[[constants.CASE_STEP_NUMBER_COLUMN]])
    number_data[constants.AMOUNT_REQUESTED_COLUMN] = amount_requested_normalizer.transform(number_data[[constants.AMOUNT_REQUESTED_COLUMN]])
    # number_data['case end count'] = case_end_count_normalizer.transform(number_data[['case end count']])
    # number_data['time_until_next_holiday'] = time_until_next_holiday_normalizer.transform(number_data[['time_until_next_holiday']])
    # number_data['week_start'] = week_start_normalizer.transform(number_data[['week_start']])
    # number_data['work_hours'] = work_hours_normalizer.transform(number_data[['work_hours']])
    number_data[constants.ACTIVE_CASES] = workrate_normalizer.transform(number_data[[constants.ACTIVE_CASES]])

    print(number_data_train)
    print(number_data)

    X_test = keras.layers.concatenate([vector_case_position, number_data, vector_event_lag_test, vector_event_second_lag_test])

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
    full_data = pd.read_csv(constants.PIPELINED_DATASET_PATH)
    # full_data = correct_data(full_data)
    # full_data.is_weekend = full_data.is_weekend.replace({True: 1, False: 0})
    # full_data.is_work_time = full_data.is_work_time.replace({True: 1, False: 0})
    # full_data.is_holiday = full_data.is_holiday.replace({True: 1, False: 0})

    train_data, test_data = splitter.split_dataset(full_data, 0.2)

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    enc = get_one_hot_encoder(full_data[[constants.CASE_POSITION_COLUMN]])
    enc_next = get_one_hot_encoder(full_data[['next event']].append(pd.DataFrame(['END'], columns=['next event'])))
    # label_encoder = get_label_encoder(full_data['next event'])
    
    max_case_len = int(full_data['activity number in case'].max())

    X_train, X_test = preprocess_event_X(train_data, test_data, enc, max_case_len)
    y_train, y_test = preprocess_event_y(train_data, test_data, enc_next, max_case_len)

    return X_train, y_train, X_test, y_test, enc_next

def train_event(X_train, y_train_clf, y_train_reg, epochs):
    """
    param X_train: array of shape (total # of events,
                                   maximum # of events in largest case,
                                   # of features),
                   containing only the training data. The number of features
                   is the total number of distinct events plus the number of
                   other columns we use as predictors.
    type X_train: 3D numpy array

    param y_train_clf: array of shape (total # of events,
                                       total # of distinct events),
                      containing the one-hot-encoded vectors of events
    type y_train_clf: 2D numpy array

    param y_train_reg: array of shape (total # of activities),
                    containing the time until next event column of training data
    type y_train_reg: 1D numpy array
    """


    # maximum sequence of events length
    max_sequence_len = np.shape(X_train)[1]

    # number of features from X_train shape (as above)
    num_of_features = np.shape(X_train)[2]

    # number of all possible event names (including an end-of-sequence delimiter)
    num_of_event_types = np.shape(y_train_clf)[1]

    main_input = keras.layers.Input(shape=(max_sequence_len, num_of_features), name='main_input')
    # train a 2-layer LSTM with one shared layer
    l1 = keras.layers.LSTM(100, implementation=2, return_sequences=True, dropout=0.2)(main_input) # the shared layer
    b1 = keras.layers.BatchNormalization()(l1)
    l2_1 = keras.layers.LSTM(100, implementation=2, return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    b2_1 = keras.layers.BatchNormalization()(l2_1)
    l2_2 = keras.layers.LSTM(100, implementation=2, return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
    b2_2 = keras.layers.BatchNormalization()(l2_2)
    event_output = keras.layers.Dense(num_of_event_types, activation='softmax', kernel_initializer='glorot_uniform', name='event_output')(b2_1)
    time_output = keras.layers.Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

    model = keras.Model(inputs=[main_input], outputs=[event_output, time_output])

    # Nadam - Much like Adam is essentially RMSprop with momentum, Nadam is Adam with Nesterov momentum
    opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'event_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=42)
    model_checkpoint = keras.callbacks.ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    history = model.fit(X_train, {'event_output':y_train_clf, 'time_output':y_train_reg}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=max_sequence_len, epochs=epochs)
    ##################   OLD CODE   ##############################
    # model = Sequential()
    # model.add(keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.LSTM(units = 50, return_sequences = True, activation="softmax"))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.LSTM(units = 50, return_sequences = True, activation="softmax"))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.LSTM(units = 50, activation="softmax"))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Dense(units = y_train.shape[1]))
    # model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    # # FOR LABEL ENCODER
    # # model.add(keras.layers.Dense(units = 1))
    # # model.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy'])

    # history = model.fit(X_train, y_train, epochs = 7, batch_size = 100)
    ################################################################
    loss_values = history.history['loss']
    epochs = range(1, len(loss_values)+1)

    plt.plot(epochs, loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    return model


def test(X_test, y_test, model, enc):
    predictions = model.predict(X_test)
    print(predictions)
    # score = model.evaluate(X_test, y_test, verbose=0)
    predictions = enc.inverse_transform(predictions)
    y_test = enc.inverse_transform(y_test)

    y_test = pd.DataFrame(y_test,columns=['next event']).join(pd.DataFrame(predictions, columns=['predicted next event']))

    classification_performance(y_test, "Confusion_Matrices/LSTM.png")

def load_model(file):
    model = keras.models.load_model(file)
    return model

def train_model(X_train, y_train, file=None):
    model = train_event(X_train, y_train_clf, y_train_reg, epochs)
    if file:
        model.save(file)
    return model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, enc = preprocess_event()
    model = train_model(X_train, y_train,'LSTM_models/class_model_more_inputs2.h5')
    # model = load_model('LSTM_models/class_model_more_inputs.h5')
    test(X_test, y_test, model, enc)
