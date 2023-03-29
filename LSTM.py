import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import keras.layers
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import splitter
import constants
from sklearn.metrics import accuracy_score
from performance_measures import classification_performance, regression_performance

plt.style.use('fivethirtyeight')


# Encodes string data as one-hot vectors
def get_one_hot_encoder(data):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data)
    return enc

def normalize(train_data, test_data):
    train_mean = train_data.mean()
    train_std = train_data.std()
    return ((train_data - train_mean) / train_std), ((test_data - train_mean) / train_std)

def preprocess_event_X(data, enc, max_case_len):
    num_of_predictors = 2    # CHANGE THIS AS NEEDED, rn its time and amount requested
    
    # number of features is the total number of distinct events plus the
    # number of other columns we use as predictors
    num_of_features = len(enc.get_feature_names_out()) + num_of_predictors
    grouped = data.groupby([constants.CASE_ID_COLUMN])
    X = np.zeros((len(data.index), max_case_len, num_of_features), dtype=np.float32)

    row_counter = 0
    for case_id, group in grouped:
        event_sequence = []
        for index, row in group.iterrows():
            current_event = list(enc.transform(row[[constants.CURRENT_EVENT]].to_numpy().reshape(1, -1)).toarray()[0])
            # print(current_event)
            # print(row[[constants.CURRENT_EVENT]])
            current_event.append(row['amount requested normalized'])
            current_event.append(row['time since previous event'])

            event_sequence.append(current_event)
            event_index = max_case_len - 1
            for event in reversed(event_sequence):
                np.copyto(X[row_counter, event_index], np.array(event).astype('float32'))
                event_index -= 1
            row_counter += 1
    return X



def preprocess_event():
    full_data = pd.read_csv(constants.PIPELINED_DATASET_PATH)
    full_data[constants.CASE_TIMESTAMP_COLUMN] = pd.to_datetime(
        full_data[constants.CASE_TIMESTAMP_COLUMN])

    full_data[constants.TIME_SINCE_PREVIOUS_EVENT] = full_data.groupby(
        constants.CASE_ID_COLUMN)[constants.TIME_DIFFERENCE].shift(1, fill_value = 0)

    # full_data = correct_data(full_data)
    # full_data.is_weekend = full_data.is_weekend.replace({True: 1, False: 0})
    # full_data.is_work_time = full_data.is_work_time.replace({True: 1, False: 0})
    # full_data.is_holiday = full_data.is_holiday.replace({True: 1, False: 0})

    full_data.dropna(inplace=True)
    train_data, test_data = splitter.split_dataset(full_data, 0.2)
    
    enc = get_one_hot_encoder(train_data[[constants.CURRENT_EVENT]].to_numpy())
    enc_next = get_one_hot_encoder(train_data[['next event']].append(pd.DataFrame(['END'], columns=['next event'])).to_numpy())
    
    max_case_len = int(full_data[constants.CASE_STEP_NUMBER_COLUMN].max())
    # max_case_len = 174
    train_data['amount requested normalized'], test_data['amount requested normalized'] = normalize(train_data[constants.AMOUNT_REQUESTED_COLUMN], test_data[constants.AMOUNT_REQUESTED_COLUMN])
    train_data['time since previous event'], test_data['time since previous event'] = normalize(train_data[constants.TIME_SINCE_PREVIOUS_EVENT], test_data[constants.TIME_SINCE_PREVIOUS_EVENT])

    X_train, X_test = preprocess_event_X(train_data, enc, max_case_len), preprocess_event_X(test_data, enc, max_case_len)

    y_train_reg = train_data[constants.TIME_DIFFERENCE]
    y_test_reg = test_data[constants.TIME_DIFFERENCE]
    
    y_train_clf = enc_next.transform(train_data[['next event']]).toarray()
    y_test_clf = enc_next.transform(train_data[['next event']]).toarray()

    return X_train, y_train_clf, y_train_reg, X_test, y_test_clf, y_test_reg, enc_next

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

    history = model.fit(X_train, {'event_output':y_train_clf, 'time_output':y_train_reg}, validation_split=0.2, verbose=1, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=max_sequence_len, epochs=epochs)
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
    
def test_model(X_test, y_test_clf, y_test_reg, enc):
    model = keras.models.load_model("LSTM_models/model_07-27919.88.h5")
    pred_clf, pred_reg = model.predict(X_test)
    pred_clf = enc.inverse_transform(pred_clf)
    y_test_clf_dec = enc.inverse_transform(y_test_clf)

    y_test_clf = pd.DataFrame(y_test_clf_dec, columns=[constants.NEXT_EVENT]).join(pd.DataFrame(pred_clf, columns=[constants.NEXT_EVENT_PREDICTION]))
    classification_performance(y_test_clf, "Confusion_Matrices/conf_matrix_LSTM_test.png")

    y_test_reg = pd.DataFrame(y_test_reg.array, columns=[constants.TIME_DIFFERENCE]).join(pd.DataFrame(pred_reg.flatten(), columns=[constants.TIME_DIFFERENCE_PREDICTION]))
    regression_performance(y_test_reg)

def load_model(file):
    model = keras.models.load_model(file)
    return model

def train_model(X_train, y_train_clf, y_train_reg, epochs, file=None):
    model = train_event(X_train, y_train_clf, y_train_reg, epochs)
    if file:
        model.save(file)
    return model

if __name__ == "__main__":
    X_train, y_train_clf, y_train_reg, X_test, y_test_clf, y_test_reg, enc = preprocess_event()
    # X_test = np.load("X_test.npy")
    #test_model(X_test, y_test_clf, y_test_reg, enc)
    model = train_model(X_train, y_train_clf,y_train_reg, 50)
    # model = load_model('LSTM_models/class_model_more_inputs.h5')
    #test(X_test, y_test, model, enc)
