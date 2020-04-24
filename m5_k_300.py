"""
autohpo example that optimizes a neural network classifier configuration for the
MNIST dataset using Keras.

In this example, we optimize the validation accuracy of MNIST classification using
Keras. We optimize the filter and kernel size, kernel stride and layer activation.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python keras_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`autohpo create-study --direction maximize --storage sqlite:///example.db`
    $ autohpo study optimize keras_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

from keras.backend import clear_session
from keras.datasets import mnist
from keras.layers import Conv1D,Input
from keras.layers import Dense
from keras.layers import Flatten, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam
import pandas as pd
import numpy as np
import autohpo, os
from keras import layers
import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras import callbacks
from keras.utils import multi_gpu_model
from keras import optimizers
N_TRAIN_EXAMPLES = 300
N_TEST_EXAMPLES = 100
#BATCHSIZE = 128
CLASSES = 10
EPOCHS = 300
pred_steps = 28
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    # print(x_train.shape)
    # print(x_train.shape[1])
    # print(x_train.shape[2])
    #(60000, 28,28)
    x_train, y_train, x_test, y_test = load_data()
    #x_train, y_train, x_test, y_test = load_data()
    scaler = StandardScaler()
    scaler.fit(pd.concat([x_train, x_test]))
    x_train[:] = scaler.transform(x_train)
    x_test[:] = scaler.transform(x_test)

    #X_train = x_train.as_matrix()
    #X_test = x_test.as_matrix()
    X_test = x_test.iloc[:,:].values
    X_train = x_train.iloc[:,:].values

    x_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    x_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(x_train.shape)
    print(x_train.shape[1])
    print(x_train.shape[2])
    print(x_train.shape)
    print(x_train.shape[0])
    print(x_train.shape[1])

    
    BATCHSIZE = trial.suggest_categorical('BATCHSIZE',[32768,46000])
    '''
     units1 = trial.suggest_categorical('units1', [128, 256,512])
    units2 = trial.suggest_categorical('units2', [8, 16, 32])
    input = Input(shape=(25,))
    x = Dense(int(units1), activation=trial.suggest_categorical('activation', ['relu', 'linear']))(input)
    x = Dense(int(units2), activation='relu')(x)
    y = Dense(1, activation='softmax')(x)
    model = keras.Model(inputs=input, outputs=y)
    model.compile(Adam(), loss='mean_squared_error', metrics=['mse'])
    '''
    units1 = trial.suggest_categorical('units1', [512, 1024])
    units2 = trial.suggest_categorical('units2', [128,256,512])
    units3 = trial.suggest_categorical('units3', [64,128])
    units4 = trial.suggest_categorical('units4', [32, 64])
    units5 = trial.suggest_categorical('units5', [16, 32])

    lr = trial.suggest_categorical('lr', [0.001, 0.005, 0.01])
    drop_rate1 = trial.suggest_categorical('drop_rate1', [0.08,0.1, 0.2])
    drop_rate2 = trial.suggest_categorical('drop_rate2', [0.01, 0.05, 0.008])
    model = Sequential()
    model.add(LSTM(units1, input_shape=(x_train.shape[1], x_test.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate1))

    model.add(Dense(units2))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate1))

    model.add(Dense(units2))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate1))

    model.add(Dense(units3))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate2))

    model.add(Dense(units3))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate2))

    model.add(Dense(units4))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate2))

    model.add(Dense(units5))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate2))

    model.add(Dense(1))
    model=multi_gpu_model(model, gpus=4)
    #model = build_model()
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
    # img_x, img_y = 1, x_train.shape[1]
    # x_train = x_train.reshape(-1, img_x, img_y)[:N_TRAIN_EXAMPLES].astype('float32') / 255
    # x_test = x_test.reshape(-1, img_x, img_y)[:N_TEST_EXAMPLES].astype('float32') / 255
    # y_train = y_train[:N_TRAIN_EXAMPLES]
    # y_test = y_test[:N_TEST_EXAMPLES]
    # # # input_shape = (img_x, img_y, 1)
    # #
    # # model = Sequential()
    # # # model.add(
    # # #     Conv2D(filters=trial.suggest_categorical('filters', [32, 64]),
    # # #            kernel_size=trial.suggest_categorical('kernel_size', [1, 1]),
    # # #            strides=trial.suggest_categorical('strides', [1, 2]),
    # # #            activation=trial.suggest_categorical('activation', ['relu', 'linear']),
    # # #            input_shape=input_shape))
    # # #model.add(Flatten())
    # # model.add(Dense(units=64, activation='relu'))
    # # model.add(Flatten())
    # # model.add(Dense(1464593, activation="linear", name='output'))
    # # #model.add(Activation('softmax'))
    # #
    # # # We compile our model with a sampled learning rate.
    # # lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    # # model.compile(loss='mean_squared_error',
    # #               optimizer=RMSprop(lr=lr),
    # #               metrics=['mse'])
    # # convolutional operation parameters
    # # Maybe try less? Was 32 but due to memory issues trying 4
    # n_filters = 4  # 32
    # filter_width = 2  # Was 2
    # # Maybe tree less? Was 8,  tried 4, trying 16... Overfitting for sure...
    # #  Due to limited memory, will be restrained to 4
    # dilation_rates = [2 ** i for i in range(4)] * 2
    #
    # # define an input history series and pass it through a stack of dilated causal convolution blocks.
    # # Note the feature input dimension corresponds to the raw series and all exogenous features
    # history_seq = Input(shape=(None,  x_train.shape[-1]))
    # x = history_seq
    #
    #
    # skips = []
    # for dilation_rate in dilation_rates:
    #     # preprocessing - equivalent to time-distributed dense
    #     x = layers.Conv1D(32, 1, padding='same', activation='relu')(x)
    #
    #     # filter convolution
    #     x_f = layers.Conv1D(filters=n_filters,
    #                         kernel_size=filter_width,
    #                         padding='causal',
    #                         dilation_rate=dilation_rate)(x)
    #
    #     # gating convolution
    #     x_g = layers.Conv1D(filters=n_filters,
    #                         kernel_size=filter_width,
    #                         padding='causal',
    #                         dilation_rate=dilation_rate)(x)
    #
    #     # multiply filter and gating branches
    #     z = layers.Multiply()([layers.Activation('tanh')(x_f),
    #                            layers.Activation('sigmoid')(x_g)])
    #
    #     # postprocessing - equivalent to time-distributed dense
    #     z = layers.Conv1D(32, 1, padding='same', activation='relu')(z)
    #
    #     # residual connection
    #     x = layers.Add()([x, z])
    #
    #     # collect skip connections
    #     skips.append(z)
    #
    # # add all skip connection outputs
    # out = layers.Activation('relu')(layers.Add()(skips))
    #
    # # final time-distributed dense layers
    # out = layers.Conv1D(128, 1, padding='same')(out)
    # out = layers.Activation('relu')(out)
    # out = layers.Dropout(.2)(out)
    # out = layers.Conv1D(1, 1, padding='same')(out)
    #
    # def slice(x, seq_length):
    #     return x[:, -seq_length:, :]
    #
    # #pred_seq_train = layers.Lambda(slice, arguments={'seq_length': pred_steps})(out)
    #
    # model = keras.Model(history_seq, out)
    # model.compile(Adam(), loss='mean_squared_error', metrics=['mse'])
    #
    # model.summary()
    #
    es = keras.callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0, patience=10, verbose=0, mode='min',
                                            baseline=None, restore_best_weights=True)
    model.fit(x_train,
              y_train.tolist(),
              validation_data=(x_test, y_test.tolist()),
              shuffle=True,
              batch_size=BATCHSIZE,
              epochs=EPOCHS,
              verbose=True,
              callbacks=[es])

    # Evaluate the model accuracy on the test set.

    score = model.evaluate(x_test, y_test.tolist(), verbose=0)

    return score[1],model, 'dl/'

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id", "sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(
                lambda x: x.rolling(win).mean())

    date_features = {

        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
        #         "ime": "is_month_end",
        #         "ims": "is_month_start",
    }

    #     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")

def load_data():
    print('loading......')
    model_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    df = pd.read_pickle(model_dir + '/m5_data_FIRST_DAY_1.pkl')
    #df = pd.read_pickle('/Users/apple/automl/auto-hpo/examples/m5faccuracy/m5_data.pkl')
    # df.shape
    # df.head()
    #
    df.info()
    #
    create_fea(df)
    # df.shape
    #
    # df.info()
    #
    df.head()
    #
    df.dropna(inplace=True)
    # df.shape
    #df = pd.read_pickle('/Users/apple/automl/auto-hpo/examples/m5faccuracy/m5_data_1000_test.pkl')

    cat_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2",
                                                                                "event_type_1", "event_type_2"]
    useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]

    train_cols = df.columns[~df.columns.isin(useless_cols)]

    X_train = df[train_cols]
    y_train = df["sales"]
    np.random.seed(777)

    fake_valid_inds = np.random.choice(X_train.index.values, 2000000, replace=False)
    train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
    train_x = X_train.loc[train_inds]
    train_y = y_train.loc[train_inds],
    #categorical_feature=cat_feats, free_raw_data=False)
    valid_x = X_train.loc[fake_valid_inds]
    valie_y = y_train.loc[fake_valid_inds]
    #categorical_feature=cat_feats,free_raw_data=False)
    print('end.....')
    #return np.asarray(train_x),np.asarray(train_y), np.asarray(valid_x),np.asarray(valie_y)
    return  train_x, np.asarray(train_y), valid_x, np.asarray(valie_y)
if __name__ == '__main__':
    study = autohpo.create_study()
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

