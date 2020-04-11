"""
autohpo example that optimizes a classifier configuration for cancer dataset using LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python lightgbm_pfs_cpu.py


(2) Execute through CLI.
    $ STUDY_NAME=`autohpo create-study --direction maximize --storage sqlite:///example.db`
    $ autohpo study optimize lightgbm_pfs_cpu.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import os,time,gc
import pandas as pd
import autohpo
import pickle
import xgboost as xgb

# FYI: Objective functions can take additional arguments
# (https://autohpo.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    model_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    df = pd.read_pickle(model_dir + '/m5/m5_data_FIRST_DAY_1.pkl')
    #df = pd.read_pickle("/Users/apple/automl/auto-hpo/examples/m5faccuracy/m5_data.pkl")
    df.shape
    df.head()

    df.info()

    create_fea(df)
    df.shape

    df.info()

    df.head()

    df.dropna(inplace=True)
    df.shape

    cat_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2",
                                                                            "event_type_1", "event_type_2"]
    useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]
    X_train = df[train_cols]
    y_train = df["sales"]

    # train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats, free_raw_data=False)
    # fake_valid_inds = np.random.choice(len(X_train), 1000000)
    # fake_valid_data = lgb.Dataset(X_train.iloc[fake_valid_inds], label=y_train.iloc[fake_valid_inds],
    #                               categorical_feature=cat_feats,
    #                               free_raw_data=False)  # This is just a subsample of the training set, not a real validation set !
    np.random.seed(777)

    fake_valid_inds = np.random.choice(X_train.index.values, 2000000, replace=False)
    train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
    train_data = lgb.Dataset(X_train.loc[train_inds], label=y_train.loc[train_inds],
                             categorical_feature=cat_feats, free_raw_data=False)
    fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label=y_train.loc[fake_valid_inds],
                                  categorical_feature=cat_feats,
                                  free_raw_data=False)  # This is a random sample, we're not gonna apply any time series train-test-split tricks here!

    # params = {
    #         "objective" : "poisson",
    #         "metric" :"rmse",
    #         "force_row_wise" : True,
    #         "learning_rate" : 0.075,
    # #         "sub_feature" : 0.8,
    #         "sub_row" : 0.75,
    #         "bagging_freq" : 1,
    #         "lambda_l2" : 0.1,
    # #         "nthread" : 4
    #         "metric": ["rmse"],
    #     'verbosity': 1,
    #     'num_iterations' : 2500,
    #     'num_leaves': NUM_LEAVES
    # }
    # params = {
    #     "objective": "poisson",
    #     "metric": "rmse",
    #     "force_row_wise": True,
    #     "learning_rate": 0.075,
    #     #         "sub_feature" : 0.8,
    #     "sub_row": 0.75,
    #     "bagging_freq": 1,
    #     "lambda_l2": 0.1,
    #     #         "nthread" : 4
    #     "metric": ["rmse"],
    #     'verbosity': 1,
    #     'num_iterations': 1200,
    #     'num_leaves': 128,
    #     "min_data_in_leaf": 100,
    # }
    params = {
        'objective': 'poisson',
        'metric': 'rmse',
        "force_row_wise": True,
        'verbosity': 1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 50, 256),
        'max_depth':trial.suggest_int('max_depth', 3, 20),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',50, 200),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'num_iterations': 1200,
    }
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        params['max_drop'] = trial.suggest_int('max_drop', 10, 150)
        params['xgboost_dart_mode'] = True

    m_lgb = lgb.train(params, train_data, valid_sets=[fake_valid_data], verbose_eval=20,early_stopping_rounds=20)

    preds = m_lgb.predict(fake_valid_data.data)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.mean_squared_error(fake_valid_data.label, pred_labels)
    return accuracy, m_lgb

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

def test_predict():
    directory = "/Users/apple/automl/auto-hpo/input/data/PredictFutureSales/"
    test = pd.read_csv(directory + 'test.csv').set_index('ID')
    data = pd.read_pickle('/Users/apple/automl/auto-hpo/examples/predictfuturesales/data.pkl')

    data = data[[
        'date_block_num',
        'shop_id',
        'item_id',
        'item_cnt_month',
        'city_code',
        'item_category_id',
        'type_code',
        'subtype_code',
        'item_cnt_month_lag_1',
        'item_cnt_month_lag_2',
        'item_cnt_month_lag_3',
        'item_cnt_month_lag_6',
        'item_cnt_month_lag_12',
        'date_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_2',
        'date_item_avg_item_cnt_lag_3',
        'date_item_avg_item_cnt_lag_6',
        'date_item_avg_item_cnt_lag_12',
        'date_shop_avg_item_cnt_lag_1',
        'date_shop_avg_item_cnt_lag_2',
        'date_shop_avg_item_cnt_lag_3',
        'date_shop_avg_item_cnt_lag_6',
        'date_shop_avg_item_cnt_lag_12',
        'date_cat_avg_item_cnt_lag_1',
        'date_shop_cat_avg_item_cnt_lag_1',
        # 'date_shop_type_avg_item_cnt_lag_1',
        # 'date_shop_subtype_avg_item_cnt_lag_1',
        'date_city_avg_item_cnt_lag_1',
        'date_item_city_avg_item_cnt_lag_1',
        # 'date_type_avg_item_cnt_lag_1',
        # 'date_subtype_avg_item_cnt_lag_1',
        'delta_price_lag',
        'month',
        'days',
        'item_shop_last_sale',
        'item_last_sale',
        'item_shop_first_sale',
        'item_first_sale',
    ]]

    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    del data
    gc.collect();

    ts = time.time()
    print(ts)
    model = pickle.load(open('/Users/apple/automl/auto-hpo/output/xgbmodel/1_model_train.pkl','rb'))
    Y_test = model.predict(X_test).clip(0, 20)

    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    submission.to_csv('gbm_submission.csv', index=False)
def test_cfp_predict():
    # directory = "/Users/apple/automl/auto-hpo/input/data/PredictFutureSales/"
    #
    # data = pd.read_pickle('/Users/apple/automl/auto-hpo/examples/predictfuturesales/cfp_data.pkl')

    model_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_pickle(model_dir+'/cfp_data.pkl')
    test = pd.read_csv('/pfs/auto-hpo/auto-hpo/input/data/PredictFutureSales/test.csv')  # .set_index('ID')

    # test.loc[test.shop_id == 0, "shop_id"] = 57
    #
    # test.loc[test.shop_id == 1, "shop_id"] = 58
    #
    # test.loc[test.shop_id == 11, "shop_id"] = 10
    #
    # test.loc[test.shop_id == 40, "shop_id"] = 39
    # test["date_block_num"] = 34
    # test["date_block_num"] = test["date_block_num"].astype(np.int8)
    # test["shop_id"] = test.shop_id.astype(np.int8)
    # test["item_id"] = test.item_id.astype(np.int16)
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    del data
    gc.collect();

    ts = time.time()
    print(ts)
    model = pickle.load(open('/pfs/auto-hpo/auto-hpo/output/xgbmodel/118_xgb_train.pkl','rb'))
    Y_test = model.predict(xgb.DMatrix(X_test)).clip(0, 20)

    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    submission.to_csv('cfp_118_submission.csv', index=False)
if __name__ == '__main__':
    study = autohpo.create_study()
    study.optimize(objective, n_trials=1000)
    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial
    print(str(trial.number))
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    #test_predict()
    #test_cfp_predict()
