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


# FYI: Objective functions can take additional arguments
# (https://autohpo.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    model_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_pickle(model_dir + '/predictfuturesales/data.pkl')

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
    # print(data)
    #
    # for num in range(4488710, 6488710):
    #     data = data.drop([num])

    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    print(len(X_train))
    del data
    gc.collect();

    ts = time.time()
    #dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = lgb.Dataset(X_valid, label=Y_valid)
    dtrain = lgb.Dataset(X_train, label=Y_train)

    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),


    }

    gbm = lgb.train(param, dtrain,valid_sets=[dtest], early_stopping_rounds=10)
    preds = gbm.predict(X_valid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.mean_squared_error(Y_valid, pred_labels)
    return accuracy, gbm


if __name__ == '__main__':
    study = autohpo.create_study()
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
