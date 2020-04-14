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
from  datetime import datetime, timedelta
# FYI: Objective functions can take additional arguments
# (https://autohpo.readthedocs.io/en/stable/faq.html#objective-func-additional-args).

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

pd.options.display.max_columns = 50
NUM_LEAVES = 76

h = 28
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25)
fday
def objective(trial):
    # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    #model_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    #df = pd.read_pickle(model_dir + '/m5_data_FIRST_DAY_1.pkl')
    #df = pd.read_pickle("/Users/apple/automl/auto-hpo/examples/m5faccuracy/m5_data.pkl")
    '''
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
    '''
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
	#'device':'gpu',
	#'gpu_platform_id':0,
	#'gpu_device_id': 1,
        'boosting_type':'gbdt', #trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 50, 256),
        'max_depth':trial.suggest_int('max_depth', 3, 20),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        #'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',10, 150),
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
    return accuracy, m_lgb,'gbm/'

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

def create_dt(is_train=True, nrows=None, first_day=1200):
    #directory = "/Users/apple/automl/auto-hpo/input/data/m5faccuracy"
    directory = '/pfs/auto-hpo/auto-hpo/input/data/m5'
    prices = pd.read_csv(directory+"/sell_prices.csv", dtype=PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    cal = pd.read_csv(directory+"/calendar.csv", dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()

    start_day = max(1 if is_train else tr_last - max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day, tr_last + 1)]
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    dtype = {numcol: "float32" for numcol in numcols}
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(directory+"/sales_train_validation.csv",
                     nrows=nrows, usecols=catcols + numcols, dtype=dtype)

    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()

    if not is_train:
        for day in range(tr_last + 1, tr_last + 28 + 1):
            dt[f"d_{day}"] = np.nan

    dt = pd.melt(dt,
                 id_vars=catcols,
                 value_vars=[col for col in dt.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")

    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    return dt



def m5_predict():
    # directory = "/Users/apple/automl/auto-hpo/input/data/PredictFutureSales/"
    #
    # data = pd.read_pickle('/Users/apple/automl/auto-hpo/examples/predictfuturesales/cfp_data.pkl')
    data_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    df = pd.read_pickle(data_dir + '/m5_data_FIRST_DAY_1.pkl')
    df = create_dt(False)
    df.to_pickle('m5_test_data.pkl')
    useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]
    #lgb = pickle.load(open('/Users/apple/automl/auto-hpo/output/xgbmodel/18_model_train.pkl', 'rb'))
    lgb = pickle.load(open('/pfs/auto-hpo/auto-hpo/output/model/gbm/18_model_train.pkl', 'rb'))

    alphas = [1.035, 1.03, 1.025, 1.02]
    weights = [1 / len(alphas)] * len(alphas)
    sub = 0.

    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

        te = create_dt(False) 
        cols = [f"F{i}" for i in range(1, 29)]

        for tdelta in range(0, 28):
            day = fday + timedelta(days=tdelta)
            print(icount, day)
            tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
            create_fea(tst)
            tst = tst.loc[tst.date == day, train_cols]
            te.loc[te.date == day, "sales"] = alpha * lgb.predict(tst)  # magic multiplier by kyakovlev

        te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
        #     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h),
        #                                                                           "id"].str.replace("validation$", "evaluation")
        te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
        te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
        te_sub.fillna(0., inplace=True)
        te_sub.sort_values("id", inplace=True)
        te_sub.reset_index(drop=True, inplace=True)
        te_sub.to_csv(f"submission_m5_{icount}.csv", index=False)
        if icount == 0:
            sub = te_sub
            sub[cols] *= weight
        else:
            sub[cols] += te_sub[cols] * weight
        print(icount, alpha, weight)

    sub2 = sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv("m5_18_submission.csv", index=False)

    sub.head(10)

    sub.id.nunique(), sub["id"].str.contains("validation$").sum()

    sub.shape
if __name__ == '__main__':
    '''
    study = autohpo.create_study()
    study.optimize(objective, n_trials=100)
    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial
    print(str(trial.number))
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    '''
    #test_predict()
    m5_predict()
