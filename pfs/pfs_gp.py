import sys
print(sys.path)
from autohpo.samplers.gp.autohpo import AUTOHPORegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import os,gc,time
import pandas as pd

autohpo_config = {
    # 'sklearn.naive_bayes.GaussianNB': {
    # },
    #
    # 'sklearn.naive_bayes.BernoulliNB': {
    #     'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
    #     'fit_prior': [True, False]
    # },
    #
    # 'sklearn.naive_bayes.MultinomialNB': {
    #     'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
    #     'fit_prior': [True, False]
    # },

    'xgboost.XGBClassifier': {
        'n_estimators': [10],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    }
    # ,
    # 'sklearn.neighbors.KNeighborsClassifier': {
    #     'n_neighbors': range(1, 101),
    #     'weights': ["uniform", "distance"],
    #     'p': [1, 2]
    # }
}
model_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data = pd.read_pickle(model_dir+'/predictfuturesales/data.pkl')

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

autohpo = AUTOHPORegressor(generations=5,verbosity=2, max_time_mins=3, population_size=50,random_state=42)#config_dict=autohpo_config)
autohpo.fit(X_train, Y_train)

print('----')
print(autohpo.score(X_valid, Y_valid))

autohpo.export('autohpo_pfs_pipeline.py')