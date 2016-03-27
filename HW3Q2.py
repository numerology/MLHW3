import numpy as np
import scipy.sparse
import pickle
import pandas as pd
import xgboost as xgb
#from xgb_util import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston

rng = np.random.RandomState(31337)

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

dtrain = xgb.DMatrix('train.csv')

param = {'max_depth':2, 'eta':0.5, 'silent':1, 'objective':'binary:logistic'}

num_round = 300

#print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
# result = xgb.cv(param, dtrain, num_round, nfold=5,
#        metrics={'error'}, seed = 1)
# print(result)
# I am going to assume the impact of different params are decoupled
# Try iterate thru max depth, eta, lambda, gamma

# max_depth_result = []
# for max_depth in range(5,100,7):
# 	param = {'max_depth': max_depth, 'eta':1, 'gamma':0, 'lambda':1, 'silent':1, 'objective':'binary:logistic'}
# 	result = xgb.cv(param, dtrain, num_round, nfold=3,
#        metrics={'error'}, seed = 0)
# 	result_mat = result.as_matrix()
# 	max_depth_result.append(np.mean(result_mat[:,0]))

#print(max_depth_result)
#result shows max_depth should be larger

# eta_result = []
# for eta in frange(0.1, 1, 0.1):
# 	param = {'max_depth':100, 'eta':eta, 'gamma':0, 'lambda':1, 'silent':1, 'objective':'binary:logistic'}
# 	result = xgb.cv(param, dtrain, num_round, nfold=5,
#        metrics={'error'}, seed = 0)
# 	result_mat = result.as_matrix()
# 	eta_result.append(np.mean(result_mat[:,0]))

# print(eta_result) # not significant still, take 0.7

# gamma_result = []
# for log_gamma in frange(-2,2,0.5):
# 	param = {'max_depth':50, 'eta':0.7, 'gamma':10 ** log_gamma, 'lambda':1, 'silent':1, 'objective':'binary:logistic'}
# 	result = xgb.cv(param, dtrain, num_round, nfold=3,
#        metrics={'error'}, seed = 0)
# 	result_mat = result.as_matrix()
# 	gamma_result.append(np.mean(result_mat[:,0]))

# print(gamma_result) #take 1

# lambda_result = []
# for log_lamb in frange(-0,4,0.5):
# 	param = {'max_depth':50, 'eta':0.7, 'gamma':1, 'lambda':10 ** log_lamb, 'silent':1, 'objective':'binary:logistic'}
# 	result = xgb.cv(param, dtrain, num_round, nfold=3,
#        metrics={'error'}, seed = 0)
# 	result_mat = result.as_matrix()
# 	lambda_result.append(np.mean(result_mat[:,0]))

# print(lambda_result) #take 1

param = {'max_depth':100, 
		'silent':1, 'objective':'binary:logistic',
		'max_delta_step':1.5, 'n_estimator':200, 'colsample_bytree':0.3,
		'scale_pos_weight':0.0614, }

#no access to true test set, separate the training set


# dtest = xgb.DMatrix('test.csv')
# train_df = pd.read_csv('train.csv', header = 0)
# y = train_df['ACTION']
# X = train_df[train_df.columns.values[1:]]
# test_df = pd.read_csv('test.csv', header = 0)
# #dtrain = xgb.DMatrix(train_df[train_df.columns.values[1:]], label = y)
# dtrain = xgb.DMatrix(train_df[['ROLE_FAMILY','ROLE_ROLLUP_1','ROLE_TITLE','RESOURCE','ROLE_DEPTNAME','ROLE_ROLLUP_2','MGR_ID','ROLE_FAMILY_DESC']], 
# 			label = train_df['ACTION'])
# print(dtrain.num_row())
# train_mat = dtrain.slice(range(0,dtrain.num_row() - 10000))
# test_mat = dtrain.slice(range(dtrain.num_row() - 9999, dtrain.num_row()))
# print(train_mat.num_row())
# # #print(dtrain[0:22770,:])
# # print(dtrain.feature_names)
# watch_list = [(test_mat, 'eval'), (train_mat, 'train')]

# #Piazza: By using the gridsearchCV function and just tuning Max depth, nestimators, and colsamplebytree, 
# #i got 0.88. Everything else is default. Make sure to use the predict_proba instead of predict.
clsf = xgb.XGBClassifier()
# # clf = GridSearchCV(clsf, {'max_depth':[100], 'n_estimators':[200],
# # 							'colsample_bytree':[0.3], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100],
# # 							'max_delta_step':[0.1, 1, 3, 10, 30], 'subsample':[0.5, 1],
# # 							'colsample_bylevel':[0.5, 0.8, 1]}, verbose = 1)
# # clf.fit(X,y)
# # print(clf.best_score_)
# # print(clf.best_params_)

# #best comb = n estimator = 200, colsample 0.3 max_depth 100

# bst = xgb.train(param, dtrain, num_round)
# bst.save_model('test.model')
# #try different max_delta_step
# # max_delta_step_result = []
# # for max_delta_step in frange(1, 2, 0.1):
# # 	param = {'max_depth':100,
# # 		'silent':1, 'objective':'binary:logistic',
# # 		'max_delta_step':max_delta_step, 'n_estimator':200, 'colsample_bytree':0.3,
# # 		'scale_pos_weight':0.0614, 'n_estimators':200}
# # 	result = xgb.cv(param, dtrain, num_round, nfold=3,
# #        metrics={'error'}, seed = 0)
# # 	result_mat = result.as_matrix()
# # 	max_delta_step_result.append(np.mean(result_mat[:,0]))
# # 	#print(bst.eval(test_mat))
# # print(max_delta_step_result)
# print(bst.get_fscore())


# test_X = xgb.DMatrix(test_df[['ROLE_FAMILY','ROLE_ROLLUP_1', 'ROLE_TITLE','RESOURCE','ROLE_DEPTNAME','ROLE_ROLLUP_2','MGR_ID','ROLE_FAMILY_DESC']])
# #print(test_X.feature_names)
# predictions1 = bst.predict(test_X)
# predictions1_df = pd.DataFrame({'id':test_df['id'], 'ACTION':predictions1})


# dtrain2 = xgb.DMatrix(train_df[['ROLE_FAMILY','ROLE_CODE','ROLE_TITLE','RESOURCE','ROLE_DEPTNAME','ROLE_ROLLUP_2','MGR_ID','ROLE_FAMILY_DESC']], 
# 			label = train_df['ACTION'])
# bst = xgb.train(param, dtrain2, num_round)
# test_X2 = xgb.DMatrix(test_df[['ROLE_FAMILY','ROLE_CODE', 'ROLE_TITLE','RESOURCE','ROLE_DEPTNAME','ROLE_ROLLUP_2','MGR_ID','ROLE_FAMILY_DESC']])
# predictions2 = bst.predict(test_X2)
# predictions2_df = pd.DataFrame({'id':test_df['id'], 'ACTION':predictions2})
# #save
# predictions1_df.to_csv('pred1.csv', index = False)
# predictions2_df.to_csv('pred2.csv', index = False)

#load prev results
predictions_logis = pd.read_csv('submission_logis.csv', header = 0)
predictions_logis = predictions_logis['ACTION']
predictions_fm = pd.read_csv('sumission_FM_best.csv', header = 0)
predictions_fm = predictions_fm['ACTION']

predictions1 = pd.read_csv('pred1.csv', header = 0)
predictions1 = predictions1['ACTION']
predictions2 = pd.read_csv('pred2.csv', header = 0)
predictions2 = predictions2['ACTION']

#Tune the coef
predictions = 0.44 * (0.35 * predictions1 + 0.15 * predictions2 + 0.5 * predictions_logis) + 0.56 * predictions_fm

print(predictions)
test_df = pd.read_csv('test.csv', header = 0)
submission = pd.DataFrame({'id':test_df['id'], 'ACTION':predictions})
submission.to_csv('submission.csv', index = False)