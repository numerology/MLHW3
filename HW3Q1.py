__author__ = 'Jiaxiao Zheng'


from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston

enc = OneHotEncoder()
#param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'intercept_scaling': [0.01, 0.1, 1, 10, 100]}
model = LogisticRegression(C = 1, penalty = 'l2', intercept_scaling = 0.01, solver = 'liblinear', 
		class_weight = 'balanced', n_jobs = -1)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
trainSet = train_df.as_matrix()
testSet = test_df.as_matrix()
trainX = trainSet[:, 1:]
trainY = trainSet[:, 0]
testX = testSet[:, 1:]
#testY = testSet[:, 0]

enc.fit(np.concatenate((trainX, testX)))
trainX = enc.transform(trainX)
testX = enc.transform(testX)


print('finished reading' + 'size of training set' + str(trainX.shape[0]))
model.fit(trainX,trainY)
print('finished fitting')

trainEst = model.predict_proba(trainX)
trainEst = trainEst[:,1]
print(trainEst)

testEst = model.predict_proba(testX)
testEst = testEst[:, 1]

submission = pd.DataFrame({'id':test_df['id'], 'ACTION':testEst})
submission.to_csv('submission_logis.csv', index = False)

fpr, tpr, thresholds = metrics.roc_curve(trainY, trainEst, pos_label = 1)
auc = metrics.auc(fpr, tpr)

print(auc)

