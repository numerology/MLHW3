import pandas as pd
import numpy as np
import xgboost as xgb

data=pd.read_csv('train.csv' ).values
testdata=pd.read_csv('test.csv').values

r,c=data.shape
x_train=data[: , 1:c]
y_train=data[: , 0]

rt, ct=testdata.shape
x_test=testdata[:, 1:ct]

dtrain=xgb.DMatrix(x_train, label=y_train)

param={'objective':'binary:logistic', 'bst:max_delta_step': 2}

num_round=10

bst=xgb.train(param, dtrain, num_round)

dtest=xgb.DMatrix(x_test)
y_test=bst.predict(dtest)

df=pd.DataFrame({'Id': testdata[:,0], 'Action': y_test}, columns=['Id', 'Action'])
df.to_csv('submission2.csv', index=False)



