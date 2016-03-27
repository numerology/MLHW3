from pyfm import pylibfm
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

fm = pylibfm.FM(num_factors = 25, num_iter = 50)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
trainSet = train_df.as_matrix()
testSet = test_df.as_matrix()
trainX = trainSet[:, 1:]
trainY = trainSet[:, 0]
testX = testSet[:, 1:]

enc = OneHotEncoder()
enc.fit(np.concatenate((trainX, testX)))
trainX = enc.transform(trainX)
testX = enc.transform(testX)

fm.fit(trainX, trainY)
testY = fm.predict(testX)
print(testY)

submission = pd.DataFrame({'id':test_df['id'], 'ACTION':testY})
submission.to_csv('submission_FM.csv', index = False)