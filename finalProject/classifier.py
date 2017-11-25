# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd
import numpy as np
from  sklearn.metrics import accuracy_score
from sklearn.utils import resample

def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
 
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 
def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)
 
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]

print("hii5")
df = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print("hii4")

df_majority=df[df.target==0]
df_minority=df[df.target==1]
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=573518,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
y= df_upsampled['target']
df_upsampled.drop('target',axis=1)
df_upsampled.drop('id',axis=1)

ids = df_test['id'].as_matrix()
df_test.drop('id',axis=1)

# normalize data
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X_test_new = my_imputer.fit_transform(df_test)

df_norm = (df_upsampled - df_upsampled.mean()) / df_upsampled.std()


#X_test_new = (X_test_new - df_test.mean())/df_test.std()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_test_new)
# Now apply the transformations to the data:
#X_train = scaler.transform(X_train)
X_test_new = scaler.transform(X_test_new)

from sklearn import decomposition
pca = decomposition.PCA(n_components=4)
#pca.fit(df_norm)

X = pca.fit_transform(df_norm)

#df_test_norm = df_test_norm.dropna()

X_test_1 = pca.fit_transform(X_test_new)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.10)

print("hii")
from xgboost import XGBClassifier
print("hii7")
n_estimators = 200
clf = XGBClassifier(n_estimators=n_estimators,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=.1, 
                        subsample=.8, 
                        colsample_bytree=.8,
                        gamma=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        nthread=2)




print("hii3")
# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)
print("hii5")

#predictions = clf.predict(X_test)
predictions_prob = clf.predict_proba(X_test)

predictions_prob_test = clf.predict_proba(X_test_1)

df2 = pd.DataFrame({'id' : ids})
df2['target'] = predictions_prob_test[:,1]
df2.to_csv("submission.csv",index= False)
print(df2)
#print(predictions_prob)
#print(predictions_prob[:,1])
'''
new_pred = []
counter = 0

for pred in predictions_prob:
    new_pred[counter] = pred[1]
    counter += 1
    
print (new_pred)

'''
print("Full OOF score : %.6f" % gini_normalized(y_test,predictions_prob[:,1]))

#print(accuracy_score(y_test, predictions))



#print(accuracy_score(y_test, predictions))

#print(pd.DataFrame(pca.components_,columns=df_norm.columns,index = ['PC-1','PC-2']))
#list(X)
# Any results you write to the current directory are saved as output.
