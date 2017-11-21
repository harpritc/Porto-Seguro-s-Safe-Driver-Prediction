"""
Created on Mon Nov 20 16:31:24 2017

@author: kavitha rajendran
"""
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.utils import resample
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"train.csv")

df_majority=df[df.target==0]
df_minority=df[df.target==1]
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=573518,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# Display new class counts
#print(df_upsampled.target.value_counts())

featureVector = df_upsampled[["ps_ind_01","ps_ind_02_cat","ps_ind_03","ps_ind_04_cat","ps_ind_05_cat","ps_ind_06_bin","ps_ind_07_bin","ps_ind_08_bin","ps_ind_09_bin","ps_ind_10_bin","ps_ind_11_bin","ps_ind_12_bin","ps_ind_13_bin","ps_ind_14","ps_ind_15","ps_ind_16_bin","ps_ind_17_bin","ps_ind_18_bin","ps_reg_01","ps_reg_02","ps_reg_03","ps_car_01_cat","ps_car_02_cat","ps_car_03_cat","ps_car_04_cat","ps_car_05_cat","ps_car_06_cat","ps_car_07_cat","ps_car_08_cat","ps_car_09_cat","ps_car_10_cat","ps_car_11_cat","ps_car_11","ps_car_12","ps_car_13","ps_car_14","ps_car_15","ps_calc_01","ps_calc_02","ps_calc_03","ps_calc_04","ps_calc_05","ps_calc_06","ps_calc_07","ps_calc_08","ps_calc_09","ps_calc_10","ps_calc_11","ps_calc_12","ps_calc_13","ps_calc_14","ps_calc_15_bin","ps_calc_16_bin","ps_calc_17_bin","ps_calc_18_bin","ps_calc_19_bin","ps_calc_20_bin"]]
classVector = df_upsampled[["target"]]
X_train, X_test, y_train, y_test = train_test_split(featureVector, classVector, test_size=0.30)
#print(featureVector)
#print(classVector)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print("\nSVM:")
svc = svm.SVC(C=2.0, kernel='poly', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=1000, decision_function_shape='ovo', random_state=100)
svc.fit(X_train,y_train)
predictions = svc.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print("accuracy:",accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))
