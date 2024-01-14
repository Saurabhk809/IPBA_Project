import os
import matplotlib.pyplot as plt
import seaborn as sbrn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import sklearn.model_selection as model_selection
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smfi
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import  variance_inflation_factor
import statsmodels.api as sm
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import datetime as dt
import statistics
import sweetviz as sv
from sklearn.ensemble import *
os.chdir('C:/ProgramData/Anaconda3/Scripts/IPBA_Project')
print(os.getcwd())

# Ensemble Models
# collection of Model - for classification task you are building Logistic Regression,Decision Tree, SVM,KNN & final prediction is decided by the voting approach
# collection of Model - for Regression task you are building Linear Regression,Decision Tree, SVM,KNN & final prediction is decided by the average of all predicted values

# Three Types of Ensemble using base learner
# 1  Boosting Models
# 2 Random Forest Models
# 3 Bagging Models: Different Trees Learning is happening on the different type of records

# Difference is based on type of Sampling Approach that is used to create the model

# 1. Bootstrap sampling -> This is used in Bagging model and also in Random Forest Model : Sampling With Replacement
# 2. Hyper Parameters for Bagging Models
#               -> Number of trees you want (equivalent to bootstrap Sample
#               ->Depth of tree
#               -> Number of observation on each node before it split

# 3. Hyper Parameters for Random Forest Models
#               -> Number of trees you want (equivalent to bootstrap Sample
#               ->Depth of tree
#               -> Number of observation on each node before it split
#               -> Number of features for each in-sample and out-sample

# Advantages of Bagging and Random Forest as compared to decision Tree
#               -> More Accurate
#               -> Less Biased
# Disadvantage of Bagging and Random Forest
#               -> Resource extensive (CPU Utilisation and Time to Predict
#               -> Explainability of the prediction or results (Use lterature behing LIME and SHAP)

# Objective is to build a bagging classifier using this dataset
#set the display options
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)


hr_data=pd.read_csv('hr.csv')
print(hr_data.head())
print(hr_data.info())
print(hr_data.left.nunique())
print(hr_data.isnull().sum())
print(hr_data.salary.value_counts())
print(hr_data.sales.value_counts())
hr_data.rename(columns={'sales': 'department'}, inplace=True)
print(hr_data['salary'].nunique(),print(hr_data['left'].nunique()))

# check frequency distribution of 0 & 1 in target
print(hr_data.left.value_counts())
print(hr_data.department.value_counts())

print(hr_data.shape)

X=hr_data.drop('left',axis=1)
y=hr_data['left']
print(type(y))

print(X.head())
print(y.head())
X=pd.get_dummies(X)
print(X.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)

print(X.department_accounting.value_counts()/X.shape[0])
print(X_train.department_accounting.value_counts()/X_train.shape[0])
print(X_test.department_accounting.value_counts()/X_test.shape[0])

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier # Specifying which Base learner should be used

#Initialise ensemble model

clf = BaggingClassifier(oob_score=True, n_estimators=5,random_state=300, base_estimator=DecisionTreeClassifier())
clf.fit(X_train, y_train)
oob = clf.oob_score_
print('\n clf_oob_score5_Train',clf.oob_score_)
print("Out of bage score is5", clf.oob_score_,str(oob))
print('clf_oob_score5_test',clf.score(X_test, y_test)) # Accuracy of the test data-set

clf = BaggingClassifier(oob_score=True, n_estimators=10,random_state=300, base_estimator=DecisionTreeClassifier())
clf.fit(X_train, y_train)
oob = clf.oob_score_
print('\n clf_oob_score10_Train',clf.oob_score_)
print("Out of bage score is10", clf.oob_score_,str(oob))
print('clf_oob_score10_test',clf.score(X_test, y_test)) # Accuracy of the test data-set

clf = BaggingClassifier(oob_score=True, n_estimators=15,random_state=300,base_estimator=DecisionTreeClassifier())
clf.fit(X_train, y_train)
oob = clf.oob_score_
print('\n clf_oob_score15_Train',clf.oob_score_)
print("Out of bage score is15", clf.oob_score_,str(oob))
print('clf_oob_score15_test',clf.score(X_test, y_test)) # Accuracy of the test data-set

clf = BaggingClassifier(oob_score=True, n_estimators=20,random_state=300,base_estimator=DecisionTreeClassifier())
clf.fit(X_train, y_train)
oob = clf.oob_score_
print('\n clf_oob_score20_Train',clf.oob_score_)
print("Out of bage score is20", clf.oob_score_,str(oob))
print('clf_oob_score20_test',clf.score(X_test, y_test)) # Accuracy of the test data-set


clf = BaggingClassifier(oob_score=True, n_estimators=25,random_state=300,base_estimator=DecisionTreeClassifier())
clf.fit(X_train, y_train)
oob = clf.oob_score_
print('\n clf_oob_score25_Train',clf.oob_score_)
print("Out of bage score is25", clf.oob_score_,str(oob))
print('clf_oob_score25_test',clf.score(X_test, y_test)) # Accuracy of the test data-set


#oob = clf.oob_score_   # Accuracy on the out of bag sample
#print('clr base estimator',clf.base_estimator_)

# print(" the number of estimators = ",)
#print("Out of bage score is", str(oob))


"""
for i in range(10,300,20):  # Try 10, 30,50, 70, 90.... 300
    clf = BaggingClassifier(oob_score=True, n_estimators=i, random_state=300, base_estimator=DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    oob = clf.oob_score_
    print(" the number of estimators = ", str(i))
    print("Out of bage score is", str(oob))
    print("----------------------------------")
    print('\n clf_oob_score_Train', clf.oob_score_)
    print("Out of bage score is25", clf.oob_score_, str(oob))
    print('clf_oob_score_test', clf.score(X_test, y_test))  # Accuracy of the test data-set

"""

clf = BaggingClassifier(oob_score=True, n_estimators=70, random_state=300, base_estimator=DecisionTreeClassifier())
B1=clf.fit(X_train,y_train)
B1.predict(X_test)

print('\n clf_oob_scoreB1',clf.oob_score_)
print("Out of bage score isB1", clf.oob_score_,str(oob))
print('clf_oob_scoreB1',clf.score(X_test, y_test)) # Accuracy of the test data-set

import  sklearn.metrics as metrics
print(metrics.confusion_matrix(y_test,B1.predict(X_test)))

# Display Feature Importance
print(clf.estimators_)    # display the number of trees
print(X.columns,clf.estimators_[0].feature_importances_) # Feature that were important for first tree
print(X.columns,clf.estimators_[1].feature_importances_) # Feature that were important for second tree

# Get feature importances
#feature_importances = clf.feature_importances_

# Calculate the average feature importance
#average_importance = np.mean(feature_importances)
df=pd.DataFrame()
for x in range(0,69):
    result_dic = {'Params': X_train.columns, 'Dec_Tree_Features': clf.estimators_[x].feature_importances_}
    Dec_Tree_Features = pd.DataFrame(result_dic)
    df=pd.concat([df,Dec_Tree_Features],axis=1)
    print(Dec_Tree_Features)

print(df.head())
# Calculate the average feature importance
average_importance = np.mean(df['satisfaction_level'])
print(average_importance)


