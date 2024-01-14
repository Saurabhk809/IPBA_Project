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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
warnings.filterwarnings("ignore")
from mlxtend.feature_selection import SequentialFeatureSelector
import statsmodels.api as smapi
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


#pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.float_format',lambda x:'%.4f' %x)

#change the directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filepath=os.getcwd()

#Load the Data
def dataloading(file1,file2):
    try:
        os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
        train_data=pd.read_csv(file1,na_values=[' ','NA','NULL'])
        test_data=pd.read_csv(file2,na_values=[' ','NA','NULL'])
        #set the display options
        pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
        return train_data,test_data
    except:
        FileNotFoundError
        print('File not present in filepath',filepath)

train_data,test_data=dataloading('HR_comma_sep.csv','HR_comma_sep.csv')
print(train_data.head())
print(train_data.info())
print(train_data.isnull().sum())
train_data.rename(columns={'sales':'dept'},inplace=True)
print(train_data.head())


y=train_data['left']
X=train_data.drop('left',axis=1)
print(X.head())

X=pd.get_dummies(X)
print(X.head())


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=400)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# Subsample = 1.0 (default): If subsample is set to 1.0 (the default value),
#     it means that each base learner is trained on the entire dataset.
#     This is equivalent to traditional gradient boosting,
#     where each tree in the ensemble is fit to the full dataset.
#     While this can lead to strong predictive performance
#     on the training data, it may also make the model prone to overfitting.

# Subsample < 1.0: When subsample is set to a value less than 1.0 (e.g., 0.8),
#     it means that each base learner is trained on a random subset of the data.
#     For example, if subsample is set to 0.8, each tree is trained on 80%
#     of the data, randomly selected without replacement.
#     This introduces randomness into the training process
#     and can help prevent overfitting, especially when the dataset is large.

clf1=AdaBoostClassifier(n_estimators=80,random_state=400)
clf=GradientBoostingClassifier(n_estimators=80,random_state=400)
clf0 = xgb.XGBClassifier(n_estimators=80,random_state=400)

clf1.fit(X_train,y_train)
clf.fit(X_train,y_train)
clf0.fit(X_train,y_train)

#print('Model score',clf1.score(X_test,y_test))
#print('Model score',clf.score(X_test,y_test))
#print('Model score',clf0.score(X_test,y_test))

from sklearn.model_selection import GridSearchCV
#mod1=GridSearchCV(clf1,param_grid={'n_estimators':[60,80,100,120,140,160]})
#mod1.fit(X_train,y_train)

#mod=GridSearchCV(clf,param_grid={'n_estimators':[60,80,100,120,140,160]})
#mod.fit(X_train,y_train)

#mod0=GridSearchCV(clf,param_grid={'n_estimators':[60,80,100,120,140,160]})
#mod.fit(X_train,y_train)

#mod0=GridSearchCV(clf,param_grid={'n_estimators':[60,80,100,120,140,160,10000],'learning_rate': [0.01, 0.1, 0.2, 0.3,0.4,0.5],})
#mod0.fit(X_train,y_train)
#print(mod0.best_estimator_)


clf1=xgb.XGBClassifier(learning_rate=0.1,n_estimators=160,random_state=400)
clf1.fit(X_train,y_train)

clf2=xgb.XGBClassifier(learning_rate=0.3,n_estimators=160,random_state=400)
clf2.fit(X_train,y_train)

clf3=xgb.XGBClassifier(learning_rate=0.4,n_estimators=160,random_state=400)
clf3.fit(X_train,y_train)

clf4=xgb.XGBClassifier(learning_rate=0.5,n_estimators=160,random_state=400)
clf4.fit(X_train,y_train)

print('Model score',clf1.score(X_test,y_test))
print('Model score',clf2.score(X_test,y_test))
print('Model score',clf3.score(X_test,y_test))
print('Model score',clf4.score(X_test,y_test))

#print(clf0.feature_importances_)

feature_imp1=pd.Series(clf0.feature_importances_,index=X.columns)
#print(feature_imp1.sort_values(ascending=False))

feature_imp1.sort_values(ascending=False).plot(kind='bar')
plt.show()

#y_pred=clf3.predict(X_test).map(lambda x:1 if x>0.5 else 0)
y_pred=clf3.predict(X_test)
#print (list(y_pred))
Metric=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Metric is \n',Metric)

# ROC Curve to check the tpr sensitivity 1 and fpr 1-specificity 0
y_score=clf0.predict(X_test)
#FPR is 1-specificity , TPR is sensitivity both should be maximised
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_score)
x,y=np.arange(0,1.1,0.1),np.arange(0,1.1,0.1)

plt.plot(fpr,tpr,"-")
plt.plot(x,y,'b--')
plt.show()

# AUC, Average accuracy of my model , compares  the confusion metric way of comparing models
print('Area under curve is',metrics.roc_auc_score(y_test,y_score))