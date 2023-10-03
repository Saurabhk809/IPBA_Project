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

#pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.float_format',lambda x:'%.4f' %x)

#change the directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filepath=os.getcwd()

file1='train.csv'
file2='test.csv'
try:
    os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
    train_data = pd.read_csv(file1, na_values=[' ', 'NA', 'NULL'])
    test_data = pd.read_csv(file2, na_values=[' ', 'NA', 'NULL'])
    # set the display options
    pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows', 1000)
except:
    FileNotFoundError
    print('File not present in filepath', filepath)

# Build the model variables
Y = train_data['price']
#X = train_data.drop('price', axis=1)
X=train_data[['id','squareMeters','made','floors']]

def IQR_UL_LL(train_data):
    pdf_IQR, pdf_Result = pd.DataFrame(), pd.DataFrame()
    for columns in train_data:
        if train_data[columns].isnull().sum()==0:
            pass
        else:
            train_data[columns].fillna(0)
        if columns not in ['id']:
            #pdf_IQR['Param']=columns
            pdf_IQR = (train_data[columns].describe())
            pdf_IQR['median'] = statistics.median(train_data[columns])
            pdf_IQR['mode']=statistics.mode(train_data[columns])
            pdf_IQR['Q1']=np.quantile(train_data[columns],.25)
            pdf_IQR['Q3'] = np.quantile(train_data[columns],.75)
            pdf_IQR['95%'] = np.quantile(train_data[columns],.25)
            pdf_IQR['IQR']=pdf_IQR['Q3']-pdf_IQR['Q1']
            UL=pdf_IQR['Q3']+(1.5)*pdf_IQR['IQR']
            print (columns,UL)
            LL=pdf_IQR['Q1']-(1.5)*pdf_IQR['IQR']
            print(columns,LL)
            train_data[columns]=np.where(train_data[columns]>UL,UL,train_data[columns])
            print(train_data[columns].max())
            train_data[columns] = np.where(train_data[columns]<LL,LL, train_data[columns])
            pdf_IQR['UL_count']=train_data[train_data[columns] > UL][columns].count()
            pdf_IQR['LL_count'] = train_data[train_data[columns] < LL][columns].count()
            pdf_IQR['UL_Per'] = pdf_IQR['UL_count']/train_data[columns].count() * 100
            pdf_IQR['LL_Per'] = pdf_IQR['LL_count']/train_data[columns].count() * 100
            #print(pdf_IQR.head(20))
            pdf_Result = pd.concat([pdf_Result,pdf_IQR], axis=1)
        else:
            pass
    print('Train_Data_Descriptive Stats :\n',pdf_Result.head(20))
    pdf_Result.to_html('Descriptve_Stats_IQR_Report.html')
    return  pdf_Result


# Build the train & test data set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.head(), Y.head())

IQR_UL_LL(X_train)
IQR_UL_LL(X_test)

from sklearn.ensemble import GradientBoostingRegressor
"""
clf=GradientBoostingRegressor(n_estimators=80,random_state=400)
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'n_estimators':[60,80,100,120,140,160]})
mod.fit(X_train,Y_train)
#print(mod.best_estimator_)

clf=GradientBoostingRegressor(n_estimators=80,random_state=400)
clf.fit(X_train,Y_train)
Y_pred_test=clf.predict(X_test)
print(clf.score(X_test,Y_test))
actual = Y_test
MSE = mean_squared_error(actual, Y_pred_test, squared=True)
RMSE = sqrt(mean_squared_error(actual, Y_pred_test, squared=True))
print('MSE',MSE)
print('RMSE',RMSE)

"""

# check with Random forest
from sklearn.ensemble import RandomForestRegressor
for x in range(1,2,1):
    print(x)
    model = RandomForestRegressor(n_estimators=70,max_depth=6)
    model.fit(X_train,Y_train)
    R2=model.score(X_test,Y_test)
    Y_pred_test = model.predict(X_test)
    #print(Y_pred_test)
    #print('R2 Score of Train Data :\n', regressor.score(X_train,Y_pred_train))
    actual = Y_test
    train= Y_train
    MSE=mean_squared_error(actual, Y_pred_test, squared=True)
    RMSE = sqrt(mean_squared_error(actual, Y_pred_test, squared=True))
    print('MSE :',MSE)
    print('RMSE :',RMSE)
    Y_test=np.array([Y_test])
    Y_pred_test=np.array([Y_pred_test])
    Y_pred_test=(Y_pred_test).reshape(-1, 1)
    Y_test=(Y_test).reshape(-1, 1)
    #print(Y_pred_test.shape)
    #print('R2 Test Data :\n',model.score(X_test,Y_test))
    n = X_test.shape[0]
    p = X_test.shape[1]
    AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
    print('MSE R :',MSE,'RMSE R:',RMSE)
    print('R R2 :',R2,'R R2',AdjR2)
    print ('---*30')
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor # Specifying which Base learner should be used

    clf = BaggingRegressor(oob_score=True,n_estimators=70,random_state=400,base_estimator=DecisionTreeRegressor())
    B1 = clf.fit(X_train, Y_train)
    print('\n clf_oob_score R2', clf.oob_score_)
    Y_pred_test = B1.predict(X_test)
    model.fit(X_train, Y_train)
    R2 = model.score(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    #print(Y_pred_test)
    # print('R2 Score of Train Data :\n', regressor.score(X_train,Y_pred_train))
    actual = Y_test
    train = Y_train
    MSE = mean_squared_error(actual, Y_pred_test, squared=True)
    RMSE = sqrt(mean_squared_error(actual, Y_pred_test, squared=True))
    Y_test = np.array([Y_test])
    Y_pred_test = np.array([Y_pred_test])
    Y_pred_test = (Y_pred_test).reshape(-1, 1)
    Y_test = (Y_test).reshape(-1, 1)
    #print(Y_pred_test.shape)
    R2= model.score(X_test, Y_test)

    n = X_test.shape[0]
    p = X_test.shape[1]
    AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
    #print('AdjR2Test Data :\n', AdjR2)
    print('MSE BR :',MSE,'RMSE BR :',RMSE)
    print('BRR2 :',R2,'AdjR2',AdjR2)
    print('---*30')

"""
# create submission file
print('creating the submission file ....')
print(test_data.head())
Y1_pred = regressor.predict(test_data)
submission_df = pd.DataFrame({'Id': test_data['id'], 'price': Y1_pred})
print(submission_df.head())
submission_df.to_csv('submission.csv', index=False)
print('Regressor Score of Test Data :\n', regressor.score(test_data, Y1_pred))
"""
