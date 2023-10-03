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

pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.float_format',lambda x:'%.4f' %x)

#change the directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filepath=os.getcwd()

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
def model(mod,X_train, X_test, Y_train, Y_test,test_data):
    #for i in range(10,500,20):
    model=mod(n_estimators=90,oob_score=True,n_jobs=-1,random_state=100,base_estimator=DecisionTreeRegressor())
    model.fit(X_train,Y_train)
    print(":",model.oob_score_)
    predict = model.predict(X_test)
    actual = Y_test
    prediction = predict
    MSE = mean_squared_error(actual, prediction, squared=False)
    RMSE = sqrt(mean_squared_error(actual, prediction, squared=False))
    print('Model:', model)
    print('MSE for test', MSE, 'RMSE for test', RMSE)
    Y1_pred = model.predict(test_data)
    print('Regressor Score of Test Data :\n', model.score(test_data, Y1_pred))
    #impfeature=pd.Series(model.feature_importance_,index=X_train.columns.tolist())

def main():
    # Load the Data
    train_data, test_data = dataloading('train_hp.csv', 'test_hp.csv')
    tid = test_data['id']
    print(train_data.head())
    print(test_data.head())
    X = train_data[['squareMeters', 'numberOfRooms', 'floors', 'cityPartRange', 'made', 'numPrevOwners', 'isNewBuilt', 'basement','garage', 'hasGuestRoom']]
    Y= train_data['price']
    test_data=test_data[['squareMeters', 'numberOfRooms', 'floors', 'cityPartRange', 'made', 'numPrevOwners', 'isNewBuilt', 'basement','garage', 'hasGuestRoom']]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.10, random_state=100)
    model(RandomForestRegressor,X_train, X_test, Y_train, Y_test,test_data)

main()
