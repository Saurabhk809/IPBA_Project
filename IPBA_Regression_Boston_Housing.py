# Script to find the price dependency of Car based on horse, Cyl,Disp,fuel and combination

# Import section for Imports
import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
try:
    filepath='C:/ProgramData/Anaconda3/Scripts/IPBA_Project/'
    filename='BostonHousing.csv'
except:
    FileNotFoundError
    print('File',filename,'is not present at',filepath)

bsData=pd.read_csv(filepath+filename)
Hd=bsData.head()
#print(Hd)

Y=bsData.medv
X=bsData.drop(['medv'],axis=1)

# Create a Train and Test Data Set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)

# Check the train data dimension
print('Dim of train data',X_train.shape,Y_train.shape)

# check the test data dimension
X_test.shape,Y_test.shape
print('Dim of test data',X_test.shape,Y_test.shape)

model = linear_model.LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))
























