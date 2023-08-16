#  Script for  OLS MLR to predict Daimond Price
# Author : Saurabh Kamble , IPBA Batch 16

import os
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# set the default working directory and read the file contents
try:
    os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
    filename = 'cubic_zirconia.csv'
    Data=pd.read_csv(filename)
except:
    FileNotFoundError
    print('File',filename,'not present in',os.getcwd())


# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

# Check the Data and perform Data Cleansing

print(Data.head())     # View the Data head
print(Data.info())       # Check the Data format
Data.drop(['Unnamed: 0'],axis=1,inplace=True) # Drop Unwanted column
print(Data.head())
print('Sum of isna :\n',Data.isna().sum(),'\n Sum of isnull :\n',Data.isnull().sum()) # Sum of all na and null
Data['depth'].fillna(0,inplace=True)
print(Data.head())
print('Sum of isna :\n',Data.isna().sum(),'\n Sum of isnull :\n',Data.isnull().sum()) # Sum of all na and null


# Convert Categorical Variables to Integer Values
Data['cut'].replace({'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5},inplace=True)
print(Data.head())
Data['color'].replace({'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7},inplace=True) # D is best and J is worst
print(Data.head())
Data['clarity'].replace({'I3':1,'I2':2,'I1':3,'SI2':4,'SI1':5,'VS2':6,'VS1':7,'VVS2':8,'VVS1':9,'IF':10,'FL':11},inplace=True)
 # Best to worst FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
print(Data.head())
print(Data.info())


# Check the seaborn pair plot to confirm relation of each parameter
#sbrn.set_style('whitegrid')
#sbrn.pairplot(Data)
#plt.title('Pair Plot of Price vs Various Params')
#plt.show()

# Check the correlation among various parameters and check heat map
DataCorrelation=Data.corr()
print('Data correlation Table is \n',DataCorrelation)
plt.Figure(figsize=(10,10))
sbrn.heatmap(data=DataCorrelation,annot=True,center=0)
plt.title('Heat Map of all Daimond Characteristics')
plt.show()

# Split the Dataset into Train and Test
# Seperate dependent and non Dependent variables
Y=Data['price']
print('Y is \n',Y)
X=Data.drop('price',axis=1)
print('X is \n',X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.80)

# Create the Model and fit it with xtrain and ytrain
Model=LinearRegression()
Model.fit(X_train,y_train)

y_predict=Model.predict(X_test)
print('Prediction is :\n ',y_predict)


# Check the Error
print('Mean Square Error :\n',mean_squared_error(y_test,y_predict))
print('Model Accuracy r2 :\n',r2_score(y_test,y_predict))