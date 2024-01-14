#  Script for  EDA Analysis & MLR to predict Daimond Price
# Author : Saurabh Kamble , IPBA Batch 16

import os
import pandas as pd
import seaborn as sbrn
import numpy as np
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
print('Data head :\n',Data.head())     # View the Data head
#print(Data.info())       # Check the Data format
Data.drop(['Unnamed: 0'],axis=1,inplace=True) # Drop Unwanted column
#print('Sum of isna :\n',Data.isna().sum(),'\n Sum of isnull :\n',Data.isnull().sum()) # Sum of all na and null
Data['depth'].fillna(0,inplace=True)

# Perform the EDA analysis for Impact of Various factor on Diamond Price
sbrn.set_style('whitegrid')
sbrn.scatterplot(data=Data,x='carat',y='price',hue='cut',palette='deep',style='cut',size='cut')
plt.title('Daimond Price vs Carat & Cut')
plt.show()

sbrn.set_style('whitegrid')
sbrn.histplot(data=Data,x='price',bins=10,kde=True)
plt.title('Count based on Daimond Price')
plt.show()

sbrn.set_style('whitegrid')
sbrn.histplot(data=Data,x='price',hue='cut',multiple='stack',bins=10)
plt.title('Stacked Count of Daimond Price based on Cut')
plt.show()

sbrn.set_style('whitegrid')
sbrn.displot(data=Data,x='price',hue='clarity',col='cut',kind="kde")
plt.title('Daimond price density based on Clarity')
plt.show()


Q1=np.percentile(Data.price,25)
Q3=np.percentile(Data.price,75)
IQR=Q3-Q1
print('IQR : ',IQR)

Upperlimit=(Q3+(1.5 *IQR))
Lowerlimit=(Q1-(1.5*IQR))

print('UpperLimit :',Upperlimit,'LowerLimit:',Lowerlimit)

# How to find count of outlieres
Outliers=Data[Data.price > Upperlimit]['price']
LowLiers=Data[Data.price < Lowerlimit]['price']
print('Outliers:\n',list(Outliers),'LowLiers:',list(LowLiers))
print(Data.head())

sbrn.boxplot(data=Data)
plt.title('Box plot of Daimond prices')
plt.show()

# Visualise data by groupby
print('Group by cut :\n',Data.groupby(['cut']).agg(['count','min','max'])['price'])
#print('Group by carat :\n',Data.groupby(['carat']).agg(['count','min','max'])['price'])

# Convert Categorical Variables using Dummies based on dtypes for absolute reference
Data[Data['price']>Upperlimit]=Upperlimit
Data[Data['price']<Lowerlimit]=Lowerlimit
print(Data.head())

sbrn.boxplot(data=Data)
plt.title('Transformed Box plot of Daimond prices')
plt.show()

Cat_ColList=['cut','color','clarity']
for cols in Cat_ColList:
    dummies=pd.get_dummies(Data[cols],prefix=cols, drop_first=True)
    Data = pd.concat([Data,dummies], axis=1)
    Data=Data.drop(cols,axis=1)

print('Data with Dummy \n',Data.head())

# Check the correlation among various parameters and check heat map
print('Correlation of multiple feature to Price is : \n',Data.corr()['price'])
DataCorPrice=Data.corr()['price']
#print('Data correlation Table is \n',DataCorrelation)
sbrn.lineplot(data=DataCorPrice,y=DataCorPrice.index,x=DataCorPrice.values)
#sbrn.heatmap(data=DataCorPrice,annot=True,centre=0)
plt.title('Feature Correlation Line Plot with Price')
plt.show()

#print('Data correlation Table is \n',DataCorrelation)
df=pd.DataFrame({'Feature':DataCorPrice.index,'Val':DataCorPrice.values})
sbrn.barplot(data=df,y='Feature',x='Val')
#sbrn.heatmap(data=DataCorPrice,annot=True,centre=0)
plt.title('Feature Correlation Bar with Price')
plt.show()

#DataCorPrice=Data.corr()['price']
plt.Figure(figsize=(10,10))
DataCorAll=Data.corr()
sbrn.heatmap(data=DataCorAll,annot=True,center=0)
plt.title('Heat Map of all Daimond Characteristics')
plt.show()


# Split the Dataset into Train and Test
# Seperate dependent and non Dependent variables
Y=Data['price']
#print('Y is \n',Y)
X=Data.drop('price',axis=1)
#print('X is \n',X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.80)

# Create the Model and fit it with xtrain and ytrain
Model=LinearRegression()
Model.fit(X_train,y_train)

Model_coef_table = pd.DataFrame(list(X_train.columns)).copy()
Model_coef_table.insert(len(Model_coef_table.columns),"Coefs",Model.coef_.transpose())
print('Model Coeff : \n',Model_coef_table,'\n')
#print('Model Coeff : \n',Model.coef_,'\n')
print('Model Intercept : \n',Model.intercept_,'\n')
y_predict=Model.predict(X_test)
print('Prediction Using Linear regression is :\n ',y_predict)

# Check the Error
print('Mean Square Error :',mean_squared_error(y_test,y_predict))
print('Model Accuracy r2 :',r2_score(y_test,y_predict))

# Plot test vs predicted
residuals=y_test-y_predict
sbrn.lineplot(residuals)
plt.title('Residuals Graph ')
plt.show()

print('creating the submission file ....')
submission_df = pd.DataFrame({'y_test': y_test, 'y_predict': y_predict})
submission_df.to_csv("submission.csv", index=False)

print('submission file .... created !')