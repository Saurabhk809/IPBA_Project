import os
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import seaborn as sbrn
import numpy as np
import warnings
import matplotlib.pyplot as plt

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
print(os.getcwd())
filename='train_Data_Prep.csv'
Data=pd.read_csv(filename)
print(Data.head())

# Data Cleansing
Data.fillna(0,inplace=True)

# Flag Numeric and Non Numeric Columns
print(Data.info())
print(Data.dtypes)

# Unqiue values
print(Data.Street.unique())
print(Data.Street.value_counts())

#Precentage of records for both categories of Street
print(Data.Street.value_counts(normalize=True))

# Find out outlieres in a numeric column
print(Data.SalePrice.describe())

# Find the outlier using commands
Q1 = np.percentile(Data.SalePrice,25, method='midpoint')
Q3 = np.percentile(Data.SalePrice,75, method='midpoint')
IQR = Q3 - Q1
print('IQR',IQR)

Upperlimit=Q3+(1.5 *IQR)
Lowerlimit=Q1-(1.5*IQR)

print('UpperLimit',Upperlimit,'LowerLimit',Lowerlimit)

# How to find count of outlieres
Outliers=Data[Data.SalePrice > Upperlimit]['SalePrice']
LowLiers=Data[Data.SalePrice < Lowerlimit]['SalePrice']
print('Outliers',Outliers,'LowLiers',LowLiers)

# Find the outliers using Distplot and box plot
sbrn.displot(Data.SalePrice)
plt.show()
sbrn.boxplot(Data.SalePrice)
plt.show()

dummies=pd.get_dummies(Data)
print(dummies.head())

#Correlation based on dtypes
Integer={}
for cols in Data.columns:
    if Data[cols].dtypes == 'int64' or Data[cols].dtypes == 'float64':
        Integer[cols]=Data[cols]
    elif Data[cols].dtypes == 'object':
        print('Non Integer data')
    else:
        print(cols,'pass')

df=DataFrame(Integer)
print('my df is \n',df)
Data_Correlation=df.corr()
print(Data_Correlation)

#Plot a heat map
plt.Figure(figsize=(10,10))
sbrn.heatmap(data=Data_Correlation,annot=True,center=0)
plt.title('Correlation')
plt.show()

# For linear regression - IDVs should not be correlated
# Write a program to list all pairs of variables where corr is greater than n or less than -n
"""
nmax=95
nmix=-5
print(Data.columns,type(Data.columns))

for cols in list(Data.columns):
    print('Cols',cols,Data.cols.dtype())
"""


