import os
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
pd.options.display.width = 0

# change the directory
os.chdir('C:/ProgramData/Anaconda3/Scripts/IPBA_Project')

# read the input data file
filename='mktmix.csv'
filepath='C:/ProgramData/Anaconda3/Scripts/IPBA_Project'
try:
    Data=pd.read_csv(filename)
except:
    FileNotFoundError
    print('File',filename,'is not present at',filepath)

# Check the data and the data types
print ('Data Head \n ',Data.head(5))
print('Data Shape \n ',Data.shape)
print('Data Types \n',Data.dtypes)
print('Data Description \n',Data.describe())
Data.boxplot(column='Base_Price')
plt.show

# Change the price in Base_Price to Avg for price less than .10
q=Data['Base_Price'].quantile(.01)
Avg=Data['Base_Price'].mean()
Data.loc[(Data['Base_Price']<q),'Base_Price']=Avg


# Check the correlation between the data
#print('Data Correlation \n', Data.corr())


# check the describe for individual columns
#for cols in Data.columns:
    #print('col=', cols, 'col describe \n',Data[cols].describe())

# Run the Model
model=smf.ols("NewVolSales~Base_Price+InStore+NewspaperInserts+Discount+TV+Stout+Website_Campaign",data=Data)
results=model.fit()
print(results.summary())


