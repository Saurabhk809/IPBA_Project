import os
import pandas as pd
import seaborn as sbrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename1 = 'test.csv'
test_data=pd.read_csv(filename1,na_values=['NA','N/A','nan',' '])
filename2 = 'train.csv'
train_data=pd.read_csv(filename2,na_values=['NA','N/A','nan',' '])

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

# Info of Train data
print(train_data.head())
print(train_data.info())

# convert categorical variables to categories
train_data['cut']=train_data['cut'].astype('category').cat.codes
train_data['color']=train_data['color'].astype('category').cat.codes
train_data['clarity']=train_data['clarity'].astype('clarity').cat.codes
test_data['cut']=test_data['cut'].astype('category').cat.codes
test_data['color']=test_data['color'].astype('category').cat.codes
test_data['clarity']=test_data['clarity'].astype('clarity').cat.codes

# Descriptive statistic of train_data
print(train_data.describe())
print(train_data.info())
