import os
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from urllib.request import  urlretrieve
import json
import urllib

#print('get current workind directory',os.getcwd())
mydir='C:\ProgramData\Anaconda3\Scripts\IPBA_Project'
if os.getcwd() !=mydir:
    os.chdir(mydir)
else:
    pass
print(os.getcwd())


# Import Flat file
Data_txt=pd.read_csv('real_estate_sales.txt',sep='\t',header=0)
print('Text data: \n',Data_txt.head(5))

# Import csv file
Data_csv=pd.read_csv('HistoricalQuotes.csv',sep=',',header=0)
print('CSV data: \n',type(Data_csv),Data_csv.head(5))

x=Data_csv['close']
y=Data_csv['volume']
print('x is \n',x.describe(),'\n',type(x),x)
print('y is \n',y)


# Import csv online from a url
url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# Read file into a DataFrame df
Data=pd.read_csv(url,sep=';')
print('Wine Details :\n',Data.head())

# Save file locally
urlretrieve(url,'winequality-red.csv')

# Read a file from Json Format from a url
#with urllib.request.urlopen("http://map.googleapis.com/maps/api/geocode/json?adress=google") as url:
with urllib.request.urlopen("https://run.googleapis.com/$discovery/rest?version=v2") as url:
    data=json.loads(url.read().decode())
    print('json data \n',data.keys())
    print('json values \n',data.values())

