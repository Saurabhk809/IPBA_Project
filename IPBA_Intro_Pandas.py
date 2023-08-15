import os
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')

# Panda Series are columns of a dataframe in pandas it is a 1 dimensional numpy array
# Extracting a single columns of a dataframe return a series , each series has its own methods and attributes
# Data Frames are data structures that are used widely in data science
# Numpy Arrays - is standard for storing numerical data

#Create a series object
S=Series([1,2,3,4,5])
print(S[0],S.values,S.index,list(S),type(S))

# Add values to series
s=Series([8,9,10],index=[1,2,3])
print(s)
s=s.drop(1)
print(s)

#Series can also be created from Dictionary
d={'a':1,'b':2,'c':3}
print(d.keys())

S1=Series(d)
print(S1[0],S1.values,S1.index,list(S1),type(S1))
print('\n',S1+1,'\n',S1*2,'\n',S1**2)

d={'a':10,'b':20,'c':30,'d':40}
S2=Series(d)
print(S2[0],S2.values,S2.index,list(S2),type(S2))
print('\n',S1+1,'\n',S1*2,'\n',S1**2)

#log of series can be
S3=np.log(S1)
print('logseries',S2[0],S2.values,S2.index,list(S2),type(S2))
S4=(S1+S2)
S4=S4.astype('float64')
print('additive series','\n',S4,type(S4))

# Pandas Data Frame are a tabular structure that is used to work with data
# Dataframes can be rolled by hand using dictionaries , Keys of dictionary can be column labels , and values column data

# Data Frame using Dic Key with list as values
mydic={'Price':[10,20,30,40,50],'Sales':[1,2,3,4,5]}
myframe=DataFrame(mydic)
print(myframe,type(myframe))

# Data Frame from Dic of  Dic
mydic={'Price':{'r1':10,'r2':20},'Sales':{'r1':600,'r2':700}}
myframe=DataFrame(mydic)
print(myframe,type(myframe))

# Data Frame from Series
mydic={'price':Series([10,20,30,40,50]),'Sales':Series([1,2,3,4,5])}
myframe=DataFrame(mydic)
print(myframe,type(myframe))

# Data Frame from list of Dictionary
mydic=[{'price':19,'Sales':600,'qty':300},{'price':22,'Sales':700,'qty':900}]
myframe=DataFrame(mydic)
print(myframe,type(myframe))

# Extracting Data from Data Series
print(myframe['price']) #Single column
print(myframe[['price','qty']]) # two column

# rename columns of DataFrame, Always pass inplace=True for replace
myframe.rename(columns={'price':'Base_price','Sales':'Unit_Sales'})
print(myframe)
myframe.rename(columns={'price':'Base_price','Sales':'Unit_Sales'},inplace=True)
print(myframe)

# Pandas Method
print('print dataframe cols',myframe.columns)
print('print dataframe head',myframe.head())
print('print dataframe describe',myframe.describe())
print('print dataframe info',myframe.info())
print('print dataframe dtypes',myframe.dtypes)

# change datatype of each dataframe using series
myframe['Base_price']=Series(myframe['Base_price'].values.astype(float))
print(myframe.dtypes)

myframe['Unit_Sales']=myframe['Unit_Sales'].values.astype(float)
print(myframe.dtypes)

# Using Iloc and Loc using the rows and columns
mydf=pd.DataFrame({'price':[14,32,43],'Sales':[400,456,526]},index=list('abc'))
print('mydf is \n',mydf)
print('mydf.loc[a] : \n',mydf.loc['a'])# loc is label based
print('mydf.iloc[0] :\n',mydf.iloc[0])# iloc is Index based
print('mydf.loc[[a,c]] : \n',mydf.loc[['a','c']])# loc is label based
print('mydf.iloc[[0,2]] :\n',mydf.iloc[[0,2]])# iloc is index  based
print('mydf.loc price[a][price]:\n',mydf.loc['a']['price'])
print('mydf.iloc price[0][price]:\n',mydf.iloc[0]['price'])
print('mydf.loc price[a,b][price]:\n',mydf.loc[['a','b']]['price'])
print('mydf.iloc price[0,2][price]:\n',mydf.iloc[[0,2]]['price'])

# create a pd dataframe from multiple list using zip ->dic ->dframe
store=['Walmart','Safeway','Total','Trader_Joe']
sales=[100,3400,6727,5618]
visitor_per_hr=[139,132,87,73]
Location=['AL','UT','TX','AR']
# create a list of columns
List_labels=['store','sales','visitor_per_hr','Location']
List_cols=[store,sales,visitor_per_hr,Location]
zippedlist=list(zip(List_labels,List_cols))
print('Zipped list :\n',type(zippedlist),'\n',zippedlist)
mydict=dict(zippedlist)
print('Dic from zipped list',mydict)
mydframe=pd.DataFrame(mydict)
print('My df from zipped list and dic\n',mydframe)


