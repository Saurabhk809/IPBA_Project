# Class Notes 12-Aug-2023

import os
import pandas as pd
from pandas import DataFrame

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='creditDefaultData.csv'
Data=pd.read_csv(filename,sep=',',header=0)
Data1=Data
#Data1.index.name='ID'
print(Data.head())
print(Data1.head())
print(Data.info())
print(Data.shape,Data.shape[0],Data.shape[1])
print(type(Data.columns),'\n',Data.columns)
print(Data.iloc[0:10,]) # First ten rows
print(Data.iloc[:,0:4]) # First 5 columns
print('4th record onwards',Data.iloc[4:])# From 4 rows onwards
print(Data.iloc[-4:]) # last 4 rows
print(Data.iloc[:,[0,1,5,7]])  # Access by Column Names
print(Data.loc[:,["EDUCATION","PAY_3","BILL_AMT6"]])
#Subset the Data
print('Age <50',Data[Data['AGE']<50]['AGE'])
print('Age <50 & Age >70',Data[(Data.AGE<50) & (Data.AGE>25)])
mydata=Data[(Data.AGE<50) & (Data.AGE>25)]
print('mydata \n ',mydata,'type DF',type(mydata))
print('Age <50 & Age !=70',Data[(Data.AGE<50) & (Data.AGE==25)])
print('Age <50 & Age !=70',Data[(Data.AGE<50) & (Data.AGE!=70)])
print('Age <50 | Age !=70',Data[(Data.AGE<50)| (Data.AGE!=70)])
credit_data=Data[Data.BILL_AMT6 > 20000]
print('credit data',credit_data,type(credit_data))
print(credit_data.iloc[0:2,],type(credit_data))
credit_data=credit_data.reset_index()
print(credit_data,type(credit_data))
print(credit_data.iloc[0:2,],type(credit_data))

# How to combine DataFrame
# Approach1 Row Binding #Vertical Stacking # Both DF should have same number of columns
# Apprach 2 Column Binding # Horizontal Stacking # Both DF should have same numbers of rows
# Approach 3 Use Join based on columns
# Inner Join :- Intersection Make use of Common Keys , takes only common keys & igore non matching keys
     #pd.merge(df1,df2,on ="EmpID")
#left Join :- Take All record of left table and add matching recording from right table
     #pd.merge(df1,df2,on ="EmpID",how='left')
#right Join :- Take All record of right table and add matching recording from left table
     #pd.merge(df1,df2,on ="EmpID",how='right')
# Full outer Join :- Unioun

# Approach 3
d1={'Dep':['Pankaj','Megha','Lisa'],'ID':[1,2,3],'country':['India','India','USA'],'Role':['CEO','CTO','CTO']}
df1=pd.DataFrame(d1)

d2=pd.DataFrame({'ID':[1,2,31],'Name':['Pankaj','Anupam','Sumit']})
df2=pd.DataFrame(d2)

df3=(df1.merge(df2,on='ID'))
print('Inner join\n',df3)

df4=(df1.merge(df2,on='ID',how='left'))
print('left join\n',df4)

df5=(df1.merge(df2,on='ID',how='right'))
print('right\n',df5)

df6=(df1.merge(df2,on='ID',how='outer'))
print('outer\n ',df6)

#Apprach1 , row and column binding
d11={'Name':['Pankaj','Megha','Lisa'],'ID':[1,2,3],'country':['India','India','USA'],'Role':['CEO','CTO','CTO']}
df11=pd.DataFrame(d11)

d12=pd.DataFrame({'Name':['Pankaj1','Megha1','Lisa1'],'ID':[10,20,30],'country':['India1','India1','USA1'],'Role':['CEO1','CTO1','CTO1']})
df12=pd.DataFrame(d12)

# axis = 0 means operating happends on row i.e row binding i.e vertical stacking
df6=pd.concat([df11,df12],axis=0)
df6=df6.reset_index(drop=True)
print('Row binding \n ',df6)

# axis = 1 means operating happends on cols i.e cols binding i.e horizontal stacking
df7=pd.concat([df11,df12],axis=1)
df7=df7.reset_index(drop=True)
print('Col binding \n ',df7)
