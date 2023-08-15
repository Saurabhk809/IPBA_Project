import os
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='creditDefaultData.csv'
Data=pd.read_csv(filename,sep=',',header=0)
Data1=Data.set_index('ID')
#Data1.index.name='ID'
print(Data.head())
#print(Data1.head())
#print(Data.info())

#Approach 1 , Row Indices

d0=Data[['ID']]
df0=pd.DataFrame(d0)
d1=Data[['ID','LIMIT_BAL']]
df1=pd.DataFrame(d1)
df11=df1.iloc[5:10,]
df22=df1.iloc[15:20,]
print('df11',df11.head())
d2=Data[['ID','AGE']]
df2=pd.DataFrame(d2)
d3=Data[['ID','SEX']]
df3=pd.DataFrame(d3)
d4=Data[['ID','BILL_AMT1']]
df4=pd.DataFrame(d4)
d5=Data[['ID','BILL_AMT2']]
df5=pd.DataFrame(d5)
d6=Data[['ID','BILL_AMT3','BILL_AMT4','BILL_AMT5']]
df6=pd.DataFrame(d6)

# One way to do column addition using axis=1 is using concat withouth changing Index
#print('Limit Bal \n',df1.head())
#print('Age \n',df2.head())

df12=pd.concat([df1,df2],axis=1)
# remove duplicate columns using df.T.drop_duplicates().T
df12=df12.T.drop_duplicates().T
print('column Binding with no set index \n',df12.head())
df1234=pd.concat([df1,df2,df3,df4],axis=1)
df1234=df1234.T.drop_duplicates().T
print('column Binding with no set index \n',df1234.head())
df123456=pd.concat([df1,df2,df3,df4,df5,df6],axis=1)
df123456=df123456.T.drop_duplicates().T
print('column Binding with no set index \n',df123456.head())

# One way to get all data in one column is using set_index
df1=df1.set_index('ID')

#print('Limit Bal \n',df1.head())

df2=df2.set_index('ID')
df3=df3.set_index('ID')
df4=df4.set_index('ID')

#print('Age \n',df2.head())
df12=pd.concat([df1,df2],axis=1)
print('column Binding with set index \n',df12.head())
df1234=pd.concat([df1,df2,df3,df4],axis=1)
print('column Binding with set index \n',df1234.head())
df123456=pd.concat([df1,df2,df3,df4,df5,df6],axis=1)
print('column Binding with set index \n',df123456.head())

# One way to do row addition using axis=0 is using concat withouth changing Index
#print(df11.head())
#print(df22.head())
df33=pd.concat([df11,df22],axis=0)
print(df33)
df333=df33.reset_index(drop=True)
print(df333)

#Another way to join the pandas rows using merge
print(df1.head())
print(df2.head())
df44=(df1.merge(df2,on='ID',how='left'))
print('left \n',df44.head(10))
df45=(df1.merge(df2,on='ID',how='right'))
print('right \n',df44.head(10))
df46=(df1.merge(df2,on='ID',how='outer'))
print('outer \n',df44.head(10))
