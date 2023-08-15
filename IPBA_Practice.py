import os
import pandas as pd
from pandas import DataFrame
from pandas import to_datetime

txt='TENET'
reversed=txt[::-1]
if reversed==txt:
    print('True')
else:
    print('False')

# change the datetime in Month and Year Format
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='credit.csv'
Data=pd.read_csv(filename)
print(Data.head())
print(Data.info())
print(Data.shape)
#print(Data.info)
Data['Date']=pd.to_datetime(Data['Date'],format='%d-%m-%Y')
print(Data['Date'].head())
#print(Data.info())
Data['Date']=Data['Date'].dt.strftime('%b-%y')
print(Data['Date'])
print(Data.head())

# flag a indicator
#Data['Ind']=Data['Amount'].str.contains('Cr')
#print(Data.head(10))
Data['Amount1']=Data['Amount'].str.replace(' Cr','')
Data['Amount1']=Data['Amount1'].str.replace(',','')
print(Data.head(10))
Data['Amount1']=Data['Amount1'].astype('float')
print(Data.head(10))
print(Data['Amount1'].dtypes)
