import pandas as pd
import xlrd
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

import os
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='US Superstore data.xls'


Data=pd.read_excel(filename)
print(Data.head(2),'\n',type(Data.info()))

print(Data.head(2),'\n',type(Data))
df2=Data.groupby('Sales').agg('count')['Order ID']/(Data.shape[0])
Data2=pd.DataFrame(df2)


Data2['per']=Data2[:1].mul(100).round(2).astype(str).add(' %')
Data2['per'].fillna(0,inplace=True)
print(Data2)
