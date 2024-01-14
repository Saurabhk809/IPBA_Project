import os
import pandas as pd
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='telecom_churn.csv'

Data=pd.read_csv(filename)
print(Data.head())
print(Data.info())
print(Data.isna().sum())
print(Data['Churn'].astype(bool))

