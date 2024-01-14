#B

import os
import pandas as pd
import seaborn as sbrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore')
#from IPython import display,HTML
from IPython.display import display,HTML

try:
    os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
    filename1 = 'loan.csv'
    df_loan=pd.read_csv(filename1,na_values=['NA','N/A','nan',' '],low_memory=False)
    #filename2 = 'Data_Dictionary1.csv'
    #data_dic=pd.read_csv(filename2,na_values=['NA','N/A','nan',' '])
except:
    FileNotFoundError
    print('File','not present in',os.getcwd())

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
#print(loan.shape)
#print(loan_df.head())
#print(loan.info())
print(df_loan.loan_status.nunique())
print(df_loan.loan_status.unique())

df_loan=df_loan[(df_loan.loan_status=='Fully Paid') | (df_loan.loan_status=='Charged Off')]
print(df_loan.shape)

print(sum(df_loan.duplicated(subset='id')))

# drop duplicates
df_loan.drop_duplicates()
print(df_loan.shape)

# print % of missing values in each columns
#print((df_loan.isnull().sum()/df_loan.shape[0],4)*100)

# Remove columns where 100 % values are missing
df_loan.dropna(axis=1,how='all',inplace=True)
#print(df_loan.shape)

# Figure out the columns where only 1 values is present all over ??
print(df_loan.pymnt_plan.unique())
unique=df_loan['pymnt_plan'].unique()
print(unique)

#for all columns
uniqs=df_loan.apply(lambda x:x.nunique())
print(uniqs[uniqs < 2])

#df_loan=df_loan.drop(uniqs[uniqs < 2][0],axis=1)
#print(df_loan.shape)

df_loan=df_loan.drop(uniqs[uniqs < 2].index,axis=1)
print(df_loan.shape)

# Drop the columns which missing value count > 50 % missing value
#print((df_loan.isnull().sum()/df_loan.shape[0],4)*100)

df_loan.drop(['mths_since_last_record','desc','mths_since_last_delinq'],axis=1,inplace=True)
#print((df_loan.isnull().sum()/df_loan.shape[0],4)*100)

df_loan.dropna(axis=0,inplace=True)
print((df_loan.isnull().sum()/df_loan.shape[0],4)*100)
print(df_loan.shape)
print(df_loan.head())

# EDA Analysis
#charged off means = default
# Q1 : Charged off % by each state of applicant
# Q2 : Charged off % by each zip code of applicant
# Q3 : Charged off % by purpose

# create dummy variable from loan_status
df_loan['charged_off']= df_loan['loan_status'].apply(lambda x:1 if x=='Charged off' else 0)
df_loan['fully_paid']=df_loan['loan_status'].apply(lambda x:1 if x=='Fully Paid' else 0)
print(df_loan.head())

"""
print(df_loan.groupby('addr_state').agg({'charged_off':'sum','fully_paid':'sum'}))
df=df_loan.groupby('addr_state').agg({'charged_off':'sum','fully_paid':'sum'}).rename(columns={'charged_off':'charged off count','fully_paid':'fully_paid count'})
df['Totalcount']=df['charged off count']+df['fully_paid count']
df['percharoff']=round(100*df['charged off count']/df['Totalcount'],2)
df.sort_values(by=['percharoff'],inplace=True)
#print(df.head())

print(df_loan.groupby('home_ownership').agg({'charged_off':'sum','fully_paid':'sum'}))
df=df_loan.groupby('addr_state').agg({'charged_off':'sum','fully_paid':'sum'}).rename(columns={'charged_off':'charged off count','fully_paid':'fully_paid count'})
df['Totalcount']=df['charged off count']+df['fully_paid count']
df['percharoff']=round(100*df['charged off count']/df['Totalcount'],2)
df.sort_values(by=['percharoff'],inplace=True)
#print(df.head())
"""

#create a function in python
def perfrom_univariate_analysis(col):
    df_loan.groupby('addr_state').agg({'charged_off': 'sum', 'fully_paid': 'sum'})
    df = df_loan.groupby('addr_state',as_index=False).agg({'charged_off': 'sum', 'fully_paid': 'sum'}).rename(columns={'charged_off': 'charged off count', 'fully_paid': 'fully_paid count'})
    print('df is ',df)
    df['Totalcount'] = df['charged off count'] + df['fully_paid count']
    df['percharoff'] = round(100 * df['charged off count'] / df['Totalcount'], 2)
    df.sort_values(by=['percharoff'], inplace=True)
    return df

mydf=perfrom_univariate_analysis('home_ownership')
print('mydf\n',mydf)

sbrn.barplot(x='addr_state',y='Totalcount',data=mydf)
plt.show()

#pip install pandas.profiling
#pip install autoviz , # can also use Switveez
"""
from pandas_profiling import ProfileReport
profile=ProfileReport(df_loan,title='Report')
profile.to_file("Report.html")

import sweetviz as sv
sv_report=sv.analyze(df_loan)
sv_report.show_html('sv_report.html')

"""
