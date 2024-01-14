import os
import pandas as pd
from pandas import DataFrame

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='countries.csv'
pd.options.display.width = 0 # Auto detects the window

Data=pd.read_csv(filename,sep=',',header=0,skiprows=None)
print(Data.columns)
Data2=pd.read_csv('creditDefaultData.csv')
Data3=pd.read_csv('airquality.csv',na_values=['NA','N/A','nan',' '])
Data4=pd.read_csv('Churn_Modelling.csv')
# Pandas String related Function
#Find all the rows where continent = North America
#print(Data[Data['CONTINENT']=='North America'])

# Do a Replace option ISO 3 , XIX
Data['ISO3']=Data.ISO3.replace('AIA','XIX')
Data['ISO3']=Data['ISO3'].replace(['ABW','ALA'],['BBW','BLA']) #both ways are same
#Data['ISO3'].replace(['ABW','ALA'],['BBW','BLA'],inplace=True) #In Place == True
print(Data.head())

#rename columns in Pandas data frame
Data.rename(columns={'NAME':'Fame','NAME_LOCAL':'FAME_LOCAL'},inplace=True)
print('replaced columns',Data.head())
Data.rename(columns={'FAME':'NAME','FAME_LOCAL':'NAME_LOCAL'},inplace=True)
# Find rows that contain a value

print(Data.NAME_FAO.str.find("Angola"))
print(Data.NAME_FAO.str.findall("Angola"))
print(Data.NAME_FAO.str.contains("Angola"))

print(Data[Data['NAME_FAO']=='Angola'])

# Frequency Distribution
print(Data.CONTINENT.value_counts(),type(Data.CONTINENT.value_counts()),Data.CONTINENT.value_counts()['Asia'])

# Counts of Values
print(Data.CONTINENT.value_counts())

# Count of Unique values
print(Data.CONTINENT.unique(),type(Data.CONTINENT.unique()))
print(Data['CONTINENT'].unique(),type(Data['CONTINENT'].unique()))
print(Data.CONTINENT.nunique(),type(Data.CONTINENT.unique()))# Count of Unique Values

# Convert the column into upper case or lower case
print(Data.CONTINENT.str.upper()) # Upper case
print(Data.CONTINENT.str.lower())  # Lower case
print(Data.CONTINENT.str.title())     # Proper case

print(Data.CONTINENT.str.strip(' '))     # Proper case
print(Data.head())

# Cross Tabulation
print(pd.crosstab(Data['CONTINENT'],Data['UNREGION1']))

# Ascending or Descending Sort
print(Data2.sort_values(['AGE']))
print(Data2['AGE'].sort_values()) # Sorts only 1 columns
print(Data2.sort_values(['EDUCATION','AGE'],ascending=[True,False]).head())
print('Describe1',Data2['SEX'].describe())
print('Describe2',Data2.describe().head())

# Group by
print(Data2.groupby(['SEX','AGE','EDUCATION']).agg(['min','max','count','mean'])[['LIMIT_BAL','BILL_AMT4']].head(10))

print(Data2.groupby(['SEX','AGE']).agg(['count','mean'])[['LIMIT_BAL','BILL_AMT4']].head(10))
df12= Data2.groupby(['SEX','AGE']).agg(['count','mean'])[['LIMIT_BAL','BILL_AMT4']].head(10)
df12 = df12.T.drop_duplicates().T
print('df12 is',df12)


#26-08-2023 Find missing values
print(Data3.head())
# Missing values in Python is NAN
print(Data3.isnull())
print(Data3.isnull().values.any())

for cols in Data3.columns:
    if Data3[cols].isnull().sum() > 0:
        Data3[cols].fillna(0,inplace=True)
print(Data3.head())
print(Data3.isnull().sum())

# Treatment of Missing Values
#Input, #Ignore # Drop the col

print(Data3.shape)

# Drop the row with 0 Values
Data3.drop(0)

# Impute the missing records
Data3.Ozone.fillna(Data3.Ozone.mean(),inplace=True)
Data3['Solar.R']=Data3['Solar.R'].fillna(Data3['Solar.R'].mean())
print(Data3['Solar.R'])

# WOE (Weight of Evidence ) & IV (Information Value)
# Impact of Age, Salary , years in service of employee Attrition (Y/N)
# Look at IV value of a variable

# IV <0.02 # useless for prediction
# .02 to 0.1 # weak predictor
# 0.1 to 0.3 # Medium pretictive power
# 0.3 to 0.5 # Strong predictive power
# > 0.5         # to good to be trye very suspicious variable (not a good candidate)

print(Data4.head())

# WOE & IV
# Create Bins using qcut
#temp=pd.qcut(Data4['Age'],3).value_counts()
#print(temp)

#Data4['Agebins']=pd.qcut(Data4['Age'],3,labels=['Age1','Age2','Age3'])
#print(Data4.head(50))
#
def calcuate_woe_iv(Data4,cols):
    #print(Data4[col])
    temp=pd.cut(Data4[cols],5).value_counts()
    print(temp)
    df=Data4[cols]
    return df

EscapeColList=['RowNumber','CustomerId','Surname','Geography','Gender','Exited','Unamed']

print('head of Data',Data4.head())
for cols in Data4.columns:

    #print(cols,Data4[cols].info())
    if cols in EscapeColList: pass
    else:
        #print('woe & IV for columne []',format(cols))
        #df,iv=calculate_woe_iv(data,col,'Exited')
        Data4[cols+'_bins']=pd.cut(Data4[cols],5,labels=['Bin1','Bin2','Bin3','Bin4','Bin5'])
        #df=calcuate_woe_iv(Data4,cols)
        #print(df)
        #print('IV score:'{:2f})

print('head of Data',Data4.head())
