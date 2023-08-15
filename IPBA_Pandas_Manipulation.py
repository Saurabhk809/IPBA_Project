import os
import pandas as pd
from pandas import DataFrame

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='countries.csv'
pd.options.display.width = 0 # Auto detects the window

Data=pd.read_csv(filename,sep=',',header=0,skiprows=None)
print(Data.columns)
Data2=pd.read_csv('creditDefaultData.csv')

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

