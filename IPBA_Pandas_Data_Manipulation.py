import os
import pandas as pd
import numpy as np
from pandas import DataFrame

# Data Manipulation using Pandas
# Filtering
# Selecting
# Sorting
# Adding new columns
# Group by operations
# Handle data and times
# Treat missing values
# Merge Pandas Data Frame

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename='sales.csv'

Data=pd.read_csv(filename,sep=',',header=0)
print(Data.head())
print(Data.shape,Data.info())

# Slicing the data
columns=Data.columns # Filter columns
print('columns \n',columns)
print('Unique_states np Array \n',Data['State'].unique()) # Filter unique columns
print('Unique_states  List\n',Data['State'].unique().tolist()) # Filter unique columns list

# Filter only for specific States
print('Filter for state\n',Data[Data['State']=='Texas'])# Filter columns based on condition
print('Filter for specific state',Data.query("State=='Texas'"))# Filter columns based on Query
print('Individual Sales where state=Texas \n',Data.query("State=='Texas'")['Sales'])
print('Sum of Sales for a State Texas \n',Data.query("State=='Texas'")['Sales'].sum())# Sum of Sales for State Texas
print('Sum of profit for a State Texas \n',Data.query("State=='Texas'")['Profit'].sum())# Sum of profit for State Texas

# Write a list comprehension to print sales per State
uniquestates=Data['State'].unique().tolist()
salesdic,profitdic={},{}
[print(state,': Sum of Profit:',Data[Data['State']==state]['Profit'].sum()) for state in uniquestates]
[print(state,': Sum of sales:',Data[Data['State']==state]['Sales'].sum()) for state in uniquestates]

for state in uniquestates:
    salesdic[state]=Data[Data['State']==state]['Sales'].sum()
    profitdic[state]=Data[Data['State']==state]['Profit'].sum()


# Sorting the values

print('reverse Sorted top 5 sales \n',Data.sort_values('Sales',ascending=False).head(5))
print('reverse Sorted top 5 profit \n',Data.sort_values('Profit',ascending=False).head(5))

#Sort Sales by each State
print('Sales Dic',salesdic,DataFrame([salesdic]))
print('Profit Dic',profitdic,DataFrame([profitdic]))
print('Sorted by Sales Values',dict(sorted(salesdic.items(),key=lambda item:item[1],reverse=True)))
print('Sorted by Profit Values',dict(sorted(profitdic.items(),key=lambda item:item[1],reverse=True)))

#Sort Sales by each State and product ID
for state in uniquestates:
    print('Sorted Sales \n',state,'ProductId \n',Data[Data['State']==state].sort_values('Sales',ascending=False)['ProductId'].head(10))
    print('Reverse Sorted Sales \n', state, 'ProductId \n',Data[Data['State'] == state].sort_values('Sales', ascending=True)['ProductId'].head(10))
    print('Sorted Profit \n', state, 'ProductId \n',Data[Data['State'] == state].sort_values('Profit', ascending=False)['ProductId'].head(10))
    print('Reverse Sorted Profit \n', state, 'ProductId \n',Data[Data['State'] == state].sort_values('Profit', ascending=True)['ProductId'].head(10))

# Group by the sales by different criteria
# as index = False creates the output in Data Frame
print('Data Group by mean Sales \n ',Data.groupby('State',as_index=False).agg({'Sales':np.mean}))
print('Data Group by mean profit \n ',Data.groupby('State',as_index=False).agg({'Profit':np.mean}))
print('Data Group by min Sales \n ',Data.groupby('State',as_index=False).agg({'Sales':np.min}))
print('Data Group by min profit \n ',Data.groupby('State',as_index=False).agg({'Profit':np.min}))
print('Data Group by max Sales \n ',Data.groupby('State',as_index=False).agg({'Sales':np.max}))
print('Data Group by max profit \n ',Data.groupby('State',as_index=False).agg({'Profit':np.max}))
print('Data Group by Sum Sales \n ',Data.groupby('State',as_index=False).agg({'Sales':np.sum}))
print('Data Group by Sum profit \n ',Data.groupby('State',as_index=False).agg({'Profit':np.sum}))
print('Data Group by Sales + profit \n ',Data.groupby('State')[['Sales','Profit']].agg(['max','min','mean']))
print('Data Group by Sales + profit \n ',Data.groupby('State')[['Sales','Profit']].agg({'Sales':np.sum,'Profit':np.max}))
