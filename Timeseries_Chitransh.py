import pandas as pd
from prophet import Prophet
import seaborn as sns
import os
import matplotlib.pyplot as plt

frame=pd.read_csv('C:\ProgramData\Anaconda3\Scripts\IPBA_Project\Copy of Superstore RAW DATA  2023.csv',parse_dates=['Order Date'])
print(frame.head())

columns_to_drop = ['Row ID', 'Order ID','Ship Date','Customer ID','City','State','Postal Code','Region']

frame1 = frame.drop(columns= columns_to_drop, index = [1] , inplace= True)

del frame['Country/Region']

frame.head()

#frame['Order Date'] = pd.to_datetime(frame['Order Date'])

df = frame[frame['Category'] == 'Furniture']

df = frame[frame['Category'] == 'Furniture'].copy()

df.sort_values('Order Date')

df.drop('Category', axis = 1 , inplace = True)

df1 = df.sort_values('Order Date')

df1.columns = ['ds','y']

y = df1['y']

m = Prophet(interval_width=0.95)

training_run = m.fit(df1)

future = m.make_future_dataframe(periods=365 , freq='D')

future.head()

future.tail()

forecast = m.predict(future)

plt.plot(df['actual'], df['forecast'], label='Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Time Series Actual vs Predicted')
plt.legend()
plt.show()