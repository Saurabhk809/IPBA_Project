import pandas as pd
import os
from prophet import Prophet
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import logging
import seaborn as sbrn
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tools.sm_exceptions import  ValueWarning
warnings.simplefilter('ignore', ValueWarning)
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#Load the data
os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")
Data=pd.read_csv('train_ts.csv')

#View the data
print(Data.head())

#Basic Data check
print(Data.info())

# Convert the date column into a Datetime Object
Data['date']=pd.to_datetime(Data['date'],format='%d-%m-%Y')
print(Data.dtypes)


# Descriptive Statistics
print(Data.describe())

#Check for missing info
print(Data.isnull().sum())

# Sort the data by date
Data=Data.sort_values(by='date')

Data['transactions']=np.log(Data['transactions'])
print(Data.head())

# view the data for outliers
Q1 = np.quantile(Data['transactions'], .25)
Q3 = np.quantile(Data['transactions'], .75)
IQR = Q3 - Q1
UL =   Q3 + (0.5) * IQR
LL =   Q3 - (0.5) *  IQR

#Data[Data['transactions']>UL]
Data['transactions'] = np.where(Data['transactions'] > UL, Data['transactions'].median(), Data['transactions'])
Data['transactions'] = np.where(Data['transactions'] < UL, Data['transactions'].median(), Data['transactions'])

#Scatter plot to check the transactions vs Date transactions
sbrn.scatterplot(data=Data,x='date',y='transactions')
plt.title('transactions vs date')
plt.show()

#Line plot to check the transactions vs Date transactions
sbrn.lineplot(data=Data,x='date',y='transactions')
plt.title('transactions vs date')
plt.show()


# Prophet has a specific requirement: the time column needs to be named as ‘ds’ and the value as ‘y’.
df_p=Data.reset_index()[['date','transactions']].rename(columns={'date':'ds','transactions':'y'})

#Fit the model
model=Prophet()
model.fit(df_p)
print(model)

# create date to predict
future_dates=model.make_future_dataframe(periods=365)

# Make prediction
predictions=model.predict(future_dates)
print(predictions.head())
print(predictions[['ds','yhat','yhat_lower','yhat_upper']].head())

# Plot the predictions
model.plot(predictions)
graph=model.plot_components(predictions)
trend=add_changepoints_to_plot(graph.gca(),model,predictions)
pyplot.title('Data Prediction')
pyplot.show()

# For Accuracy #choose between 'mse', 'rmse', 'mae', 'mape', 'coverage'

from prophet.diagnostics import cross_validation, performance_metrics
# Perform cross-validation with initial 365 days for the first training data and the cut-off for every 180 days.
df_cv = cross_validation(model, initial='400 days', period='180 days', horizon = '365 days')
print(df_cv.head())

# Calculate evaluation metrics
res = performance_metrics(df_cv)
plot_cross_validation_metric(df_cv, metric= 'mape')
pyplot.show()