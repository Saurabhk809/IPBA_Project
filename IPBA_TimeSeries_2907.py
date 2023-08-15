import pandas as pd
import os
import seaborn as sbrn
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
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

# Read the Data
os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")
Data=pd.read_csv('HistoricalQuotes.csv')
#print(Data.info())
#print(Data['date'],type(Data['date']))

# Convert the Month column into a Datetime Object and sort the data
#Data['date']=pd.to_datetime(Data['date'],format='%d-%m-%Y')
Data['date']=pd.to_datetime(Data['date'],format='%Y/%m/%d')
#print(Data.head(5))
#print(Data.tail())
Data1=Data.sort_values('date',axis=0,ascending=True,ignore_index=True)
#print(Data1.head(5))
#print(Data1.tail())

# check the data for missing values
#print('\n missing value before treatment:\n', Data.isnull().any())
Data1.fillna(0,inplace=True)
#print('\n missing value after treatment:\n', Data.isnull().any())

# columns date	close	volume	open	high	low

# Use Normal Data and then Later use Log Data to check the difference
Data1.plot(x='date',y='close')
plt.show()

# How to create log of data
Data1['ln_cls']=np.log(Data1.close)
Data1['ln_open']=np.log(Data1.open)
Data1['ln_high']=np.log(Data1.high)
Data1['ln_low']=np.log(Data1.low)
#print(Data.head())

# Check ADF Test for stationarity
DicFuller=adfuller(Data1['ln_cls'],autolag='AIC')
#print(DicFuller)
Output_DFull=pd.DataFrame({"Values":[DicFuller[0],DicFuller[1],DicFuller[2],DicFuller[3],DicFuller[4]['1%'],DicFuller[4]['5%'],DicFuller[4]['10%']],
'Metric':['Test Stats','pvalue','NoofLags','NoofObs','criticalvalue(1%)','criticalvalue(5%)','criticalvalue(10%)']})
print('Normal Data Dickey Fuller Test O/P for stationarity : \n',Output_DFull)

# If p Values is greater than .05 Data is not Stationary else it is stationary
Data1['Diff1_ln_cls']=Data1.ln_cls.diff()
Data1['Diff1_ln_cls'].fillna(0,inplace=True)
#print(Data['Diff1_ln_cls'].to_string)

# Check ADF Test for stationarity
DicFuller=adfuller(Data1['Diff1_ln_cls'],autolag='AIC')
#print(DicFuller)
Output_DFull=pd.DataFrame({"Values":[DicFuller[0],DicFuller[1],DicFuller[2],DicFuller[3],DicFuller[4]['1%'],DicFuller[4]['5%'],DicFuller[4]['10%']],
'Metric':['Test Stats','pvalue','NoofLags','NoofObs','criticalvalue(1%)','criticalvalue(5%)','criticalvalue(10%)']})
print('Log Normal Data Dickey Fuller Test O/P for stationarity :  \n',Output_DFull)

# If p Value is greater than .05 apply a Diff of 1 and Recheck ADF
# Take help of Auto ARIMA function

# Plot the ACF and PACF
# Find the MA term from the ACF plot # Find the AR terms from PACF plot
lag_acf = 15
lag_pacf = 15
height = 4
width = 12
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2*height))
plot_acf(Data1['close'],lags=lag_acf, ax=ax[0])
plot_pacf(Data1['close'],lags=lag_pacf, ax=ax[1], method='ols')
plt.tight_layout()
plt.title('Normal Data ACF:MA and PACF:AR values')
plt.show()

lag_acf = 15
lag_pacf = 15
height = 4
width = 12
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2*height))
plot_acf(Data1['Diff1_ln_cls'],lags=lag_acf, ax=ax[0])
plot_pacf(Data1['Diff1_ln_cls'],lags=lag_pacf, ax=ax[1], method='ols')
plt.tight_layout()
plt.title('Log normal Data ACF:MA and PACF:AR values')
plt.show()

# Fit a model using StatsForecast Method
df=Data1[['date','Diff1_ln_cls']]
# constant value required by the Model
df['unique_id']='1'
#print(df)
# Date need to be renamed as ds and data as y
df.columns=['ds','y','unique_id']
sf=StatsForecast(models=[AutoARIMA(season_length=12)],freq='M')
sf.fit(df)
prediction=sf.predict(h=10,level=[95])
#print(df.tail())
#print('prediction',prediction)
#print(sf.forecast(h=12))

# Modelling using  AUTO ARIMA Function of pmdarima with normal data ##

import pmdarima as pm
model1=pm.auto_arima(Data1['close'],m=7,seasonal=True,start_p=0,start_q=0,max_order=5,test='adf',error_action='ignore',
                    suppress_warnings=True,stepwise=True,trace=True)
print('Normal Model',model1.summary())

# Modelling using  AUTO ARIMA Function of pmdarima with normal data ##

model2=pm.auto_arima(Data1['Diff1_ln_cls'],m=7,seasonal=True,start_p=0,start_q=0,max_order=5,test='adf',error_action='ignore',
                    suppress_warnings=True,stepwise=True,trace=True)
print('Log Normal',model2.summary())

# Model 3 using MA component as 0
model3=ARIMA(Data1['close'],order=(1,1,1))
model_fit=model3.fit()
print('Normal Data ARI model with MA 0:\n',model_fit.summary())

model4=ARIMA(Data1['Diff1_ln_cls'],order=(1,1,1))
model_fit=model4.fit()
print('Log Normal Data ARI model with MA 0:\n',model_fit.summary())

X=Data1['close']
OrgSize=len(X)
Size=int(len(X)*0.60)
train,test=X[0:Size],X[Size:len(X)]
print('train',train.size,'test',test.size)

#model1.fit(Data1['close'])
model1.fit(Data1['close'])
forecast1=model1.predict(n_periods=OrgSize,return_conf_int=False)
#print('\n Forecast1\n',forecast1)
forecast1_df=pd.DataFrame(forecast1,columns=['forecast'])
#print(Data1.head())
#print(Data1.tail())
#print(forecast1_df.head())
#print(forecast1_df.tail())

print('\n normal forecast is \n',forecast1_df.head())
#forecast2=model2.predict(n_periods=10,return_conf_int=True)

#evaluate forecasts, Goodness of Fit
rmse=sqrt(mean_squared_error(Data1['close'],forecast1))
print('Test RMSE=%f',rmse)

#mpe=mean_absolute_percentage_error(Data1['close'],forecast1)
mpe=mean_absolute_percentage_error(Data1['close'],forecast1)
print('Test mpe=%f',mpe)

# Plot Actual vs Predicted
pd.concat([Data1['close'],forecast1_df],axis=0).plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()






