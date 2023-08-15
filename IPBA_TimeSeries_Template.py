import pandas as pd
import os
import seaborn as sbrn
import matplotlib.pyplot as plt
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


os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")
Data=pd.read_csv('AirPassengers.csv')
#print(Data.to_string)

# Convert the Month column into a Datetime Object

Data['Month']=pd.to_datetime(Data['Month'],format='%Y-%m')
#print(Data.to_string)

# Convert the Date column into Index to allow us to work with some packages later

Data.index=Data['Month']
del Data['Month']
#print(Data.to_string)

# Lets look at Data using Seaborn Plot
sbrn.set_style('darkgrid')
sbrn.lineplot(Data)
plt.ylabel('Number of Passenger')
#plt.show()

# Plot original Data with rolling 7 month mean and Std Dev
rolling_mean=Data.rolling(7).mean()
rolling_std=Data.rolling(7).std()

plt.plot(Data,color='blue',label="Original passenger Data")
plt.plot(rolling_mean,color='red',label="Rolling Mean Passenger Number")
plt.plot(rolling_std,color='black',label="Rolling Std Passenger Number")
plt.title("Passenger Time Series, Rolling Mean, Standard Deviation")
plt.show()

# Check for Stationarity # Stationary time series will not have any trends or Seasonality
# Check for stationarity using Dickey Fuller test and Hypothesis
# Null Hypo # No Stationarity , Alt Hypo # There is stationarity

# If P value > .05 then data is not Stationary

DicFuller=adfuller(Data,autolag='AIC')
#print(DicFuller)
Output_DFull=pd.DataFrame({"Values":[DicFuller[0],DicFuller[1],DicFuller[2],DicFuller[3],DicFuller[4]['1%'],DicFuller[4]['5%'],DicFuller[4]['10%']],
'Metric':['Test Stats','pvalue','NoofLags','NoofObs','criticalvalue(1%)','criticalvalue(5%)','criticalvalue(10%)']})
print('Dickey Fuller Test Ouput : \n',Output_DFull)

# Check for (ACF) AutoCorrelation of Data on different timeframes for MA components
"""
AutoCorrelationLag1=Data['#Passengers'].autocorr(lag=1)
x=pd.plotting.autocorrelation_plot(Data['#Passengers'])
x.plot()
plt.title('ACF : Auto Correlation Factor')
plt.show()
print('One Month Lag:',AutoCorrelationLag1)
AutoCorrelationLag3=Data['#Passengers'].autocorr(lag=1)
print('One Month Lag:',AutoCorrelationLag3)
AutoCorrelationLag6=Data['#Passengers'].autocorr(lag=6)
print('One Month Lag:',AutoCorrelationLag6)
"""

#Check for ACF : Auto correlation for MA &
# PACF  PartialAutoCorrelation of Data on different timeframes for RA components

lag_acf = 15
lag_pacf = 15
height = 4
width = 12

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2*height))
plot_acf(Data['#Passengers'],lags=lag_acf, ax=ax[0])
plot_pacf(Data['#Passengers'],lags=lag_pacf, ax=ax[1], method='ols')
plt.tight_layout()
plt.show()

# Decompose the data to verify randomness, Trend, Seasonality

decompose1=seasonal_decompose(Data['#Passengers'],model='additive',period=7)
decompose1.plot()
plt.title("Passenger Additive Decomposition")
decompose2=seasonal_decompose(Data['#Passengers'],model='multiplicative',period=7)
decompose2.plot()
plt.title("Passenger Multiplicative Decomposition")
plt.show()

# Build a ARIMA Model
# For Non-Seasonal Data p=1,d=1,Q=0/1
#model=ARIMA(Data['#Passengers'],order=(1,1,1))
#X=Data['#Passengers']
log_X=np.log(Data['#Passengers'])
#print(X.head)
#print(log_X.head)

model=ARIMA(Data['#Passengers'],order=(1,1,1))
model_fit=model.fit()
print('Model Summary Normal:\n',model_fit.summary())

model2=ARIMA(log_X,order=(1,1,1))
model2_fit=model2.fit()
print('Model Summary LogNormal:\n',model2_fit.summary())

# Line Plot of residuals
residuals=pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('Residuals Line Plot')
plt.show()

# density plot of residuals
residuals.plot(kind='kde')
plt.title('Residuals Density Plot')
plt.show()
# summary stats of residuals
print(residuals.describe())

# How to Make ARIMA forecasts
X=Data['#Passengers']
Size=int(len(X)*0.66)
train,test=X[0:Size],X[Size:len(X)]
history=[x for x in train]
prediction=list()
# Walk-Forward Validaiton
for t in range(len(test)):
    model=ARIMA(history,order=(0,1,1))
    model_fit=model.fit()
    output=model_fit.forecast()
    yhat=output[0]
    prediction.append(yhat)
    obs=test[t]
    history.append(obs)
    print('predicted=%f,expected=%f' %(yhat,obs))

#evaluate forecasts, Goodness of Fit
rmse=sqrt(mean_squared_error(test,prediction))
print('Test RMSE=%f',rmse)

mpe=mean_absolute_percentage_error(test,prediction)
print('Test mpe=%f',mpe)

# plot forecasts againts actual outcomes
pyplot.plot(test)
pyplot.plot(prediction,color='red')
pyplot.show()
print(np.array[test].head())
print(np.array[test].tail())
print(np.array[prediction].tail())

plt.plot(test,label='testdata')
plt.plot(prediction,label='predicteddata')
plt.grid(True)
plt.show()

# Log Transformed Forecast

model2=ARIMA(log_X,order=(1,1,1))
model2_fit=model2.fit()
print('Model Summary LogNormal:\n',model2_fit.summary())

# How to Make ARIMA forecasts
X=Data['#Passengers']
Size=int(len(X)*0.66)
train,test=X[0:Size],X[Size:len(X)]
history=[x for x in train]
prediction=list()
# Walk-Forward Validaiton
for t in range(len(test)):
    model=ARIMA(np.log(history),order=(0,1,1))
    model_fit=model.fit()
    output=model_fit.forecast()
    yhat=output[0]
    prediction.append(yhat)
    obs=test[t]
    history.append(obs)
    print('predicted=%f,expected=%f' %(yhat,obs))

#evaluate forecasts, Goodness of Fit
rmse=sqrt(mean_squared_error(test,prediction))
print('Test RMSE=%f',rmse)

mpe=mean_absolute_percentage_error(test,prediction)
print('Test mpe=%f',mpe)

# plot forecasts againts actual outcomes
pyplot.plot(test)
pyplot.show()
pyplot.plot(prediction,color='red')
pyplot.show()




