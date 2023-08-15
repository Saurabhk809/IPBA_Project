import pandas as pd
import os
from prophet import Prophet
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
from matplotlib import pyplot
import logging

os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")
Data=pd.read_csv('HistoricalQuotes.csv')
print(Data.info())

# Convert the Month column into a Datetime Object and sort the data
Data['date']=pd.to_datetime(Data['date'])
Data=Data.sort_values(by='date')
#Data.index=Data['date']
#Data.drop('date',axis=1,inplace=True)
#print(Data.head())

# Prophet has a specific requirement: the time column needs to be named as ‘ds’ and the value as ‘y’.
df_p=Data.reset_index()[['date','close']].rename(columns={'date':'ds','close':'y'})

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
pyplot.show()

# For Accurary #choose between 'mse', 'rmse', 'mae', 'mape', 'coverage'
from prophet.diagnostics import cross_validation, performance_metrics
# Perform cross-validation with initial 365 days for the first training data and the cut-off for every 180 days.
df_cv = cross_validation(model, initial='400 days', period='180 days', horizon = '365 days')
print(df_cv.head())

# Calculate evaluation metrics
res = performance_metrics(df_cv)
plot_cross_validation_metric(df_cv, metric= 'mape')
pyplot.show()