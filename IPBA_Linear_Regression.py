# Script for Linear Regression
#python

# Import section for Imports
import scipy
import numpy as np
import statistics as stats
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import statsmodels.formula.api as smf
import time
import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression

# Reading the data Set
filepath='C:/Users/E1077195/OneDrive - FIS/0 - NSE Trading/IPBA/02 Documentation/'
filename='Healthcare.csv'
Health_Care=pd.read_csv(filepath+filename)

# Cleaning the Data Set
HltCr_df=Health_Care[['AGE','Cost of Treatment']]
HltCr_df.fillna(0)


# Fetching the Cleanesed data from Data Frames
Age=HltCr_df['AGE']
Cost=HltCr_df['Cost of Treatment']

# Reshape the data for Linear Regression
X=np.array(Age).reshape(-1,1)
Y=np.array(Cost).reshape(-1,1)

# Different ways to do linear Regression In Python
# Method 1:#  scikit learn's linear mode for Linear Regression , sklearn.linear_model import LinearRegression

# create the instance of the Model
model=LinearRegression()
# Provide the datasets to Modle
model.fit(X,Y)
# Print the model Score
print('Model Prediction Score is:',model.score(X,Y))
# Print the Model Prediction
Y_pred=model.predict(X)
Y2=model.predict(np.array(11).reshape(-1,1))
sbrn.set_style('whitegrid')
sbrn.regplot(x=Y,y=Y_pred).set(title='Actual Vs Predicted')
plt.show()

#print(Y2)
# Print all Values
"""
for i in range(len(X)):
    print('Prediction of X is ',X[i],Y_pred[i])

# for individual data create a loop
#for i in range(len(X)):
    #print(X[i],Y_pred[i])
"""

# Method 2:# scipy stats module linregress to get slope ,interscept and R^2

slope,intercerpt,rvalue,pvalue,stderr=scipy.stats.linregress(Age,Cost)
print('slope B1:',slope,'\nintersecpt B0 :',intercerpt,'\nr^2 :',rvalue**2,'\npval :',pvalue,'\nstderr :',stderr)


# Method 3 # ge the filt details with polyfit numpy
result=np.polyfit(Age,Cost,deg=1)
print('Numpy Polyfit result',result)

# Method 4:statsmodels.api.ols() "Ordinary least Square method to estimate coefficient to perform Hypothesis if Age ~ cost
# Define constant for Predictor Variable
Age=smapi.add_constant(Age)
# Fit a linear regression Model
model=smapi.OLS(Cost,Age).fit()
#result=model.fit()
#prediction=model.predict(Cost)
#sbrn.regplot(x=)
print('smapi',model.summary())
time.sleep(5)

#print('Prediction',prediction)
#print('Prediction',predict)

#Plot  regression Line with seaborn
sbrn.set_style('whitegrid')
sbrn.lmplot(x='AGE',y='Cost of Treatment',hue='AGE',data=Health_Care).set(title='SeaBorn lmplot')
plt.show()

# Plot withouth regression Line
sbrn.set_style('darkgrid')
palette=['o','b','g']
sbrn.lmplot(x='AGE',y='Cost of Treatment',fit_reg=True,data=Health_Care,palette=palette).set(title='SeaBorn LMPlot with Fit')
plt.show()

# Reg plot with regression line
sbrn.set_style('whitegrid')
sbrn.regplot(x='AGE',y='Cost of Treatment',fit_reg=True,data=Health_Care).set(title='SeaBorn XY Scatter')
plt.show()

#Multiple linear regression
file2='TRP_data.csv'
TRP_data=pd.read_csv(filepath+file2)

# Cleaning the Data Set
TRP_data.fillna(0)

result = smf.ols(formula='R ~ CTRP + P', data=TRP_data).fit()
print(result.summary())
sbrn.set_style('whitegrid')
sbrn.regplot(x='R',y=[['CTRP', 'P']],data=TRP_data)
#sbrn.lmplot(x='R',y='CTRP + P',data=TRP_data)
plt.show()
result = smf.ols(formula='R ~ CTRP', data=TRP_data).fit()
print(result.summary())
result = smf.ols(formula='R ~ P', data=TRP_data).fit()
print(result.summary())