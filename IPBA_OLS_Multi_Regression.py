# Script for OLS Ordinary Least Square Method for Multiple regression

# Import section for Imports

import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import statsmodels.formula.api as smf
import scipy

# Read the Input
filepath='C:/Users/E1077195/OneDrive - FIS/0 - NSE Trading/IPBA/02 Documentation/'
filename='TRP_data.csv'
try:
    Data=pd.read_csv(filepath+filename)
    TRPData=Data.fillna(0)
except:
    FileNotFoundError
    print('File :', filename, 'is not present at', filepath)


# Multiple OLS Regression
# Null Hypo there is no Impact , Alt Hypo there is Impact
model = smf.ols(formula='R ~ CTRP + P', data=TRPData).fit()
print(model.summary())
#sbrn.set_style('whitegrid')
#sbrn.regplot(x='R',y=['CTRP','P'],data=TRPData,fit_reg=True,color='darkgreen')
#plt.show()
model = smf.ols(formula='R ~ CTRP', data=TRPData).fit()
print(model.summary())
model = smf.ols(formula='R ~ P', data=TRPData).fit()
print(model.summary())

# Another way
Y=TRPData['R']
X=TRPData[['CTRP','P']]
result=smapi.OLS(Y,X).fit()
print(result.summary())



