# Script for OLS Ordinary Least Square Method for regression

# Import section for Imports
import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import scipy
from scipy import datasets
import statsmodels.formula.api as smf
import time

# Open the file
try:
    filepath='C:/ProgramData/Anaconda3/Scripts/IPBA_Project/'
    filename='birthsmoke.csv'
    Data=pd.read_csv(filepath+filename)
except:
    FileNotFoundError
    print('File',filename,'not presnt at',filepath)

# Check and clean the data
Data=Data.fillna(0)
head=Data.head()
shape=Data.shape
#print(head)

Y=Data['Wgt']
X=Data[['Gest','Smoke']]
X1=smapi.add_constant(X)
model = smapi.OLS(Y,X1).fit()
print(model.summary())
time.sleep(5)
X_test=np.array([1,20,bool(1)])
prd=model.predict(X_test)
print('pred is',prd)

# Plot withouth regression Line
sbrn.set_style('darkgrid')
palette=['o','b','g']
sbrn.lmplot(x='Gest',y='Wgt',hue='Smoke',fit_reg=True,data=Data).set(title='SeaBorn LMPlot with Fit')
plt.show()





