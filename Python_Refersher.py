# Script for Python Refresher #30 June 2023
#python
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbrn
import numpy as np
import scipy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

filepath='C:/Users/E1077195/OneDrive - FIS/0 - NSE Trading/IPBA/02 Documentation/'
filename='Healthcare.csv'
Health_care=pd.read_csv(filepath+filename)
AGE=Health_care['AGE']
cost=Health_care['Cost of Treatment']


slope,intercerpt,rvalue,pvalue,stderr=stats.linregress(AGE,cost)
print('slope=',slope,'intersecpt=',intercerpt,'rvalue=',rvalue*2,'pvalue=',pvalue,'stderr=',stderr)

#Hypo_lm = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

#Plot  with regression Line
#sbrn.lmplot(x='AGE',y='Cost of Treatment',hue='AGE',data=Health_care)
#plt.show()

result=np.polyfit(AGE,cost,deg=1)
print(result)

# X, Y Scatter with trend line
sbrn.regplot(x='AGE',y='Cost of Treatment',fit_reg=True,data=Health_care)
#sbrn.jointplot(x='AGE',y='Cost of Treatment',kind="reg",data=Health_care)
plt.show()

# Plot withouth regression Line
#sbrn.lmplot(x='AGE',y='Cost of Treatment',fit_reg=False,data=Health_care,hue='AGE')
#plt.show()

# check the fit with statmodles
#import statsmodels.api as ms
#import statsmodels.formula.api as smf
# fit a regression module
#results=smf.ols('AGE ~ cost').fit()
#print(results.summary())




