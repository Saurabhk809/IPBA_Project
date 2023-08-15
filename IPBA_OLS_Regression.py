# Script for OLS Ordinary Least Square Method for regression

# Import section for Imports
import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import scipy

# Read the Input
try:
    filepath='C:/Users/E1077195/OneDrive - FIS/0 - NSE Trading/IPBA/02 Documentation/'
    filename='Healthcare.csv'
    Data=pd.read_csv(filepath+filename)
    Data = Data.fillna(0)  # Cleanse the data
    Age=Data['AGE']
    Cost=Data['Cost of Treatment']
except:
    FileNotFoundError
    print('File :',filename ,'is not present at',filepath)


#Plot  regression Line with seaborn
sbrn.set_style('whitegrid')
sbrn.lmplot(x='AGE',y='Cost of Treatment',hue='AGE',data=Data).set(title='SeaBorn lmplot')
plt.show()

# OLS regression
# define the constant variable
# Null Hypo there is no Impact , Alt Hypo there is Impact

x=smapi.add_constant(Age)
result=smapi.OLS(Cost,x).fit()
print(result.summary())

# Find the Values for Slope and Intercept
slope,intercerpt,rvalue,pvalue,stderr=scipy.stats.linregress(Age,Cost)

# Fit the regression Line using seaborn
sbrn.set_style('whitegrid')
sbrn.regplot(x='AGE',y='Cost of Treatment',fit_reg=True,data=Data,color='darkgreen').set(title='SeaBorn Reg Plot')

# Add Equation to the plot
plt.text(50,145966,'y='+str(round(slope,3))+'x'+'+'+str(round(intercerpt,3)))

# show the plot
plt.show()

X_test=np.array([1,11])
prd=result.predict(X_test)
print('prediction is',prd)



