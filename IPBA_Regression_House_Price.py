# Script to find House price based on # cols PriceInTh	 ,Sqmt,DistanceToCity ,HouseAge,Builder

# Import section for Imports
import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import statsmodels as stats
import scipy
import statsmodels.formula.api as smfi
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import  variance_inflation_factor

try:
    filepath = 'C:/ProgramData/Anaconda3/Scripts/IPBA_Project/'
    filename='HousePrices.csv'
except:
    FileNotFoundError
    print('File',filename,'not present in filepath',filepath)

HouseData=pd.read_csv(filepath+filename)
Hd=HouseData.head()
print(Hd)

# View the data
sbrn.lmplot(x='Sqmt',y='PriceInTh' ,data=HouseData)
plt.show()

sbrn.lmplot(x='DistanceToCity',y='PriceInTh' ,data=HouseData)
plt.show()

sbrn.lmplot(x='Sqmt',y='PriceInTh',hue='Builder' ,markers=['o','v','x'],data=HouseData)
plt.show()

# Test House price PriceInTh dependency on Builder, Sqmt , HouseAge,DistanceToCity
Y=HouseData['PriceInTh']
X=HouseData['DistanceToCity']
X=smapi.add_constant(X)

model=smapi.OLS(Y,X).fit()
print(model.summary())

x=HouseData['DistanceToCity']
slope,intercerpt,rvalue,pvalue,stderr=scipy.stats.linregress(x,Y)
print('Equation is','y='+str(round(slope,3))+'x'+'+'+str(round(intercerpt,3)))

# Test House Price PriceInTh dependency on Builder, Sqmt , HouseAge,DistanceToCity
Y=HouseData['PriceInTh']
X=HouseData[['DistanceToCity','Sqmt','HouseAge']]
X=smapi.add_constant(X)
model=smapi.OLS(Y,X).fit()
print(model.summary())
X_test=np.array([1,5,120,3])
result=model.predict(X_test)
print('prediction is ',result)

# Different way
model=smfi.ols("PriceInTh~DistanceToCity+Sqmt+HouseAge",data=HouseData)
results=model.fit()
print(results.summary())
prediction=results.predict(X)
Actuals=Y

# Plot Actual Vs Predicted
plt.plot(Actuals,'b')
plt.plot(prediction,'r')
plt.show()

residuals=results.resid
type(residuals)

residualssdf=pd.DataFrame(residuals)
residualssdf.rename(columns={0:"res"},inplace=True)

# PLot of residual vs Prediction to confirm HomoScadisticity or HetroScadasticity
plt.scatter(residualssdf,prediction)
plt.xticks([])
plt.show()

# Find absolute mean Error
mae=metrics.mean_absolute_error(Actuals,prediction)
print (mae)

MAPE=np.mean(abs((Actuals-prediction)/Actuals))# MAPE
print('MAPE is',MAPE)

# Find VIF

VIF=pd.DataFrame()
VIF["VIF Factor"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
VIF["features"]=X.columns
VIF.round(1)
print(VIF)

df3 = pd.get_dummies(HouseData,columns=['Builder'])
df3['Sqmt'] = df3['Sqmt'].astype(float)
print(df3)
print('correlation is',df3.corr())



