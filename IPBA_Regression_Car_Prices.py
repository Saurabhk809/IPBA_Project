# Script to find the price dependency of Car based on horse, Cyl,Disp,fuel and combination

# Import section for Imports
import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import statsmodels as stats
import scipy
import statsmodels.formula.api as smfi
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')


# File reading
try:
    filepath='C:/ProgramData/Anaconda3/Scripts/IPBA_Project/'
    filename='CarPrices.csv'
    CarData=pd.read_csv(filepath+filename)
except:
    FileNotFoundError
    print('File',filename,'not present on filepath',filepath)

Hd=CarData.head()
Sh=CarData.shape
#print(CarData.dtypes) # Check the Data Types of columns
#print(CarData.describe()) # Statistical description for the Data
CarData.fillna(0)


#print(Hd)
#print(Data.info())
#for cols in CarData:
    #print(cols)

# Visual the initial data
#sbrn.set_style('whitegrid')
sbrn.scatterplot(x='horse',y='MSRP',data=CarData,palette="deep").set(title='MSRP vs Horse Pow')
plt.show()

sbrn.set_style('whitegrid')
sbrn.lmplot(y='MSRP',x='horse',hue='Cyl',data=CarData,row='SUV',col='Non-SUV')
plt.show()

sbrn.set_style('whitegrid')
sbrn.lmplot(y='MSRP',x='horse',hue='Cyl',row='SUV',data=CarData)
plt.show()

#Q1 Test impact of singular variable on MSRP : example , horse, Cyl,Disp,fuel
Y=CarData['MSRP']
X=CarData['Cyl']
X=smapi.add_constant(X)
model=smapi.OLS(Y,X).fit()
X_Test=np.array([1,15])
print(model.summary())
price=model.predict(X_Test)
print('pred ',price)

# Find the Values for Slope and Intercept
x=CarData['fuel']
slope,intercerpt,rvalue,pvalue,stderr=scipy.stats.linregress(x,Y)
print('Equation is','y='+str(round(slope,3))+'x'+'+'+str(round(intercerpt,3)))

# Regplot
sbrn.set_style('whitegrid')
sbrn.regplot(x='fuel',y='MSRP',fit_reg=True,data=CarData,color='darkgreen')

# Add Equation to the plot
plt.text(385,9380000,'y='+str(round(slope,3))+'x'+'+'+str(round(intercerpt,3)))
plt.show()

# Q2 Train  and Test the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model=smapi.OLS(Y_train,X_train).fit()
Y_Pred=model.predict(X_test)
print('Trained Model result is \n', Y_Pred)

#Y_test=np.array(Y_test)
#Y_Pred=np.array(Y_Pred)
Result=pd.DataFrame()
Result=[[Y_Pred],[X_test]]


sbrn.scatterplot(x=Y_test,y=Y_Pred,color='darkgreen')
plt.show()

# Q3 Test impact of two variables on MSRP : example Horse+ Cyl , Cyl+Fuel
Y=CarData['MSRP']
X=CarData[['horse','fuel','Disp','luggage']]
#X=CarData[['horse','Cyl','fuel','Disp','luggage']]
#print(Z)

#X=CarData[['horse','Cyl','fuel']]
X=smapi.add_constant(X)

model=smapi.OLS(Y,X).fit()
print(model.summary())
time.sleep(5)
#X_Test=np.array([1,250,2,3,4,14])
#pred=model.predict(X_Test)
#print('Pred is',pred)

# Predict the Result
Z=X.copy()
Z_test=X.mul(Z,fill_value=100)
Z_test=np.array(Z_test)

#print(X.shape)
#print(Z_test.shape)

pred2=model.predict(Z_test)
print('Final Pred is :\n',pred2)






















































