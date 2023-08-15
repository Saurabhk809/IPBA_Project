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

tips=sbrn.load_dataset("tips")

#print(tips.info())
#print(tips.columns)
#print('\n',tips.head)
mycat=[]
for cols in tips.columns:
    dty = tips.dtypes[cols]
    #print (cols,dty)
    if dty=='category':
        mycat.append(cols)
    else:
        pass

#print('category cols',mycat)
#pd.get_dummies(tips[['sex','smoker','day','time']])
#print('after \n', tips.head)
Y=tips['total_bill']
X=tips[['sex','smoker','day']]
print('x is',X)
X=smapi.add_constant(X)
model=smapi.OLS(Y,X).fit()
model.summary()