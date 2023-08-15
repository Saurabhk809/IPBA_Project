import warnings
warnings.filterwarnings('ignore')
from kite_trade import *
import seaborn as sbrn
import numpy as np
import time
import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import scipy
#from scipy.stats import f_oneway
#import backtrader as bt

# Get Historical data

"""
enctoken="IvNRqfOVgcFaFxE950xrdYXSQe9YSPIHxHK9cSg9Bdvr4Z/svGaGqP/krV+ewBrv3LXQdRgSDHq4rho6boiji2fkkPtofK/+n2aZGNVFb1MX2uSW7fDX5Q=="
# created a SSL less session by making change at   line number 73 self.session.get(self.root_url, headers=self.headers,verify=False)
kite=KiteApp(enctoken=enctoken)


import datetime
instrument_token = 3060993
from_datetime = datetime.datetime.now() - datetime.timedelta(days=30)     # From last & days
to_datetime = datetime.datetime.now()
interval = "30minute"
IDFC=kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False)
IDFC_DF=pd.DataFrame(IDFC)
#IDFC_DF.to_csv('IDFC.csv')

IDFC_Close=IDFC_DF['close']
IDFC_Volume=IDFC_DF['volume']

"""
try:
    filepath='C:/ProgramData/Anaconda3/Scripts/IPBA_Project/'
    filename='IDFC.csv'
    IDFC_DF=pd.read_csv(filepath+filename)
except:
    FileNotFoundError
    print('file not present ')

#Build a linear regression
X=smapi.add_constant(IDFC_DF['volume'])
Y=IDFC_DF['close']
model=smapi.OLS(Y,X).fit()
print('Summary for IDFC volume vs Close price Regression',model.summary())

x_pred=np.array([1,840417])
pred=model.predict(x_pred)
print('price prediction',pred)

# Fit the regression Line using seaborn
sbrn.set_style('whitegrid')
sbrn.regplot(x='volume',y='close',fit_reg=True,data=IDFC_DF,color='darkgreen').set(title='SeaBorn Reg Plot')

# Find the Values for Slope and Intercept
slope,intercerpt,rvalue,pvalue,stderr=scipy.stats.linregress(IDFC_DF['volume'],IDFC_DF['close'])

# Add Equation to the plot
plt.text(50,145966,'y='+str(round(slope,3))+'x'+'+'+str(round(intercerpt,3)))

plt.show()




