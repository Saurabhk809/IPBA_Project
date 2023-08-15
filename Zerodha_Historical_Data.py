import warnings
warnings.filterwarnings('ignore')
from kite_trade import *
import pandas as pd
import seaborn as sbrn
import numpy as np
import time
import matplotlib.pyplot as plt
#from scipy.stats import f_oneway
#import backtrader as bt

enctoken="tRLz2yjounH1Ar83kC9XXmZ/zjvXfIejgMz592y2D8pfGIchOrpkdcUmzwGSaM+Ss9ffD+OfzUr7TX5mZ3lAUd/2ODBWAuEDeK+fTrmyhzzGwe5G/p1SsA=="
# created a SSL less session by making change at   line number 73 self.session.get(self.root_url, headers=self.headers,verify=False)
kite=KiteApp(enctoken=enctoken)

#print(kite.instruments('NSE'))
#print (kite.instruments('NFO'))

# Get instruments
#FO_Instruments=pd.DataFrame(kite.instruments('NFO'))
#FO_Instruments.to_csv('Option_instrument.csv',encoding='utf-8', index=False)
#Eqty_Instruments=pd.DataFrame(kite.instruments('NSE'))
#Eqty_Instruments.to_csv('Equity_instrument.csv',encoding='utf-8',index=False)

# Get Historical data
###
import datetime
instrument_token = 260105
from_datetime = datetime.datetime.now() - datetime.timedelta(days=2)     # From last & days
to_datetime = datetime.datetime.now()
interval = "30minute"
#print(kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False))
Bnifty=kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False)
#print(Bnifty)
#print(pd.DataFrame(Bnifty))
Bnifty=pd.DataFrame(Bnifty)
#my_historical_data_frame=pd.DataFrame(my_historical_data)
# call options data
instrument_token = 11090178
BNiftyOptCall=kite.historical_data(instrument_token,from_datetime, to_datetime, interval, continuous=False, oi=False)
BNiftyOptCall=pd.DataFrame(BNiftyOptCall)
#BNiftyOptCall=pd.DataFrame(BNiftyOptCall)['close']
# Put options data
instrument_token = 11090434
BNiftyOptPut=kite.historical_data(instrument_token,from_datetime, to_datetime, interval, continuous=False, oi=False)
BNiftyOptPut=pd.DataFrame(BNiftyOptPut)

Datalist=[260105,11090178,11090434]
from_datetime = datetime.datetime.now() - datetime.timedelta(days=2)     # From last & days
to_datetime = datetime.datetime.now()
interval = "15minute"
AllData=pd.DataFrame()
mydic={}
for i in range(len(Datalist)):
    instrument_token=Datalist[i]
    Data=kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False)
    #print(pd.DataFrame(Data))
    AllData1=(pd.DataFrame(Data))
    AllData1['Symbol']=Datalist[i]
    #print(AllData1)
    AllData=AllData._append(AllData1)
#print(AllData)
    #AllData[i]=pd.DataFrame(Data)

# Seaborn PairPlot
palette = ['tab:blue', 'tab:green', 'tab:orange']
#AllData.to_csv('data.csv')
sbrn.axes_style("whitegrid")
sbrn.pairplot(AllData,hue='Symbol',palette=palette)
#sbrn.pairplot(AllData,hue='Symbol',palette=palette)
plt.show()

#sbrn.lmplot(x='AGE',y='Cost of Treatment',hue='AGE',data=Health_care)
#plt.show()

instrument_token=11090178
from_datetime = datetime.datetime.now() - datetime.timedelta(days=2)     # From last & days
to_datetime = datetime.datetime.now()
interval = "5minute"
Data=kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False)
OptData=pd.DataFrame(Data)

# Normal without fit
#sbrn.lmplot(x='close',y='volume',fit_reg=False,data=OptData,hue='close')
#plt.show()

# Lm plot With fit
#sbrn.lmplot(x='volume',y='close',fit_reg=True,data=OptData)
#plt.show()

# regplot with fit
vol=OptData['volume']
cls=OptData['close']
result=np.polyfit(x=vol,y=cls,deg=1)
print(result)

sbrn.regplot(x='volume',y='close',fit_reg=True,data=OptData)
plt.show()


#print(BNiftyOpt)
#BNifty_Call=np.polyfit(Bnifty,BNiftyOptCall,deg=1)
#print(BNifty_Call)

#BNifty_Put=np.polyfit(Bnifty,BNiftyOptPut,deg=1)
#print(BNifty_Put)

#sbrn.regplot(x=Bnifty,y=BNiftyOptCall,fit_reg=True)
#plt.show()

# Demo plot of multiple values in table
#data=sbrn.load_dataset("iris")
#print (data)
#sbrn.pairplot(data=data,hue='species')
#plt.show()

#sbrn.pairplot(data=AllData,hue='close')
#plt.show()

#sbrn.regplot(x=Bnifty,y=BNiftyOptCall,fit_reg=True,label='Bnifty vs 44700CALL')
#plt.show()

#sbrn.regplot(x=Bnifty,y=BNiftyOptPut,fit_reg=True)
#plt.show()

#sbrn.regplot(x='AGE',y='Cost of Treatment',fit_reg=True,data=Health_care)
#plt.show()

"""
#print(my_historical_data_frame)
#calculate the MACD values using pandas
print(my_historical_data_frame['close'].describe())
trend=(my_historical_data_frame['close']-my_historical_data_frame['open'])
my_historical_data_frame['trend']=trend
print(my_historical_data_frame)


#plt.plot(my_historical_data_frame['trend'],label="BNIFTY Trend",linestyle="--")
#plt.legend()
#plt.show()
#f_oneway(my_historical_data_frame['close'],)my_historical_data_frame['open']

"""