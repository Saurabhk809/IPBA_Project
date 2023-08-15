# Script to connect to Zerodha using API

# # First Way to Login
# # You can use your Kite app in mobile
# # But You can't login anywhere in 'kite.zerodha.com' website else this session will disconnected
import warnings
warnings.filterwarnings('ignore')
from kite_trade import *
import pandas as pd
import time

#enctoken=input('enter the enctoken :')
enctoken="8lmTjkrqnGq+ShNdV5x3GWtNP+5wAgpmhntb1ahLIe8T7V5fBcj0a2UJgx9hy7vMOfL5o6Lq36SAy070KUQbsyOccwxvqx8TC357o2/wuH7fsVlU1HfUdA=="
# created a SSL less session by making change at   line number 73 self.session.get(self.root_url, headers=self.headers,verify=False)
kite=KiteApp(enctoken=enctoken)

# global declaration
symbol = ''
lastprice = 0
Netchange = 0
pricestats = 0
Open = 0
High = 0
Low = 0
Close = 0
# Basic calls
#print(kite.margins())
#print(kite.orders())
#print(kite.positions())

# Get the Instruments or exchange
#print(kite.instruments('NSE'))
#print (kite.instruments('NFO'))
# making a Panda dataframe for NSEFO instruments
#NSEFO_Data = pd.DataFrame(kite.instruments('NFO'))

# Get live data
#instriment token for Nifty Bank 260105
#print(kite.ltp(["NSE:NIFTY BANK"]))
#print(kite.quote("NSE:NIFTY BANK"))
#print(Quote)
# Function to get data for Index and Underlying
def getlivedata(symbol):
    Quote=kite.quote(symbol)
    for keys,value in Quote.items():
        symbol=symbol
        lastprice=value['last_price']
        Netchange=value['net_change']
        pricestats=value['ohlc']
        Open=pricestats['open']
        High = pricestats['high']
        Low = pricestats['low']
        Close = pricestats['close']
    print('[symbol]:',symbol,'[lastprice]:',lastprice,'[Netchange]:',
          Netchange,'[Open]:',Open,'[High]:',High,'[Low]:',Low,'Close:',Close)
    return
# Take entry , SL or Take Profit while checking for Price

while True:
    getlivedata("NSE:NIFTY BANK")
    if lastprice> 44320:
        print ('Price is high',lastprice)
    getlivedata('NSE:IRCTC')
    if lastprice > 620:
        print('Price is high',lastprice)
    getlivedata('NSE:INFY')
    if lastprice < 1300:
        print('Price is low',lastprice)
    print('sleep5'+'-'*100)
    time.sleep(2)

#print(kite.quote("NFO:BANKNIFTY23JUN44300CE"))

'''
while True:
    print(kite.ltp(["NSE:NIFTY BANK"]))
    print(kite.ltp(["NFO:BANKNIFTY23JUN44300CE"]))
    print(kite.ltp(["NFO:BANKNIFTY23JUN44400CE"]))
    print(kite.ltp(["NFO:BANKNIFTY23JUN44500CE"]))
    data1 = kite.ltp(["NFO:BANKNIFTY23JUN44300CE"])
    data2 = kite.ltp(["NFO:BANKNIFTY23JUN44400CE"])
    data3 = kite.ltp(["NFO:BANKNIFTY23JUN44500CE"])
    #print(data1['NFO:BANKNIFTY23JUN44300CE']['last_price'])
    #print(data2['NFO:BANKNIFTY23JUN44400CE']['last_price'])
    print(data3['NFO:BANKNIFTY23JUN44500CE']['last_price'])
    data3=data3['NFO:BANKNIFTY23JUN44500CE']['last_price']
    time.sleep(5)
    if data3 < 0.10:
        order = kite.place_order(variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NFO,
                                 # exchange=kite.EXCHANGE_NSE,
                                 # tradingsymbol="ACC",
                                 tradingsymbol="BANKNIFTY23JUN44500CE",
                                 transaction_type=kite.TRANSACTION_TYPE_BUY,
                                 quantity=25,
                                 product=kite.PRODUCT_MIS,
                                 order_type=kite.ORDER_TYPE_LIMIT,
                                 price=0.10,
                                 validity=kite.VALIDITY_DAY,
                                 disclosed_quantity=None,
                                 trigger_price=None,
                                 squareoff=None,
                                 stoploss=None,
                                 trailing_stoploss=None,
                                 tag="TradeViaPython")
        print(order)
        break
    else:
        print("price not reached")
while order:
    data3 = kite.ltp(["NFO:BANKNIFTY23JUN44500CE"])
    data3 = data3['NFO:BANKNIFTY23JUN44500CE']['last_price']
    if data3 < 0.05 :
        order = kite.place_order(variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NFO,
                                 # exchange=kite.EXCHANGE_NSE,
                                 # tradingsymbol="ACC",
                                 tradingsymbol="BANKNIFTY23JUN44500CE",
                                 transaction_type=kite.TRANSACTION_TYPE_SELL,
                                 quantity=25,
                                 product=kite.PRODUCT_MIS,
                                 order_type=kite.ORDER_TYPE_LIMIT,
                                 price=0.20,
                                 validity=kite.VALIDITY_DAY,
                                 disclosed_quantity=None,
                                 trigger_price=None,
                                 squareoff=None,
                                 stoploss=None,
                                 trailing_stoploss=None,
                                 tag="TradeViaPython")
        print(order)
        break
    elif data3>0.10:
        order = kite.place_order(variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NFO,
                             # exchange=kite.EXCHANGE_NSE,
                             # tradingsymbol="ACC",
                             tradingsymbol="BANKNIFTY23JUN44400CE",
                             transaction_type=kite.TRANSACTION_TYPE_SELL,
                             quantity=25,
                             product=kite.PRODUCT_MIS,
                             order_type=kite.ORDER_TYPE_LIMIT,
                             price=0.20,
                             validity=kite.VALIDITY_DAY,
                             disclosed_quantity=None,
                             trigger_price=None,
                             squareoff=None,
                             stoploss=None,
                             trailing_stoploss=None,
                             tag="TradeViaPython")
        print(order)
        break
    else:
        print("price not reached")

'''
"""
#print(kite.ltp(["NFO:BANKNIFTY23JUN44400CE"]))
#print(kite.ltp(["NFO:BANKNIFTY23JUN44300CE"]))
#print(type(kite.ltp(["NFO:BANKNIFTY23JUN44300CE"])))
data1=kite.ltp(["NFO:BANKNIFTY23JUN44300CE"])
data2=kite.ltp(["NFO:BANKNIFTY23JUN44400CE"])
data3=kite.ltp(["NFO:BANKNIFTY23JUN44500CE"])
print(data1['NFO:BANKNIFTY23JUN44300CE']['last_price'])
print(data2['NFO:BANKNIFTY23JUN44400CE']['last_price'])
print(data3['NFO:BANKNIFTY23JUN44400CE']['last_price'])
#get_last_price=kite.ltp(["NFO:BANKNIFTY23JUN44300CE"])['last_price']
#print (get_last_price)

# Get Historical data

import datetime
instrument_token = 260105
from_datetime = datetime.datetime.now() - datetime.timedelta(days=7)     # From last & days
to_datetime = datetime.datetime.now()
interval = "15minute"
print(kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False))

# Place Order

# Modify order
kite.modify_order(variety=kite.VARIETY_REGULAR,
                  order_id="order_id",
                  parent_order_id=None,
                  quantity=5,
                  price=200,
                  order_type=kite.ORDER_TYPE_LIMIT,
                  trigger_price=None,
                  validity=kite.VALIDITY_DAY,
                  disclosed_quantity=None)

# Cancel order
kite.cancel_order(variety=kite.VARIETY_REGULAR,order_id="order_id",parent_order_id=None)
"""