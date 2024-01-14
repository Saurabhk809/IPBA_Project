# Script to connect to Zerodha using API
import warnings
warnings.filterwarnings('ignore')
from kite_trade import *
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sbrn
import numpy as np

#from scipy.stats import f_oneway
#import backtrader as bt

enctoken="lgE0klojQQgw0lSJYLpMFAsXM660QQxXi0WhuIKTN7OFyQCrqpbtqvKW6ES0ZLHNKPBJm0sPHU+VePBZfIcWxUneaT0cEomw0wX0USIj6kYVOKCdKhbU7w=="
# created a SSL less session by making change at   line number 73 self.session.get(self.root_url, headers=self.headers,verify=False)
kite=KiteApp(enctoken=enctoken)

class Zerodha_Trade:
    def __init__(self,symbol,entry_price,stoploss,exit_price,segment):
        self.symbol=symbol
        self.entry_price=entry_price
        self.stoploss=stoploss
        self.exit_price=exit_price
        self.segment=segment
    def Get_Prices(self):
        my_symbol_dic = {}
        Quote = kite.quote(self.symbol)
        for keys, value in Quote.items():
            my_symbol_dic['symbol'] = self.symbol
            my_symbol_dic['lastprice'] = value['last_price']
            my_symbol_dic['Netchange'] = value['net_change']
            pricestats = value['ohlc']
            my_symbol_dic['Open'] = pricestats['open']
            my_symbol_dic['High'] = pricestats['high']
            my_symbol_dic['Low'] = pricestats['low']
            my_symbol_dic['Close'] = pricestats['close']
        return my_symbol_dic
    def order_treatment_eqty(self,segment,qty,side,product):
        while True:
            #myobj=self
            NSE_Price_Stat= NSE_Eqty.Get_Prices()
            #print(NSE_Price_Stat)
            Entry_Price = 153.90
            StopLoss = 153.50
            Exit_Price = 154
            if NSE_Price_Stat['lastprice'] < Entry_Price:
                Order = NSE_Eqty.order_action(NSE_Price_Stat['symbol'], Entry_Price, StopLoss,
                                              Exit_Price,segment,qty,side,product)
                print('Entry success entry price', Order)
                break
            else:
                print('Wait,','Entry Price :',Entry_Price,NSE_Price_Stat)
            time.sleep(5)
        while Order:
            side=kite.TRANSACTION_TYPE_SELL
            NSE_Price_Stat = NSE_Eqty.Get_Prices()
            if NSE_Price_Stat['lastprice'] < StopLoss:
                Order = NSE_Eqty.order_action(NSE_Price_Stat['symbol'], Entry_Price, StopLoss,
                                              Exit_Price,segment,qty,side,product)
                print('StopLoss taken', Order, 'loss=', NSE_Price_Stat['lastprice'] - StopLoss)
                break
            elif NSE_Price_Stat['lastprice'] > Exit_Price:
                Order = NSE_Eqty.order_action(NSE_Price_Stat['symbol'], Entry_Price, StopLoss,
                                              Exit_Price,segment,qty,side,product)
                print('Entry success entry price', Order)
                break
            else:
                print('Order waiting SL or Exit', Order, NSE_Price_Stat['lastprice'], Entry_Price,
                      Exit_Price)
            time.sleep(5)
    def order_treatment_FO(self,segment,qty,side,product):
        while True:
            #myobj=self
            NSE_Price_Stat= NSE_FO.Get_Prices()
            #print(NSE_Price_Stat)
            Entry_Price = 153
            StopLoss = 152.85
            Exit_Price = 154
            if NSE_Price_Stat['lastprice'] < Entry_Price:
                Order = NSE_FO.order_action(NSE_Price_Stat['symbol'], Entry_Price, StopLoss,
                                              Exit_Price,segment,qty,side,product)
                print('Entry success entry price', Order)
                break
            else:
                print('Wait,','Entry Price :',Entry_Price,NSE_Price_Stat)
            time.sleep(5)
        while Order:
            side=kite.TRANSACTION_TYPE_SELL
            NSE_Price_Stat = NSE_FO.Get_Prices()
            if NSE_Price_Stat['lastprice'] < StopLoss:
                Order = NSE_FO.order_action(NSE_Price_Stat['symbol'], Entry_Price, StopLoss,
                                              Exit_Price,segment,qty,side,product)
                print('StopLoss taken', Order, 'loss=', NSE_Price_Stat['lastprice'] - StopLoss)
                break
            elif NSE_Price_Stat['lastprice'] > Exit_Price:
                Order = NSE_FO.order_action(NSE_Price_Stat['symbol'], Entry_Price, StopLoss,
                                              Exit_Price,segment,qty,side,product)
                print('Entry success entry price', Order)
                break
            else:
                print('Order waiting SL or Exit', Order, NSE_Price_Stat['lastprice'], Entry_Price,
                      Exit_Price)
            time.sleep(5)
    def order_action(self,symbol,entry_price,stoploss,exit_price,segment,qty,side,product):
        order = kite.place_order(variety=kite.VARIETY_REGULAR,
                                 exchange=segment,
                                 #exchange=kite.EXCHANGE_NFO
                                 #exchange=kite.EXCHANGE_NSE,
                                 tradingsymbol=self.symbol,
                                 #tradingsymbol="BANKNIFTY23JUN44500CE",
                                 transaction_type=side,
                                 quantity=qty,
                                 product=product,
                                 order_type=kite.ORDER_TYPE_LIMIT,
                                 price=entry_price,
                                 validity=kite.VALIDITY_DAY,
                                 disclosed_quantity=None,
                                 trigger_price=None,
                                 squareoff=None,
                                 stoploss=None,
                                 trailing_stoploss=None,
                                 tag="TradeViaPython")
        #print('Entry success entry price',self.symbol,entry_price,order)
        return order


# Initialise the Object of the instrument
NSE_Eqty=Zerodha_Trade('NSE:KALYANKJIL',151.10,150.70,151.20,'EXCHANGE_NSE')
#Bnifty=Zerodha_Trade('NSE:NIFTY BANK',1,1,1,'EXCHANGE_NSE')
#NSE_FO=Zerodha_Trade('NFO:BANKNIFTY2370645400CE',30,10,100,'EXCHANGE_NFO')

# Get the data and compare with trade params
# exchange=kite.EXCHANGE_NFO
# exchange=kite.EXCHANGE_NSE,
#kite.PRODUCT_MIS.kite.PRODUCT_CNC

Qty,side,product,segment=1,kite.TRANSACTION_TYPE_BUY,kite.PRODUCT_MIS,kite.EXCHANGE_NSE
NSE_Eqty.order_treatment_eqty(segment,Qty,side,product)
#Qty,side,product,segment=25,kite.TRANSACTION_TYPE_BUY,kite.PRODUCT_MIS,kite.EXCHANGE_NFO
#NSE_FO.order_treatment_FO(segment,Qty,side,product)
