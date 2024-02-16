# Script to generate a result format from latency output
#Author Saurabh Kamble : Date: 08-09-2023

#Import section
import os
import pandas as pd
import sweetviz as sv
import seaborn as sbrn
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series,DataFrame
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import logging

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")

#LogFile for logging
logfile='Result.log'
logging.basicConfig(filename=logfile,filemode='w',level=logging.INFO)
log=logging.getLogger()

# Read the file or throw an error
filename='timestamps.csv_result_with_wait.csv'
def file_read(filename):
    try:
        Data=pd.read_csv(filename,na_values=['na','NAN',' '])
        return Data
    except FileNotFoundError:
        print('File :',filename,'not found at :/n',os.getcwd())
        log.info('File :',filename,'not found at :/n',os.getcwd())


def clean_data(Data):
    print(Data.head())
    #log.info(Data.head())
    print(Data.isnull().sum())
    #log.info(Data.isnull().sum())
    Data.fillna(0,inplace=True)

def encode_data(Data):
    for cols in Data:
        Data[cols] = pd.to_numeric(Data[cols], errors='coerce')
        Data[cols] = Data[cols].fillna(0,inplace=True)
        print(Data.head())
        log.info(Data.head())
        print(Data.isnull().sum())
        log.info(Data.isnull().sum())
        return Data

    # Data Preparation for EDA
def prep_Analyse_data(Data):
    pass


def main():
    import seaborn as sbrn
    Data=file_read(filename)
    clean_data(Data)
    encode_data(Data)
    #Lat_Data, Out_Data, In_Data=prep_Analyse_data(Data)
    print(type(Out_Data),Out_Data)
    sbrn.displot(data=Out_Data, rug=True, bins=50)
    plt.title('Displot of outbound Latency')
    plt.tight_layout(pad=3.0)
    plt.savefig('Displot of outbound Latency')
    plt.show()
    sbrn.displot(data=In_Data, bins=50,rug=True)
    plt.title('Displot of Inbound Latency')
    plt.tight_layout(pad=3.0)
    plt.savefig('Displot of Inbound Latency')
    plt.show()


if __name__ == '__main__':
    main()
