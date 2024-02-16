import os
import pandas as pd
import sweetviz as sv
import seaborn as sbrn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import fpdf
import sweetviz as sv
from pandas import Series,DataFrame
import statistics

import warnings
warnings.filterwarnings('ignore')

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

filename='timestamp.csv_result_with_wait.csv'
os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")
Data=pd.read_csv(filename,na_values=['na','NAN',' '])
for cols in Data.columns:
    Data[cols] = pd.to_numeric(Data[cols], errors='coerce')
    Data[cols] = Data[cols].fillna(Data[cols].mean())

# Check Data using sv
sv_report = sv.analyze(Data)
sv_report.show_html('sv_latency_report.html')


# Latency Description
Result=pd.DataFrame()
for cols in Lat_Data.columns:
    Desc=DataFrame(Data[cols].describe())
    Desc.loc['median']=statistics.median(Data[cols])
    Desc.loc['95%']=Data[cols].quantile(0.95)
    Q1=np.percentile(Data[cols],25)
    Q3=np.percentile(Data[cols],75)
    Desc.loc['Q1']=Q1
    Desc.loc['Q3']=Q3
    IQR=Q3-Q1
    Desc.loc['IQR']=IQR
    UL=Q3+(1.5*IQR)
    LL = Q1 - (1.5 * IQR)
    Desc.loc['UL']=Q3+(1.5*IQR)
    Desc.loc['LL']=Q1 - (1.5 * IQR)
    UL_Outliers = Data[Data[cols] > UL][cols]
    #print('outliers',cols,UL_Outliers)
    LL_LowLiers = Data[Data[cols] < LL][cols]
    Per_UL_Outliers=UL_Outliers.count() / Data[cols].count() * 100
    Per_LL_Outliers = LL_LowLiers.count() / Data[cols].count() * 100
    Desc.loc['%_UL_Outliers']=Per_UL_Outliers
    Desc.loc['%_LL_Outliers']=Per_LL_Outliers
    Result=pd.concat([Result,Desc],axis=1)

Result.to_html('Result_Description.html')
Result.to_csv('Result.csv',sep=',', encoding='utf-8', header='true')

palette={'green','plum','orange','purple'}
sbrn.displot(data=Out_Data,rug=True,bins=50)
plt.title('Displot of outbound Latency')
plt.tight_layout(pad=3.0)
plt.savefig('outbound Latency.webp')
#plt.savefig('Displot of outbound Latency')
plt.xlabel('Latency')
plt.show()

sbrn.displot(data=In_Data,rug=True,bins=50)
plt.title('Displot of Inbound Latency')
plt.tight_layout(pad=3.0)
plt.savefig('Inbound Latency.webp')
plt.xlabel('Latency')
plt.show()
