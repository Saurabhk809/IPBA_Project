import os
import pandas as pd
import numpy as np

osfilename='Bank Data running case study.xlsx'
osfilepath="C://Users//E1077195"

file_to_use=osfilepath+osfilename
print(file_to_use)
#load the data from the excel
data=pd.read_excel("C:\\Users\\E1077195\\Bank Data running case study.xlsx")
data.head(2)



