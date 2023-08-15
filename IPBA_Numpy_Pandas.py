import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Set Current working Directory
os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")
print(os.getcwd())

# Basic Numpy array
mylist=[30,40,50,60]
list2=[[1,2,3],[10,20,30]]
nparray1=np.array(mylist)
nparr2=np.array(list2)
print(type(mylist),type(nparray1))
print('size',nparray1.size)
print('size',nparr2.size)
print(nparr2.shape)
print(nparray1+10)
print(nparray1*10)
print(nparr2+10)

com=[4,5,6,7]
nparray2=np.array(com)
print('addition',nparray1+nparray2)
print('mult',nparray1*nparray2)
print('divis ',nparray1/nparray2)

# Generate a new range, # last value is not included
RanArr1=np.arange(10)
print(RanArr1)
RanArr2=np.arange(1,10,1)
print(RanArr2)

# Operation on the Np Array
print(RanArr2.mean())
print(RanArr2.max())
print(RanArr2>5)

# Indexing positive and negative
print(RanArr2[4])
print(RanArr2[-1])
print(RanArr2[RanArr2>5])
print(RanArr2[RanArr2<5])

# Reading the file
scores=np.loadtxt('survey_large.txt',dtype='int')
print(scores)
print(scores.size)

print(scores.min())
print(scores.max())
print(scores.mean())

# Detractors
det=scores[scores<=6]
print('detractors',det)
prm=scores[scores>=9]
print('promoters',prm)
print('size',det.size,prm.size)
nps=(prm.size-det.size)/scores.size
print(nps*100)

# Pandas
import pandas as pd
Data=pd.read_csv('mckinsey.csv')
print(Data.to_string)
print(Data.head())
print(Data.tail())
print('size',Data.size,'shape',Data.shape)
print(Data.info())
print(Data.describe())

# Filter the data
#print(Data[[Data['year']==2007],[Data['country']=='Angola']])
df_2007=Data[Data['year']==2007]
print(df_2007.min())
lemin=(df_2007['life_exp'].min())
print(df_2007[df_2007['life_exp']==lemin])
lemax=(df_2007['life_exp'].max())
print(df_2007[df_2007['life_exp']==lemax])

lemin=(df_2007['gdp_cap'].min())
print(df_2007[df_2007['gdp_cap']==lemin])
lemax=(df_2007['gdp_cap'].max())
print(df_2007[df_2007['gdp_cap']==lemax])

# How to deal with categorical data
print(df_2007['country'].value_counts()) # count the number of rows per country
print(df_2007['continent'].value_counts()) # count the number of rows per continent

# How India's Life expectancy and GDP have grown with time
Df_Ind=Data[Data['country']=='India']
print(Df_Ind)
df_yr_gdp=Df_Ind[['year','gdp_cap']]
Df_Ind.plot(x='year',y='gdp_cap',color='red')
plt.title('linePlot')
plt.show()
