import os
import matplotlib.pyplot as plt
import seaborn as sbrn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import sklearn.model_selection as model_selection
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smfi
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import  variance_inflation_factor
import statsmodels.api as sm
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import datetime as dt
import statistics
import sweetviz as sv
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
warnings.filterwarnings("ignore")
from mlxtend.feature_selection import SequentialFeatureSelector
import statsmodels.api as smapi
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
#change the directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filepath=os.getcwd()

def dataloading(file1):
    try:
        os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
        data=pd.read_csv(file1,na_values=[' ','NA','NULL'])
        #test_data=pd.read_csv(file2,na_values=[' ','NA','NULL'])
        #set the display options
        pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
        #return train_data,test_data
        return  data
    except:
        FileNotFoundError
        print('File not present in filepath',filepath)

data=dataloading('kc_housingdata.csv')
print(data.head())

print(data.dtypes)
data_num=data[['price','bedrooms','bathrooms','sqft_living']]

# scale the data
def scale(x):
    return (x-np.mean(x)/np.std(x))
data_scaled=data_num.apply(scale,axis=0)

print(data_scaled.head())

#Another way of doing the same is
import sklearn.preprocessing as preprocessing
dat_scaled=preprocessing.scale(data_num,axis=0)

print(data_scaled)
print('Type of Output is '+str(type(dat_scaled)))
print('Shape of the object is '+str(dat_scaled))

#create a cluster model
import sklearn.cluster as cluster
kmeans=cluster.KMeans(n_clusters=3,init="k-means++")
kmeans=kmeans.fit(dat_scaled)

print(kmeans.labels_)
print(kmeans.cluster_centers_)

# Create Elbow plot
from scipy.spatial.distance import  cdist
K=range(1,20)
wss=[]
for k in K:
    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(dat_scaled)
    wss.append(sum(np.min(cdist(dat_scaled,kmeans.cluster_centers_,'euclidean'),axis=1))/dat_scaled.shape[0])

plt.plot(K,wss,'bx')
plt.xlabel('k')
plt.ylabel('Avg Distortion')
plt.title('selecting K with Elblow Method')
plt.show()






