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

data=dataloading('CLV.csv')
print(data.head())
print(data.dtypes)
print(data.describe())
print('duplicated',data.duplicated().sum())

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 10):
    model = KMeans(n_clusters=i)
    model.fit(data)
    wcss.append(model.inertia_) # Inertia is the sum of square of the distance between the data points and its centroid
    print("cluster:",i,'SSE',model.inertia_)

plt.plot(range(1, 10), wcss, marker="o")
plt.xlabel('K means Clusters')
plt.ylabel('Inertia:Within Cluster sum of square/ Distortion')
plt.show()

model1=KMeans(n_clusters=3)
model1.fit(data)
predict=model1.predict(data)
print('model prediction for labels:\n',predict)
print('model property for label:\n',model1.labels_)
data['Label']=pd.Series(model1.labels_) # Each Data Sample is assigned a label
print(data.head())

centre=model1.cluster_centers_ # centroid of each cluster with x & y cordinates
#data=data.values
"""
plt.scatter(data[:,0][predict==0],data[:,1][predict==0],label="cluster 0")
plt.scatter(data[:,0][predict==1],data[:,1][predict==1],label="cluster 1")
plt.scatter(data[:,0][predict==2],data[:,1][predict==2],label="cluster 2")
plt.scatter(centre[:,0],centre[:,1],marker="D")
"""
plt.scatter(data['INCOME'][predict==0],data['SPEND'][predict==0],label="cluster 0")
plt.scatter(data['INCOME'][predict==1],data['SPEND'][predict==1],label="cluster 1")
plt.scatter(data['INCOME'][predict==2],data['SPEND'][predict==2],label="cluster 2")
plt.scatter(centre[:,0],centre[:,1],marker="D") # X and Y columns for all 3 clusters 0,1,2

plt.xlabel("Income")
plt.ylabel("Spend")
plt.legend()
plt.show()

# To validate the K means between cluster label and user defined labels (#user defined lables should be present in data)
cross_tab=pd.crosstab(data['Label'],data['Label'])
print(cross_tab)


"""
data.columns=model1.feature_names
data['Actual']=x1.target
data['predicted']=model1.labels_
dict(zip(data['Actual'].unique(),data['predicted'].unique()))
"""

# Hierarchical Clustering
import numpy as np
#x=np.array([[5,3],[15,23],[45,3],[85,3]])
x = np.array([[5, 3],[5, 23],[45, 3],[85, 3],[5, 53],[95, 35],[16, 13],[25, 31],[35, 31],[35, 32]])
print(x.shape)
labels=range(1,12)
plt.scatter(x[:,0],x[:,1])
for labels,X,y in zip(labels,x[:0],x[:,1]):
    plt.annotate(labels,xy=(X,y))

from scipy.cluster.hierarchy import linkage,dendrogram
l1=linkage(x,method="complete")
print('linkage \n',l1)
dendrogram(l1)
plt.show()


# AgglomerativeClustering
from sklearn import cluster
print(dir(cluster))
from sklearn.cluster import AgglomerativeClustering

model2=AgglomerativeClustering
model2.fit(data.iloc[:,:-2])
AgglomerativeClustering(n_clusters=3)
print('model2',model2.labels_)


"""
#create a cluster model
import sklearn.cluster as cluster
kmeans=cluster.KMeans(n_clusters=3,init="k-means++")
kmeans=kmeans.fit(data)

print(kmeans.labels_)
print(kmeans.cluster_centers_)

# Create Elbow plot
from scipy.spatial.distance import  cdist
K=range(1,10)
wss=[]
for k in K:
    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(data)
    wss.append(sum(np.min(cdist(data,kmeans.cluster_centers_,'euclidean'),axis=1))/data.shape[0])

plt.plot(K,wss,'bx')
plt.xlabel('k')
plt.ylabel('Avg Distortion')
plt.title('selecting K with Elblow Method')
plt.show()

from sklearn import datasets
from sklearn.datasets import  load_iris
x1=load_iris()
print(x1.keys())
data=x1.data
"""