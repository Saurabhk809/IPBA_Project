#clustering for loan data based on income

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
import warnings
warnings.filterwarnings("ignore")

path='C:\ProgramData\Anaconda3\Scripts\IPBA_Project'
os.chdir(path)
data=pd.read_csv('clustering.csv')

print(data.head())
data=data[['ApplicantIncome','LoanAmount']]
data=data.fillna(0)
print(data.isnull().sum())
print(data.head())

# Create a loop and find the K from Elbow method

wcss=[]
for i in range(1,10):
    model=KMeans(n_clusters=i)
    model.fit(data)
    wcss.append(model.inertia_)

plt.plot(range(1,10),wcss,marker='o')
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.show()

model1=KMeans(n_clusters=3)
model1.fit(data)
predict=model1.predict(data)
label=model1.labels_

centre=model1.cluster_centers_
plt.scatter(data['ApplicantIncome'][predict==0],data['LoanAmount'][predict==0],label='cluster0')
plt.scatter(data['ApplicantIncome'][predict==1],data['LoanAmount'][predict==1],label='cluster1')
plt.scatter(data['ApplicantIncome'][predict==2],data['LoanAmount'][predict==2],label='cluster2')
plt.scatter(centre[:,0],centre[:,1])
plt.show()
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.legend()


















