import pandas as pd
import numpy as np
import seaborn as sbrn
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn.metrics as mtrcs
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import os

data_dir='C:/ProgramData/Anaconda3/Scripts/IPBA_Project/'
os.chdir(data_dir)
filename='dm.csv'

# Open The file
try:
    Data=pd.read_csv(filename,na_values=[""," ","NA","N/A"])
except:
    FileNotFoundError
    print('File',filename,'is not present at' ,os.getcwd())
    NameError
    print('File name is not correct ')

# Check the Data
#print(Data.head)
#print(Data.shape)
#print(Data.dtypes,Data.columns)

# Problem Statement Find out People who are good Customers so to focus on them and not good Customers
# Assume people who spend more than Average are good customers vs not so good Customers

# Lambda derivation for adding the columns
Amnt_Mean=Data['AmountSpent'].mean()
for rows in Data['AmountSpent']:
    if rows > Amnt_Mean:
        pass
        #print (rows,Amnt_Mean, 'good customer 1')
    else:
        pass
        #print(rows,Amnt_Mean, 'bad customer 0')

# Create a Target Data Column to Map Average Good , Bad Customer
Data['Target']=Data['AmountSpent'].map(lambda x:1 if x>Data['AmountSpent'].mean() else 0)

# Drop AmountSpent as it is considered in Target Variable
Data=Data.drop("AmountSpent",axis=1)
#print(Data.head)

# Check for Categorical Variables for missing Data
print(Data['History'].value_counts(),Data['History'].isnull().sum())
Data['History']=Data['History'].fillna('NewCust')
#print(Data.head)

# Split the Data in Train and Test
data_train=Data.sample(frac=0.70,random_state=200)
data_test=Data.drop(data_train.index)

# Build the Logistic Model and check its Summary
model=smf.glm('Target~C(Age)+C(Gender)+C(OwnHome)+C(Married)+C(Location)+Salary+Children+C(History)+Catalogs',data=data_train,family=sm.families.Binomial()).fit()
print(model.summary())
print('Model AIC :',model.aic)

# Variables to remove
# Age,Gender,OwnHome,Married,History[T.low,T.NewCust]
# History need to map for T.Medium

data_train['Hist_Low']=data_train['History'].map(lambda x: 1 if x=="Low" else 0)
data_test['Hist_Low']=data_test['History'].map(lambda x: 1 if x=="Low" else 0)
data_train['Hist_Med']=data_train['History'].map(lambda x: 1 if x=="Medium" else 0)
data_test['Hist_Med']=data_test['History'].map(lambda x: 1 if x=="Medium" else 0)

model2=smf.glm('Target~Children+Catalogs+Salary+Hist_Med',data=data_train,family=sm.families.Binomial()).fit()
print(model2.summary())
print('Model AIC :',model2.aic)
# AIC is likelihood fit for the model

# Lets Check Confusion Matrix and AUC - Area Under the curve
y_actual=data_test['Target']
y_pred=model2.predict(data_test)
#print (list(y_pred))

# set the probability % cut to classify probabilities in 1 and 0

y_pred=model2.predict(data_test).map(lambda x:1 if x>0.5 else 0)
#print (list(y_pred))
Metric=metrics.confusion_matrix(y_actual,y_pred)
print('Confusion Metric is \n',Metric)

# ROC Curve to check the tpr sensitivity 1 and fpr 1-specificity 0
y_score=model2.predict(data_test)
#FPR is 1-specificity , TPR is sensitivity both should be maximised
fpr,tpr,thresholds=metrics.roc_curve(y_actual,y_score)
x,y=np.arange(0,1.1,0.1),np.arange(0,1.1,0.1)

plt.plot(fpr,tpr,"-")
plt.plot(x,y,'b--')
plt.show()

# AUC, Average accuracy of my model , compares  the confusion metric way of comparing models
print('Area under curve is',metrics.roc_auc_score(y_actual,y_score))


# Find the list of Customer whom we should Target
data_test['prob']=model2.predict(data_test)
data_test['prob'].head()
print(data_test.sort_values("prob",ascending=False)[['Cust_Id']].head(90))## These are the people to target


# Find cumulative Gains from events
data_test['prob_deciles']=pd.qcut(data_test['prob'],q=10)
data_test.sort_values('prob',ascending=False).head()
gains=data_test.groupby("prob_deciles",as_index=False)['Target'].agg(['sum','count']).reset_index().sort_values("prob_deciles",ascending=False)
#print(list(gains))
#print(list(gains.columns))
gains.columns=["index","Deciles","TotalEvents","NumberObs"]
gains["PercEvents"]=gains['TotalEvents']/gains['TotalEvents'].sum()
gains["CumulativeEvents"]=gains.PercEvents.cumsum()
print('Percent Event',gains['PercEvents'])
print('Cumulative Events',gains['CumulativeEvents'])







