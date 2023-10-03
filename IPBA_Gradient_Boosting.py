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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

#pd.options.display.float_format = '{:,.4f}'.format
#pd.set_option('display.float_format',lambda x:'%.4f' %x)

pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
#change the directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filepath=os.getcwd()

train_data=pd.read_csv('train_hp.csv')
test_data=pd.read_csv('test_hp.csv')
train_data.drop_duplicates(keep=False, inplace=True)


"""
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
train_data=scaler.fit_transform((train_data))
X=train_data.drop('price',axis=1)
Y=np.log(train_data['price'])

"""

def IQR_UL_LL(train_data):
    pdf_IQR, pdf_Result = pd.DataFrame(), pd.DataFrame()
    for columns in train_data:
        if train_data[columns].isnull().sum()==0:
            pass
        else:
            train_data[columns].fillna(0)
        if columns not in ['id','made']:
            pdf_IQR = (train_data[columns].describe())
            pdf_IQR['median'] = statistics.median(train_data[columns])
            pdf_IQR['mode']=statistics.mode(train_data[columns])
            pdf_IQR['Q1']=np.quantile(train_data[columns],.25)
            pdf_IQR['Q3'] = np.quantile(train_data[columns],.75)
            pdf_IQR['95%'] = np.quantile(train_data[columns],.25)
            pdf_IQR['IQR']=pdf_IQR['Q3']-pdf_IQR['Q1']
            UL=pdf_IQR['Q3']+(1.5)*pdf_IQR['IQR']
            #print (columns,UL)
            LL=pdf_IQR['Q1']-(1.5)*pdf_IQR['IQR']
            #train_data[columns]=np.where(train_data[columns]>UL,train_data[columns].median(),train_data[columns])
            #train_data[columns] = np.where(train_data[columns]<LL,train_data[columns].median(), train_data[columns])
            train_data=train_data.drop(train_data[train_data[columns]>UL].index)
            train_data=train_data.drop(train_data[train_data[columns]<LL].index)
            pdf_IQR['min2']=train_data[columns].min()
            pdf_IQR['max2'] = train_data[columns].max()
            pdf_IQR['UL_count']=train_data[train_data[columns] > train_data[columns].median()][columns].count()
            pdf_IQR['LL_count'] = train_data[train_data[columns] < train_data[columns].median()][columns].count()
            pdf_IQR['UL_Per'] = pdf_IQR['UL_count']/train_data[columns].count() * 100
            pdf_IQR['LL_Per'] = pdf_IQR['LL_count']/train_data[columns].count() * 100
            #print(pdf_IQR.head(20))
            pdf_Result = pd.concat([pdf_Result,pdf_IQR], axis=1)
        else:
            pass
    #print('Train_Data_Descriptive Stats :\n',pdf_Result.head(20))
    pdf_Result.to_html('Descriptve_Stats_IQR_Report.html')
    return  pdf_Result,train_data

IQRreport,train_data=IQR_UL_LL(train_data)

#IQR_UL_LL(test_data)


#X=train_data[['squareMeters','numberOfRooms','floors','cityPartRange','made','numPrevOwners','isNewBuilt','basement','garage','hasGuestRoom']]
#test_data=test_data[['squareMeters','numberOfRooms','floors','cityPartRange','made','numPrevOwners','isNewBuilt','basement','garage','hasGuestRoom']]

train_data['squareMeters']=np.log(train_data['squareMeters'])
train_data['made']=np.log(train_data['made'])
train_data['price']=np.log(train_data['price'])
train_data['basement']=np.log(train_data['basement'])
train_data['numberOfRooms']=((train_data['numberOfRooms'])/10)
test_data['numberOfRooms']=((test_data['numberOfRooms'])/10)
test_data['basement']=np.log(test_data['basement'])
test_data['squareMeters']=np.log(test_data['squareMeters'])
test_data['cityCode']=np.log(test_data['cityCode'])
test_data['attic']=np.log(test_data['attic'])


#pdf_Result,train_data=IQR_UL_LL(train_data)
train_data['made'].to_csv('made1.csv')
train_data['made']=np.where(train_data['made'] ==10000,2021,train_data['made'])
train_data = train_data.drop(train_data[train_data.made < 7.62].index)
"""
train_data = train_data.drop(train_data[train_data.squareMeters < 10.71].index)
train_data = train_data.drop(train_data[train_data.price < 15.51].index)
train_data = train_data.drop(train_data[train_data.price > 15.60].index)
train_data = train_data.drop(train_data[train_data.cityPartRange < 2].index)
train_data = train_data.drop(train_data[train_data.cityPartRange > 9].index)
#train_data = train_data.drop(train_data[train_data.attic < 10000].index)
"""
train_data=train_data.sort_values(by='made')

sbrn.lineplot(data=train_data,x='made',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='squareMeters',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='numPrevOwners',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='numberOfRooms',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='floors',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='cityCode',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='cityPartRange',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='basement',y='price')
plt.show()

sbrn.lineplot(data=train_data,x='attic',y='price')
plt.show()


Y=train_data['price']
X=train_data[['squareMeters','made','numPrevOwners','cityPartRange']]
test_data=test_data[['squareMeters','made','numPrevOwners','cityPartRange']]

#Y=(train_data['price'])
#X=train_data['squareMeters']
#X=train_data.drop('price',axis=1)

#X=train_data[['squareMeters','made']]
Xtrain,Xtest,Ytrain,Ytest=model_selection.train_test_split(X,Y,test_size=0.10,random_state=100)

print(train_data.head())
print(test_data.head())
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor  # Specifying which Base learner should be used
Bgr = BaggingRegressor(oob_score=True, n_estimators=90, random_state=300, base_estimator=DecisionTreeRegressor())
Bgr.fit(Xtrain,Ytrain)
print(Bgr.base_estimator)
#Xtest=np.array([Xtest]).reshape(-1,1)
#print('Bgr:Training Score:',Bgr.score(Xtrain,Ytrain))
prediction=Bgr.predict(Xtest)
prediction=np.exp(prediction)
#print('Bgr:Testing Score:',Bgr.score(Xtest,Ytest))
print('Bgr:RMSE:',sqrt(mean_squared_error(Ytest,prediction,squared=False)))

from sklearn.tree import DecisionTreeRegressor  # Specifying which Base learner should be used

Model=DecisionTreeRegressor()
Model.fit(Xtrain,Ytrain)

prediction=Model.predict(Xtest)
prediction=np.exp(prediction)
#print(prediction)
print('DT:RMSE:',sqrt(mean_squared_error(prediction,Ytest,squared=False)))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
#for i in range(10,1000,100):
model=GradientBoostingRegressor(n_estimators=100,random_state=500,max_depth=1)
model.fit(Xtrain,Ytrain)
prediction=model.predict(Xtest)
prediction=np.exp(prediction)
print('GB:RMSE:',sqrt(mean_squared_error(prediction,Ytest,squared=False)))
print('GBR:Training Score:',model.score(Xtrain,Ytrain))

#

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

'''
"""
for items in mylist:
    sbrn.relplot(data=train_data,y='price',x=items)
    plt.title('price vs square meters')
    plt.show()

train_data['made']=pd.to_datetime(train_data['made']).dt.year
sbrn.relplot(data=train_data,y='price',x='made')
plt.title('price vs made')
plt.show()

sbrn.relplot(data=train_data,y='price',x='floors')
plt.title('price vs floors')
plt.show()
"""

pipeline = Pipeline([('std_scalar', StandardScaler())])
Xtrain = pipeline.fit_transform(Xtrain)
Xtest = pipeline.transform(Xtest)

# Import the gradient boost
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
#for i in range(10,1000,100):
model=GradientBoostingRegressor(n_estimators=100,random_state=500,max_depth=1)
mod=GridSearchCV(model,param_grid={'n_estimators':[60,80,100,120,140,60]})
mod.fit(Xtrain,Ytrain)
print(mod.best_estimator_)
#Xtrain=np.array([Xtrain]).reshape(-1,1)
#Ytrain=np.array([Ytrain]).reshape(-1,1)
model.fit(Xtrain,Ytrain)
print('GBR:Training Score:',model.score(Xtrain,Ytrain))
#Xtest=np.array([Xtest]).reshape(-1,1)
prediction=model.predict(Xtest)
print('GBR:Testing Score:',model.score(Xtest,Ytest))
print('GBR:RMSE:',np.sqrt(mean_squared_error(prediction,Ytest,squared=False)))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

regressor = RandomForestRegressor(n_estimators=1000,max_depth=6)
#Xtrain=np.array([Xtrain]).reshape(-1,1)
#Ytrain=np.array([Ytrain]).reshape(-1,1)
regressor.fit(Xtrain, Ytrain)
#Xtest=np.array([Xtest]).reshape(-1,1)
prediction=regressor.predict(Xtest)
#print('RF:Training Score:',regressor.score(Xtrain,(Ytrain)))
prediction=regressor.predict(Xtest)
#print('RF:Testing Score:',regressor.score(Xtest,Ytest))
print('RF:RMSE:',np.sqrt(mean_squared_error(prediction,Ytest,squared=False)))

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor  # Specifying which Base learner should be used
#Xtrain=np.array([Xtrain]).reshape(-1,1)
#Ytrain=np.array([Ytrain]).reshape(-1,1)
# for i in range(1, 2, 1):

Ratios=pd.DataFrame()
Ratios['PrcSqMt']=train_data['price']/train_data['squareMeters']
Ratios['PrcSqMade']=train_data['price']/train_data['made']
Ratios['PrcFloor']=train_data['price']/train_data['floors']

#print('Ratios:\n',Ratios.head())
#mylist=['squareMeters','cityPartRange','numPrevOwners','basement']
#train_data['price']=np.log(train_data['price'])
#train_data['squareMeters']=np.log(train_data['squareMeters'])

'''

#print('*******',i,'********'*20)



























