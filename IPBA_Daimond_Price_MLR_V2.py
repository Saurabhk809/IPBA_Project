import os
import pandas as pd
import seaborn as sbrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# set the default working directory and read the file contents
try:
    os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
    filename1 = 'test.csv'
    test_data=pd.read_csv(filename1,na_values=['NA','N/A','nan',' '])
    filename2 = 'train.csv'
    train_data=pd.read_csv(filename2,na_values=['NA','N/A','nan',' '])
except:
    FileNotFoundError
    print('File','not present in',os.getcwd())

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

# Print the train Dataset
print(train_data.head())
print(test_data.head())

print(test_data.columns)
print(test_data.shape)
print(train_data.columns)
print(train_data.shape)

col1=['cut','clarity','color']
for col in col1:
    print(train_data[col].unique())

for col in col1:
    print(train_data[col].unique())


Cat_ColList=['cut','color','clarity']
for cols in Cat_ColList:
    dummies1=pd.get_dummies(test_data[cols],prefix=cols, drop_first=True)
    test_data= pd.concat([test_data,dummies1], axis=1)
    test_data=test_data.drop(cols,axis=1)
    dummies2 = pd.get_dummies(train_data[cols], prefix=cols, drop_first=True)
    train_data = pd.concat([train_data, dummies2], axis=1)
    train_data = train_data.drop(cols, axis=1)

print('Test Data with Dummy \n',test_data.head(),'\n',test_data.shape)
print('Train Data with Dummy \n',train_data.head(),'\n',train_data.shape)

Y=train_data['price']
#print('Y is \n',Y)
X=train_data.drop('price',axis=1)
#print('X is \n',X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.80)

# Create the Model and fit it with xtrain and ytrain
Model=LinearRegression()
Model.fit(X_train,y_train)

Model_coef_table = pd.DataFrame(list(X_train.columns)).copy()
Model_coef_table.insert(len(Model_coef_table.columns),"Coefs",Model.coef_.transpose())
print('Model Coeff : \n',Model_coef_table,'\n')
#print('Model Coeff : \n',Model.coef_,'\n')
print('Model Intercept : \n',Model.intercept_,'\n')
y_predict=Model.predict(X_test)
print('Prediction Using Linear regression is :\n ',y_predict)

# Check the Error
print('Mean Square Error :',mean_squared_error(y_test,y_predict))
print('Model Accuracy r2 :',r2_score(y_test,y_predict))

My_predict=Model.predict(test_data)
print('creating the submission file ....')
#submission=pd.DataFrame({'id':X_test.index,'price':y_predict})
submission_df = pd.DataFrame({'Id': test_data['id'], 'y_predict': My_predict})
print(submission_df.head())
print(submission_df.tail())
#submission_df.to_csv("submission.csv", index=False)
#print('submission file .... created !')
