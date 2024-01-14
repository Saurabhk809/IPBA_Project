import os
import pandas as pd
import seaborn as sbrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
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

# Distributed Train & Test Data
Y=train_data['price']
X_train=train_data.drop('price',axis=1)

# convert categorical variables into categories
catcols=['cut','color','clarity']
LE = LabelEncoder()
for cols in catcols:
    X_train[cols]=LE.fit_transform(X_train[cols])
    test_data[cols] = LE.fit_transform(test_data[cols])
    print(X_train[cols].unique())
    print(X_train[cols].head())

print(X_train.info())
print(X_train.head())

# Construct a random forest classifier
random_forest_model = RandomForestRegressor(n_estimators=9, random_state=5)
model = BaggingRegressor(base_estimator=random_forest_model, n_estimators=4, random_state=5)
model.fit(X_train, Y)
print(model.summary())


RFE_predict = model.predict(test_data)
cols=['RFE_Pred']
Final_Predictions=pd.DataFrame(data=RFE_predict.T,columns=cols)
print(Final_Predictions.head())

"""
# List of Importance Features
Features=model.feature_names_in_
#Features=random_forest_model.feature_importances_
Sorted_Feature = np.argsort(Features)[::-1]
print(*X_train.columns[Features], sep = "\n")
"""