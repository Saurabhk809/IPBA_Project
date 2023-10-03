import os
import pandas as pd
import seaborn as sbrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

import os
import pandas as pd
import seaborn as sbrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filename1 = 'test.csv'
test_data=pd.read_csv(filename1,na_values=['NA','N/A','nan',' '])
filename2 = 'train.csv'
train_data=pd.read_csv(filename2,na_values=['NA','N/A','nan',' '])

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

# Info of Train data
print(train_data.head())
print(train_data.info())

# convert categorical variables to categories
train_data['cut']=train_data['cut'].astype('category').cat.codes
train_data['color']=train_data['color'].astype('category').cat.codes
train_data['clarity']=train_data['clarity'].astype('category').cat.codes
test_data['cut']=test_data['cut'].astype('category').cat.codes
test_data['color']=test_data['color'].astype('category').cat.codes
test_data['clarity']=test_data['clarity'].astype('category').cat.codes

# Descriptive statistic of train_data
print(train_data.describe())
print(train_data.info())

#
Y=train_data['price']
X=train_data.drop('price',axis=1)

# Check the Train and Test
X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.80)
model = LinearRegression()
from sklearn.feature_selection import RFE

reg=LinearRegression()
reg.fit(X_train,y_train)

# Create the RFE object and specify the number of features
selector = RFE(model, n_features_to_select=2)

# Fit the RFE object to the data
selector = selector.fit(X, Y)

# Print the selected features
print(X.columns[selector.support_])
print(reg.support_)
print(train_data.columns,reg.coef_.transpose())
data={'column':train_data.columns,'coeff':reg.coef_.transpose()}
summary=pd.DataFrame(data)
print('Coefficients:',type(reg.coef_),summary)
print('Variance score: {}'.format(reg.score(X, Y)))

# plotting residual errors in test data
plt.scatter(reg.predict(X_test),
            reg.predict(X_test) - y_test,
            color="blue", s=10,
            label='Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

# plotting legend
plt.legend(loc='upper right')
plt.show()