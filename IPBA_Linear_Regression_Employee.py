# Script for OLS Ordinary Least Square Method for regression

# Import section for Imports
import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import scipy
from scipy import datasets
import statsmodels.formula.api as smf
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Open the file
try:
    filepath='C:/ProgramData/Anaconda3/Scripts/IPBA_Project/'
    filename='dataset.csv'
    Data=pd.read_csv(filepath+filename)
except:
    FileNotFoundError
    print('File',filename,'not presnt at',filepath)

# Check and clean the data
Data=Data.fillna(0)
head=Data.head()
shape=Data.shape
#print(head)

# Plot withouth regression Line how the data looks like
sbrn.set_style('darkgrid')
palette=['o','b','g']
sbrn.lmplot(x='Sales',y='Employees',hue='Manufacturing',fit_reg=True,data=Data).set(title='SeaBorn LMPlot with Fit')
#plt.show()

# Impact of  All Predictors on Salary withouth any Intercept
Y=Data['Salary']
X=Data[['Sales','Employees','Cap','Manufacturing']]
result = smapi.OLS(Y,X).fit()
print('Impact of All Predictors w/o Intercept ' ,'\n',result.summary())

# Impact of  All Predictors on Salary with  any Intercept
X1=smapi.add_constant(X)
model = smapi.OLS(Y,X1).fit()
print('Impact of All Predictors with Intercept ','\n',model.summary())

# Find which Predictor  is increasing the P Value using Step Wise Approach

Y=Data['Salary']
mydata = pd.DataFrame()
i=0
X3=[]

for cols in Data.columns:                                     # Iterate the columns of DataFrame for predictor
    if cols =='Salary':                                              # Ignore the dependent Y Variable
        pass
    else:
        mydata[cols]=Data[cols]                               # Add the fitted predictor in a dataframe for checking P Vale
        X2=smapi.add_constant(mydata)
        model1 = smapi.OLS(Y, X2).fit()
        for data in list(mydata):                                      # Iterate for loop to check if cols added is adding or reducing p Value
            try:
                if data=='const':                                  # Ignore the constant
                    pass
                else:
                    if model1.pvalues.loc[data] > .055 :    # If P value is higher than 0.5 drop the predictor else preserve it
                        if data in X3:
                            pass
                        else:
                            #print('removed',data,model1.pvalues.loc[data])
                            X2=X2.drop(data,axis=1)
                            mydata=mydata.drop(data,axis=1)
                    else:
                        #print('added', data, model1.pvalues.loc[data])
                        X3.append(data)
                        X4=list(set(X3))
                        X5=mydata[X4]
                        model2 = smapi.OLS(Y, X5).fit()
                        print(model2.summary())
            except:
                ValueError
                #print('Nothing to remove', data, model1.pvalues.loc[data])

#print(X3)
#print(Data[X3])
print('Predictor using stewise method',list(set(X3)))
finalpredictor=list(set(X3))
X4=Data[finalpredictor]
#X4=smapi.add_constant(Data[finalpredictor])
X4=smapi.add_constant(X4)
#print(X4)
model2 = smapi.OLS(Y,X4).fit()
print('Impact of All Predictors with Intercept ' ,'\n',model2.summary())

#Q3 Train the Model with Data Set
X1=Data['Sales']
X2=Data['Employees']
Y=Data['Salary']
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y,test_size=1)
X2_train, X2_test, Y1_train, Y1_test = train_test_split(X2,Y,test_size=1)
#print(X1_train,type(X1_train))
#print(X2_train,type(X2_train))
#print(X1_test)
#print(X2_test)

TestData=pd.DataFrame(X1_train,X2_train)
TrainData=pd.DataFrame(X1_train,X2_train)
#print(TrainData.ndim)
#print(type(TestData))
#print(type(TrainData))
result=model2.predict(X4)
#print('prediction result \n',result)

## Find which Predictor  is increasing the Accuracy using a Forward Value Approach
Y=Data['Salary']
X=Data[['Sales','Employees','Cap','Manufacturing']]

#print(Y.shape)
#print(X.shape)

# check for Null data
#Y.isnull()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5,random_state=80)
print(X_train.shape)
print(Y_train.shape)

#try:
forward_selector=SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),cv='',k_features=2,forward=True,floating=False,
    verbose=2,scoring="accuracy").fit(X,Y)
print(forward_selector.k_feature_idx_)
print(forward_selector.k_feature_names_)

#except:
    # ValueError

model5=RandomForestClassifier(n_estimators=40)
model5.fit(X,Y)
print('Model Score is ','\n',model5.score(X_test,Y_test))

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
clf = LinearRegression()

# Build step forward feature selection
sfs1 = sfs(clf,k_features = 2,forward=True,floating=False, scoring='r2',cv=5)

# Perform SFFS
sfs1 = sfs1.fit(X_train, Y_train)

# Create a linear regression estimator
estimator = LinearRegression()

from sklearn.feature_selection import RFE
# splitting X and y into training and testing sets
X_train, X_test,y_train, y_test = train_test_split(X, Y,test_size=0.4,random_state=1)

reg=LinearRegression()
reg.fit(X_train,y_train)


# Create the RFE object and specify the number of features
selector = RFE(estimator, n_features_to_select=2)

# Fit the RFE object to the data
selector = selector.fit(X, Y)

# Print the selected features
print(X.columns[selector.support_])
print('Coefficients:',reg.coef_)
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

# plot title
plt.title("Residual errors")

# method call for showing the plot
plt.show()

# calculating VIF for each feature

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

print(vif_data)

#
