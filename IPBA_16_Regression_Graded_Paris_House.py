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
pd.set_option('display.float_format',lambda x:'%.4f' %x)
#pd.options.display.float_format = '{:,.2f}'.format

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')

train_data=pd.read_csv('train.csv',na_values=[' ','NA','NULL'])
test_data=pd.read_csv('test.csv',na_values=[' ','NA','NULL'])

#set the display options
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

#check the data
#print(train_data.head())
#print(test_data.head())

#check the Data Types
#print(train_data.info())

# Check the Null or NA
#print(train_data.isnull().sum())

# Check the data description
#print(train_data.describe())

Y=train_data['price']
X=train_data.drop(['price'],axis=1)
print(X.head(),Y.head())


#Visulaise the data
#sbrn.lmplot(data=train_data,x='squareMeters',y='price')
#plt.show()

# Build the train & test data set
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.2,random_state=2)

# OLS model
# Add a constant #x=sm.add_constant(X_train)

#Fit the model
def mymodel(xtrain,ytrain):
    model=sm.OLS(xtrain,ytrain).fit()
    print(model.summary())
    return model

# Call prediction
def Prediction(mode,xtest,ytest):
    prediction = model.predict(xtest)
    actual = ytest

    # Find Root Mean Square Error
    rms = sqrt(mean_squared_error(actual, prediction, squared=False))
    print('Root Mean Square from OLS Iteration:'':', rms)
    return prediction

#call the model
model=mymodel(Y_train,X_train)
Prediction(model,X_test,Y_test)

d,j={},{}
for i in X_train.columns.tolist():
    #d[i]=model.pvalues[i]
    d[i]='{:.10f}'.format(model.pvalues[i])
#df_pvalue= pd.DataFrame(d.items(), columns=['Var_name', 'p-Value']).sort_values(by = 'p-Value',ascending=False).reset_index(drop=True)
df_pvalue= pd.DataFrame(d.items(), columns=['OLS_Var_name', 'p-Value'])
df_pvalue['p-Value']=df_pvalue['p-Value'].astype(float)
df_pvalue.sort_values('p-Value',ascending=True,inplace=True)
#print(df_pvalue)

#Important features having p values less than 0.05, significant variables
FSet=train_data[['squareMeters','numberOfRooms','cityCode','made','hasStormProtector','basement','garage']]
#print(FSet.shape)

# Features having p value greater than .05
NonFset=train_data[['hasYard','hasPool','cityPartRange','numPrevOwners','isNewBuilt','attic','hasStorageRoom','hasGuestRoom']]

# Build the train & test data set
F_train,F_test,Y_train,Y_test=model_selection.train_test_split(FSet,Y,test_size=0.2,random_state=2)

# Call the model
model=mymodel(Y_train,F_train)
prediction=Prediction(model,F_test,Y_test)

# Plot test vs predicted
cons_result=pd.DataFrame()
residuals=Y_test-prediction
cons_result['Ytest']=Y_test
#prediction='{:.10f}'.format(prediction)
cons_result['OLS_Predic']=prediction
cons_result['Ytest-OLS']=(prediction-Y_test)
sbrn.lineplot(residuals)
plt.title('Residuals Graph ')
plt.show()

# Create the findal prediction
test_data1=test_data[['squareMeters','numberOfRooms','cityCode','made','hasStormProtector','basement','garage']]
Final_Pred=model.predict(test_data1)
#print('Prediction Using OLS Linear regression is :\n ','Id :',test_data['id'],'price :',Final_Pred)
print('OLS prediction Accuracy is :','\n Model AIC:',model.aic ,'\n Model R2:',model.rsquared,'\n Model RSqAddj :',model.rsquared_adj)
OLS_Features=pd.DataFrame({'OLS_Features':test_data1.columns,'Pvalues':model.pvalues})

#print(OLS_Features['OLS_Features'])
#print(type(OLS_Features['Pvalues']),OLS_Features['Pvalues'])

OLS_Features.to_html('OLS Feature.html')
#print('OLS Feature',OLS_Features)

# sample code to generate decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
# Initialise the decision tree classifier

dt = DecisionTreeRegressor(max_depth=6, ccp_alpha=0.01)
dt.fit(X_train,Y_train)
y_pred = dt.predict(X_test)
#y_pred='{:.10f}'.format(y_pred)
cons_result['Dct_Y_Pred']=y_pred
cons_result['Ytest-Dct']=y_pred-Y_test
print('Decision Tree Score :\n',dt.score(X_test,Y_test))
print('Mean Square Error DCT :',mean_squared_error(y_pred,Y_test))
plt.figure(figsize=(10,10))
tree.plot_tree(dt, feature_names = list(X_train.columns),max_depth=6, filled=True)
plt.show()

# How to optimise the model using hypertuning using GridSearch Cross Validation
#[{'criterion':['entropy','gini']},{'max_depth':[2,3,4,5,6,7,8,9,10]}])
mod = model_selection.GridSearchCV(dt,param_grid=[{'criterion':['entropy','gini']},{'max_depth':[2,3,4,5,6,7,8,9,10]}])
#mod = model_selection.GridSearchCV(dt,param_grid={'max_depth':[2,3,4,5,6,7,8,9,10]})
mod.fit(X_train,Y_train)
#print(mod.best_estimator_)
#print(mod.best_score_)
#print(mod.best_params_)

# sample code to display the results
print('mod best estimator \n',mod.best_estimator_)
print('mod score',mod.best_score_)
result_dic={'Params':X_train.columns,'Dec_Tree_Features':dt.feature_importances_}
Dec_Tree_Features=pd.DataFrame(result_dic)
#Dec_Tree_Features.sort_values('Dec_Tree_Features',ascending=False,inplace=True)
#print(Dec_Tree_Features.sort_values('Dec_Tree_Features',ascending=False))
plt.barh(X_train.columns,dt.feature_importances_)
plt.title('Decision Tree: Feature Importance')
plt.show()


# check with Random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train,Y_train)
#regressor.fit(F_train,Y_train)
Y_pred = regressor.predict(X_test)
print(X_test.head())
#Y_pred='{:.10f}'.format(y_pred)
cons_result['RFE_Reg']=Y_pred
cons_result['Ytest-RFEReg']=Y_pred-Y_test
result_dic2={'Variables':X_train.columns,'RFE_Features':regressor.feature_importances_}
RFR_Feature=pd.DataFrame(result_dic2)
#RFR_Feature.sort_values('RFE_Features',ascending=False,inplace=True)
print('Regressor Score of Train Data :\n',regressor.score(X_test,Y_pred))
#print(RFR_Feature.sort_values('RFE_Features',ascending=False))

#OLS_Features
#print('df before \n',df_pvalue)
#Final_Out=[RFR_Feature,Dec_Tree_Features,df_pvalue]
#print(type(Final_Out))
Final_Features=pd.concat([RFR_Feature,Dec_Tree_Features,df_pvalue],axis=1,ignore_index=False,sort=False)
#columnsTitles=['RFR_Feature','Dec_Tree_Features','df_pvalue']
#Final_Features= Final_Features.reindex(columns=columnsTitles)

#Final_Features.sort_values(by=[RFR_Feature,Dec_Tree_Features,df_pvalue])
#print('df after \n',Final_Features.head())
print('consolidated result \n',cons_result.head())
cons_result.to_html('consolidated_results.html')
print('Cons result',cons_result.describe())
Final_Features.fillna(0)
#Final_Features['model_aic']=pd.Series(model.aic)
#Final_Features.fillna(0)
#Final_Features['Model R2']=pd.Series(model.rsquared)
#Final_Features.fillna(0)
#Final_Features['Model RSqAddj']=pd.Series(model.rsquared_adj)
#Final_Features.fillna(0)

Final_Features.to_html('Final_Feature.html')
#print('Final Feature Set',Final_Features)

plt.barh(X_train.columns,regressor.feature_importances_)
plt.title('Random Forest: Feature Importance')
plt.show()

"""
def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })

dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()  # %age of good record for each bin
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()    # %age of bad record for each bin
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])  # woe of each bin
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return dset, iv

for col in data.columns:
    if col == 'Exited': continue
    else:
        print('WoE and IV for column: {}'.format(col))
        df, iv = calculate_woe_iv(data, col, 'Exited')
        print(df)
        print('IV score: {:.2f}'.format(iv))
        print('\n')

"""