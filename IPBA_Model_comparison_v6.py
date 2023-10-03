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


#pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.float_format',lambda x:'%.4f' %x)

#change the directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filepath=os.getcwd()

#Load the Data
def dataloading(file1,file2):
    try:
        os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
        train_data=pd.read_csv(file1,na_values=[' ','NA','NULL'])
        test_data=pd.read_csv(file2,na_values=[' ','NA','NULL'])
        #set the display options
        pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
        return train_data,test_data
    except:
        FileNotFoundError
        print('File not present in filepath',filepath)

def feature(Data):
    i = 0
    X3 = []
    mydata = pd.DataFrame()
    Y=Data['price']
    X=Data.drop('price',axis=1)
    for cols in Data.columns:  # Iterate the columns of DataFrame for predictor
        if cols == 'id':  # Ignore the dependent Y Variable
            pass
        else:
            mydata[cols] = Data[cols]  # Add the fitted predictor in a dataframe for checking P Vale
            X2 = smapi.add_constant(mydata)
            model1 = smapi.OLS(Y, X2).fit()
            for data in list(mydata):  # Iterate for loop to check if cols added is adding or reducing p Value
                try:
                    if data == 'const':  # Ignore the constant
                        pass
                    else:
                        if model1.pvalues.loc[
                            data] > .055:  # If P value is higher than 0.5 drop the predictor else preserve it
                            if data in X3:
                                pass
                            else:
                                # print('removed',data,model1.pvalues.loc[data])
                                X2 = X2.drop(data, axis=1)
                                mydata = mydata.drop(data, axis=1)
                        else:
                            # print('added', data, model1.pvalues.loc[data])
                            X3.append(data)
                            X4 = list(set(X3))
                            X5 = mydata[X4]
                            model2 = smapi.OLS(Y, X5).fit()
                            #print(model2.summary())
                except:
                    ValueError
                    # print('Nothing to remove', data, model1.pvalues.loc[data])

    # print(X3)
    # print(Data[X3])
    #print('Predictor using stewise method\n', list(set(X3)))

def Prediction(mymodels,X_train, X_test, Y_train,Y_test,test_data,submission_df):
    print()
    #result1=pd.DataFrame()
    Model_Comparison=pd.DataFrame(columns=['Model','MeanDiff','RegScore','RMSE','MSE','R2Adj','R2'])
    #new_record=pd.DataFrame([{'Model':'OLS1','R2':model.rsquared,'R2Adj':model.rsquared_adj,'MSE':MSE,'RMSE':RMSE}])
    #Model_Comparison=pd.concat([Model_Comparison,new_record],ignore_index=True)
    for model in mymodels:
        model.fit(X_train,Y_train)
        prediction = model.predict(X_test)
        actual = Y_test
        #prediction=np.exp(prediction)
        #actual=np.exp(actual)
        print('Building model...:',str(model).split('(')[0])
        MSE=mean_squared_error(actual, prediction, squared=False)
        RMSE = sqrt(mean_squared_error(actual, prediction, squared=False))
        result1 = model.predict(test_data)
        #result1=np.exp(result1)
        expmean=mean(result1)
        bestmean=4574265
        #print('Expmean-bestMean',expmean-bestmean)
        #result2 = pd.DataFrame
        submission_df[str(model).split('(')[0]]=result1
        #result = np.exp(result)
        #print('Regressor Score of Test Data :', model.score(test_data, result1))
        if model==LinearRegression():
            #print([{'R2':model.rsquared,'R2Adj':model.rsquared_adj,'RMSE':RMSE,'MSE':MSE}])
            new_record = pd.DataFrame([{'Model':str(model).split('(')[0] ,'MeanDiff':expmean-bestmean,'RegScore':model.score(test_data, result1),'RMSE': RMSE,'MSE': MSE, 'R2Adj': model.rsquared_adj,'R2': model.rsquared}])
            Model_Comparison = pd.concat([Model_Comparison, new_record], ignore_index=True)
        elif model==BaggingRegressor():
            R2 = model.oob_score_
            n = X_test.shape[0]
            p = X_test.shape[1]
            AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
            #print([{'R2': R2, 'R2Adj': AdjR2,'RMSE':RMSE,'MSE':MSE}])
            new_record = pd.DataFrame([{'Model': str(model).split('(')[0], 'MeanDiff': expmean - bestmean,'RegScore': model.score(test_data, result1),'RMSE': RMSE ,'MSE': MSE,'R2Adj': AdjR2,'R2': R2}])
            Model_Comparison = pd.concat([Model_Comparison, new_record], ignore_index=True)
        else:
            R2=model.score(X_test,Y_test)
            n=X_test.shape[0]
            p=X_test.shape[1]
            AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
            #print([{'R2':R2,'R2Adj':AdjR2,'RMSE':RMSE,'MSE':MSE}])
            new_record = pd.DataFrame([{'Model': str(model).split('(')[0], 'MeanDiff': expmean - bestmean,'RegScore': model.score(test_data, result1),'RMSE': RMSE,'MSE': MSE,'R2Adj': AdjR2, 'R2': R2,}])
            Model_Comparison = pd.concat([Model_Comparison, new_record], ignore_index=True)
        #return model
    submission_df.to_csv('consolidated_submission.csv')
    Model_Comparison.to_html('consolidated_Model_summary.html')
    print(Model_Comparison)

def IQR_UL_LL(train_data):
    pdf_IQR, pdf_Result = pd.DataFrame(), pd.DataFrame()
    for columns in train_data:
        if train_data[columns].isnull().sum()==0:
            pass
        else:
            train_data[columns].fillna(0)
        if columns not in ['id','made']:
            #pdf_IQR['Param']=columns
            pdf_IQR = (train_data[columns].describe())
            pdf_IQR['median'] = statistics.median(train_data[columns])
            pdf_IQR['mode']=statistics.mode(train_data[columns])
            pdf_IQR['Q1']=np.quantile(train_data[columns],.25)
            pdf_IQR['Q3'] = np.quantile(train_data[columns],.75)
            pdf_IQR['95%'] = np.quantile(train_data[columns],.25)
            pdf_IQR['IQR']=pdf_IQR['Q3']-pdf_IQR['Q1']
            UL=pdf_IQR['Q3']+(0.5)*pdf_IQR['IQR']
            LL=pdf_IQR['Q3']-(0.5)*pdf_IQR['IQR']
            train_data[columns] = np.where(train_data[columns] > UL, train_data[columns].median(), train_data[columns])
            train_data[columns] = np.where(train_data[columns] < LL, train_data[columns].median(), train_data[columns])
            pdf_IQR['UL_count']=train_data[train_data[columns] > UL][columns].count()
            pdf_IQR['LL_count'] = train_data[train_data[columns] < LL][columns].count()
            pdf_IQR['UL_Per'] = pdf_IQR['UL_count']/train_data[columns].count() * 100
            pdf_IQR['LL_Per'] = pdf_IQR['LL_count']/train_data[columns].count() * 100
            #train_data['price'] = np.log(train_data['price'])
            #train_data['squareMeters'] = np.log(train_data['squareMeters'])
            #train_data.squareMeters = np.where(train_data.squareMeters < 10.71, train_data.squareMeters.median(), train_data.squareMeters)
            #train_data.price = np.where(train_data.price < 15.35, train_data.price.median(), train_data.price)
            #print(pdf_IQR.head(20))
            #pdf_Result = pd.concat([pdf_Result,pdf_IQR], axis=1)
            #train_data['price']=np.log10(train_data['price'])
            #train_data['squareMeters']=np.log10(train_data['squareMeters'])
            #sbrn.relplot(data=train_data,x=columns,y='price')
            #plt.show()
        else:
            pass
    #print('Train_Data_Descriptive Stats :\n',pdf_Result.head(20))
    pdf_Result.to_html('Descriptve_Stats_IQR_Report.html')
    return  pdf_Result
#Correlation
def main():
        # Load the Data
        train_data, test_data = dataloading('train_hp.csv','test_hp.csv')
        tid=test_data['id']
        feature(train_data)
        IQR_report = IQR_UL_LL(train_data)
        #train_data['price']=np.log(train_data['price'])
        #train_data['squareMeters'] = np.log(train_data['squareMeters'])
        #train_data['basement'] = np.log(train_data['basement'])
        #train_data['attic'] = np.log(train_data['attic'])
        #train_data['garage'] = np.log(train_data['garage'])

        #train_data.squareMeters = np.where(train_data.squareMeters < 10.71, train_data.squareMeters.median(), train_data.squareMeters)
        #train_data.price = np.where(train_data.price < 15.35, train_data.price.median(), train_data.price)
        #train_data = train_data.drop(train_data[train_data.squareMeters < 10.71].index)
        #train_data = train_data.drop(train_data[train_data.price< 15.35].index)
        Y = (train_data['price'])
        train_data['made'] = pd.to_datetime(train_data['made']).dt.year
        #X=train_data.drop('price',axis=1)
        # Kallal Sir's Features
        #X=train_data[['squareMeters','numberOfRooms','made','garage','hasStorageRoom']]
        #X=train_data[['cityPartRange', 'basement', 'made', 'squareMeters', 'garage', 'cityCode', 'numberOfRooms', 'floors']]
        # np.log features
        #X = train_data[['squareMeters', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasYard', 'hasPool','basement','attic','garage']]
        # Stepwise features
        #X=train_data[['made','floors', 'garage', 'basement', 'cityCode', 'numberOfRooms', 'squareMeters', 'cityPartRange']]
        # Best Score
        #X = train_data[['squareMeters', 'numberOfRooms', 'made', 'garage', 'hasStorageRoom']]
        train_data['squareMeters']=np.log(train_data['squareMeters'])
        test_data['squareMeters']=np.log(test_data['squareMeters'])

        # Ajay Feature list
        X = train_data[['squareMeters','numberOfRooms','floors','cityPartRange','made','numPrevOwners','isNewBuilt','basement','garage','hasGuestRoom']]
        #X=train_data[['squareMeters','made']]

        temp=['id','isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasYard', 'hasPool','basement','attic','garage']
        for item in temp:
            #sbrn.relplot(data=train_data,x='squareMeters',y='price',hue=item)
            #plt.show()
            train_data['squareMeters'].describe()
        IQR_report = IQR_UL_LL(train_data)
        lr = LinearRegression()
        dt = DecisionTreeRegressor(max_depth=6)
        print(dt._check_feature_names(X))
        #Rf = RandomForestRegressor(n_estimators=80,criterion="squared_error", min_samples_leaf=3)
        Rf = RandomForestRegressor(n_estimators=100,max_depth=6,random_state=500)
        Brgsr = BaggingRegressor(oob_score=True, n_estimators=80, random_state=300,estimator=DecisionTreeRegressor())
        #Brgsr = BaggingRegressor(oob_score=True, n_estimators=100, random_state=1000, estimator=DecisionTreeRegressor())
        adaboost = AdaBoostRegressor(n_estimators=100, base_estimator=None, learning_rate=1, random_state=1)
        GR = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=1)
        #mymodels = [lr, dt, Rf,Brgsr,adaboost,GR]
        mymodels=[dt]
        #Rf = RandomForestRegressor(n_estimators=80)
        #Brgsr = BaggingRegressor(oob_score=True, n_estimators=80, random_state=1000,
                                 #estimator=DecisionTreeRegressor())
        X_train, X_test, Y_train,Y_test = model_selection.train_test_split(X, Y, test_size=0.10, random_state=100)
        test_data = test_data[['squareMeters', 'numberOfRooms', 'floors', 'cityPartRange', 'made', 'numPrevOwners', 'isNewBuilt', 'basement','garage', 'hasGuestRoom']]
        #test_data = test_data[['squareMeters','made']]
        #test_data=test_data[['squareMeters','numberOfRooms','made','garage','hasStorageRoom']]
        #test_data=test_data[['squareMeters', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasYard', 'hasPool','basement','attic','garage']]
        #test_data=test_data[['squareMeters', 'numberOfRooms', 'made', 'garage', 'hasStorageRoom']]
        #model=Prediction(mymodels, X_train, X_test, Y_train, Y_test,test_data)
        #test_data['squareMeters']=np.log(test_data['squareMeters'])
        #test_data['basement'] = np.log(test_data['basement'])
        #test_data['attic'] = np.log(test_data['attic'])
        #test_data['garage'] = np.log(test_data['garage'])
        submission_df = pd.DataFrame({'id': tid})
        submission_df.set_index(['id'], inplace=True)
        Prediction(mymodels,X_train,X_test,Y_train,Y_test,test_data,submission_df)
        #print(test_data.head())
        #result=model.predict(test_data)
        #result=np.exp(result)
        #print(result)
        #print(test_data)
        #submission_df=pd.DataFrame({'id':tid,'price':result})
        #submission_df.set_index(['id'],inplace=True)
        #print(submission_df.head())
        #print('Regressor Score of Test Data :', model.score(test_data, result))
        #submission_df.to_csv('C:/Users/E1077195/OneDrive - FIS/000- IPBA/IPBA/submission.csv')
main()