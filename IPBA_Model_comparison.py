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


def Prediction(mymodels,X_train, X_test, Y_train,Y_test):
    print()
    for model in mymodels:
        model.fit(X_train,Y_train)
        prediction = model.predict(X_test)
        actual = Y_test
        prediction=np.exp(prediction)
        actual=np.exp(actual)
        print('model:',model)
        MSE=mean_squared_error(actual, prediction, squared=True)
        RMSE = sqrt(mean_squared_error(actual, prediction, squared=True))
        #print('MSE',MSE,'RMSE',RMSE)
        #Plot the residuals
        #residuals = actual - prediction
        #sbrn.relplot(residuals)
        #plt.title('Residuals Graph')
        #plt.show()
        if model==LinearRegression():
            #print('OLS model Summary :\n', model.summary())
            print([{'R2':model.rsquared,'R2Adj':model.rsquared_adj,'MSE':MSE,'RMSE':RMSE}])
        elif model==BaggingRegressor():
            R2 = model.oob_score_
            n = X_test.shape[0]
            p = X_test.shape[1]
            AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
            print([{'R2': R2, 'R2Adj': AdjR2, 'MSE': MSE, 'RMSE': RMSE}])
        else:
            R2=model.score(X_test,(Y_test))
            n=X_test.shape[0]
            p=X_test.shape[1]
            AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
            print([{'R2':R2,'R2Adj':AdjR2,'MSE':MSE,'RMSE':RMSE}])
        #return prediction

def IQR_UL_LL(train_data):
    pdf_IQR, pdf_Result = pd.DataFrame(), pd.DataFrame()
    #train_data['made'] = pd.to_datetime(train_data['made'])
    #train_data['basement'] = np.log(train_data['basement'])
    #train_data['squareMeters'] = (train_data['squareMeters'])
    #train_data['price'] = np.log(train_data['price'])
    #train_data['numberOfRooms'] = np.log(train_data['numberOfRooms'])

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
            #print(pdf_IQR.head(20))
            #pdf_Result = pd.concat([pdf_Result,pdf_IQR], axis=1)
            #train_data['price']=np.log10(train_data['price'])
            #train_data['squareMeters']=np.log10(train_data['squareMeters'])
            #sbrn.relplot(data=train_data,x=columns,y='price')
            #plt.show()
        else:
            pass
    print('Train_Data_Descriptive Stats :\n',pdf_Result.head(20))
    pdf_Result.to_html('Descriptve_Stats_IQR_Report.html')
    return  pdf_Result
#Correlation

def WOE_IV(train_data):
    df=train_data
    df_woe_iv = (pd.crosstab(df['squareMeters','numberOfRooms','hasYard','hasPool','floors','cityCode','cityPartRange','numPrevOwners'
                                                                                                                       ,'made','isNewBuilt','hasStormProtector'], df['price'])
                 .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                 .assign(iv=lambda dfx: np.sum(dfx['woe'] *
                                               (dfx[1] - dfx[0]))))

    print(df_woe_iv)
def main():
        # Load the Data
        train_data, test_data = dataloading('train_hp.csv','test_hp.csv')
          #X =np.log(train_data['squareMeters'])
        IQR_report = IQR_UL_LL(train_data)
        #temp_data = train_data[train_data['squareMeters'] < 5000]
        #print(temp_data['price'].mean(),temp_data['price'].mode(),temp_data['price'].max())
        #train_data = train_data[train_data['squareMeters'] > 5000]
        #print(train_data['price'].mean())
        #train_data['made']=pd.to_datetime(train_data['made']).dt.year
        #print(train_data['made'].dtypes)
        train_data['price']=np.log(train_data['price'])
        train_data['squareMeters'] = np.log(train_data['squareMeters'])

        #sbrn.relplot(data=train_data, x='made', y='price')
        #sbrn.scatterplot(data=train_data, x='made', y='price')
        #plt.show()
        #train_data = train_data[['id','price','squareMeters','made','numberOfRooms']]
        #train_data.set_index(['id'],inplace=True)

        #train_data=train_data[['price','squareMeters']]
        #train_data=train_data[train_data['squareMeters'] >10.8]

        train_data = train_data.drop(train_data[train_data.squareMeters < 10.71].index)
        train_data = train_data.drop(train_data[train_data.price< 15.35].index)

        Y = (train_data['price'])
        X = train_data[['squareMeters', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasYard', 'hasPool']]
        #X = train_data['squareMeters']
        temp=['id','isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasYard', 'hasPool']
        for item in temp:
            sbrn.relplot(data=train_data,x='squareMeters',y='price',hue=item)
            plt.show()
            train_data['squareMeters'].describe()

        #Y=(train_data['price'])
        #X=train_data.drop('price',axis=1)
        #X =(train_data[['squareMeters','made','numberOfRooms']])

        IQR_report = IQR_UL_LL(train_data)
        lr = LinearRegression()
        dt = DecisionTreeRegressor()
        Rf = RandomForestRegressor(n_estimators=100)
        Brgsr = BaggingRegressor(oob_score=True, n_estimators=80, random_state=300,
                                 base_estimator=DecisionTreeRegressor())
        mymodels = [lr, dt, Rf, Brgsr]
        Rf = RandomForestRegressor(n_estimators=100)
        Brgsr = BaggingRegressor(oob_score=True, n_estimators=80, random_state=300,
                                 base_estimator=DecisionTreeRegressor())
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=100)
        Prediction(mymodels, X_train, X_test, Y_train, Y_test)

        """
        #X = train_data[['squareMeters', 'floors', 'made', 'cityCode']]
        #X=train_data[['squareMeters','numberOfRooms','floors','made','basement','garage','cityCode','hasStormProtector']]
        #X=train_data.drop('price',axis=1)
        X_train, X_test, Y_train, Y_test=model_selection.train_test_split(X,Y,test_size=0.20,random_state=100)

        X_train=np.array(X_train).reshape(-1,1)
        Y_train=np.array(Y_train).reshape(-1,1)
        #print(X_train.shape,Y_train.shape)
        X_test=np.array(X_test).reshape(-1,1)
        Y_test=np.array(Y_test).reshape(-1,1)

        #print(X_test.shape, Y_test.shape)

        from sklearn.preprocessing import MinMaxScaler
        # check descriptive statistics on Data
        IQR_report = IQR_UL_LL(train_data)
        lr=LinearRegression()
        dt=DecisionTreeRegressor()
        Rf = RandomForestRegressor(n_estimators=90)
        Brgsr = BaggingRegressor(oob_score=True, n_estimators=80, random_state=300,base_estimator=DecisionTreeRegressor())
        mymodels = [lr,dt,Rf, Brgsr]
        #for i in range(1,100,10):
        Rf=RandomForestRegressor(n_estimators=80)
        #for x in range(1,100,10):
        Brgsr = BaggingRegressor(oob_score=True, max_features=2, n_estimators=80, random_state=300,base_estimator=DecisionTreeRegressor())
            #Brgsr = BaggingRegressor(oob_score=True, n_estimators=x, random_state=500,base_estimator=DecisionTreeRegressor())
        #mymodels = [lr,dt,Rf,Brgsr,Rf]
        Prediction(mymodels, X_train, X_test, Y_train,Y_test)
        """
main()