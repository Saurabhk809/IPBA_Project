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
#check the data
def data_check(train_data,test_data):

    # check the head of data
    print('train data \n',train_data.head())
    print('test data \n',test_data.head())

    # Check the data description
    print('train data describe \n',train_data.describe())

    # Check the Null or NA
    print('Check for Null chars:\n',train_data.isnull().sum())

    #check the Data Types
    print('Info of train data:\n',train_data.info())

# Descriptive Statistics ,IQR and LL
def IQR_UL_LL(train_data):
    pdf_IQR, pdf_Result = pd.DataFrame(), pd.DataFrame()
    for columns in train_data:
        if train_data[columns].isnull().sum()==0:
            pass
        else:
            train_data[columns].fillna(0)
        if columns not in ['id']:
            #pdf_IQR['Param']=columns
            pdf_IQR = (train_data[columns].describe())
            pdf_IQR['median'] = statistics.median(train_data[columns])
            pdf_IQR['mode']=statistics.mode(train_data[columns])
            pdf_IQR['Q1']=np.quantile(train_data[columns],.25)
            pdf_IQR['Q3'] = np.quantile(train_data[columns],.75)
            pdf_IQR['95%'] = np.quantile(train_data[columns],.25)
            pdf_IQR['IQR']=pdf_IQR['Q3']-pdf_IQR['Q1']
            UL=pdf_IQR['Q3']+(1.5)*pdf_IQR['IQR']
            print (columns,UL)
            LL=pdf_IQR['Q1']-(1.5)*pdf_IQR['IQR']
            print(columns,LL)
            #train_data[columns]=np.where(train_data[columns]>UL,UL,train_data[columns])
            print(train_data[columns].max())
            #train_data[columns] = np.where(train_data[columns]<LL,LL, train_data[columns])
            pdf_IQR['UL_count']=train_data[train_data[columns] > UL][columns].count()
            pdf_IQR['LL_count'] = train_data[train_data[columns] < LL][columns].count()
            pdf_IQR['UL_Per'] = pdf_IQR['UL_count']/train_data[columns].count() * 100
            pdf_IQR['LL_Per'] = pdf_IQR['LL_count']/train_data[columns].count() * 100
            #print(pdf_IQR.head(20))
            pdf_Result = pd.concat([pdf_Result,pdf_IQR], axis=1)
        else:
            pass
    print('Train_Data_Descriptive Stats :\n',pdf_Result.head(20))
    pdf_Result.to_html('Descriptve_Stats_IQR_Report.html')
    return  pdf_Result
#Correlation
def correlation(train_data):
    # check correlation with target variables
    corr_matrix = train_data.corr()
    #corr_matrix.sort_values('price',ascending=True)
    fig, ax = plt.subplots(figsize=(20,15))
    print('correlation Matrix :\n',corr_matrix)
    #sbrn.heatmap(corr_matrix,cmap='coolwarm',vmin=-1,vmax=1,annot=True,square=True,ax=ax,linewidths=0.5,fmt="0.2f")
    #plt.show()
    print('correlation Table for Target Variable :\n',train_data.corr()['price'].sort_values(ascending=False))

# EDA
def histogram_scatter_plots(train_data):
    # convert price data to log
    train_data_copy=train_data.copy()
    train_data_copy['log_price'] = np.log(train_data_copy['price'])
    train_data_copy.drop('price', axis=1, inplace=True)
    train_data_copy['log_squareMeters'] = np.log(train_data_copy['squareMeters'])
    train_data_copy.drop('squareMeters', axis=1, inplace=True)

    # pass variable for scatter plot
    x_values = ['log_squareMeters', 'made', 'numberOfRooms', 'floors', 'cityCode', 'cityPartRange', 'numPrevOwners','attic', 'garage', 'hasStorageRoom', 'hasGuestRoom']
    y_values = ['log_price']
    hue = ['isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasYard', 'hasPool']

    # Histogram
    def histogram(train_data_copy, hue):
        for values in hue:
            sbrn.displot(data=train_data_copy, x='log_price', hue=values, multiple='stack')
            plt.show()

    # Scatter plot
    def scatterplot(train_data_copy, hue):
        for xvalues in x_values:
            for hvalues in hue:
                sbrn.relplot(data=train_data_copy, x=xvalues, y="log_price", col="isNewBuilt", hue=hvalues,
                             kind="scatter")
                plt.xlabel(xvalues)
                plt.ylabel('log_price')
                plt.show()

    histogram(train_data_copy, hue)
    scatterplot(train_data_copy, hue)

def Prediction(model,xtest,ytest,type,df_mod_cm):
    prediction = (model.predict(xtest))
    actual = (ytest)
    MSE=mean_squared_error(actual, prediction, squared=True)
    RMSE = sqrt(mean_squared_error(actual, prediction, squared=True))
    #Plot the residuals
    residuals = actual - prediction
    result=pd.DataFrame([actual,prediction,residuals])
    result.to_csv('myresult.csv')
    sbrn.lineplot(y=residuals,x=prediction)
    #plt.title('Residuals Graph')
    #plt.show()
    if type=='OLS':
        #print('OLS model Summary :\n', model.summary())
        new_record=pd.DataFrame([{'Model':type,'R2':model.rsquared,'R2Adj':model.rsquared_adj,'MSE':MSE,'RMSE':RMSE}])
        df_mod_cm=pd.concat([df_mod_cm,new_record],ignore_index=True)
    elif type=='BRGRS':
        R2 = model.oob_score_
        n = xtest.shape[0]
        p = xtest.shape[1]
        AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
        new_record = pd.DataFrame([{'Model': type, 'R2': R2, 'R2Adj': AdjR2, 'MSE': MSE, 'RMSE': RMSE}])
        df_mod_cm = pd.concat([df_mod_cm, new_record], ignore_index=True)
    else:
        R2=model.score(xtest,ytest)
        n=xtest.shape[0]
        p=xtest.shape[1]
        AdjR2 = (1 - (1 - R2)) * ((n - 1) / (n - p))
        new_record=pd.DataFrame([{'Model':type,'R2':R2,'R2Adj':AdjR2,'MSE':MSE,'RMSE':RMSE}])
        df_mod_cm=pd.concat([df_mod_cm,new_record],ignore_index=True)
    return prediction,df_mod_cm

def consolidate_result(cons_result,Y_test,prediction,type,):
    cons_result[type+'_Ytest'] = Y_test
    cons_result[type+'_Predic'] = prediction
    cons_result[type+'_Ytest-OLS'] = (prediction - Y_test)

def OLSFeatures_NonFeatures(model,X_train,):
    Fset,NonFset={},{}
    for cols in X_train.columns:
        if model.pvalues[cols] <0.05:
            Fset[cols] = model.pvalues[cols]
        else:
            NonFset[cols]=model.pvalues[cols]
    return Fset,NonFset

def main():

    # Load the Data
    train_data, test_data = dataloading('train_hp.csv', 'test_hp.csv')

    # View basic Data
    data_check(train_data, test_data)

    # check descriptive statistics on Data
    IQR_report = IQR_UL_LL(train_data)
    #IQR_report= IQR_UL_LL(train_data[['id','squareMeters']])

    # Build correlation
    correlation(train_data)

    # EDA with histogram & scatter plot
    #histogram_scatter_plots(train_data)

    # Build the model variables

    Y = train_data['price']
    X = train_data.drop('price', axis=1)

    #Y = train_data['price']
    #X=(train_data[['id','squareMeters','made']])
    #X['made'] = pd.to_datetime(X['made']).dt.year
    #print(X['made'].info())
    #X = (train_data[['squareMeters','floors','made']])
    #X.set_index('id', inplace=True)
    #X=train_data[['squareMeters', 'numberOfRooms', 'floors', 'cityCode', 'made', 'hasStormProtector', 'basement', 'garage']]
    #X = train_data.drop('price',axis=1)
    #X=train_data[['squareMeters', 'floors']]
    #X = train_data[['squareMeters','floors','made','numberOfRooms']]
    #X = train_data[['squareMeters','floors', 'made']] #126014 ,125190,124267
    #X = train_data[['squareMeters', 'numberOfRooms', 'floors', 'made', 'basement', 'garage', 'cityCode', 'hasStormProtector']]
    #X = train_data[['squareMeters','floors','made','cityCode']]
    # Build the train & test data set

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2, random_state=300)

    # Build  OLS Model with All Features
    #model=sm.OLS(Y_train.astype(float),X_train.astype(float)).fit()
    model = sm.OLS(Y_train, X_train).fit()
    print(model.summary())

    # Generate the Model Predicition & consolidate the result
    df_mod_cm = pd.DataFrame(columns=['Model', 'R2', 'R2Adj', 'MSE', 'RMSE'])
    prediction,df_mod_cm = Prediction(model,X_test,Y_test,'OLS',df_mod_cm)

    # Print the Model prediction
    df_cons_result = pd.DataFrame()
    consolidate_result(df_cons_result,Y_test,prediction,'OLS1')
    print('Consolidated Model Prediction :\n',df_cons_result.head())

    # Print the Model Accuracy
    print('Consolidated Model Error \n',df_mod_cm.head())

    # check the feature list from the model by finding pvalues returned
    Fset, NonFset = OLSFeatures_NonFeatures(model, X_train)
    print('1st iteration Features vs. Non Features \n','Features:\n',list(Fset.keys()),'\n NonFeatures:\n',list(NonFset.keys()))

    # Important features having p values less than 0.05, significant variables
    Fset = train_data[Fset.keys()]

    # Re-Build the Model with new train & test data set
    F_train, F_test, Y_train, Y_test = model_selection.train_test_split(Fset, Y, test_size=0.2, random_state=2)

    # 2 # ReCall the model with subsidised features
    model = sm.OLS(Y_train, F_train).fit()
    print(model.summary())

    # Generate the Model Predicition & consolidate the result
    prediction, df_mod_cm = Prediction(model,F_test, Y_test,'OLS', df_mod_cm)

    # Print the Model prediction
    consolidate_result(df_cons_result,Y_test, prediction,'OLS')
    print('Consolidated Model Prediction :\n', df_cons_result.head())

    # Print the Model Accuracy
    print('Consolidated Model Error \n', df_mod_cm.head())

    # Consolidate the OLS feature set
    Fset, NonFset = OLSFeatures_NonFeatures(model, F_train)
    print('2nd iteration Features vs. Non Features \n', 'Features:\n', list(Fset.keys()), '\n NonFeatures:\n',list(NonFset.keys()))
    Features = pd.DataFrame({'OLS_Features':Fset.keys(),'Pvalues': Fset.values()})
    Features.to_html('Corrr_OLS Feature.html')


    # 3 # Repeat all the steps by Build another model with DecisionTree to improve the accuracy
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    from sklearn import tree
    import graphviz

    # Create the model & tune the Model using the Grid Search & cross Validation
    dt = tree.DecisionTreeRegressor(max_depth=1,random_state=200)
    dt.fit(X_train,Y_train)
    mydepth=list(range(1,10,1))
    mod = model_selection.GridSearchCV(dt,param_grid={'max_depth': mydepth})
    mod.fit(X_train,Y_train)
    # sample code to display the results
    print('mod best estimator \n', mod.best_estimator_)
    print('mod best params \n', mod.best_params_)
    print('mod score', mod.best_score_)
    print('mod score', dt.score(X_test,Y_test))
    result_dic = {'Params': X_train.columns,'Dec_Tree_Features': dt.feature_importances_}
    Dec_Tree_Features = pd.DataFrame(result_dic)
    print('best params',Dec_Tree_Features)

    # Initialise the decision tree classifier
    dt = DecisionTreeRegressor(max_depth=1, random_state=200)
    #dt = DecisionTreeRegressor(max_depth=1, ccp_alpha=0.01)
    dt.fit(X_train,Y_train)

    # Generate the Model Predicition & consolidate the result
    prediction, df_mod_cm = Prediction(dt,X_test,Y_test, 'DCT',df_mod_cm)

    # Print the Model prediction
    consolidate_result(df_cons_result,Y_test, prediction,'DCT')
    print('Consolidated Model Prediction :\n', df_cons_result.head())

    # Print the Model Accuracy
    print('Consolidated Model Error \n', df_mod_cm.head())

    # consolidate the feature list of Decision Tree & Plot them
    result_dic = {'Params': X_train.columns,'Dec_Tree_Features': dt.feature_importances_}
    Dec_Tree_Features = pd.DataFrame(result_dic)
    plt.barh(X_train.columns, dt.feature_importances_)
    plt.title('Decision Tree: Feature Importance')
    plt.show()

    # generate the view of the DCT feature list
    """
    from IPython.display import display
    # display(graphviz.Source(export_graphviz(dt)))
    display(graphviz.Source(export_graphviz(dt, out_file=None, feature_names=X.columns, filled=True, rounded=True)))
    plt.figure(figsize=(20, 15))
    plt.show()
    plt.savefig('my_dt_plot.png')
    plt.figure(figsize=(20, 15))
    tree.plot_tree(dt, feature_names=list(X_train.columns), max_depth=6, filled=True)
    plt.show()
    """

    # 4 # Repeat all the steps by Build another model with Random Forest to improve the accuracy

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    for x in range(1,1000,100):
        regressor = RandomForestRegressor(n_estimators=x)
        regressor.fit(X_train, Y_train)

        # Generate the Model Predicition & consolidate the result
        prediction,df_mod_cm = Prediction(regressor, X_test, Y_test, 'RFR',df_mod_cm)

        # # Print the Model prediction
        consolidate_result(df_cons_result, Y_test, prediction, 'RFR')
        print('Consolidated Model Prediction :\n', df_cons_result.head())

        # Print the Model Accuracy
        print('Fit',x)
        print('Consolidated Model Error \n', df_mod_cm.head())

        # consolidate the model features
        result_dic2 = {'Variables': X_train.columns,'RFE_Features': regressor.feature_importances_}
        RFR_Feature = pd.DataFrame(result_dic2)


    # 5 # Repeat all the steps by building another model with Random Forest Bagging Regressor

    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor # Specifying which Base learner should be used

    #for i in range(1, 2, 1):
    clf = BaggingRegressor(oob_score=True,n_estimators=1000,random_state=300,base_estimator=DecisionTreeRegressor())
    B1 = clf.fit(X_train, Y_train)

    # Generate the Model Predicition & consolidate the result
    prediction, df_mod_cm = Prediction(clf, X_test, Y_test, 'BRGRS', df_mod_cm)

    print('\n clf_oob_scoreB1', clf.oob_score_)
    print("Out of bage score isB1", clf.oob_score_)
    print('clf_oob_scoreB1', clf.score(X_test, Y_test))  # Accuracy of the test data-set

    # # Print the Model prediction
    consolidate_result(df_cons_result, Y_test, prediction, 'BRGRS')
    print('Consolidated Model Prediction :\n', df_cons_result.head())
    df_cons_result.to_html('consolidated_Model_results.html')

    # Print the Model Accuracy
    print('Consolidated Model Error \n', df_mod_cm.head())
    df_mod_cm.to_html('Consolidated_Model_Errors.html')

    # create submission file
    # check with Random forest
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    print(X_test.head())
    #cons_result['RFE_Reg'] = Y_pred
    #cons_result['Ytest-RFEReg'] = Y_pred - Y_test
    #result_dic2 = {'Variables': X_train.columns, 'RFE_Features': regressor.feature_importances_}
    #RFR_Feature = pd.DataFrame(result_dic2)
    #print('Regressor Score of Train Data :\n', regressor.score(X_test, Y_pred))

    print('creating the submission file ....')
    print(test_data.head())
    Y1_pred = regressor.predict(test_data)
    submission_df = pd.DataFrame({'Id': test_data['id'], 'price': Y1_pred})
    print(submission_df.head())
    submission_df.to_csv('submission.csv', index=False)
    print('Regressor Score of Test Data :\n', regressor.score(test_data, Y1_pred))


main()