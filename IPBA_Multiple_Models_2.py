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

#pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.float_format',lambda x:'%.4f' %x)

#change the directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
filepath=os.getcwd()

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
        print('File not present in filepath', filepath)
#check the data
def data_check(train_data,test_data):

    # check the head of data
    print('train data \n',train_data.head())
    print('test data \n',test_data.head())

    # Check the data description
    print('train data describe \n',train_data.describe())

    # Check the Null or NA
    print(train_data.isnull().sum())

    #check the Data Types
    print(train_data.info())

    # check the sweetviz report
    #sv_report = sv.analyze(train_data)
    #sv_report.show_html('sv_report.html')

    #convert the datetimedate to datetime
    #train_data['made'] = pd.to_datetime(train_data['made'],format='mixed')

# Descriptive Statistics ,IQR and LL
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
            UL=pdf_IQR['Q3']+(1.5)*pdf_IQR['IQR']
            LL=pdf_IQR['Q3']-(1.5)*pdf_IQR['IQR']
            train_data=np.where(train_data[train_data]>UL,train_data.mean(),train_data)
            train_data.apply(lambda x:x.mean())
            train_data = (train_data[train_data]>UL).apply(lambda x:x.mean())
            pdf_IQR['UL_count']=train_data[train_data[columns] > UL][columns].count()
            pdf_IQR['LL_count'] = train_data[train_data[columns] < LL][columns].count()
            pdf_IQR['UL_Per'] = pdf_IQR['UL_count']/train_data[columns].count() * 100
            pdf_IQR['LL_Per'] = pdf_IQR['LL_count']/train_data[columns].count() * 100
            #print(pdf_IQR.head(20))
            pdf_Result = pd.concat([pdf_Result,pdf_IQR], axis=1)
        else:
            pass
    print(pdf_Result .head(20))
    pdf_Result.to_html('IQR_Report.html')
    return  pdf_Result

#Correlation
def correlation(train_data):
    # check correlation with target variables
    corr_matrix = train_data.corr()
    #corr_matrix.sort_values('price',ascending=True)
    fig, ax = plt.subplots(figsize=(20,15))
    print('correlation Matrix :\n',corr_matrix)
    sbrn.heatmap(corr_matrix,cmap='coolwarm',vmin=-1,vmax=1,annot=True,square=True,ax=ax,linewidths=0.5,fmt="0.2f")
    plt.show()

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
    def histogram(train_data_copy,hue):
        for values in hue:
            sbrn.displot(data=train_data_copy,x='log_price',hue=values,multiple='stack')
            plt.show()

# Scatter plot
    def scatterplot(train_data_copy,hue):
        for xvalues in x_values:
            for hvalues in hue:
                sbrn.relplot(data=train_data_copy, x=xvalues, y="log_price",col="isNewBuilt", hue=hvalues,kind="scatter")
                plt.xlabel(xvalues)
                plt.ylabel('log_price')
                plt.show()

    histogram(train_data_copy, hue)
    scatterplot(train_data_copy, hue)

#Fit the OLS Model
def mymodel(xtrain,ytrain):
    model=sm.OLS(xtrain.astype(float),ytrain.astype(float)).fit()
    print(model.summary())
    return model

# Give the prediction
def Prediction(model,xtest,ytest):
    prediction = model.predict(xtest)
    actual = ytest
    # Print the model Summary
    print('OLS model Summary :\n',model.summary())
    # Print the Model Accuracy
    print('OLS prediction Accuracy is :', '\n Model AIC:', model.aic, '\n Model R2:', model.rsquared,'\n Model RSqAddj :', model.rsquared_adj)
    # Find Root Mean Square Error
    print('actual size :\n',actual.shape)
    print('predicted size :\n', prediction.shape)
    MSE=mean_squared_error(actual, prediction, squared=True)
    RMSE = sqrt(mean_squared_error(actual, prediction, squared=True))
    print('Mean Square from OLS Iteration: \n', MSE)
    print('Root Mean Square from OLS Iteration:\n', RMSE)
    #Plot the residuals
    residuals = actual - prediction
    sbrn.lineplot(residuals)
    plt.title('Residuals Graph ')
    plt.show()
    return prediction

def OLSFeatures_NonFeatures(model,X_train,):
    Fset,NonFset={},{}
    for cols in X_train.columns:
        if model.pvalues[cols] <0.05:
            Fset[cols] = model.pvalues[cols]
        else:
            NonFset[cols]=model.pvalues[cols]
    return Fset,NonFset

def main():

    #Load the Data
    train_data,test_data=dataloading('train.csv','test.csv')

    # View basic Data
    data_check(train_data, test_data)

    #check descriptive statistics on Data
    IQR_report = IQR_UL_LL(train_data)

    #Build correlation
    correlation(train_data)

    #EDA with histogram
    #histogram_scatter_plots(train_data)

    # Build the model variables
    Y = train_data['price']
    X = train_data.drop('price', axis=1)

    # Build the train & test data set
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=2)
    print(X.head(), Y.head())

    # call the model with all the variables
    model = mymodel(Y_train, X_train)
    Prediction(model,X_test, Y_test)

    # check the feature list from the model by finding pvalues returned
    Fset,NonFset=OLSFeatures_NonFeatures(model,X_train)
    print('1st iteration Features\ n', Fset.keys())
    print('Type', type(Fset.keys()))
    print('1st Iteration Non Features\ n', NonFset.keys())

    # Important features having p values less than 0.05, significant variables
    #FSet=df_pvalue[['p-Value']<0.050]
    Fset=train_data[Fset.keys()]

    # Re-Build the Model with new train & test data set
    F_train, F_test, Y_train, Y_test = model_selection.train_test_split(Fset,Y,test_size=0.2, random_state=2)

    # ReCall the model with subsidised features
    model = mymodel(Y_train, F_train)
    prediction = Prediction(model,F_test, Y_test)

    Fset,NonFset=OLSFeatures_NonFeatures(model,F_train)
    print('2nd Iteration Features\ n', Fset.keys())
    print('Type', type(Fset.keys()))
    print('2nd Iteration Non Features\ n', NonFset.keys())
    Fset2 = train_data[Fset.keys()]

    # Plot test vs predicted
    cons_result = pd.DataFrame()
    #prediction=Prediction(model,F_train, Y_test)
    cons_result['Ytest'] = Y_test
    cons_result['OLS_Predic'] = prediction
    cons_result['Ytest-OLS'] = (prediction - Y_test)

    # Create the final prediction based on final features
    #Final_Pred = model.predict(test_data[Fset])

    # print('Prediction Using OLS Linear regression is :\n ','Id :',test_data['id'],'price :',Final_Pred)
    OLS_Features = pd.DataFrame({'OLS_Features': Fset.keys(), 'Pvalues': Fset.values()})
    OLS_Features.to_html('OLS Feature.html')

"""
    for values in Fset2:
        Z=X_train[values]
        # Build the train & test data set
        Z_train, Z_test, Y_train, Y_test = model_selection.train_test_split(Z, Y, test_size=0.2, random_state=2)
        print(Z.head(), Z.head())
        # call the model with all the variables
        model = mymodel(Z_train, Y_train)
        Prediction(model,Z_test, Y_test)
"""



    # Build Another model based on Feature sample code to generate decision tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor,export_graphviz
    from sklearn import tree
    import graphviz
    # Initialise the decision tree classifier

    dt = DecisionTreeRegressor(max_depth=6, ccp_alpha=0.01)
    dt.fit(X_train,Y_train)
    y_pred = dt.predict(X_test)
    cons_result['Dct_Y_Pred'] = y_pred
    cons_result['Ytest-Dct'] = y_pred - Y_test
    print('Decision Tree Score :\n', dt.score(X_test, Y_test))
    print('Mean Square Error DCT :', mean_squared_error(y_pred, Y_test))
    from IPython.display import display
    display(graphviz.Source(export_graphviz(dt)))
    plt.savefig('my_dt_plot.png')

    """
    plt.figure(figsize=(20, 15))
    tree.plot_tree(dt, feature_names=list(X_train.columns), max_depth=6, filled=True)
    plt.show()
    """
    # How to optimise the model using hypertuning using GridSearch Cross Validation
    # [{'criterion':['entropy','gini']},{'max_depth':[2,3,4,5,6,7,8,9,10]}])
    mod = model_selection.GridSearchCV(dt, param_grid=[{'criterion': ['entropy', 'gini']}, {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}])
    # mod = model_selection.GridSearchCV(dt,param_grid={'max_depth':[2,3,4,5,6,7,8,9,10]})
    mod.fit(X_train, Y_train)
    #print(mod.best_estimator_)
    # print(mod.best_score_)
    # print(mod.best_params_)

    # sample code to display the results
    print('mod best estimator \n', mod.best_estimator_)
    print('mod score', mod.best_score_)
    result_dic = {'Params': X_train.columns, 'Dec_Tree_Features': dt.feature_importances_}
    Dec_Tree_Features = pd.DataFrame(result_dic)
    plt.barh(X_train.columns, dt.feature_importances_)
    plt.title('Decision Tree: Feature Importance')
    plt.show()

    # check with Random forest
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train,Y_train)
    Y_pred = regressor.predict(X_test)
    print(X_test.head())
    cons_result['RFE_Reg'] = Y_pred
    cons_result['Ytest-RFEReg'] = Y_pred - Y_test
    result_dic2 = {'Variables': X_train.columns, 'RFE_Features': regressor.feature_importances_}
    RFR_Feature = pd.DataFrame(result_dic2)
    print('Regressor Score of Train Data :\n', regressor.score(X_test, Y_pred))

    # Final Feature set
    A=pd.DataFrame(Fset.keys(),columns=['OLSFeatures'])
    Final_Features = pd.concat([RFR_Feature,Dec_Tree_Features,A],axis=1, ignore_index=False, sort=False)
    print('consolidated result \n', cons_result.head())
    cons_result.to_html('consolidated_results.html')
    print('Cons result', cons_result.describe())
    Final_Features.fillna(0)
    Final_Features.to_html('Final_Feature.html')

    #plot the file features
    plt.barh(X_train.columns, regressor.feature_importances_)
    plt.title('Random Forest: Feature Importance')
    plt.show()

    #plot the file features
    plt.bar(x=X_train.columns,height= regressor.feature_importances_)
    plt.title('Random Forest: Feature Importance')
    plt.show()

    # create submission file
    print('creating the submission file ....')
    print(test_data.head())
    Y1_pred = regressor.predict(test_data)
    submission_df = pd.DataFrame({'Id': test_data['id'], 'price': Y1_pred})
    print(submission_df.head())
    submission_df.to_csv('submission.csv', index=False)
    print('Regressor Score of Test Data :\n', regressor.score(test_data, Y1_pred))


if __name__ == '__main__':
    main()