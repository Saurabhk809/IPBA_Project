import os
import pandas as pd
import logging
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbrn
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn.metrics as mtrcs
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA


# Logger section
logging.basicConfig(filename='MLlogger.log',filemode='w')
logging.debug('This is message from debug level for debugging')
logging.info('This is message from info level')
logging.warning('This is message from warning')
logging.error('This is message from error level')
logging.critical('This is message from critical level')

#set the display options
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
pd.set_option('display.float_format',lambda x:'%.4f' %x)

# set the working directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')

# Function to load the data
def dataloading(file1,file2):
    try:
        train_data=pd.read_csv(file1,na_values=[' ','NA','N/A'])
        test_data = pd.read_csv(file2, na_values=[' ', 'NA', 'N/A'])
        print('train data:\n', train_data.head())
        print('test data:\n', test_data.head())
        return train_data, test_data
    except FileNotFoundError:
        print('file not present at path :',os.getcwd())
        logging.error('file not present at path :',os.getcwd())
def data_check(train_data):
    print('data type:\n',train_data.info())
    #print('data describe:\n', train_data.describe())
    #print('data check for null:\n',train_data.isnull().sum())
def data_cleansing(train_data):
    IQR_Report=pd.DataFrame()
    for column in train_data:
        if column in ['id','Attrition'] or train_data[column].dtype==object:
            pass
        else:
            #print(column,train_data[column].describe())
            IQR_Report[column]=train_data[column].describe()
            IQR=train_data[column].quantile(0.75)-train_data[column].quantile(0.25)
            UL=train_data[column].quantile(0.75)+(0.5)*IQR
            LL=train_data[column].quantile(0.25)-(0.5)*IQR
            train_data[column]=np.where(train_data[column]>UL,UL,train_data[column])
            train_data[column] = np.where(train_data[column]< LL,LL,train_data[column])
    print("IQR_Report : \n",IQR_Report)
    print("Cleansed Data describe:\n",train_data.describe())
    return train_data
def correlation(X):
    correlation = X.corr()
    #print('correlation:\n',correlation)
    fig, ax = plt.subplots(figsize=(20, 15))
    #print('\n correlation Matrix :\n', correlation)
    #sbrn.heatmap(correlation, cmap='coolwarm', vmin=-1, vmax=1, annot=True, square=True, ax=ax, linewidths=0.5,fmt="0.2f")
    #plt.show()
def roc_curve(ytest,ypredict,model):
    # FPR is 1-specificity , TPR is sensitivity both should be maximised
    fpr, tpr, thresholds = metrics.roc_curve(ytest, ypredict)
    #print('fpr:',fpr,'tpr:',tpr,'threshold:',thresholds)
    x, y = np.arange(0, 1.0, 0.1), np.arange(0, 1.0, 0.1)
    plt.plot(fpr, tpr, "-")
    plt.xlabel('1-specificity')
    plt.ylabel('sensitivty')
    plt.plot(x, y,'b--')
    plt.title('ROC Curve')
    #plt.show()
    # AUC, Average accuracy of my model , compares  the confusion metric way of comparing models
    print('Area under curve of',model,metrics.roc_auc_score(ytest, ypredict))
def EDAplot(train_data):
    for columns in train_data:
        if columns=='id':
            pass
        else:
           break
           sbrn.countplot(data=train_data,hue='Attrition',x=columns)
           plt.xlabel(columns)
           plt.ylabel('attrition')
           plt.show()
           #print(train_data.groupby([columns,'Attrition']).agg({'count','sum'}))
def model_test(X_train,X_test, Y_train,Y_test,mlist):
    for model in mlist:
        model.fit(X_train,Y_train)
        if model=="LogisticRegression":
            # Print model summary
            scores, pvalues = chi2(X_train, Y_train)
            pvalues = {'X': X_train.columns, 'pvalues': pvalues}
            summary = pd.DataFrame(pvalues)
            summary = summary.sort_values(by=['pvalues'], ascending=True)
            print('Model Summary :\n', summary)
            featurelist = pd.DataFrame(summary[summary['pvalues'] < .05])
            featurelist.to_html('Feature_pvalues.html')
            print('Model Feature list \n', featurelist)
            nonfeaturelist = summary[summary['pvalues'] > .05]
            correlation(X_train[featurelist.iloc[:, 0]])
        Y_pre = model.predict(X_test)
        CfnMatrix = confusion_matrix(Y_test, Y_pre)
        print("Confusion Matrix of : ",model,"\n",CfnMatrix)
        print("Accuracy of : ",model, accuracy_score(Y_test, Y_pre))
        roc_curve(Y_test,Y_pre,model)
    return model
def ensemble_classifier(X_train,X_test, Y_train,Y_test):
    clf1 = DecisionTreeClassifier()
    clf2 = SVC(probability=True)
    clf3 = LogisticRegression()
    clf4 = RandomForestClassifier(n_estimators=200, random_state=300, bootstrap=True)
    clf5 = GradientBoostingClassifier(n_estimators=100, random_state=500, max_depth=3)
    clf6 = BaggingClassifier(oob_score=True, n_estimators=90, random_state=300, base_estimator=DecisionTreeClassifier())
    ensemble_clf = VotingClassifier(estimators=[('dt', clf1), ('svm', clf2), ('lr', clf3),('Rf', clf4),('GB',clf5),('GBC',clf6)],voting='hard')
    ensemble_clf.fit(X_train, Y_train)
    y_pred = ensemble_clf.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy:", accuracy)
    print('Area under curve of',ensemble_clf,metrics.roc_auc_score(Y_test, y_pred))
def bestestimators(X_train,X_test, Y_train,Y_test,Dt):
    from sklearn.model_selection import GridSearchCV
    Dt.fit(X_train, Y_train)
    param_grid = {'criterion': ['gini', 'entropy'],'max_depth': [None, 5, 10, 15],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}
    grid_search = GridSearchCV(Dt, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)
    #result_dic = {'Params': X_train.columns, 'Dec_Tree_Features': Dt.feature_importances_}
    #Dec_Tree_Features = pd.DataFrame(result_dic)
    #Dec_Tree_Features.sort_values(by='Dec_Tree_Features', ascending=False, inplace=True)
    #print(Dec_Tree_Features)
    #Dec_Tree_Features.to_html('Decision_Tree_Features.html')

def main():
    train_data,test_data=dataloading('train_ml.csv','test_ml.csv')
    data_check(train_data)
    train_data=data_cleansing(train_data)
    Y=train_data['Attrition']
    X=train_data.drop('Attrition',axis=1)
    #correlation(train_data)
    X=pd.get_dummies(X,dtype=int,drop_first=True)
    print('\n Transformed Dummies Data :\n',X.head())
    X['MonthlyIncome']=np.log(X['MonthlyIncome'])
    X['MonthlyRate'] = np.log(X['MonthlyRate'])

    # DT Feature list
    #X=X[['id','MonthlyRate','OverTime_Yes','MonthlyIncome','PercentSalaryHike','StockOptionLevel','DailyRate','DistanceFromHome','JobInvolvement',
        #'JobRole_Laboratory Technician','EnvironmentSatisfaction']]

    #DT Feature + P values
    X = X[['id', 'MonthlyRate','OverTime_Yes','MonthlyIncome','PercentSalaryHike','StockOptionLevel','DailyRate',
           'DistanceFromHome', 'JobInvolvement','JobRole_Laboratory Technician', 'EnvironmentSatisfaction','HourlyRate','YearsWithCurrManager',
           'YearsAtCompany','MaritalStatus_Single','NumCompaniesWorked','JobRole_Sales Representative','BusinessTravel_Travel_Frequently','YearsInCurrentRole','TotalWorkingYears']]
    print(X.shape,X.shape[0],X.shape[1])

    """
    X=X[['id','MonthlyIncome','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager','StockOptionLevel','OverTime_Yes','JobRole_Sales Representative','MaritalStatus_Single',
         'BusinessTravel_Travel_Rarely','HourlyRate','JobRole_Manufacturing Director','JobRole_Laboratory Technician','YearsSinceLastPromotion',
         'JobLevel','JobRole_Research Director','EnvironmentSatisfaction','DistanceFromHome','Education','MaritalStatus_Married','RelationshipSatisfaction','NumCompaniesWorked']]
    """
    pca = PCA(n_components=X.shape[1])
    correlation(X)
    pca.fit(X)
    X = pca.transform(X)

    # Call the EDA
    EDAplot(train_data)

    # Perform train,test split
    X_train,X_test, Y_train,Y_test = model_selection.train_test_split(X, Y,test_size=0.80, random_state=100)

    # Build the Logistic Model and check its Summary

    LR=LogisticRegression(random_state=0)
    Dt = DecisionTreeClassifier()
    #Dt=DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=1,min_samples_split=5)
    bestestimators(X_train,X_test, Y_train,Y_test,Dt)
    Rf=RandomForestClassifier(n_estimators=200, random_state=300,bootstrap=True)
    GBC = GradientBoostingClassifier(n_estimators=100, random_state=500, max_depth=3)
    Bgc=BaggingClassifier(oob_score=True, n_estimators=90, random_state=300, base_estimator=DecisionTreeClassifier())
    ensemble_classifier(X_train, X_test, Y_train, Y_test)
    mlist = [LR,Rf,GBC,Bgc,Dt]
    model=model_test(X_train,X_test, Y_train,Y_test,mlist)

    # create submission file
    print('creating the submission file ....')
    test_data=pd.get_dummies(test_data,dtype=int,drop_first=True)

    test_data=test_data[['id', 'MonthlyRate', 'OverTime_Yes', 'MonthlyIncome', 'PercentSalaryHike', 'StockOptionLevel', 'DailyRate',
           'DistanceFromHome', 'JobInvolvement','JobRole_Laboratory Technician', 'EnvironmentSatisfaction','HourlyRate','YearsWithCurrManager',
           'YearsAtCompany','MaritalStatus_Single','NumCompaniesWorked','JobRole_Sales Representative','BusinessTravel_Travel_Frequently','YearsInCurrentRole','TotalWorkingYears']]

    """
    test_data=test_data[['id','MonthlyIncome','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager','StockOptionLevel','OverTime_Yes','JobRole_Sales Representative','MaritalStatus_Single',
         'BusinessTravel_Travel_Rarely','HourlyRate','JobRole_Manufacturing Director','JobRole_Laboratory Technician','YearsSinceLastPromotion',
         'JobLevel','JobRole_Research Director','EnvironmentSatisfaction','DistanceFromHome','Education','MaritalStatus_Married','RelationshipSatisfaction','NumCompaniesWorked']]
    """

    #print(test_data.head())
    Y1_pred = model.predict(test_data)
    submission_df = pd.DataFrame({'Id':test_data['id'],'Attrition':Y1_pred})
    print(submission_df.head())
    submission_df.to_csv('submission.csv', index=False)

main()
