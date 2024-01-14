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
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import datetime
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, recall_score,precision_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# set the working directory
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')

#set the display options
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
pd.set_option('display.float_format',lambda x:'%.10f' %x)

# Function to load the data
def dataloading(file1):
    try:
        train_data=pd.read_csv(file1,na_values=[' ','NA','N/A'],parse_dates=['Effective To Date'])
        #print(train_data.isnull().sum())
        #print('\n count of nqunique',train_data.nunique())
        train_data.fillna(0)
        train_data = train_data.rename(columns={'Customer Lifetime Value': 'CLTV'})
        CLTV = train_data['CLTV']
        Customer=train_data['Customer']
        train_data=train_data.drop('Customer',axis=1)
        numcols=train_data.select_dtypes('number').columns
        catcols=train_data.select_dtypes('object').columns
        numcols=list(set(numcols))
        numcols.remove('CLTV')
        catcols=list(set(catcols))
        #print('\n Shape of Train Data',train_data.shape)
        #print('\n Head of Train data:\n',train_data.head())
        return train_data,numcols,catcols
    except FileNotFoundError:
        #print('file not present at path :',os.getcwd())
        logging.error('file not present at path :',os.getcwd())

# Function to check the data
def data_check(train_data):
    print('\n Shape of Train Data',train_data.shape)
    print('\n Train data data type:\n',train_data.info())
    print('\n Train data describe:\n',train_data.describe())
    print('\n Train data check for null:\n',train_data.isnull().sum())
    print('\n count of nqunique',train_data.nunique())
    print('\n Count of duplicate',train_data[train_data.duplicated()])

# Function to clean the data with Capping
def data_cleansing(train_data):
    IQR_Report=pd.DataFrame()
    for column in train_data:
        if column in ['id','Effective To Date','Customer'] or train_data[column].dtype==object:
            pass
        else:
            IQR_Report[column]=train_data[column].describe()
            IQR=train_data[column].quantile(0.75)-train_data[column].quantile(0.25)
            min=train_data[column].min()
            max=train_data[column].max()
            mean=train_data[column].mean()
            UL=round(train_data[column].quantile(0.75)+(1.5)*IQR)
            LL=round(train_data[column].quantile(0.25)-(1.5)*IQR)
            outliers=[x for x in train_data[column] if x < LL or x > UL]
            if column == 'CLTV':
                train_data['Priority']=pd.qcut(train_data[column],q=3,labels=['Low','Medium','High'])
                train_data = train_data.drop(['CLTV'], axis=1)
            else:
                percent_outlier= len(outliers)/len(train_data[column])*100
                IQR_Report.loc[len(IQR_Report)] = percent_outlier
                if percent_outlier > 5:
                    print('outlier > 5 % for ',column)
                    train_data[column]=np.where(train_data[column]>UL,UL,train_data[column])
                    train_data[column] = np.where(train_data[column]<LL,LL,train_data[column])
    #print("IQR_Report : \n",IQR_Report)
    #print("Cleansed Data describe:\n",train_data.describe())
    print('\n Head of Train data:\n', train_data.head())
    print ('\n Churn % is ',train_data['Priority'].value_counts(normalize=True))
    return train_data

# Function to check EDA for numerical and cat cols
def EDA(train_data,numcols,catcols):
    Y=train_data['Priority']
    for cols in train_data[catcols]:

        # Pie Chart for Demographics
        plt.figure(figsize=(10,5))
        plt.subplot(2,2,1)
        plot1 = train_data[cols].value_counts().plot(kind='pie',autopct='%1.2f%%')
        plt.title('Pie Chart for Demographics')

        # Sorted Count plot for Params
        plt.subplot(2,2,2)
        plot2 = sbrn.countplot(data=train_data,x=cols,order=train_data[cols].value_counts(ascending=False).index)
        for label in plot1.containers:
            plot1.bar_label(label)
        plt.title('Sorted Bar Chart for Demographics')

        # Count plot based on Priority
        plt.subplot(2,2,3)
        ho = ['High','Medium','Low']
        plot3 = sbrn.countplot(data=train_data,x=cols,hue='Priority',hue_order=ho)
        for label in plot3.containers:
            plot3.bar_label(label)
        plt.legend(loc='upper right')
        plt.title('Count plot grouped by Priority')

        # Count plot based on State
        plt.subplot(2, 2, 4)
        plot4 = sbrn.countplot(data=train_data, x=cols, hue='State')
        for label in plot4.containers:
            plot4.bar_label(label)
        plt.legend(loc='upper right')
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')
        plt.title('Bar Chart for state wise Distribution')

        plt.show()

        # Construct the dynamic file name
        #file_name = cols+".png"

        # Save the plot with the dynamic file name
        #plt.savefig(file_name)

        # Close the plot to free up memory
        #plt.close()


    for cols in train_data[numcols]:

        # Set color options for the box plot
        box_colors = ['skyblue', 'lightgreen', 'lightpink']
        train_data.boxplot(column=cols,by='Priority',sym='',patch_artist=True,boxprops=dict(facecolor=box_colors[0]),
                           capprops=dict(color=box_colors[1]),
                           whiskerprops=dict(color=box_colors[2]),
                           flierprops=dict(markeredgecolor=box_colors[0], marker='o'),
                           medianprops=dict(color='black')
                           )
        plt.show()

        # Construct the dynamic file name
        #file_name = cols+".png"

        # Save the plot with the dynamic file name
        #plt.savefig(file_name)

        # Close the plot to free up memory
        #plt.close()

def Data_To_SQL(train_data):
    import pandas as pd
    import mysql.connector

    # Connect to the MySQL database
    cnx = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Saurabh123',
        database='byop_data'
    )

    # Create a cursor object to execute SQL queries
    cursor = cnx.cursor()

    # Read the CSV file into a pandas DataFrame
    data = train_data

    # Iterate over each row in the DataFrame
    for _, row in data.iterrows():
        # Prepare the SQL query
        query = "INSERT INTO table_name (column1, column2, column3) VALUES (%s, %s, %s)"

        # Get the values from the current row
        values = (row['column1'], row['column2'], row['column3'])

        # Execute the query
        cursor.execute(query, values)

    # Commit the changes to the database
    cnx.commit()

    # Close the cursor and connection
    cursor.close()
    cnx.close()

# Function to check relationship between dependent categorical & independent continuos variables
def annova_test(numcols,train_data):

# Annova test is done between numerical output & a categorical Input
    Annova_test_df = pd.DataFrame(columns=['Hypo_Stat','Parameter','Hypo_Result','p_value'])

    # specify the order of the categories
    priority_map = {'Low':1,'Medium':2,'High':3}

    # perform ordinal encoding on the 'quality' column
    train_data['Pr'] = train_data['Priority'].map(priority_map)

    """
    from sklearn.preprocessing import LabelEncoder
    labelencoder= LabelEncoder() #initializing an object of class LabelEncoder
    train_data['Priority'] = labelencoder.fit_transform(train_data['Priority'])
    """

#fitting and transforming the desired categorical column.

    for columns in train_data:
        if columns in numcols and columns !='Customer':

            # Perform One-way ANNOVA

            Y=train_data['Pr']
            X=train_data[columns]

            f_value, p_value = stats.f_oneway(Y,X)
            #model = ols('Priority ~ columns',data=train_data).fit()

            # Print results
            #print('Parameter:',columns)
            #print("F-value:", f_value)
            #print("P-value:", p_value)


            # Print the ANOVA table
            #print("\n ANOVA value for:", columns)
            #print("\nHo "Avg CLV is Same for all:",cols,"\nH1 Avg CLV is Not Same for all:",cols)
            #print('p value of',columns,'is',p_value)

            if p_value < 0.05:
                #print('Reject Ho/Accept H1',"Avg CLV is Not Same for all:",columns)
                Annova_test_df.loc[len(Annova_test_df)]=['Priority has no impact due to',columns,'Reject Ho/Accept H1',p_value]
            else:
                #print('Accept Ho,Reject Ho',"Avg CLV is Same for all:",columns)
                Annova_test_df.loc[len(Annova_test_df)]= ['Priority has impact due to',columns,'Accept H0/Reject H1',p_value]

    Annova_test_df.to_html('Annova_test_df.html')
    print('\n Hypothesis_Annova_result :\n', Annova_test_df)
    return Annova_test_df

    #crosstab=pd.crosstab(Y,X)
    #print('Crosstab of Annova',columns,crosstab)

# Function to check relationship between dependent categorical & independent categorical variables
def crosstab_ChiSquare(catcols,traindata):
    from scipy.stats import chi2_contingency

    X=traindata[catcols]
    Y=traindata['Priority']
    Cat_Cross_Tab,Chi_Sqr_Result=pd.DataFrame(),pd.DataFrame(columns=['Param','ChiSqStat','Pvalue','Dof'])
    #print('cols in X \n',X.columns)
    for cols in X:
        Cat_CrossTab_Rows =pd.crosstab(X[cols],traindata['Priority'])
        Cat_CroosTab_Row_Per=pd.crosstab(X[cols],traindata['Priority'],normalize='index')
        sampledf=pd.concat([Cat_CrossTab_Rows,Cat_CroosTab_Row_Per],axis=1)
        #print('\n crosstab \n',sampledf)
        Cat_CrossTab_Cols = pd.crosstab(X[cols], traindata['Priority'],normalize='columns')
        #Cat_CrossTab_Cols_Per = pd.crosstab(X[cols], traindata['Priority'], normalize='index')
        #print('Col Cross Tab of ', cols, 'vs Priority\n', Cat_CrossTab_Cols)
        Cat_Cross_Tab=pd.concat([Cat_Cross_Tab,Cat_CrossTab_Rows],axis=0)
        #print('1st',Cat_Cross_Tab)
        Cat_Cross_Tab=pd.concat([Cat_Cross_Tab,Cat_CroosTab_Row_Per],axis=1)
        #print('2nd', Cat_Cross_Tab)

        #print('\n Row Cross Tab of ', cols, 'vs Priority\n', Cat_Cross_Tab)

        # Create a contingency table from the DataFrame
        #contingency_table = pd.crosstab(data['Category1'], data['Category2'])

        # Perform the chi-square test
        chi2, p_value, dof, expected = chi2_contingency(Cat_CrossTab_Rows)

        # Print the results
        #print("Chi-square statistic for: ",cols,{chi2})
        #print("P-value: ",{p_value})
        #print("Degrees of freedom: ",{dof})
        #print("Expected frequencies:")
        #print(expected)

        #Create a dictionary representing the new row data
        new_row = {'Param': cols, 'ChiSqStat': chi2,'Pvalue':p_value,'Dof':dof}

        # Append the new row to the DataFrame
        Chi_Sqr_Result.loc[len(Chi_Sqr_Result)] = new_row

    #print('Cat Cross Tab is :\n',Cat_Cross_Tab)
    Cat_Cross_Tab.to_html('Cat_Cross_Tab.html')
    Chi_Sqr_Result=Chi_Sqr_Result.sort_values(by='Pvalue',ascending=True)
    Chi_Sqr_Result.to_html('Chi_Sqr_Result.html')
    print('\n Chi_Square_Result :\n',Chi_Sqr_Result)
    return Chi_Sqr_Result,Cat_Cross_Tab

# Function for crosstab
def crosstab(train_data,numcols,catcols):
    RowWise_Crosstab,ColWise_Crosstab,NumRowWise_Crosstab,NumColWise_Crosstab,\
    NumRow_Col_Aggregated=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

    # Define the data for test
    for columns in catcols:
        if columns in ['Customer']:
            pass
        else:
            catcrosstabrows=pd.crosstab(train_data[columns],train_data['Priority'],rownames=[columns])
            Aggcatcrosstabrows = pd.crosstab(train_data[columns],train_data['Priority'],normalize='index')
            sampledf1=pd.concat([catcrosstabrows,Aggcatcrosstabrows],axis=1)
            RowWise_Crosstab=pd.concat([RowWise_Crosstab,sampledf1],axis=0)
            print('\n Row Wise Cross tabulation of cat cols',columns,'vs priority :\n',sampledf1,'\n')
            sampledf1=()
            #print('\n Row Wise Aggregated Cross tabulation of cat cols', columns, 'vs priority :\n',Aggcatcrosstabrows)
            catcrosstabcols = pd.crosstab(train_data[columns], train_data['Priority'],normalize='columns')
            Aggcatcrosstabcols = pd.crosstab(train_data[columns], train_data['Priority'], normalize='columns')
            sampledf2 = pd.concat([catcrosstabcols, Aggcatcrosstabcols], axis=1)
            ColWise_Crosstab = pd.concat([ColWise_Crosstab, sampledf2], axis=0)
            Row_Col_Aggregated=pd.concat([RowWise_Crosstab,ColWise_Crosstab],axis=1)
            #print('\n Col Wise Cross tabulation of cat cols', columns, 'vs priority :\n', catcrosstabcols)
            #print('\n Col Wise Aggregated Cross tabulation of cat cols', columns, 'vs priority :\n', Aggcatcrosstabcols)

    #print ('\n Consolidated cross tab & Agg CLTV Mean Row_wise:\n',RowWise_Crosstab)
    #print('\n Consolidated cross tab & Agg CLTV Mean Col_wise:\n', ColWise_Crosstab)
    RowWise_Crosstab.to_html('Cat_col_Row_Wise_Crosstab.html')
    ColWise_Crosstab.to_html('Cat_col_Col_Wise_Crosstab.html')
    Row_Col_Aggregated.to_html('Row_cols_Aggregated.html')

    for columns in numcols:
        if columns in ['Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Total Claim Amount','Income']:
            pass
        else:
            numcrosstabrows = pd.crosstab(train_data[columns], train_data['Priority'],rownames=[columns])
            Aggnumcrosstabrows = pd.crosstab(train_data[columns], train_data['Priority'], normalize='index')
            sampledf3 = pd.concat([numcrosstabrows, Aggnumcrosstabrows],axis=1)
            NumRowWise_Crosstab = pd.concat([NumRowWise_Crosstab, sampledf3],axis=0)
            print('\n Row Wise Cross tabulation of Numeric cols', columns, 'vs priority :\n', sampledf3,'\n')
            sampledf3=()
            #print('\n Row Wise Aggregated Cross tabulation of cat cols', columns, 'vs priority :\n',Aggnumcrosstabrows)
            numcrosstabcols = pd.crosstab(train_data[columns], train_data['Priority'], normalize='columns')
            Aggnumcrosstabcols = pd.crosstab(train_data[columns], train_data['Priority'], normalize='columns')
            sampledf4 = pd.concat([numcrosstabcols, Aggnumcrosstabcols], axis=1)
            NumColWise_Crosstab = pd.concat([NumColWise_Crosstab, sampledf4], axis=0)
            NumRow_Col_Aggregated = pd.concat([NumRowWise_Crosstab, NumColWise_Crosstab], axis=1)
            #print('\n Col Wise Cross tabulation of cat cols', columns, 'vs priority :\n', numcrosstabcols)
            #print('\n Col Wise Aggregated Cross tabulation of cat cols', columns, 'vs priority :\n', Aggnumcrosstabcols)
        #print('Consolidated cross tab & Agg CLTV Mean Row_wise:\n', NumRowWise_Crosstab)
        #print('Consolidated cross tab & Agg CLTV Mean Col_wise:\n', NumColWise_Crosstab)
        NumRowWise_Crosstab.to_html('Num_col_Row_Wise_Crosstab.html')
        NumColWise_Crosstab.to_html('Num_col_Col_Wise_Crosstab.html')
        NumRow_Col_Aggregated.to_html('Num_Row_cols_Aggregated.html')

# Function for groupby
def groupby(train_data, numcols, catcols):
    RowWise_groupby = pd.DataFrame()
    # Define the data for test
    for columns in catcols:
        if columns in ['Priority']:
            pass
        else:
            df=train_data.groupby(['Priority',columns]).agg({'Priority':['count']})
            print('Group by is:\n',df)
            RowWise_groupby=pd.concat([RowWise_groupby,df])
    print('\n Group by Df is:\n',RowWise_groupby)
    RowWise_groupby.to_html('RowWise_groupby.html')

# Function for correlaion of final features
def correlation(data,thr):
    col_corr=set()
    corr_matrix=data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>thr:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    print('\n Correlated Features\n',col_corr)
    correlation = data.corr()

    # Assuming you have your predictor variables in a DataFrame called 'X'
    # Add a constant term to the predictor variables
    X_with_constant = add_constant(data)

    # Calculate VIF for each predictor variable
    vif = pd.DataFrame()
    vif["Variable"] = X_with_constant.columns
    vif["VIF"] = [variance_inflation_factor(X_with_constant.values, i) for i in range(X_with_constant.shape[1])]

    # Print the VIF values
    print(vif)
    #print('\n correlation', correlation,'\n')

    # Generate the heatmap
    #sbrn.heatmap(correlation, annot=True, cmap='coolwarm')
    # Display the plot
    #plt.show()
    return col_corr

# Function for cross validation of Decision Tree
def bestestimators(X_train,X_test, Y_train,Y_test,model):
    from sklearn.model_selection import GridSearchCV
    model.fit(X_train, Y_train)
    param_grid = {'criterion': ['gini', 'entropy'],'max_depth': [None, 5, 10, 15],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}
    grid_search = GridSearchCV(model,param_grid,cv=5)
    grid_search.fit(X_train, Y_train)
    print("\n Decision Tree Best Param Found : ", grid_search.best_params_)
    print(" Best score found: ", grid_search.best_score_)
    return grid_search

# Function for evaluating best model based on Accuracy & ROC
def model_test(X_train,X_test, Y_train,Y_test,mlist):
    try:
        FinalFeaturelist=pd.DataFrame()
        for model in mlist:
            model.fit(X_train,Y_train)
            Y_pre = model.predict(X_test)
            CfnMatrix = confusion_matrix(Y_test,Y_pre)
            print("\n Confusion Matrix of : ",model,"\n",CfnMatrix)
            print("Accuracy of : ",model,accuracy_score(Y_test,Y_pre))
            recall = recall_score(Y_test,Y_pre, pos_label='positive',average='micro')
            f1 = f1_score(Y_test,Y_pre,pos_label='positive',average='micro')
            precision = precision_score(Y_test, Y_pre,pos_label='positive',average='micro')
            print("Recall Score : ",model,recall)
            print("f1 score is : ",model,f1)
            print("precision score is : ",model,precision)
            #roc_curve(Y_test,Y_pre,model)
    except Exception as e :
        print("An error occurred: {str(e)}")
    return model

# Function for evaluating ensemble voting classifier based on aggregation
def ensemble_classifier(X_train,X_test,Y_train,Y_test):
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
    CfnMatrix = confusion_matrix(Y_test,y_pred)
    print("\n Confusion Matrix of : Ensemble Classifier ",'\n',CfnMatrix)
    print("Accuracy of Ensemble Classifier :",accuracy)
    recall = recall_score(Y_test, y_pred, pos_label='positive', average='micro')
    f1 = f1_score(Y_test, y_pred, pos_label='positive', average='micro')
    precision = precision_score(Y_test, y_pred, pos_label='positive', average='micro')
    print("Recall Score : ",ensemble_clf, recall)
    print("f1 score is : ",ensemble_clf, f1)
    print("precision score is : ",ensemble_clf, precision)
    #print('Area under curve of',ensemble_clf,metrics.roc_auc_score(Y_test, y_pred))

def mystreamlit():
    pass

def Prin_Com_Analysis(X,Y):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principal_components,columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf,Y],axis=1)
    print('Final DF is',finalDf)

# Main function to call all above functions
def main():
    import numpy as np

    # Load Train data
    train_data,numcols,catcols=dataloading('BYOP_Data_Set.csv')
    print('\n Numeric cols are:',len(numcols),'\n',numcols)
    print('\n Categorical cols are:',len(catcols),'\n',catcols)

    # Check the data
    data_check(train_data)

    #cleanse the data
    train_data=data_cleansing(train_data)

    # write train data to a csv file
    train_data.to_csv('BYOP_train_data.csv')

    # EDA
    #EDA(train_data,numcols,catcols)

    # Data to SQL
    #Data_To_SQL(train_data)

    # Do cross tab & check the data
    crosstab(train_data, numcols, catcols)

    # Do Groupby & check the data
    groupby(train_data, numcols, catcols)

    # Def for cross tab of cat cols & chisquare test
    Chi_Sqr_Result,Cat_Cross_Tab=crosstab_ChiSquare(catcols,train_data)

    # Perform an Annova Test on the data for Ordinal to numeric cols
    Annova_test_df=annova_test(numcols,train_data)

    # distribute feature and not features based on p value
    Features=Chi_Sqr_Result[Chi_Sqr_Result['Pvalue']<=0.05]
    NonFeatures=Chi_Sqr_Result[Chi_Sqr_Result['Pvalue']>=0.05]
    print('\n Non Features List as per ChiSqTest :\n', NonFeatures)
    print('\n Features List as per ChiSqTest :\n',Features)
    Annova_Features=Annova_test_df[Annova_test_df['p_value']<0.05]
    print('\n Annova Feature List :\n',Annova_Features)
    Final_Features=pd.concat([Features['Param'],Annova_Features['Parameter']])
    print('\n Final Feature list from ChiSquare & Annova :\n',Final_Features)

    # split the data in train and test
    Y = train_data['Priority']
    X = train_data[Final_Features]

    # check for Correlation & remove features which are above correlation threshold of 0.7
    # corr_features = correlation(X_train[numcols],0.7)

    data = X[Final_Features]

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()  # initializing an object of class LabelEncoder
    catcols = data.select_dtypes('object').columns

    for columns in catcols:
        data[columns] = labelencoder.fit_transform(data[columns])
    corr_features = correlation(data,0.50)
    #print('\n correlation features are:',corr_features)

    # create model 2 after dropping correlated features
    # Perform train,test split
    X=X.drop(corr_features,axis=1)

    catcols = X.select_dtypes('object').columns
    for columns in catcols:
        X[columns] = labelencoder.fit_transform(X[columns])
        X[columns] = X[columns].astype('category')
        print('\n',columns,'has categorical Data type :',pd.api.types.is_categorical_dtype(X[columns]))

    #Perform a SMOTE to oversample minority & undersample Majority
    import numpy as np
    from sklearn.datasets import make_classification

    # Perform PCA
    Prin_Com_Analysis(X,Y)

    # Perform train,test split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.20,random_state=100)

    # create a simple logistic regression model and check the feature

    model = LogisticRegression(random_state=0)
    model.fit(X_train,Y_train)
    pvalues = {'X': X_train.columns,'pvalues': np.squeeze(model.coef_[0])}

    summary = pd.DataFrame(pvalues)
    summary = summary.sort_values(by=['pvalues'],ascending=False)
    print('\n Summary of Logistic Regression :\n',summary)

    # Predict the value of the Priority based on Logistic regression

    Y_pre = model.predict(X_test)
    CfnMatrix = confusion_matrix(Y_test, Y_pre)
    print("\n Confusion Matrix of :",model, '\n',CfnMatrix)
    print("Accuracy of : ",model,accuracy_score(Y_test, Y_pre))
    recall = recall_score(Y_test,Y_pre,pos_label='positive',average='micro')
    f1 = f1_score(Y_test,Y_pre,pos_label='positive',average='micro')
    precision=precision_score(Y_test,Y_pre,pos_label='positive',average='micro')
    print("Recall Score : ",model,recall)
    print("f1 score is : ",model,f1)
    print("precision score is : ",model,precision)

    # Intialise a Decision Tree & check best estimator using decision tree

    dt = DecisionTreeClassifier(criterion='gini',max_depth=10,min_samples_leaf=4,min_samples_split=5)
    grid_search = bestestimators(X_train,X_test,Y_train,Y_test,dt)
    Y_pre = grid_search.predict(X_test)
    DtCfnMatrix = confusion_matrix(Y_test,Y_pre)
    print("\n Confusion Matrix of :",dt, '\n', DtCfnMatrix)
    print("Accuracy of : ", dt, accuracy_score(Y_test, Y_pre))
    recall = recall_score(Y_test, Y_pre, pos_label='positive', average='micro')
    f1 = f1_score(Y_test, Y_pre, pos_label='positive', average='micro')
    precision = precision_score(Y_test, Y_pre, pos_label='positive', average='micro')
    print("Recall Score : ",grid_search,recall)
    print("f1 score is : ",grid_search,f1)
    print("precision score is : ",dt,precision)

    result_dic = {'Params': X_train.columns,'Dec_Tree_Features': dt.feature_importances_}
    Dec_Tree_Features = pd.DataFrame(result_dic)
    plt.barh(X_train.columns,dt.feature_importances_)
    plt.title('Decision Tree: Feature Importance')

    # Construct the dynamic file name
    file_name = 'Decision_Tree_Feature'+".png"

    # Save the plot with the dynamic file name
    plt.savefig(file_name)

    # Close the plot to free up memory
    plt.show()
    plt.close()

    X1 = X[['Number of Policies', 'Monthly Premium Auto']]

    # Perform train,test split
    X1_train, X1_test, Y1_train, Y1_test = model_selection.train_test_split(X1, Y, test_size=0.20, random_state=100)

    grid_search.fit(X1_train, Y1_train)

    Y1_pre = grid_search.predict(X1_test)
    CfnMatrix = confusion_matrix(Y1_test,Y1_pre)
    print("\n Confusion Matrix of :",grid_search,'\n',CfnMatrix)
    print("Accuracy of : ", grid_search, accuracy_score(Y_test,Y_pre))
    recall = recall_score(Y_test, Y_pre, pos_label='positive', average='micro')
    f1 = f1_score(Y_test, Y_pre, pos_label='positive', average='micro')
    precision = precision_score(Y_test, Y_pre, pos_label='positive', average='micro')
    print("Recall Score : ", grid_search, recall)
    print("f1 score is : ", grid_search, f1)
    print("precision score is : ", grid_search, precision)

    # Create the Random Forest classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    rf = RandomForestClassifier()

    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [10,20,30],
        'max_depth': [None,5,10],
        'min_samples_split': [2,5,10]
    }
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid)
    grid_search.fit(X_train,Y_train)

    # Retrieve the number of trees (n_estimators)
    num_trees = grid_search.best_estimator_.get_params()['n_estimators']
    bestestimator = grid_search.best_estimator_.get_params()

    print("Number of trees (n_estimators):", num_trees, bestestimator)

    # Predict
    Y_pre = grid_search.predict(X_test)
    CfnMatrix = confusion_matrix(Y_test,Y_pre)
    print("\n Confusion Matrix of :",grid_search,'\n',CfnMatrix)
    print("Accuracy of : ",grid_search,accuracy_score(Y_test,Y_pre))
    recall = recall_score(Y_test, Y_pre, pos_label='positive',average='micro')
    f1 = f1_score(Y_test, Y_pre, pos_label='positive', average='micro')
    precision = precision_score(Y_test,Y_pre,pos_label='positive',average='micro')
    print("Recall Score : ",rf,recall)
    print("f1 score is : ",rf,f1)
    print("precision score is : ",rf,precision)

    # Print the best parameters and best score
    sel = SelectFromModel(RandomForestClassifier(n_estimators=200))
    sel.fit(X_train,Y_train)
    selected_feat = X_train.columns[(sel.get_support())]
    len(selected_feat)
    print("Best RF Parameters: ",selected_feat)

    #plot the Features
    result_dic = {'Params': X_train.columns,'Random_Forest_Features': grid_search.feature_names_in_}
    Random_Forest_Features = pd.DataFrame(result_dic)
    plt.barh(X_train.columns,grid_search.feature_names_in_)
    plt.title('Random_Forest_Features: Feature Importance')

    # Construct the dynamic file name
    file_name = 'Random_Forest_Features' + ".png"

    # Save the plot with the dynamic file name
    plt.savefig(file_name)

    # Close the plot to free up memory
    plt.show()
    plt.close()

    """
    f_i = list(zip(X_train.columns,rf.feature_importances_))
    f_i.sort(key=lambda x: x[1])
    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
    plt.show()
    """

    # Predict
    grid_search.fit(X1_train, Y1_train)
    Y2_pre = grid_search.predict(X1_test)
    RfCfnMatrix = confusion_matrix(Y1_test, Y1_pre)
    print("\n Confusion Matrix of :", grid_search, '\n', CfnMatrix)
    print("Accuracy of : ", grid_search, accuracy_score(Y1_test, Y1_pre))
    recall = recall_score(Y1_test, Y1_pre, pos_label='positive', average='micro')
    f1 = f1_score(Y1_test, Y1_pre, pos_label='positive', average='micro')
    precision = precision_score(Y1_test, Y1_pre, pos_label='positive', average='micro')
    print("Recall Score : ", grid_search, recall)
    print("f1 score is : ", grid_search, f1)
    print("precision score is : ", grid_search, precision)

    # create an ensemble classifier and find the best model

    Rf = RandomForestClassifier(n_estimators=200, random_state=300, bootstrap=True)
    GBC = GradientBoostingClassifier(n_estimators=100, random_state=500, max_depth=3)
    Bgc = BaggingClassifier(oob_score=True, n_estimators=90, random_state=300, base_estimator=DecisionTreeClassifier())
    ensemble_classifier(X1_train, X1_test, Y1_train, Y1_test)
    mlist = [Rf, GBC, Bgc]
    model = model_test(X1_train,X1_test,Y1_train,Y1_test,mlist)

    #fig, ax = plt.subplots(figsize=(20, 15))
    #roc_curve(Y_test, Y_pre)

    GBC = GradientBoostingClassifier(n_estimators=1, random_state=200, max_depth=3)

    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [5, 10, 20],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5]
    }
    # Perform grid search using cross-validation
    grid_search = GridSearchCV(GBC, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)

    # Print the best parameters and the corresponding accuracy score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Accuracy Score: ", grid_search.best_score_)


    # summarize results
    import matplotlib.pyplot as pyplot
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    n_estimators = [5, 10, 20]
    print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    GBC2 = GradientBoostingClassifier(n_estimators=5, random_state=200, max_depth=4, learning_rate=0.3)
    GBC2.fit(X1_train, Y1_train)
    GBCfnMatrix = confusion_matrix(Y1_test, Y1_pre)
    print("\n Confusion Matrix of :", GBC2, '\n', GBCfnMatrix)
    print("Accuracy of : ", GBC2, accuracy_score(Y1_test, Y1_pre))
    recall = recall_score(Y1_test, Y1_pre, pos_label='positive', average='micro')
    f1 = f1_score(Y1_test, Y1_pre, pos_label='positive', average='micro')
    precision = precision_score(Y1_test, Y1_pre, pos_label='positive', average='micro')
    print("Recall Score : ", GBC2, recall)
    print("f1 score is : ", GBC2, f1)
    print("precision score is : ", GBC2, precision)

    import seaborn as sbrn
    def plot_confusion_Matrix(title, matrix):

        pyplot.figure(figsize=(5, 3))
        sbrn.heatmap(matrix, annot=True, cmap='Blues', fmt='g', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        pyplot.xlabel('Predicted')
        pyplot.ylabel('Actual')
        pyplot.title(title)
        pyplot.show()

    mymatrix = {'Logistic': CfnMatrix, 'Decision Tree': DtCfnMatrix, 'Random Forest': RfCfnMatrix,
                'Gradient Boosting': GBCfnMatrix}
    for keys, values in mymatrix.items():
        plot_confusion_Matrix(keys, values)

main()

"""
Hypothesis From EDA
Coverage type > Basic >Extended coverage > premium contributes the highest Churn across state
Male & Female contribute equally to churn
Majority of the customer has refused for renewal
customer are buying more Midsized>small>large policies across states
Bachelors > college > High school are buying more policies in that order
offer1>offer2 > offer3 > offer 4 are buying more policies in that order
policy tye> personal L3 > Personal L2 > are buying more policies across state
Suburban contributes more > rural and urban seems equal
Sales Channel > Agent > Branch > call centre > web
Employment status > Employed > unemployed  contributing to churn
Married > Single > Divorced contribute to churn in that order
Policy type : personal Auto is sold more than corporate auto
policy type Special Auto is highest in Oregon,California,Arizona
Vehicle class : Four-Door > SUV>Two Door > contributes to churn
California contributes to highest across the states

Grouped by Priority
Number of policies > High 2 > Medium -3-7 > low 1
Total claim Amount > High claimed highest claim amount ,low and medium is same
Monthly Premium Auto > High > Low > Medium

import dtale
dtale.show(data)

"""
