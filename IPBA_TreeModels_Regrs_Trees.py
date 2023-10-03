import os
import pandas as pd
import seaborn as sbrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import sklearn.model_selection as model_selection
import warnings
warnings.filterwarnings("ignore")

# set the columns width for display
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
filename='credit_history.csv'
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
Data=pd.read_csv(filename)
print(Data.head())
print(Data.isnull().sum())
print(Data.info())
print(Data.describe())
print(Data['years'].mean())
Data.fillna(Data['years'].mean(),inplace=True)
print(Data.isnull().sum())

# create X Independent and Y Dependent variable
Y=Data['default']
X=Data.drop('default',axis=1)

print('prediction set \n',Y.head())
print('feature set \n',X.head())

# Translate categorical columns into dummies
X=pd.get_dummies(X,dtype=int)
print(X.head())

#Training and Testing dataset creation
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,Y,test_size=0.2,random_state=2)

import sklearn.tree as tree
# Initialise the decision tree classifier
clf=tree.DecisionTreeClassifier(max_depth=5,random_state=200)

#train the tree on training set
clf.fit(X_train,y_train) #passing the feature set and also corresponding labels


#check prediction on training set
prediction=clf.predict(X_train)
print(y_train,prediction)

# confusion Matrix # how to validate the classificiation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Non Defaulter=0, Defaulter=1
conf_matrix=confusion_matrix(y_train,prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=clf.classes_)
disp.plot()
plt.show()

#Accuracy (TN+TP/Total number of records in train set
#2185+1855

print(clf.score(X_train,y_train))
predict_test=clf.predict(X_test)

conf_matrix=confusion_matrix(y_test,predict_test)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=clf.classes_)
disp.plot()
plt.show()

print(clf.score(X_test,y_test))
X_train2=X_train[(X_train.grade_A <= 0.5)]
round(X_train2.shape[0]/X_train.shape[0],2)*100
print(X_train2.shape[0])

# How to optimise the model using hypertuning using GridSearch Cross Validation

mod = model_selection.GridSearchCV(clf,param_grid={'max_depth':[2,3,4,5,6,7,8,9,10]})
mod.fit(X_train,y_train)
print(mod.best_estimator_)
print(mod.best_score_)
print(mod.best_params_)

mod = model_selection.GridSearchCV(clf,param_grid=[{'criterion':['entropy','gini']},{'max_depth':[2,3,4,5,6,7,8,9,10]}])
mod.fit(X_train,y_train)
print('mod',type(clf.feature_importances_),clf.feature_importances_)
#print('mod',X_train.columns)
#print('mod',mod.best_params_)

# sample code to display the results
print('mod best estimator \n',mod.best_estimator_)
print('mod score',mod.best_score_)
result_dic={'Params':X_train.columns,'FeatImpor':clf.feature_importances_}
results=pd.DataFrame(result_dic)
print(results.sort_values('FeatImpor',ascending=False))

#Validation Data Set

#How to visualise a Decision Tree
from sklearn import tree

fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf, fontsize=10)
plt.show()


""""
#How this decision Tree look like , lets visualise the decision tree
# step1 - Install a package pydotplus !pip install pydotplus
# stet2 - Download graphviz

import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pydotplus,os
dot_data=tree.export_graphviz(clf,out_file=None,feature_names=X.columns,class_names=["0","1"],filled=True,rounded=True,special_characters=True,proportion=True)
graph=pydotplus.graph_from_dot_data(dot_data)

from IPython.display import Image
Image(graph.create_png())

"""





