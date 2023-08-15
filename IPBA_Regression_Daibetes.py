from sklearn import  datasets
from sklearn.model_selection import train_test_split

dbData=datasets.load_diabetes()
print(dbData.DESCR)
print(dbData.feature_names)

X=dbData.data
Y=dbData.target

print(X.shape,Y.shape)
hd=X.head()
print(hd)


