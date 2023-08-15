import pandas as pd
import os
os.chdir("C:\ProgramData\Anaconda3\Scripts\IPBA_Project")
df=pd.read_csv('dm.csv')
df2=pd.read_csv('dataset.csv')
# Prints the dataframe as String like txt data
#print(df.to_string())

# create a dummy data set
mydataset={
                        'cars':["BMW","Volvo","Ford"],
                        'passings':[3,7,2]
                    }

Myvar=pd.DataFrame(mydataset)
print(Myvar)
print(Myvar.to_string)
print(pd.__version__)
a=[1,7,2]
myvar2=pd.Series(a,index=["x","y","z"])
print(myvar2)
print(myvar2[["x"]])
calories={'day1':400,'day2':380,"day3":390}
myvar3=pd.Series(calories)
print(myvar3)
print(pd.Series(calories,index=['day1','day2']))
data={
          "cal":[420,380,390],
            "duraion":[50,40,45]
}
mydata=pd.DataFrame(data)
print(mydata)

# Sort Pandas DataFrame by a column Ascending
#print(mydata.sort_values('cal',ascending=True))
# Sort Pandas DataFrame by a column Descending
#print(mydata.sort_values('cal',ascending=False))
# Group Python DataFrame with index as column
#groupeddf1=(mydata.groupby('cal',as_index=False))
#print('grouped1',groupeddf1.first())
# Group Python DataFrame withouth index as column
groupeddf2=(mydata.groupby('cal',as_index=True))
#print('grouped2',groupeddf2.first())
#print('Aggregate',groupeddf2['cal'].agg(['sum','count']))

# Aggregate function on Dataset csv
#print(df2.to_string)
print('grouped******** /n',df2.groupby('Manufacturing'))
print('grouped + Agg + Sum  ******* /n',df2.groupby('Manufacturing').agg(['sum']))
print('grouped + Agg + Sum + Count ******* /n',df2.groupby('Manufacturing').agg(['sum','count']))
print('grouped + Agg + Sum + Count ******* /n',df2.groupby('Manufacturing').agg(['sum','count']).sort_values('Manufacturing',ascending=False))

#print(df2.groupby('Salary').agg(['sum','count']))
