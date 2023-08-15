import seaborn as sbrn
import matplotlib.pyplot as plt

Data=sbrn.load_dataset('tips')
#Data=Data.fillna(0)
head=Data.head()
print(head)

#Linear plot
#sbrn.set_style('darkgrid')
sbrn.set_style('whitegrid')
sbrn.lmplot(x='total_bill',y='tip',data=Data)
plt.show()

# Add Hue and Markers
sbrn.lmplot(x='total_bill',y='tip',data=Data,hue='sex',markers=['o','v'])
plt.show()

# Add Multiple plots
sbrn.lmplot(x ='total_bill', y ='tip', data = Data, col ='sex', row ='time', hue ='smoker')
plt.show()

# Adjust Aspect ratio
sbrn.lmplot(x ='total_bill', y ='tip', data = Data, col ='sex', row ='time', hue ='smoker', aspect = 0.6, palette ='coolwarm')
plt.show()