import os
import numpy as np
os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
print(os.getcwd())

# Lambdas quicker way to write functions similar to lists
Square=lambda x,y:x**y
print(Square(2,3))
print(Square(10,20))
print(Square(11,22))

Square1=[2,5,6,7,4]
Square2=map(lambda x:x**2,Square1)
print(list,list(Square2))

# Advantage of using Numpy
Square3=np.array(Square1)
print(Square3**2)
print(Square3**3)
print(Square3+1)
print(Square3+10)

# Lambda , Map, Filter
mass=[45,555,65,76]
ht=[1.65,1.70,1.55,1.80]
bmi=list(map(lambda x,y:x/y**2,mass,ht))
print('bmi',bmi)
bmi_filter=list(filter(lambda x:x>20,bmi))
print('bmi greater than 20',bmi_filter)

# Advantage of using numpy
npmass=np.array(mass)
npht=np.array(ht)
npbmi=npmass/npht**2
print('numpy bmi',npbmi)
print('numpy bmi greater than 20',npbmi[npbmi>20])
print('numpy bmi less than 20',npbmi[npbmi<20])

# lambda filter for length
cities=['Pune','Delhi','Mumbai','Cochin','Bengaluru']
city_len=list(filter(lambda x:len(x)<6,cities))
print('cities less than 6',city_len)

# Advanrage of numpy
num_cities=np.array(cities)
print('length >6 ',len(num_cities)>6) ## ??
print('length <6 ',len(num_cities)<6) ## ??

# Tuples Immutable data types
mytuple=(1,2,'ABC',3,'XYZ')
print(mytuple[0],mytuple[1])
print(dir(mytuple))

# Dictionaries
mydic={"umesh":3000,"Ramesh":4000,"Nilesh":25000}
print(mydic)
print(mydic.keys())
print(mydic.values)

# how to access key values
for keys,values in mydic.items():
    print(keys,values)

a=[item for item in mydic]
print(a)

# Dictionary of Dictionary
Income1={'Ramesh':{'Gender':'male','Income':4000},
                'Ana':{'Gender':'female','Income':6600},
                'Uma':{'Gender':'female','Income':2000},
                'Umesh':{'Gender':'male','Income':5800}
               }

# Print income of Ramesh
print(Income1['Ramesh']['Income'])

# Add dictionary
Income1['Seema']={'Gender':'female','Income':3600}

print(Income1)

# Create a dictionary with occurance of each words in string
assign_txt='Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] ((listen) ; born 17 September 1950)[b] is an Indian politician who has served as the 14th Prime Minister of India since May 2014. Modi was the Chief Minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister from outside the Indian National Congress. '
worddic={}
for words in assign_txt.split(' '):
    if words in worddic:
        worddic[words] =worddic[words]+1
    else:
        worddic[words]=1
print(worddic)