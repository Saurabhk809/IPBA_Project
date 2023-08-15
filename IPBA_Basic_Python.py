import os

os.chdir('C:\ProgramData\Anaconda3\Scripts\IPBA_Project')
print(os.getcwd())

# Basic Python
# Integers
# float
# Strings
# Boolean
# List
# Dictionary
# Integers

Data=[10,20,30,40,50,60,70,80,90,99]
Data2=[1,2,3]
for item in Data:
    print(item,item+1)
    print(item, item * 10)

import numpy as np
mydata=np.array(Data)
print('Numpy',mydata+10)
print('Numpy mult',mydata*10)
mydata=np.array(Data[4:9])


income=100
Age=25.5
profitable=True
Course=' Data Science with Python'
Ins='IIM'
print(income, type(income),Age,type(Age),Course,type(Course),profitable,type(profitable))
print(Course+Ins)
print(income+Age)

#print(Course+Age) # cannot club Str+Int
pye=23.456
print(pye,type(pye))
print(int(pye),float(pye))

# List : Single value collection which are mutable and iterable comprised of integers,strings
Data=[10,20,30,40,50,60,70,80,90,99]
Income=[1,2,3,4,5,6,7,'ABC','XYZ',49.99,True]
print(Income)
print('third element',Data[3])
print(Income[0])
print(Income[:]) # prints all the elements
print(Income[2:6]) # Print all elements from Start+1 and including end
print(Income[:2]) # Print first two elements
print(Income[0:4])
for x in range(len(Income)):
    print(Income[x])

#List of List
Income=["Sunit",5000,"Amit",75000,"Tina",6500,"Heena",45000,"Reena",25000]
Income2=[["Sunit",5000],["Amit",75000],["Tina",6500],["Reena",25000]]
print(Income)
print(Income2)
print(Income[1]+Income[3])
print(Income2[0][1])
print(Income2[1][1])
print(len(Income2),len(Income2[0]))

# Iterate a list of list using for loop
i=0
for x in range(len(Income2)):
    for y in range(len(Income2[i])):
        print(Income2[x], ':',Income2[x][y])
    i=i+1

# Positive & Negative Index
Income=["Sunit",5000,"Amit",75000,"Tina",6500,"Heena",45000,"Reena",25000]
print('first item of list',Income[0]) # First item from list
print('first and second item',Income[0 :2]) # First but Excluding last
print('Second and third last item',Income[1:3 ]) # Second but excluding last
print('All items from index',Income[2:]) #' All items from index
print('Last item of list',Income[-1]) # last item from list
print('Second last and third last item',Income[-3 :-1]) # Excluding last item
print('from last to third last ',Income[-3 : ]) # Including last item
print('from 3rd last +1 item',Income[ : -3]) # Including last item
#print('All items from index',Income[::-1]) #'*All items from index, reverse**

# program to swap two numbers
mylist=[1,2,3,4,5]
print(mylist[::-1])
Income=["Sunit",5000,"Amit",75000,"Tina",6500,"Heena",45000,"Reena",25000]
print('Income step -1',Income[::-1])
print('Income step -2',Income[::-2])
print('Income step -3',Income[::-3]) # Check for reverse
Income.reverse()
print('Income reversed',Income,'\n',list(reversed(Income)))
print(Income[-2::])

print('abc',mylist[len(mylist)-1],mylist[len(mylist)-2],mylist[len(mylist)-3],mylist[len(mylist)-4],mylist[len(mylist)-5])
x=len(Income)-1
while x >= 0:
    print('reversed',Income[x])
    x=x-1
#print(mylist[-1],mylist[-2])
if 'Amit' in Income:
    print('income',Income.index('Amit'))
else:
    pass

# List Manipulation : Change "=", Add new "+", Remove Elements "del"
Income=["Sunit",5000,"Amit",75000,"Tina",6500,"Heena",45000,"Reena",25000]
print(Income)
Income[1]=7000
Income[0]=8000
print(Income)
Income3=Income+[9000]+['Ajay']+[8000]+['Vijay']+[4000,'Sujay']
Income3.insert(1,6000) # Insert will shift the Place and insert in between existing values
Income3.insert(2,'ABCD')
print('Shifting',Income3)
Income4=Income+[9000,4000,'Sujay']
print(Income)
del(Income3[9])
del(Income3[3:5])
Income3.insert(3,'Anuj')
print(Income3)
Income3.append(10000) # Append witll Add the values at the end
print(Income3)
Income3.append(10000)
print('methods',dir(Income3))
Income3.__delitem__(2) # you can also use del(Income[0:2)
print(Income3)

# Use by Reference and Use by Value
Income4=Income3 # Used by Same memory reference
Income5=Income3.copy() # or can use Income5=Income3[:] , Picks the value not memory reference
print('Income3',Income3)
print('Income4',Income4)
print('Income5',Income5)
del(Income4[1])
Income4.insert(1,'Shamit')
print('Income3',Income3)
print('Income4',Income4)
print('Income5',Income5)

# How to check methods on any variable 'dir'
print('Inbuilt methods',dir(Income4))
print('String methods',dir(Course))

# Functions # reusable code to solve a particular tasks
print(len(Income3))
print(len(Data),max(Data),min(Data),(Data))

# Method used on Python Objects represented by '.'
Data=[10,20,30,40,50,60,70,80,90,99]
print('Data',Data)
print('Income',Income3)
print(Data.count(10))
print(Income3.count('Ajay'))
Data.append(999)
print(Data)
mystring='Life is beautiful'
print(mystring.count('i'),mystring.index('i'))
print(mystring.upper())
print(mystring.lower())
print('sorted',sorted(Data))
print('reversed list',sorted(Data,reverse=True))

#Iteratbles and Iterators # For loop
location=['Delhi','Mumbai','Chennai']  # location is an iterable
for places in location:
    print(places)

letter='Jigsaw'                                    # Jigsaw is an Iterable
for x in letter:
    print(x)

Course='Python'
it=iter(Course)
next(it)
print(*it)

# Conditional statement if else
assign_txt='Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] ((listen) ; born 17 September 1950)[b] is an Indian politician who has served as the 14th Prime Minister of India since May 2014. Modi was the Chief Minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister from outside the Indian National Congress. '
c,a,n,l,ak=0,0,0,0,0
for txt in assign_txt:
    if txt.isupper():
        c=c+1
    elif txt.isalpha():
        a=a+1
    elif txt.isdigit():
        n=n+1
    elif txt.islower():
        l=l+1
        print(txt)
    elif txt.isascii():
        ak=ak+1
print('upperletter :',c,'Alphanumeric:',a,'numbers:',n,'lower:',l,'asci',ak)

# Split the line based on character
mylist=assign_txt.split(' ')
print(mylist)
for item in mylist:
    print(item.strip(' ''[''('';'')'""'"]'':'']'))

# List comprehensions
age=[23,12,67,78,84]
new_age=[]

for x in range(len(age)):
    print(x,age[x])

for item in age:    # Done using a for loop
    new_age.append(item+1)
print(new_age)

new_age = [y+1 for y in age] # Done using list comprehension
print(new_age)

# create a list containing squares of c
c=[2,3,4,5]
D=[y**2 for y in c]  # List comprehension
print(D)

# create a list of a and b such that it looks like the following
#[[1,'a'],[2,'b'],[3,'c'],[4,'d']]
a=[1,2,3,4]
b=['a','b','c','d','e']

mylist=[[x,y] for x in a for y in b if a.index(x)==b.index(y)]
print(mylist)

# create a list comprehension of pairs
pairs=[[x,y]for x in range(0,2)for y in range(3,5)]
print(pairs)

# Zip another generator function to create pairs
a=[1,2,3,4]
b=['a','b','c','d']
c=zip(a,b)
print(list(c))
# can also create list of list
mylst=[list(x) for x in zip(a,b)]
print(mylst)

# Check string in a list
x=['P',6,1,'Q',5,'R']
strings=[y for y in x if type(y)==str]
print(strings)
Nonstrings=[y for y in x if type(y)!=str]
print(Nonstrings)
strings,Nonstrings=[],[]
# Strings non Strings
[strings.append(y) if type(y)==str else Nonstrings.append(y) for y in x]
print('string is',strings,'Non String',Nonstrings)

# Even Odd
x=range(100)
even,odd=[],[]
[even.append(y) if y%2==0 else odd.append(y) for y in x]
print('even',even,'odd',odd)

# Prime number
x=range(1,100)
prime,notprime=[],[]
#[prime.append(y) if y%(int(y/2))==0 else notprime.append(y) for y in x]
for y in x:
    if y==1 or y==2:
        #print('prime',y)
        prime.append(y)
    elif y > 2: # Y can be 3 to 10
            i=1
            while i < (y-1):
                if y%(y-i)==0:
                    #print('not prime',y)
                    notprime.append(y)
                    break
                i=i+1
            else:
                #print('prime',y)
                prime.append(y)
    else:
        pass

print('prime',prime,'notprime',notprime)





