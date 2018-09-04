# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:05:26 2018

@author: Student
"""


a = [1,2,3,4,5,9,8,0,6,7,7,6,4,7,6,76,6,6,6,6,6]

#Finding the mean of a list
def mean(a):
    mean = sum(a)/len(a)
    return mean
#a = [1,2,3,4,5]
b = mean(a)
print("mean:",b)

#Finding the mean of a list
def median(a):
    a.sort()
    #print(a)
    if(len(a)%2!=0):                            #if  odd number of entries
        index = (len(a)/2) 
        med = a[index]
        #print("median:",med)
    else:                                       #if  even number of entries
        index = (len(a)/2)
        med = float(a[index]+a[index-1])/2      #mean of 2 middlemost numbers
        #print("median:",med)
    return med
#a = [1,2,3,4,5,9,8,0,6,7]
c = median(a)
print("median:",c)
    
#Finding the mean of a list
dic = {}
def mode(a):
    for i in a:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] = dic[i]+1
        
        #dic = dict({a[i]:a.count(a[i])})        #dictionary created with list entries
        m = max(dic.values())                   #max count of dictionary values
        for k, l in dic.iteritems():
            if(l==m):
                mode = k
    #print(dic)            
    return mode
    
a = [1,2,3,4,5,9,8,0,6,7,7,6,4,7,6,76,6,6,6,6,6]
d = mode(a)
print("mode:",d)
    



































'''

#dic = {}
def mode(a):
    for i in range(len(a)):
        if a[i] is not in dic:
            dic[value] = 1
        else:
            dic[value] = dic[value]+1
            
        
        dic = dict({a[i]:a.count(a[i])})        #dictionary created with list entries
        m = max(dic.values())                   #max count of dictionary values
        for k, l in dic.iteritems():
            if(l==m):
                mode = k
    return mode
    
a = [1,2,3,4,5,9,8,0,6,7,7,6,4,7,6,76,6,6,6,6,6]
d = mode(a)
print("mode:",d)

     sum1 = 0; mean =0
    for i in range (len(a)):
        sum1 = sum1+a[i]   
a = [1,2,3,4,5,9,8,0,6,7,7,6,4,7]
for i in range (len(a)):
    b= a.count(a[i])
    #print(a[i],b)
    print(b)
    #print(b)

a = [1,2,3,4,5,9,8,0,6,7,7,6,4,7]
for i in range(len(a)):
    dic = dict({a[i]:a.count(a[i])})
    #print(dic)
c = max(dic.values())
for k, l in dic.iteritems():
    if(l==c):
        print(k)    
d = dic[max(dic.values())]
print(d)    

b = mode(a)
print(b)    

#Finding the mean of a list

    '''
    
        