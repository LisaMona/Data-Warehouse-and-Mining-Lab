# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:15:45 2018


"""
import csv
import numpy as np
import pandas as pd


data = pd.read_csv('Cancer.csv')

data['Gender'][data.Gender=='M'] = '0'
data['Gender'][data.Gender=='F'] = '1'
data['HasCancer'][data.HasCancer==False] = '0'
data['HasCancer'][data.HasCancer==True] = '1'

first_five = data.iloc[:5,:]

data['Age'].fillna(np.mean(data['Age']))




a = 0
b = 0
l = []
with open('Cancer.csv') as csv_file:
    reader = csv.reader(csv_file)
    r = next(reader)
    for r in reader:
        if r[3] == '':
            a = r[1]
with open('Cancer.csv') as csv_file:
    reader = csv.reader(csv_file)
    r = next(reader)
    
    c = 0
    for r in reader:
        if r[1] == a and r[3]!='':
            b += float(r[3])
            c+=1
            print(r)
    b = b/c
    #print()
    r[3] = b
    print(r[3])


l = []
with open('Cancer.csv') as csv_file:
    reader = csv.reader(csv_file)
    read = next(reader)
    for read in reader:
        if read[3] == '':
            read[3]= l
        print(read)
        l.append(read)
with open('Cancer2.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(l)


