# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:27:09 2018

@author: Monalisa
"""

import math
#from math import *
from decimal import Decimal
def p_root(value, root):
     
    root_value = 1 / float(root)
    return round (Decimal(value) **
             Decimal(root_value), 3)
 
def minkowski(x, y, p_value):
     
    return (p_root(sum(pow(abs(a-b), p_value)
            for a, b in zip(x, y)), p_value))
x = (2,5,13,8)
y = (4,5,18,7)
euc = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
print('Euclidean distance between x and y: ',euc)
man = sum(abs(e - s) for s,e in zip(x, y))
print('Manhattan distance between x and y: ',man)
p = 3
print('Minkowski distance between x and y: ',minkowski(x,y,p))