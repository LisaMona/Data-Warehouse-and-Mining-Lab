# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:41:10 2018

@author: Monalisa
"""


import numpy as np
import math

marks = [150,234,265,190,290]
avg = np.mean(marks)
stde = np.std(marks)
base = min(marks)
print(base)
ranges = max(marks) - base
max_min_normalized = [(x-base)/float(ranges) for x in marks]
print('After max-min normalization:',max_min_normalized)
z_score_normalized = [(x - avg)/stde for x in marks]
print('After z_score normalization:',z_score_normalized)
count = 0
num = max(marks)
while (num > 0):
  num = num//10
  count = count + 1
print(count)
new_marks = []
for x in marks:
    new_marks.append(x/(10**count))
print('After decimal_scaling normalization:',new_marks)