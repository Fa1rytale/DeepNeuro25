# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 21:58:17 2025

@author: Denis
"""

from random import randint
randlist = []
n = 100
while (len(randlist)<n):
    randlist.append(randint(1, 10))
print(randlist)
SummEl = 0
for i in randlist:
    if i%2==0:
        SummEl+=i
print('Сумма четных элементов массива равна '+str(SummEl))
    
    