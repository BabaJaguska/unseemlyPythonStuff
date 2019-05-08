# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:13:57 2019

@author: minja
"""

import numpy as np


def stochasticRound(number): 
    q = np.trunc(number)
    remnant = abs(number - q)
    
    adjust = np.random.choice([0,1], p = [1-remnant, remnant])
    
    
    signAdjust = -1 if q<0 else 1 #ne moze np.sign zbog nule
    
    rounded = q + adjust*signAdjust
    
    return rounded
    
    
    
    