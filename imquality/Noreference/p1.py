#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:44:13 2022

@author: venkat
"""
import numpy as np
try:
    x = 1
    y = 0
    assert y != 0, "Invalid Operation"
    print(x / y)
 
# the errror_message provided by the user gets printed
except AssertionError as msg:
    print(msg)