#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3 Feb 2021

@author: jeff
"""

'''
Operation Shed Empty.
Should Tim empty the shed before it floods?
'''

# Begin by importing coast and other packages
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import os,sys
coastdir = os.path.dirname('/Users/jeff/GitHub/COAsT/coast')
sys.path.insert(0, coastdir)
import coast



date_start = np.datetime64('2021-01-25')
date_end = np.datetime64('now') 

# Load in data from the Shoothill API. Gladstone dock is loaded by default
hnk = coast.TIDEGAUGE()
hnk.dataset = hnk.read_shoothill_to_xarray(stationId="1032", date_start=date_start, date_end=date_end)
hnk.plot_timeseries()
plt.xlabel('date')
plt.ylabel('Water level on Hinksey Stream (Cold Harbour) (m)')
plt.legend().remove()
plt.title('Operation Shed Empty')



