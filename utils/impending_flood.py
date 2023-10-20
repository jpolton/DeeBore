#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:55:23 2021

@author: jeff
"""

'''
This is a demonstration script for using the TIDEGAUGE object in the COAsT
package. This object has strict data formatting requirements, which are
outlined in TIDEGAUGE.py.

This tutorial uses data from the Shoothill API. This does require a key to be
setup. It is assumed that the key is privately stored in
 config_keys.py

This API aggregates data across the country for a variety of instruments but,
 requiring a key, is trickier to set up.
To discover the StationId for a particular measurement site check the
 integer id in the url or its twitter page having identified it via
  https://www.gaugemap.co.uk/#!Map
E.g  Liverpool (Gladstone Dock stationId="13482", which is read by default.

Usage:
DeeBore% python utils/impending_flood.py
'''

# Begin by importing coast and other packages
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import os,sys
coastdir = os.path.dirname('/Users/jeff/GitHub/COAsT/coast')
sys.path.insert(0, coastdir)
#import coast

import sys, os
sys.path.append(os.path.dirname(os.path.abspath("shoothill_api/shoothill_api.py")))
from shoothill_api import GAUGE
#from shoothill_api.shoothill_api import GAUGE


ndays = 5

date_end = np.datetime64('now')
date_start = date_end - np.timedelta64(ndays, 'D')



# Load in data from the Shoothill API. Gladstone dock is loaded by default
liv = GAUGE()
liv.dataset = liv.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)
#liv.plot_timeseries()

ctr = GAUGE()
ctr.dataset = ctr.read_shoothill_to_xarray(station_id="7899" ,date_start=date_start, date_end=date_end)
#ctr.plot_timeseries()

plt.close('all')
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

## Only get tides over the weir with 8.75m at Liverpool
fig.suptitle('Timing of Chester Meadows flood relative to tides. Storm Christoph, Jan 2021')
ax1.scatter(ctr.dataset.time, ctr.dataset.sea_level, s=1)
ax2.scatter(liv.dataset.time, liv.dataset.sea_level, s=1)
ax2.plot( [date_start, date_end], [8.75,8.75], 'k--')
ax1.set_ylabel('Chester Weir (m)')
ax2.set_ylabel('Gladstone Dock, Liverpool (m)')



#date_start = np.datetime64('2021-01-17')
#date_end = np.datetime64('2021-01-25')

count = 0

try:
    ctr = GAUGE()
    ctr.dataset = ctr.read_shoothill_to_xarray(station_id="7899" ,date_start=date_start, date_end=date_end)
    ctr.dataset['shft'] = -ctr.dataset.sea_level[0].values + count

    #count += 0.5
    ctr2 = GAUGE()
    ctr2.dataset = ctr.read_shoothill_to_xarray(station_id="7900" ,date_start=date_start, date_end=date_end)
    ctr2.dataset['shft'] = -ctr2.dataset.sea_level[8].values + count
    ctr2.plot_timeseries()

except:

    ctr23 = GAUGE()
    ctr23.dataset = ctr.read_shoothill_to_xarray(station_id="15563", date_start=date_start, date_end=date_end)
    ctr23.dataset['shft'] = -ctr23.dataset.sea_level[0].values + count
    ctr23.plot_timeseries()

count += 0.5
iron = GAUGE()
iron.dataset = ctr.read_shoothill_to_xarray(station_id="968" ,date_start=date_start, date_end=date_end)
iron.dataset['shft'] = -iron.dataset.sea_level[0].values + count
iron.plot_timeseries()

count += 0.5
farn = GAUGE()
farn.dataset = ctr.read_shoothill_to_xarray(station_id="972" ,date_start=date_start, date_end=date_end)
farn.dataset['shft'] = -farn.dataset.sea_level[0].values + count
farn.plot_timeseries()

count += 0.5
manh = GAUGE()
manh.dataset = ctr.read_shoothill_to_xarray(station_id="963" ,date_start=date_start, date_end=date_end)
manh.dataset['shft'] = -manh.dataset.sea_level[0].values + count
manh.plot_timeseries()

count += 0.5
chrk = GAUGE()
chrk.dataset = ctr.read_shoothill_to_xarray(station_id="957" ,date_start=date_start, date_end=date_end)
chrk.dataset['shft'] = -chrk.dataset.sea_level[0].values + count
chrk.plot_timeseries()

#count += 0.5
corwen = GAUGE()
corwen.dataset = ctr.read_shoothill_to_xarray(station_id="962" ,date_start=date_start, date_end=date_end)
corwen.dataset['shft'] = -corwen.dataset.sea_level[0].values + count
corwen.plot_timeseries()

count += 0.5
deebr = GAUGE()
deebr.dataset = ctr.read_shoothill_to_xarray(station_id="971" ,date_start=date_start, date_end=date_end)
deebr.dataset['shft'] = -deebr.dataset.sea_level[0].values + count
deebr.plot_timeseries()

count += 0.5
bala = GAUGE()
bala.dataset = ctr.read_shoothill_to_xarray(station_id="965" ,date_start=date_start, date_end=date_end)
bala.dataset['shft'] = -bala.dataset.sea_level[0].values + count
bala.plot_timeseries()



fig, ax = plt.subplots()

## Only get tides over the weir with 8.75m at Liverpool
fig.suptitle('River Dee water levels')
for var in [deebr, corwen, chrk, manh, farn, iron, ctr23]: # ctr2, ctr]:
    ax.scatter(var.dataset.time, var.dataset.sea_level  + var.dataset.shft, s=1, label=var.dataset.site_name)

ax.set_ylabel('relative river height (m)')
ax.set_xlabel('date')
myFmt = mdates.DateFormatter('%d-%a')
ax.xaxis.set_major_formatter(myFmt)
plt.legend(markerscale=6)
plt.savefig('impending_flood.png')


## Ctr + Ironbridge + Farndon

## Only get tides over the weir with 8.75m at Liverpool
fig, ax = plt.subplots()

plt.title('River Dee water levels')
for var in [farn, iron, ctr23]: #ctr2, ctr]:
    ax.scatter(var.dataset.time, var.dataset.sea_level  + 0*var.dataset.shft, s=1, label=var.dataset.site_name)

ax.set_ylabel('relative river height (m)')
#ax.set_xlabel('Jan 2021')
ax.set_xlabel('Date')
myFmt = mdates.DateFormatter('%d-%a')
ax.xaxis.set_major_formatter(myFmt)
plt.legend(markerscale=6)
plt.savefig('impending_flood_short.png')

