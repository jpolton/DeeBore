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





date_start = np.datetime64('2021-01-11')
date_end = np.datetime64('2021-01-30') 

# Load in data from the Shoothill API. Gladstone dock is loaded by default
liv = coast.TIDEGAUGE()
liv.dataset = liv.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)
#liv.plot_timeseries()

ctr = coast.TIDEGAUGE()
ctr.dataset = ctr.read_shoothill_to_xarray(stationId="7899" ,date_start=date_start, date_end=date_end)
#ctr.plot_timeseries()

plt.close('all')
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

## Only get tides over the weir with 8.75m at Liverpool
fig.suptitle('Timing of Chester Meadows flood relative to tides. Storm Christoph, Jan 2021')
ax1.scatter(ctr.dataset.time, ctr.dataset.sea_level, s=1)
ax2.scatter(liv.dataset.time, liv.dataset.sea_level, s=1)
ax2.plot( [date_start, date_end], [8.75,8.75], 'k--')
ax1.set_ylabel('Chester Weir (m)')
ax2.set_ylabel('Gladston Dock, Liverpool (m)')



date_start = np.datetime64('2021-01-17')
date_end = np.datetime64('2021-01-25') 

count = 0

ctr = coast.TIDEGAUGE()
ctr.dataset = ctr.read_shoothill_to_xarray(stationId="7899" ,date_start=date_start, date_end=date_end)
ctr.dataset['shft'] = -ctr.dataset.sea_level[0].values + count

#count += 0.5
ctr2 = coast.TIDEGAUGE()
ctr2.dataset = ctr.read_shoothill_to_xarray(stationId="7900" ,date_start=date_start, date_end=date_end)
ctr2.dataset['shft'] = -ctr2.dataset.sea_level[8].values + count
ctr2.plot_timeseries()

count += 0.5
iron = coast.TIDEGAUGE()
iron.dataset = ctr.read_shoothill_to_xarray(stationId="968" ,date_start=date_start, date_end=date_end)
iron.dataset['shft'] = -iron.dataset.sea_level[0].values + count
iron.plot_timeseries()

count += 0.5
farn = coast.TIDEGAUGE()
farn.dataset = ctr.read_shoothill_to_xarray(stationId="972" ,date_start=date_start, date_end=date_end)
farn.dataset['shft'] = -farn.dataset.sea_level[0].values + count
farn.plot_timeseries()

count += 0.5
manh = coast.TIDEGAUGE()
manh.dataset = ctr.read_shoothill_to_xarray(stationId="963" ,date_start=date_start, date_end=date_end)
manh.dataset['shft'] = -manh.dataset.sea_level[0].values + count
manh.plot_timeseries()

count += 0.5
chrk = coast.TIDEGAUGE()
chrk.dataset = ctr.read_shoothill_to_xarray(stationId="957" ,date_start=date_start, date_end=date_end)
chrk.dataset['shft'] = -chrk.dataset.sea_level[0].values + count
chrk.plot_timeseries()

#count += 0.5
corwen = coast.TIDEGAUGE()
corwen.dataset = ctr.read_shoothill_to_xarray(stationId="962" ,date_start=date_start, date_end=date_end)
corwen.dataset['shft'] = -corwen.dataset.sea_level[0].values + count
corwen.plot_timeseries()

count += 0.5
deebr = coast.TIDEGAUGE()
deebr.dataset = ctr.read_shoothill_to_xarray(stationId="971" ,date_start=date_start, date_end=date_end)
deebr.dataset['shft'] = -deebr.dataset.sea_level[0].values + count
deebr.plot_timeseries()

count += 0.5
bala = coast.TIDEGAUGE()
bala.dataset = ctr.read_shoothill_to_xarray(stationId="965" ,date_start=date_start, date_end=date_end)
bala.dataset['shft'] = -bala.dataset.sea_level[0].values + count
bala.plot_timeseries()



fig, ax = plt.subplots()

## Only get tides over the weir with 8.75m at Liverpool
fig.suptitle('Timing of Chester Meadows flood relative to tides. Storm Christoph, Jan 2021')
for var in [deebr, corwen, chrk, manh, farn, iron, ctr2, ctr]:
    ax.scatter(var.dataset.time, var.dataset.sea_level  + var.dataset.shft, s=1, label=var.dataset.site_name)

ax.set_ylabel('relative river height (m)')
ax.set_xlabel('date')
myFmt = mdates.DateFormatter('%d-%a')
ax.xaxis.set_major_formatter(myFmt)
plt.legend(markerscale=6)
plt.savefig('Dee_river_levels_Jan21.png')


## Ctr + Ironbridge + Farndon

## Only get tides over the weir with 8.75m at Liverpool
fig, ax = plt.subplots()

plt.title('Timing of Chester Meadows flood for Storm Christoph, Jan 2021')
for var in [farn, iron, ctr2, ctr]:
    ax.scatter(var.dataset.time, var.dataset.sea_level  + 0*var.dataset.shft, s=1, label=var.dataset.site_name)

ax.set_ylabel('relative river height (m)')
ax.set_xlabel('Jan 2021')
myFmt = mdates.DateFormatter('%d-%a')
ax.xaxis.set_major_formatter(myFmt)
plt.legend(markerscale=6)
plt.savefig('Dee_river_levels_Jan21_short.png')

