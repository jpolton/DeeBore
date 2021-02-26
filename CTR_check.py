#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:55:23 2021

@author: jeff
"""

'''
Check the gauges at Chester.
Plot Chester weir height (above AND below weir) + flow
Plot Gladstone and Ironbridge levels.


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

#%%  plot functions
def line_plot(ax, time, y, color, size, label=None ):
    #ax1.scatter(liv.dataset.time, liv.dataset.sea_level, color='k', s=1, label=liv.dataset.site_name)
    ax.plot(time, y, color=color, linewidth=size, label=label)
    return ax

def scatter_plot(ax, time, y, color, size, label=None ):
    #ax1.scatter(liv.dataset.time, liv.dataset.sea_level, color='k', s=1, label=liv.dataset.site_name)
    ax.scatter(time, y, color=color, s=size, label=label)
    return ax
#%%
ndays = 1

date_end = np.datetime64('now') 
date_start = date_end - np.timedelta64(ndays,'D')

## Christoph 22 Jan 21
date_end = np.datetime64('2021-01-28') 
ndays = 10
date_start = date_end - np.timedelta64(ndays,'D')

## 2021
date_end = np.datetime64('now') 
date_start = np.datetime64('2021-01-01')

#%% Load data

# Load in data from the Shoothill API. Gladstone dock is loaded by default
liv = coast.TIDEGAUGE()
liv.dataset = liv.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)
#liv.plot_timeseries()

ctrf = coast.TIDEGAUGE()
ctrf.dataset = ctrf.read_shoothill_to_xarray(stationId="7899" ,date_start=date_start, date_end=date_end, dataType=15)
#ctrf.plot_timeseries()

ctr = coast.TIDEGAUGE()
ctr.dataset = ctr.read_shoothill_to_xarray(stationId="7899" ,date_start=date_start, date_end=date_end)
#ctr.plot_timeseries()

ctr2 = coast.TIDEGAUGE()
ctr2.dataset = ctr.read_shoothill_to_xarray(stationId="7900" ,date_start=date_start, date_end=date_end)
#ctr2.plot_timeseries()

iron = coast.TIDEGAUGE()
iron.dataset = ctr.read_shoothill_to_xarray(stationId="968" ,date_start=date_start, date_end=date_end)
#iron.plot_timeseries()

#farn = coast.TIDEGAUGE()
#farn.dataset = ctr.read_shoothill_to_xarray(stationId="972" ,date_start=date_start, date_end=date_end)
#farn.plot_timeseries()




#%% Plot data

# Top: Gladstone + Ironbridge
# Lower: CTR height + flow

line_flag = True
today_only_flag = True

plt.close('all')
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

## Only get tides over the weir with 8.75m at Liverpool
fig.suptitle('Dee River heights and flow')
#ax1.scatter(liv.dataset.time, liv.dataset.sea_level, color='k', s=1, label=liv.dataset.site_name)
ax1 = scatter_plot(ax1, liv.dataset.time, liv.dataset.sea_level, 'k', 1, liv.dataset.site_name)
if line_flag:
    ax1 = line_plot(ax1, liv.dataset.time, liv.dataset.sea_level, 'k', 1)
    
ax1.plot( [date_start - np.timedelta64(1,'D'), date_end], [8.75,8.75], 'k--')
ax1b = ax1.twinx()
ax1b = scatter_plot(ax1b, iron.dataset.time, iron.dataset.sea_level, 'b', 1, iron.dataset.site_name)
if line_flag:
    ax1b = line_plot(ax1b, iron.dataset.time, iron.dataset.sea_level, 'b', 1)

ax1.set_ylabel('water level (m)', color='k')
ax1b.set_ylabel('water level (m)', color='b')
for tl in ax1b.get_yticklabels():
    tl.set_color('b')
ax1.legend(markerscale=6)
ax1b.legend(markerscale=6)


ax2 = scatter_plot(ax2, ctr.dataset.time, ctr.dataset.sea_level, 'k', 1, "Chester, above weir")
if line_flag:
    ax2 = line_plot(ax2, ctr.dataset.time, ctr.dataset.sea_level, 'k', 1)
ax2 = scatter_plot(ax2, ctr2.dataset.time, ctr2.dataset.sea_level, 'b', 1, "Chester, below weir")
if line_flag:
    ax2 = line_plot(ax2, ctr2.dataset.time, ctr2.dataset.sea_level, 'b', 1)
ax2b = ax2.twinx()
ax2b = scatter_plot(ax2b, ctrf.dataset.time, ctrf.dataset.sea_level, 'g', 1, ctrf.dataset.site_name)
if line_flag:
    ax2b = line_plot(ax2b, ctrf.dataset.time, ctrf.dataset.sea_level, 'g', 1)
ax2b.set_ylabel('flow rate (m3/s)', color='g')
for tl in ax2b.get_yticklabels():
    tl.set_color('g')
ax2.set_ylabel('water level (m)')

myFmt = mdates.DateFormatter('%H:%M') #('%d-%a')
ax2.xaxis.set_major_formatter(myFmt)
#ax2.set_xlabel( date_start.astype(datetime.datetime).strftime('%d%b%y') + \
#               '-' + date_end.astype(datetime.datetime).strftime('%d%b%y') )
ax2.set_xlabel(date_end.astype(datetime.datetime).strftime('%d%b%y'))
ax2.legend(markerscale=6)

plt.savefig('Chester_river_levels.png')

