#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:55:23 2021

@author: jeff
"""

'''
Archive the data from the Shoothill API for gauges around Chester.

Save in yearly files.

Chester weir height (above AND below weir) + flow
Gladstone and Ironbridge levels.


This does require a key to be
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

import xarray as xr

import os,sys
coastdir = os.path.dirname('/Users/jeff/GitHub/COAsT/coast')
sys.path.insert(0, coastdir)
import coast

#%% Save method
def save_method(loc, ofile=None):
    """

    Parameters
    ----------
    loc : STR
        variable name used.
    ofile : STR
        filename head if different from "loc_year"


    Returns
    -------
    None.

    """

    # Check the exported file is as you expect.
    # Load file as see that the xarray structure is preserved.
    ofile = ofile + ".nc"
    try:
        object = xr.open_dataset(ofile)
        object.close() # close file associated with this object
        file_flag = True
        loc.dataset = xr.concat([loc.dataset, object], "time").compute()

        #os.remove(ofile)
        print('loaded old file')
    except:
        print(f'{ofile} does not exist or does not load. Write a fresh file')
        file_flag = False

    try:
        loc.dataset.to_netcdf( ofile, mode="w", format="NETCDF4" )
        print(f'{ofile} saved.')

    except:
        print(f'{ofile} would not save. Create _2.nc file')
        loc.dataset.to_netcdf( ofile.replace('.nc','_2.nc'), mode="w", format="NETCDF4" )

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


#%% Save yearly data
#for year in range(2010,2020+1):
for year in range(2021,2021+1):

    print(year)
    date_start = np.datetime64(str(year)+'-01-01')
    date_end = np.datetime64(str(year)+'-12-31')



    # Load in data from the Shoothill API. Gladstone dock is loaded by default
    try:
        liv = coast.Tidegauge()
        liv.dataset = liv.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)
        #liv.plot_timeseries()
        save_method(liv, ofile='liv_'+str(year))
    except:
        print(str(year) + 'failed for liv')

    try:
        ctrf = coast.Tidegauge()
        ctrf.dataset = ctrf.read_shoothill_to_xarray(stationId="7899" ,date_start=date_start, date_end=date_end, dataType=15)
        #ctrf.plot_timeseries()
        save_method(ctrf, ofile='ctrf_'+str(year))
    except:
        print(str(year) + 'failed for ctrf')

    try:
        ctr = coast.Tidegauge()
        ctr.dataset = ctr.read_shoothill_to_xarray(stationId="7899" ,date_start=date_start, date_end=date_end)
        #ctr.plot_timeseries()
        save_method(ctr, ofile='ctr_'+str(year))
    except:
        print(str(year) + 'failed for ctr')

    try:
        ctr2 = coast.Tidegauge()
        ctr2.dataset = ctr.read_shoothill_to_xarray(stationId="7900" ,date_start=date_start, date_end=date_end)
        #ctr2.plot_timeseries()
        save_method(ctr2, ofile='ctr2_'+str(year))
    except:
        print(str(year) + 'failed for ctr2')

    try:
        iron = coast.Tidegauge()
        iron.dataset = ctr.read_shoothill_to_xarray(stationId="968" ,date_start=date_start, date_end=date_end)
        #iron.plot_timeseries()
        save_method(iron, ofile='iron_'+str(year))
    except:
        print(str(year) + 'failed for iron')

    try:
        farn = coast.Tidegauge()
        farn.dataset = ctr.read_shoothill_to_xarray(stationId="972" ,date_start=date_start, date_end=date_end)
        #farn.plot_timeseries()
        save_method(farn, ofile='farn_'+str(year))
    except:
        print(str(year) + 'failed for farn')





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
# Add empty data to ax1 to get "green flow data" in the legend
ax2 = scatter_plot(ax2, [], [], 'g', 1, "Flow, above weir")

# plot the legend
ax2.legend(markerscale=6, loc='lower left')

plt.savefig('Chester_river_levels.png')