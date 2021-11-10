#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2021

@author: jeff
"""

'''
Plot Chester weir height (above AND below weir) + flow
Plot Gladstone and Ironbridge levels.
Loads data from local shoothill files.

Gauges from upstream to downstream (sea):
Farndon, Ironbridge, Chester (above) weir, Chester (below) weir, Gladstone dock (Liverpool)
I also include Chester river flow (m^3/s), measured above the weir, because it exists. Probably not useful.
These are all on the river Dee, except Gladstone which is on the neighbouring estuary, the Mersey.

Ideally we are interested in predicting Chester weir from Gladstone dock
(Liverpool) data, using lagged upstream river data if it helps.
The below weir data has a shorter record than the above weir data.

One loaded the data are stored as "time" for time and "sea_level" for the free
variable (which is the water level (m) or river flow rate (m3/s)).
'''

# Begin by importing coast and other packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # There are many ways to read netCDF files, including this one!

#%%  plot functions
def line_plot(ax, time, y, color, size, label=None ):
    ax.plot(time, y, color=color, linewidth=size, label=label)
    return ax

def scatter_plot(ax, time, y, color, size, label=None ):
    ax.scatter(time, y, color=color, s=size, label=label)
    return ax
#%%
# Choose some arbitary dates
start_date = np.datetime64('2019-02-17')
end_date =  np.datetime64('2019-02-23')
#start_date = np.datetime64('2021-11-02')
#end_date =  np.datetime64('2021-11-04')

# location of files
dir = "archive_shoothill/" #

# load data by location.
liv = xr.open_mfdataset(dir+"liv_????.nc") # Tidal port Gladstone Dock, Liverpool
ctr_dn = xr.open_mfdataset(dir+"ctr2_????.nc") # below the Chester weir
ctr_up = xr.open_mfdataset(dir+"ctr_????.nc") # above the Chester weir
ctr_fl = xr.open_mfdataset(dir+"ctrf_????.nc") # flow rate above (and at) the weir
iron= xr.open_mfdataset(dir+"iron_????.nc") # upstream river at Ironbridge
farn= xr.open_mfdataset(dir+"farn_????.nc") # upstream river at Farndon.



#%% Plot data

# Top: Gladstone + Ironbridge
line_flag = True
today_only_flag = True

plt.close('all')
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

## Only get tides over the weir with about 8.75m at Liverpool
fig.suptitle('Dee River heights and flow')
ax1 = scatter_plot(ax1, liv.time, liv.sea_level, 'k', 1, liv.site_name)
if line_flag:
    ax1 = line_plot(ax1, liv.time, liv.sea_level, 'k', 1)

ax1.plot( [start_date - np.timedelta64(1,'D'), end_date], [8.75,8.75], 'k--')
ax1b = ax1.twinx()
ax1b = scatter_plot(ax1b, iron.time, iron.sea_level, 'b', 1, iron.site_name)
if line_flag:
    ax1b = line_plot(ax1b, iron.time, iron.sea_level, 'b', 1)

ax1.set_ylabel('water level (m)', color='k')
ax1b.set_ylabel('water level (m)', color='b')
for tl in ax1b.get_yticklabels():
    tl.set_color('b')

# plot legend. sort y limits
ax1.set_ylim([0, 12]) # Liverpool range
ax1.legend(markerscale=6)
ax1b.legend(markerscale=6)

# Lower: CTR height + flow
ax2 = scatter_plot(ax2, ctr_up.time, ctr_up.sea_level, 'k', 1, "Chester, above weir")
if line_flag:
    ax2 = line_plot(ax2, ctr_up.time, ctr_up.sea_level, 'k', 1)
ax2 = scatter_plot(ax2, ctr_dn.time, ctr_dn.sea_level, 'b', 1, "Chester, below weir")
if line_flag:
    ax2 = line_plot(ax2, ctr_dn.time, ctr_dn.sea_level, 'b', 1)
ax2b = ax2.twinx()
ax2b = scatter_plot(ax2b, ctr_fl.time, ctr_fl.sea_level, 'g', 1, ctr_fl.site_name)
if line_flag:
    ax2b = line_plot(ax2b, ctr_fl.time, ctr_fl.sea_level, 'g', 1)
ax2b.set_ylabel('flow rate (m3/s)', color='g')
for tl in ax2b.get_yticklabels():
    tl.set_color('g')
ax2.set_ylabel('water level (m)')

# Add empty data to ax1 to get "green flow data" in the legend
ax2 = scatter_plot(ax2, [], [], 'g', 1, "Flow, above weir")

# plot the legend. sort y limits
ax2.set_ylim([2, 7]) # weir range
ax2b.set_ylim([-400, 400]) # flow range
ax2.set_xlim([start_date, end_date])
ax2.legend(markerscale=6, loc='lower left')

#plt.show()
plt.savefig('Chester_river_levels.png')
