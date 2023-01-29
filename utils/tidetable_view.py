#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 2023

@author: jeff
"""

'''

NEED TO UPDATE

Check the gauges at Chester.
Plot Chester weir height (above AND below weir) + flow
Plot Gladstone and Ironbridge levels.
Uses shoothill_api package to augment COAsT

This tutorial uses data from the Shoothill API. This does require a key to be
setup. It is assumed that the key is privately stored in
 config_keys.py

This API aggregates data across the country for a variety of instruments but,
 requiring a key, is trickier to set up.
To discover the StationId for a particular measurement site check the
 integer id in the url or its twitter page having identified it via
  https://www.gaugemap.co.uk/#!Map
E.g  Liverpool (Gladstone Dock stationId="13482", which is read by default.

Env: workshop_env with coast and requests installed,
E.g.
## Create an environment with coast installed
yes | conda env remove --name workshop_env
yes | conda create --name workshop_env python=3.8
conda activate workshop_env
yes | conda install -c bodc coast=1.2.7
# enforce the GSW package number (something fishy with the build process bumped up this version number)
yes | conda install -c conda-forge gsw=3.3.1
# install cartopy, not part of coast package
yes | conda install -c conda-forge cartopy=0.20.1

## install request for shoothill server requests
conda install requests

Usage:
DeeBore% python utils/tidetable_view.py
'''

# Begin by importing coast and other packages
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.abspath("shoothill_api/shoothill_api.py")))
#from shoothill_api import GAUGE   ## WORKS WITH COMMAND LINE
from shoothill_api.shoothill_api import GAUGE  ## WORKS WITH PYCHARM


class GladstoneHarmonicReconstruction:
    """         
    if source == 'harmonic_rec': # load full tidal signal using anyTide code
    tg = GladstoneHarmonicReconstruction().to_tidegauge()
    """         
    def __init__(self, date_start=None, date_end=None): 
        tg = GAUGE()
        #date_start=np.datetime64('now')
        #ndays = 5
        #tg.dataset = tg.anyTide_to_xarray(date_start=date_start, ndays=5)
        #date_start=np.datetime64('2005-04-01')
        #date_end=np.datetime64('now','D') 
        if (date_start==None) and (date_end==None):
            date_start = np.datetime64('now','D') 
            date_end = np.datetime64('now','D') + np.timedelta64(4,'D')
        tg.dataset = tg.anyTide_to_xarray(date_start=date_start, date_end=date_end)
        tg.dataset['site_name'] = "Liverpool (Gladstone)"
        self.tg = tg

    def to_tidegauge(self):
        return self.tg
            




################################################################################
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


################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":


    filnam = '/Users/jelt/GitHub/DeeBore/data/Liverpool_2023_2025_HLW.txt'
    tg = GAUGE()


    ndays = 7

    date_end = np.datetime64('now') + np.timedelta64(ndays, 'D')
    date_start = np.datetime64('now') - np.timedelta64(ndays, 'D')

    tg.dataset = tg.read_hlw_to_xarray(filnam, date_start=date_start, date_end=date_end)
    nt = tg.dataset.sizes['time']

    # %% Load data
    #iron = GAUGE()
    #iron.dataset = ctr.read_shoothill_to_xarray(station_id="968", date_start=date_start, date_end=date_end)


    HW = []
    for i in range(nt-1):
        if (tg.dataset.sea_level[i].values > tg.dataset.sea_level[i+1].values):
            HW.append(i)

    # Clip data to HW only
    tg.dataset = tg.dataset.isel(time=HW)

    nt = tg.dataset.sizes['time']

    date = [pd.to_datetime(tg.dataset.time[i].values) for i in range(nt)]
    day = [date[i].day for i in range(nt)]
    hour = [date[i].hour for i in range(nt)]
    mins = [date[i].minute for i in range(nt)]
    day_str = [date[i].day_name()[0:2] for i in range(nt)]

    theta = [float(hour[i] + mins[i]/60.)/12.*2*np.pi for i in range(nt)]

    col = [ 'b' if theta[i] > 2*np.pi else 'r' for i in range(nt)]
    sym = [ '+' if theta[i] > 2*np.pi else 'o' for i in range(nt)]


    r = [tg.dataset.sea_level[i].values for i in range(nt)]
    x = [ r[i]*np.sin(theta[i]) for i in range(nt)]
    y = [ r[i]*np.cos(theta[i]) for i in range(nt)]

    # Circle
    R = 9.5
    plt.plot( R*np.sin(np.arange(0,360)*np.pi/180.), R*np.cos(np.arange(0,360)*np.pi/180.),'g'  )
    plt.text( 0, R*1.1, "12")
    plt.text( R*1.1, 0, "3")
    plt.text( 0, -R*1.1, "6")
    plt.text( -R*1.1, 0, "9")

    #plt.text( 0, R*1.1, "00")
    #plt.text( R*1.1, 0, "06")
    #plt.text( 0, -R*1.1, "12")
    #plt.text( -R*1.1, 0, "18")

    # Plot points
    # plot dates
    for i in range(nt):
        plt.scatter(x[i], y[i], c=col[i], s=10, marker=sym[i])
        plt.text(x[i], y[i], day_str[i]+str(day[i]))
    plt.text(0, +0.5, 'a.m', c='r')
    plt.text(0, -0.5, 'p.m', c='b')
    plt.show()

    if(0):



        #%% Load data
        iron = GAUGE()
        iron.dataset = ctr.read_shoothill_to_xarray(station_id="968" ,date_start=date_start, date_end=date_end)

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
        ax1 = scatter_plot(ax1, liv.dataset.time, liv.dataset.sea_level, 'k', 1, liv.dataset.site_name.values)
        if line_flag:
            ax1 = line_plot(ax1, liv.dataset.time, liv.dataset.sea_level, 'k', 1)

        ax1.plot( [date_start - np.timedelta64(1,'D'), date_end], [8.75,8.75], 'k--')
        ax1b = ax1.twinx()
        ax1b = scatter_plot(ax1b, iron.dataset.time, iron.dataset.sea_level, 'b', 1, iron.dataset.site_name)
        if line_flag:
            ax1b = line_plot(ax1b, iron.dataset.time, iron.dataset.sea_level, 'b', 1)

        ax1.set_ylabel('water level (m)', color='k')
        ax1b.set_ylabel('water level (m)', color='b')
        ax1b.set_ylim([4.8,8.2])
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

        # format the ticks
        #myFmt = mdates.DateFormatter('%H:%M')
        myFmt = mdates.DateFormatter('%d-%a')
        days = mdates.DayLocator()
        ax2.xaxis.set_major_locator(days)
        ax2.xaxis.set_minor_locator(mdates.HourLocator([00,6,12,18]))
        ax2.xaxis.set_major_formatter(myFmt)

        ax2.set_xlabel( date_start.astype(datetime.datetime).strftime('%d%b%y') + \
                       '-' + date_end.astype(datetime.datetime).strftime('%d%b%y') )
        #ax2.set_xlabel(date_end.astype(datetime.datetime).strftime('%d%b%y'))
        # Add empty data to ax1 to get "green flow data" in the legend
        ax2 = scatter_plot(ax2, [], [], 'g', 1, "Flow, above weir")

        # plot the legend
        ax2.legend(markerscale=6, loc='lower left')

        plt.savefig('Chester_river_levels.png')
