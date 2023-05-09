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

def addtext(ax, props, deg):
    ax.text(0.5, 0.5, 'text 0', props, rotation=deg)
    ax.text(1.5, 0.5, 'text 45', props, rotation=45)
    ax.text(2.5, 0.5, 'text 135', props, rotation=135)
    ax.text(3.5, 0.5, 'text 225', props, rotation=225)
    ax.text(4.5, 0.5, 'text -45', props, rotation=-45)
    for x in range(0, 5):
        ax.scatter(x + 0.5, 0.5, color='r', alpha=0.5)
    ax.set_yticks([0, .5, 1])
    ax.set_xticks(np.arange(0, 5.1, 0.5))
    ax.set_xlim(0, 5)
    ax.grid(True)

class thing():
    def __init__(self, tg):
        nt = tg.dataset.sizes['time']

        date = [pd.to_datetime(tg.dataset.time[i].values) for i in range(nt)]
        day = [date[i].day for i in range(nt)]
        hour = [date[i].hour for i in range(nt)]
        print(f"hour: {hour}")
        mins = [date[i].minute for i in range(nt)]
        day_str = [date[i].day_name()[0:1] for i in range(nt)]

        theta = [float(hour[i] + mins[i]/60.)/12.*2*np.pi for i in range(nt)]

        r = [tg.dataset.sea_level[i].values for i in range(nt)]
        x = [ r[i]*np.sin(theta[i]) for i in range(nt)]
        y = [ r[i]*np.cos(theta[i]) for i in range(nt)]

        self.theta = theta
        self.r = r
        self.x = x
        self.y = y

################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":


    filnam = '/Users/jelt/GitHub/DeeBore/data/Liverpool_2023_2025_HLW.txt'
    tg = GAUGE()


    ndays = 14

    # the text bounding box
    bbox = {}#'fc': '0.8', 'pad': 0}

    date_end = np.datetime64('now') + np.timedelta64(ndays, 'D')
    date_start = np.datetime64('now') - np.timedelta64(1, 'D')

    tg.dataset = tg.read_hlw_to_xarray(filnam, date_start=date_start, date_end=date_end)
    nt = tg.dataset.sizes['time']

    # %% Load data
    try:
        iron = GAUGE()
        iron.dataset = iron.read_shoothill_to_xarray(station_id="968",
                                                    date_start=np.datetime64('now') - np.timedelta64(1, 'D'),
                                                    date_end=np.datetime64('now'))
        plot_river_flag = True
    except:
        plot_river_flag = False

    # Liverpool reconstruction
    liv = GAUGE()
    liv.dataset = liv.anyTide_to_xarray(date_start=np.datetime64('now') - np.timedelta64(1, 'D'),
                                        date_end=np.datetime64('now'))
    liv.dataset['site_name'] = "Liverpool (Gladstone)"
    d = thing(liv)


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

    r = [tg.dataset.sea_level[i].values for i in range(nt)]
    x = [ r[i]*np.sin(theta[i]) for i in range(nt)]
    y = [ r[i]*np.cos(theta[i]) for i in range(nt)]


    col = [ 'k' if theta[i] > 2*np.pi else 'k' for i in range(nt)]  # am / pm
    #sym = [ '+' if theta[i] > 2*np.pi else 'o' for i in range(nt)]
    sym = [ 'o' if theta[i] > 2*np.pi else 'o' for i in range(nt)]
    siz = [ 20 if (hour[i] >= 6 and hour[i] <= 18) else 5 for i in range(nt)]
    rot_deg = [90 - (theta[i]*180/np.pi) if (hour[i]%12 >= 0 and hour[i]%12 < 6) else
               270 - (theta[i]*180/np.pi) for i in range(nt)]




    # Start figure

    fig, ax = plt.subplots()
    # draw circle
    #R_bound = 12.
    R_thresh = 9.5
    R_num = 6
    #ax.plot( R_bound*np.sin(np.arange(0,360)*np.pi/180.), R_bound*np.cos(np.arange(0,360)*np.pi/180.),'g'  )
    ax.plot( R_thresh*np.sin(np.arange(0,360)*np.pi/180.), R_thresh*np.cos(np.arange(0,360)*np.pi/180.),'g'  )
    ax.text( 0, R_num, "12",  {'ha': 'center', 'va': 'center'}, fontsize=36)
    ax.text( R_num, 0, "3",   {'ha': 'center', 'va': 'center'}, fontsize=36)
    ax.text( 0, -R_num, "6",  {'ha': 'center', 'va': 'center'}, fontsize=36)
    ax.text( -R_num, 0, "9",  {'ha': 'center', 'va': 'center'}, fontsize=36)


    # Plot last 24hr tide
    ax.plot( d.x, d.y)
    # Plot clock hour hand
    ax.plot( [0, d.x[0]], [0, d.y[0]], 'k')



    #addtext(axs[1], {'ha': 'left', 'va': 'bottom', 'bbox': bbox})
    #axs[1].set_ylabel('left / bottom')
    scale = 1.2  # radius scale for text
    for i in range(nt):
        plt.scatter(x[i], y[i], c=col[i], s=siz[i], marker=sym[i])
        ax.text(x[i]*scale, y[i]*scale,
                day_str[i]+str(day[i]),
                {'ha': 'center', 'va': 'center'},
                rotation=rot_deg[i])
    ax.axis('equal')

    # Plot river
    if (plot_river_flag):
        ww = 0.3
        hh = 0.15
        ax2 = ax.inset_axes([0.5 - 0.5*ww, 0.5 - 0.5*hh, ww, hh])
        ax2.plot(iron.dataset.time, iron.dataset.sea_level, 'g.')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(True)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['left'].set_visible(True)
        ax2.set_title('Ironbridge')
        # format the ticks
        # myFmt = mdates.DateFormatter('%H:%M')
        # myFmt = mdates.DateFormatter('%d-%a')
        myFmt = mdates.DateFormatter('%a')
        days = mdates.DayLocator()
        ax2.xaxis.set_major_locator(days)
        ax2.xaxis.set_minor_locator(mdates.HourLocator([00, 6, 12, 18]))
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.tick_params(axis="y", direction="in", pad=-28)
    plt.axis('off')

    ir = thing(iron)
    # Plot last 24hr river
    ax.plot( ir.x, ir.y, 'g')

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
