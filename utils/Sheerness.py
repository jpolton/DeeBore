#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:55:23 2021

@author: jeff
"""

'''
Check the surge at Sheerness

Uses shoothill_api package to augment COAsT



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
DeeBore% python utils/Sheerness.py

Works in env: coast-3.10
'''

# Begin by importing coast and other packages
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys, os
sys.path.append(os.path.dirname(os.path.abspath("shoothill_api/shoothill_api.py")))
try: # command line
    from shoothill_api import GAUGE
except: # pycharm
    from shoothill_api.shoothill_api import GAUGE

coastdir = os.path.dirname('/Users/jelt/GitHub/COAsT/coast')
sys.path.insert(0, coastdir)
import coast
from coast._utils.general_utils import day_of_week

# For interactive plotting / else save to .png
flag_interactive = True
if flag_interactive:
    import matplotlib as mpl
    mpl.use('macosx')


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
            

class QCdata:
    """
    tg = QCdata().to_tidegauge()
    """
    def __init__(self):
        dir = '/Users/jelt/GitHub/DeeBore/data/BODC_processed/'
        filelist = [
        'ClassAObsAfterSurgeQC2022jul.nc', # Alternative data to patch July 2022 bad data
        'ClassAObsAfterSurgeQC2023jan.nc',
        'ClassAObsAfterSurgeQC2023feb.nc',
        'ClassAObsAfterSurgeQC2023mar.nc',
        'ClassAObsAfterSurgeQC2023sep.nc',
        'ClassAObsAfterSurgeQC2023oct.nc',
        ]
        tg = coast.Tidegauge()
        for file in filelist:
            tg0 = coast.Tidegauge()
            if file.endswith(".nc"):  # if file ends .nc
                ds = xr.open_dataset(dir+file).isel(station=[16]).rename_dims({'time':'t_dim', 'station':'id_dim'})\
                                                                .rename_vars({"station_name":"site_name"})
                if (ds.site_name.values == b'EA-Sheerness'):
                    tg0.dataset = ds.rename_vars({"zos_total_observed":"ssh", "timeseries":"time"})
                    tg0.dataset = tg0.dataset.set_coords(["longitude", "latitude", "site_name", "time"])
            else:
                print(f"Did not expect file: {file}")
            if tg.dataset is None:
                tg.dataset = tg0.dataset
            else:
                # insert new data is time overlaps exist, then merge
                tg.dataset = tg.dataset.where( (tg.dataset.time < tg0.dataset.time.min()) | (tg.dataset.time > tg0.dataset.time.max()), drop=True )
                tg.dataset = xr.concat([ tg.dataset, tg0.dataset], dim='t_dim').sortby("time")
        # Use QC to drop null values
        #tg.dataset['sea_level'] = tg.dataset.sea_level.where( np.logical_or(tg.dataset.qc_flags=='', tg.dataset.qc_flags=='T'), drop=True)
        try:
            tg.dataset = tg.dataset.rename_vars({'ssh': 'sea_level'})
        except:
            pass
        try:
            tg.dataset = tg.dataset.squeeze(dim='id_dim')
        except:
            pass
        #tg.dataset = tg.dataset.where( tg.dataset.qc_flags!='N', drop=True)  # BODC flags
        tg.dataset = tg.dataset.where( tg.dataset.zos_flags==0, drop=True)
        # Fix some attributes (others might not be correct for all data)
        tg.dataset['start_date'] = tg.dataset.time.min().values
        tg.dataset['end_date'] = tg.dataset.time.max().values
        self.tg = tg

    def to_tidegauge(self):
        return self.tg

class AMM7_surge_ERA5:
    """
    Class to handle AMM7_surge data forced by ERA5
        tg = AMM7_surge_ERA5().to_tidegauge()

    """
    def __init__(self):
        fn_SHNS = "/Users/jelt/DATA/surge_hindcast_ERA5_ClassA_bysite/SHNS.nc"
        sh_era5 = GAUGE()
        sh_era5.dataset = xr.open_dataset(fn_SHNS).swap_dims({"time":"t_dim"})
        sh_era5.dataset['time'] = sh_era5.dataset.timeseries.dt.round('s') # round to the nearest second
        sh_era5.dataset['sea_level'] = sh_era5.dataset.tide_ht + sh_era5.dataset.residual_ht
        sh_era5.dataset['start_date'] = sh_era5.dataset.time.min().values
        sh_era5.dataset['end_date'] = sh_era5.dataset.time.max().values
        self.tg = sh_era5

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



    ndays = 2

    date_end = np.datetime64('now')
    date_start = date_end - np.timedelta64(ndays,'D')

    ## Christoph 22 Jan 21
    date_end = np.datetime64('2021-01-28')
    ndays = 10
    date_start = date_end - np.timedelta64(ndays,'D')

    ## 2021
    date_end = np.datetime64('now')
    date_start = np.datetime64('2021-01-01')

    ## 24 hrs
    date_end = np.datetime64('now')
    date_start = np.datetime64('now') - np.timedelta64(24,'h')
    #date_start = np.datetime64('2022-12-21')

    ## 24 hrs
    date_end = np.datetime64('now') - np.timedelta64(18,'h')
    date_start = np.datetime64('now') - np.timedelta64(23,'h')

    #%% Load data
    sh_qc = QCdata().to_tidegauge()

    date_start = sh_qc.dataset.time.min().values.astype('datetime64[s]')  # cast as seconds
    date_end   = sh_qc.dataset.time.max().values.astype('datetime64[s]')


    sh_ea = GAUGE()
    sh_ea.dataset = sh_ea.read_ea_api_to_xarray(date_start=date_start, date_end=date_end, station_id="E71539")
    sh_ea.dataset = sh_ea.dataset.sortby(sh_ea.dataset.time) # sometimes the arrives out of order

    sh_ea2 = GAUGE()
    sh_ea2.dataset = sh_ea2.read_ea_api_to_xarray(date_start=date_start, date_end=date_end, station_id="E71524")
    sh_ea2.dataset = sh_ea2.dataset.sortby(sh_ea2.dataset.time) # sometimes the arrives out of order

    sh_shoot = GAUGE()
    sh_shoot.dataset = sh_shoot.read_shoothill_to_xarray(station_id="7522", date_start=date_start, date_end=date_end, dataType=4)
    sh_shoot.plot_timeseries('sea_level')

    # Load Sheerness tide prediction from NOC Innovation api
    sh_noci = GAUGE()
    sh_noci.dataset = sh_noci.read_nocinnov_to_xarray(station_id="Sheerness", date_start=date_start, date_end=date_end)
    sh_noci.plot_timeseries('sea_level')

    # Load Sheerness from QC'd data
    sh_qc = QCdata().to_tidegauge()

    # Load Sheerness ERA5 data
    sh_nemo = AMM7_surge_ERA5().to_tidegauge()

    #%% Plot data

    plt.close('all')
    fig, ax_l = plt.subplots(1, sharex=True)

    ## Only get tides over the weir with 8.75m at Liverpool
    fig.suptitle('Sheerness water levels')

    ax_l = line_plot(ax_l, sh_qc.dataset.time, sh_qc.dataset.sea_level, 'y', 1, "QC")
    ax_l = line_plot(ax_l, sh_ea.dataset.time, sh_ea.dataset.ssh, 'm', 1, "EA:1")
    ax_l = line_plot(ax_l, sh_ea2.dataset.time, sh_ea2.dataset.ssh, 'r', 1, "EA:2")
    ax_l = line_plot(ax_l, sh_shoot.dataset.time, sh_shoot.dataset.sea_level, 'g', 1, "Shoothill")
    ax_l = scatter_plot(ax_l, sh_noci.dataset.time, sh_noci.dataset.sea_level, 'k', 1, "NOCi:Harmonic")
    # Add empty data to ax1 to get RH axis in the legend
    ax_l = line_plot(ax_l, [], [], 'b', 1, "EA2-EA1")
    ax_r = ax_l.twinx()
    ax_r = line_plot(ax_r, sh_ea.dataset.time, sh_ea2.dataset.ssh-sh_ea.dataset.ssh, 'b', 1, "EA diff")
    # Add dotted harmonic pred
    #ax_r = scatter_plot(ax_r, sh_noci.dataset.time, sh_noci.dataset.sea_level, 'b', 1, sh_noci.dataset.site_name)

    ax_l.set_ylabel('water level (m)', color='k')
    ax_r.set_ylabel('Diff (m)', color='b')
    #ax_r.set_ylim([4.8,8.2])
    for tl in ax_r.get_yticklabels():
        tl.set_color('b')

    # plot the legend
    ax_l.legend(markerscale=6, loc='upper left')


    # format the ticks
    myFmt = mdates.DateFormatter('%d-%a')
    days = mdates.DayLocator()
    ax_l.xaxis.set_major_locator(days)
    ax_l.xaxis.set_minor_locator(mdates.HourLocator([00,6,12,18]))
    ax_l.xaxis.set_major_formatter(myFmt)

    ax_l.set_xlabel( date_start.astype(datetime.datetime).strftime('%d%b%y') + \
                   '-' + date_end.astype(datetime.datetime).strftime('%d%b%y') )

    if flag_interactive:
        plt.show()
    else:
        plt.savefig('Sheerness_river_levels_measured.png')


    ####################

    plt.close('all')
    fig, ax_l = plt.subplots(1, sharex=True)

    ## Only get tides over the weir with 8.75m at Liverpool
    fig.suptitle('Sheerness water levels')

    ax_l = line_plot(ax_l, sh_nemo.dataset.time, sh_nemo.dataset.sea_level, 'y', 1, "nemo")
    ax_l = line_plot(ax_l, sh_qc.dataset.time,   sh_qc.dataset.sea_level, 'm', 1, "QC")
    ax_l.set_ylabel('water level (m)', color='k')

    # plot the legend
    ax_l.legend(markerscale=6, loc='upper left')


    # format the ticks
    myFmt = mdates.DateFormatter('%d-%a')
    days = mdates.DayLocator()
    ax_l.xaxis.set_major_locator(days)
    ax_l.xaxis.set_minor_locator(mdates.HourLocator([00,6,12,18]))
    ax_l.xaxis.set_major_formatter(myFmt)

    ax_l.set_xlabel( date_start.astype(datetime.datetime).strftime('%d%b%y') + \
                   '-' + date_end.astype(datetime.datetime).strftime('%d%b%y') )

    if flag_interactive:
        plt.show()
    else:
        plt.savefig('Sheerness_river_levels_model.png')
