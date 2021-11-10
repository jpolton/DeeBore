#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9 Nov 2021

@author: jeff
"""

'''
Patch data received as csv files from NRW to fill gap from gaugemap.

Convert these to xarray and save.

Save in yearly files.

Chester weir height (above AND below weir) + flow

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
    Edit year variable
    python archive_NRW_csv.py

'''

# Begin by importing coast and other packages
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import xarray as xr

from shoothill_api.shoothill_api import GAUGE

class NRWGauge:
    """
    """
    def __init__(self):
        self.dataset = None

    #%% Load method
    @classmethod
    def read_nrw_to_xarray(cls, fn_nrw, date_start=None, date_end=None):
        """
        For reading from a single NRW csv file into an
        xarray dataset.
        If no data lies between the specified dates, a dataset is still created
        containing information on the  gauge, but the time dimension will
        be empty.

        The data takes the form:

        Station Site:	Northern [CY]
        Station Name:	CHESTER WEIR
        Station Number:	067020
        LocalX:	---
        LocalY:	---
        Datum:	---
        Parameter Name:	SG [Stage]
        Parameter Type:	S
        Parameter Type Name:	Stage
        Time series Name:	067020.SG.15.P
        Time series Unit:	m
        GlobalX:	340846
        GlobalY:	365842
        Longitude:	-2.88535
        Latitude:	53.186092
        Date,Time,Value[m],State of value,Interpolation,Absolute value[m],State of absolute value,Interpolation of absolute value,Tags,Comments
        01/01/2021,09:00:00,4.800,G,102,---,255,101
        01/01/2021,09:15:00,4.800,G,102,---,255,101
        01/01/2021,09:30:00,4.800,G,102,---,255,101
        01/01/2021,09:45:00,4.799,G,102,---,255,101
        ...

        Parameters
        ----------
        fn_nrw (str) : path to NRW gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data

        Returns
        -------
        xarray.Dataset object.
        """
        #debug(f'Reading "{fn_nrw}" as a NRW file with {get_slug(cls)}')  # TODO Maybe include start/end dates
        try:
            header_dict = cls.read_nrw_header(fn_nrw)
            dataset = cls.read_nrw_data(fn_nrw, date_start, date_end)
        except:
            raise Exception("Problem reading NRW file: " + fn_nrw)
        # Attributes
        dataset["longitude"] = float(header_dict["Longitude"])
        dataset["latitude"] = float(header_dict["Latitude"])
        dataset["site_name"] = header_dict["Station Name"]
        del header_dict["Longitude"]
        del header_dict["Latitude"]
        del header_dict["Station Name"]

        dataset.attrs = header_dict

        return dataset

    @classmethod
    def read_nrw_header(cls, filnam):
        """
        Reads header from a NRW csv file.

        Parameters
        ----------
        filnam (str) : path to file

        Returns
        -------
        dictionary of attributes
        """

        #debug(f'Reading NRW header from "{filnam}" ')

        my_dict = pd.read_csv(filnam, nrows=15, delimiter=':', header=None, index_col=0).to_dict()
        # Strip out special characters and remove nesting from dict
        header_dict = {key.strip(): item.strip() for key, item in my_dict[1].items()} # my_dict has a nested dict. Want the nest.

        return header_dict

    @classmethod
    def read_nrw_data(cls, filnam, date_start=None, date_end=None, header_length: int = 14):
        """
        Reads NRW data from a csv file.

        Parameters
        ----------
        filnam (str) : path to HLW tide gauge file
        date_start (np.datetime64) : start date for returning data.
        date_end (np.datetime64) : end date for returning data.
        header_length (int) : number of lines in header (to skip when reading).
                            Not including column titles

        Returns
        -------
        xarray.Dataset containing times, water level values, other data
        """
        import datetime

        # Initialise empty dataset and lists
        #debug(f'Reading NRW data from "{filnam}"')
        dataset = xr.Dataset()
        #time = []
        #sea_level = []

        data = pd.read_csv(filnam, parse_dates=[['Date', 'Time']], dayfirst=True, skiprows=header_length, delimiter=',', header=1, index_col=0)
        my_dict = {data.filter(regex='Value*').columns[0]: 'sea_level'} # Captures different string endings
        data.rename(columns=my_dict, inplace=True)
        data.index.names=['time']
        dataset = data.to_xarray()
        if  date_start != None:
                dataset = dataset.where(dataset.time >= date_start)
        if date_end != None:
                dataset = dataset.where(dataset.time <= date_end)
        # Assign local dataset to object-scope dataset
        return dataset


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
    ofile = "../archive_shoothill/" + ofile + ".nc"
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
    ax.plot(time, y, color=color, linewidth=size, label=label)
    return ax

def scatter_plot(ax, time, y, color, size, label=None ):
    ax.scatter(time, y, color=color, s=size, label=label)
    return ax
#%%


#%% Load and export NRW csv files to xarray and netcdf
######################################################
dir = "data/ATI 22356a - River level & flow data at Chester on the river Dee/"

try:
    fn_nrw = dir + "067020.SG[Stage].15min.csv"
    ctr067020SG = NRWGauge()
    ctr067020SG.dataset = ctr067020SG.read_nrw_to_xarray(fn_nrw, date_start=None, date_end=None)
    save_method(ctr067020SG, ofile="ctr067020SG_2021")
    #liv.plot_timeseries()
except:
    print('failed for ctr067020SG')

try:
    fn_nrw = dir + "067033.SG[Stage].15min.csv"
    ctr067033SG = NRWGauge()
    ctr067033SG.dataset = ctr067033SG.read_nrw_to_xarray(fn_nrw, date_start=None, date_end=None)
    save_method(ctr067033SG, ofile="ctr067033SG_2021")
    #liv.plot_timeseries()
except:
    print('failed for ctr067033SG')


try:
    fn_nrw = dir + "067033.FL[FlowLogged].15min.csv"
    ctr067033FL = NRWGauge()
    ctr067033FL.dataset = ctr067033FL.read_nrw_to_xarray(fn_nrw, date_start=None, date_end=None)
    save_method(ctr067033FL, ofile="ctr067033FL_2021")
    #liv.plot_timeseries()
except:
    print('failed for ctr067033FL')


#%% Load from csv and plot
######################################################
if(1):

    date_end = np.datetime64('2021-11-08T23:59:59')
    date_start = np.datetime64('2021-11-08')

    fn_nrw = dir + "067033.SG[Stage].15min.csv"
    ctr = NRWGauge()
    ctr.dataset = ctr.read_nrw_to_xarray(fn_nrw, date_start=date_start, date_end=date_end)

    fn_nrw = dir + "067020.SG[Stage].15min.csv"
    ctr2 = NRWGauge()
    ctr2.dataset = ctr2.read_nrw_to_xarray(fn_nrw, date_start=date_start, date_end=date_end)

    fn_nrw = dir + "067033.FL[FlowLogged].15min.csv"
    ctrf = NRWGauge()
    ctrf.dataset = ctrf.read_nrw_to_xarray(fn_nrw, date_start=date_start, date_end=date_end)



    #%% Plot data
    # CTR height + flow

    line_flag = True
    today_only_flag = True

    try: date_start
    except NameError: date_start = np.datetime64(ctr2.dataset.time[0].values,'ms')

    try: date_end
    except NameError: date_end = np.datetime64(ct2.dataset.time[-1].values, 'ms')

    plt.close('all')
    fig,  ax1 = plt.subplots(1, sharex=True)

    ## Only get tides over the weir with 8.75m at Liverpool
    fig.suptitle('Dee River heights and flow')
    #ax1.scatter(liv.dataset.time, liv.dataset.sea_level, color='k', s=1, label=liv.dataset.site_name)


    ax1 = scatter_plot(ax1, ctr.dataset.time, ctr.dataset.sea_level, 'k', 1, ctr.dataset.site_name.values)
    if line_flag:
        ax1 = line_plot(ax1, ctr.dataset.time, ctr.dataset.sea_level, 'k', 1)
    ax1 = scatter_plot(ax1, ctr2.dataset.time, ctr2.dataset.sea_level, 'b', 1, ctr2.dataset.site_name.values)
    if line_flag:
        ax1 = line_plot(ax1, ctr2.dataset.time, ctr2.dataset.sea_level, 'b', 1)
    ax1b = ax1.twinx()
    ax1b = scatter_plot(ax1b, ctrf.dataset.time, ctrf.dataset.sea_level, 'g', 1, ctrf.dataset.site_name.values)
    if line_flag:
        ax1b = line_plot(ax1b, ctrf.dataset.time, ctrf.dataset.sea_level, 'g', 1)
    ax1b.set_ylabel('flow rate (m3/s)', color='g')
    for tl in ax1b.get_yticklabels():
        tl.set_color('g')
    ax1.set_ylabel('water level (m)')

    myFmt = mdates.DateFormatter('%H:%M') #('%d-%a')
    ax1.xaxis.set_major_formatter(myFmt)
    #ax1.set_xlabel( date_start.astype(datetime.datetime).strftime('%d%b%y') + \
    #               '-' + date_end.astype(datetime.datetime).strftime('%d%b%y') )
    ax1.set_xlabel(date_end.astype(datetime.datetime).strftime('%d%b%y'))
    # Add empty data to ax1 to get "green flow data" in the legend
    ax1 = scatter_plot(ax1, [], [], 'g', 1, "Flow, above weir")

    # plot the legend
    ax1.legend(markerscale=6, loc='lower left')

    plt.savefig('Chester_river_NRW_levels.png')
