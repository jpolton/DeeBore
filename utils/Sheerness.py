#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse timeseries at Sheerness water levels.
* Load timeseries from observational data and ERA5 forced AMM7_surge simulations.
* Remove annual means and compare the difference.
* Compute harmonics analysis of the difference (representing a combined description of the difference between the
observational and simulated tide)
* Plot the stationarity of the harmonic coefficients with (plot_harmonic.py)
* Reconstruct a sealevel timeseries using the model and the correction. Compare with observed sealevel timeseries.

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
import json
from socket import gethostname
import pickle

import sys, os
sys.path.append(os.path.dirname(os.path.abspath("shoothill_api/shoothill_api.py")))
try: # command line
    from shoothill_api import GAUGE
except: # pycharm
    from shoothill_api.shoothill_api import GAUGE

coastdir = os.path.dirname(os.environ["HOME"]+'/GitHub/COAsT/coast')
sys.path.insert(0, coastdir)
import coast

# For interactive plotting / else save to .png
flag_interactive = False
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
    def __init__(self, date_start=None, date_end=None, dir=None):

        if dir == None:
            if "LJOB" in gethostname().upper():
                dir = '/projectsa/surge_archive/observations/netcdf/'  #2020/ClassAObsAfterSurgeQC*.nc
            else:
                dir = '/Users/jelt/DATA/SURGE/observations/'  #2020/ClassAObsAfterSurgeQC*.nc
            print(f"Try dir:{dir}")

        # Load data from nc file. Hardwired station number and station name
        station_id = 16
        site_name = "EA-Sheerness"
        tg = coast.Tidegauge()
        ds = (xr.open_mfdataset(dir+"????/ClassAObsAfterSurgeQC*.nc", combine='nested', concat_dim='time')
                .isel(station=[station_id])
                .rename_dims({'time': 't_dim', 'station': 'id_dim'})
                .rename_vars({"station_name": "site_name"}))

        tg.dataset = ds.rename_vars({"zos_total_observed": "sea_level", "timeseries": "time"})
        tg.dataset = tg.dataset.set_coords(["longitude", "latitude", "site_name", "time"])
        #tg.dataset = tg.dataset.where( tg.dataset.qc_flags!='N', drop=True)  # BODC flags
        tg.dataset = tg.dataset.where( tg.dataset.zos_flags != 0, drop=False)
        tg.dataset['time'] = tg.dataset.time.dt.round('s')  # round to the nearest second
        #tg.dataset['site_name'] = "EA-Sheerness"  # fix issue with replication over t_dim
        #tg.dataset['latitude'] = np.unique(tg.dataset.latitude)    # fix issue with replication over t_dim
        #tg.dataset['longitude'] = np.unique(tg.dataset.longitude)  # fix issue with replication over t_dim

        # Attributes
        tg.dataset["longitude"] = ("id_dim", np.unique(tg.dataset.longitude))
        tg.dataset["latitude"] = ("id_dim", np.unique(tg.dataset.latitude))
        tg.dataset["site_name"] = ("id_dim", [site_name])
        tg.dataset = tg.dataset.set_coords(["longitude", "latitude", "site_name"])


        # Return only values between stated dates
        if ((date_start == None) | (date_end == None)):
            pass
        else:
            # sortby("time")  was important to include with xr.open_mfdataset()
            tg.dataset = tg.dataset\
                            .sortby("time")\
                            .swap_dims({'t_dim':'time'})\
                            .sel(time=slice(date_start, date_end))\
                            .swap_dims({'time': 't_dim'})

        # Fix some attributes (others might not be correct for all data)
        try:
            tg.dataset['date_start'] = tg.dataset.time.min().values
            tg.dataset['date_end'] = tg.dataset.time.max().values
        except: pass
        self.tg = tg

    def to_tidegauge(self):
        return self.tg

class ErrorHarmonics:
    """
    class to handle management of harmonic analysis for the difference between obs and modelled sealevel
    ErrorHarmonics().pickle_harmonics(ha)
    ErrorHarmonics().load_harmonics()  --> self.ha
    """
    def __init__(self):
        pass


    def pickle_harmonics(self, ha=None):
        """ save copy of ha into pickle file, if requested """
        if ha == None:
            print(f"No harmonics file specified")
            return
        else:
            self.ha = ha
        print('Pickle harmonic data.')
        os.system('rm -f '+DATABUCKET_FILE)
        if(1):
            with open(DATABUCKET_FILE, 'wb') as file_object:
                pickle.dump(self.ha, file_object)
            return True
        else:
            print("Don't save as pickle file")
        return False

    def load_harmonics(self):
        """ load harmonics from pickle file. Save to self.ha """
        print('Load pickled harmonics.')
        if os.path.exists(DATABUCKET_FILE):
            template = "...Loading (%s)"
            print(template % DATABUCKET_FILE)
            with open(DATABUCKET_FILE, 'rb') as file_object:
                self.ha = pickle.load(file_object)
            return True
        else:
            print(f"Pickle file does not exist: {DATABUCKET_FILE}")
        return False

class AMM7_surge_ERA5:
    """
    Class to handle AMM7_surge data forced by ERA5
        tg = AMM7_surge_ERA5().to_tidegauge()

    """
    def __init__(self, date_start=None, date_end=None, dir=None, site_name=None):
        if (site_name == None) or (site_name == "Sheerness"):
            site_name = "Sheerness"
            fname = "SHNS.nc"
            print(f"Extract ERA5 forced data for {site_name}")
        else:
            print(f"Not expecting that station name: {site_name}")

        if dir == None:
            if "LJOB" in gethostname().upper():
                dir = "/projectsa/surge_archive/surge_hindcast/surge_hindcast_NEMOSurge_ERA5/surge_hindcast_ERA5_ClassA_bysite/"
            else:
                dir = "/Users/jelt/DATA/surge_hindcast_ERA5_ClassA_bysite/"
            print(f"Try dir:{dir}")

        fn_SHNS = dir+fname
        tg = GAUGE()
        tg.dataset = xr.open_dataset(fn_SHNS).swap_dims({"time":"t_dim"})
        tg.dataset['time'] = tg.dataset.timeseries.dt.round('s') # round to the nearest second
        tg.dataset['sea_level'] = tg.dataset.tide_ht + tg.dataset.residual_ht
        tg.dataset = tg.dataset.expand_dims(dim={"id_dim": 1})

        # Attributes
        tg.dataset["longitude"] = ("id_dim", [-999])
        tg.dataset["latitude"] = ("id_dim", [-999])
        tg.dataset["site_name"] = ("id_dim", [site_name])
        tg.dataset = tg.dataset.set_coords(["longitude", "latitude", "site_name"])


        # Return only values between stated dates
        if ((date_start == None) | (date_end == None)):
            pass
        else:
            tg.dataset = tg.dataset.where((tg.dataset.time >= date_start) & (tg.dataset.time <= date_end), drop=True)

        tg.dataset['date_start'] = tg.dataset.time.min().values
        tg.dataset['date_start'] = tg.dataset.time.max().values

        self.tg = tg

    def to_tidegauge(self):
        return self.tg

################################################################################
#%%  plot functions
def line_plot(ax, time, y, color, size, label=None ):
    ax.plot(time, y, color=color, linewidth=size, label=label)
    return ax

def scatter_plot(ax, time, y, color, size, label=None ):
    ax.scatter(time, y, color=color, s=size, label=label)
    return ax
#%%


################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":

    #### Constants
    DATABUCKET_FILE = "sealevel_difference_harmonics.pkl"

    ## initialise dictionaries
    dict_amp = {}
    dict_pha = {}
    dict_ha = {}


    for year in range(2012,2020+1):
        yyyy = str(year)
        date_start = np.datetime64(yyyy + '-01-01')
        date_end = np.datetime64(yyyy + '-12-31')

        #%% Load data
        if(0):
            sh_ea = GAUGE()
            sh_ea.dataset = sh_ea.read_ea_api_to_xarray(date_start=date_start, date_end=date_end, station_id="E71539")
            sh_ea.dataset = sh_ea.dataset.sortby(sh_ea.dataset.time)  # sometimes the arrives out of order

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
        sh_qc = QCdata(date_start=date_start, date_end=date_end).to_tidegauge()

        # Load Sheerness ERA5 data
        sh_nemo = AMM7_surge_ERA5(date_start=date_start, date_end=date_end).to_tidegauge()
        # copy lat/lon from obs file
        sh_nemo.dataset["longitude"] = sh_qc.dataset.longitude.values
        sh_nemo.dataset["latitude"]  = sh_qc.dataset.latitude.values

        try:
            if(1):
                # Tide gauge analysis
                tganalysis = coast.TidegaugeAnalysis()

                # This routine searches for missing values in each dataset and applies them
                # equally to each corresponding dataset
                nemo, qc = tganalysis.match_missing_values(sh_nemo.dataset.sea_level, sh_qc.dataset.sea_level)

                # Subtract means from all time series
                sh_nemo = tganalysis.demean_timeseries(nemo.dataset)
                sh_qc   = tganalysis.demean_timeseries(qc.dataset)

            sh_qc.dataset['sea_level_diff'] = sh_qc.dataset['sea_level'] - sh_nemo.dataset['sea_level']

            if(1):
                # Harmonic analysis
                ha_diff = tganalysis.harmonic_analysis_utide(sh_qc.dataset.sea_level_diff, min_datapoints=1)
                dict_ha[yyyy] = ha_diff  # store complete harmonic analysis object for later

                print(f"Species:   {ha_diff[0]['name'][0:10]}")
                print(f"Amplitude: {ha_diff[0]['A'][0:10]}")

                # Write to dictionary
                dict_amp[yyyy] = {}
                dict_pha[yyyy] = {}
                for i in range(10):
                    dict_amp[yyyy][ha_diff[0]['name'][i]] =  round(ha_diff[0]['A'][i], 3)
                    dict_pha[yyyy][ha_diff[0]['name'][i]] =  round(ha_diff[0]['g'][i])

            # Treshold statistics. See https://british-oceanographic-data-centre.github.io/COAsT/docs/examples/notebooks/tidegauge/tidegauge_validation_tutorial/



            #%% Plot data
            if(0):
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
            # Plot the two timeseries on LH axis and the difference on the RH axis
            fig.suptitle('Sheerness water levels')

            ax_l = line_plot(ax_l, sh_qc.dataset.time, sh_nemo.dataset.sea_level.squeeze(), 'y', 1, "nemo")
            ax_l = line_plot(ax_l, sh_qc.dataset.time,   sh_qc.dataset.sea_level.squeeze(), 'm', 1, "QC")
            ax_l.set_ylabel('water level (m)', color='k')


            ax_r = ax_l.twinx()
            ax_r = line_plot(ax_r, sh_qc.dataset.time, sh_qc.dataset.sea_level_diff.squeeze(), 'b', 1, "QC-NEMO")

            ax_r.set_ylabel('Diff (m)', color='b')
            for tl in ax_r.get_yticklabels():
                tl.set_color('b')

            # plot the legend
            ax_l.legend(markerscale=6, loc='upper left')

            # format the ticks
            myFmt = mdates.DateFormatter('%d-%b')
            days = mdates.DayLocator()
            ax_l.xaxis.set_major_locator(days)
            ax_l.xaxis.set_minor_locator(mdates.HourLocator([00, 6, 12, 18]))
            ax_l.xaxis.set_major_formatter(myFmt)

            ax_l.set_xlabel(date_start.astype(datetime.datetime).strftime('%d%b%y') + \
                            '-' + date_end.astype(datetime.datetime).strftime('%d%b%y'))

            if flag_interactive:
                plt.show()
            else:
                plt.savefig('Sheerness_river_levels_'+yyyy+'_measured.png')

        except:
            print(f"Problem with year: {yyyy}")

        print(dict_amp)

    # Save json file of harmonics
    with open('data_amp.json', 'w', encoding='utf-8') as f:
        json.dump(dict_amp, f, ensure_ascii=False, indent=4)
    with open('data_pha.json', 'w', encoding='utf-8') as f:
        json.dump(dict_pha, f, ensure_ascii=False, indent=4)

    # pickle the harmonic data
    flag = ErrorHarmonics().pickle_harmonics(dict_ha)



    ### Just take 2020 obs. Add harmonic corrections from year 2012 - 2019 to the modelled 2020. Compare against obs
    ####################################################################################

    ##  analysis period for timeseries reconstruction
    date_start = np.datetime64('2019-12-05')
    date_end   = np.datetime64('2019-12-07')

    ##  analysis period for timeseries reconstruction
    date_start = np.datetime64('2020-12-05')
    date_end   = np.datetime64('2020-12-07')

    # Load Sheerness from QC'd data
    ref_qc = QCdata(date_start=date_start, date_end=date_end).to_tidegauge()

    # Load Sheerness ERA5 data
    ref_nemo = AMM7_surge_ERA5(date_start=date_start, date_end=date_end).to_tidegauge()
    ref_nemo.dataset["longitude"] = ref_qc.dataset.longitude.values
    ref_nemo.dataset["latitude"] = ref_qc.dataset.latitude.values

    # This routine searches for missing values in each dataset and applies them
    # equally to each corresponding dataset
    nemo, qc = tganalysis.match_missing_values(ref_nemo.dataset.sea_level, ref_qc.dataset.sea_level)

    # Subtract means from all time series
    sh_nemo = tganalysis.demean_timeseries(nemo.dataset)
    sh_qc = tganalysis.demean_timeseries(qc.dataset)

    # load harmonic data from pickle file
    ds = ErrorHarmonics()
    if(ds.load_harmonics()):
        stored_ha = ds.ha
    else:
        print(f"Cannot reload harmonics")


    ## Plot the times series without any harmonic error correction. Then overlay lines with different corrections
    fig, [ax1, ax2] = plt.subplots(2, sharex=True)

    fig.suptitle('Sheerness water levels: modified simulation')

    ax1 = line_plot(ax1, sh_qc.dataset.time, sh_nemo.dataset.sea_level.squeeze(), 'y', 1, "nemo")
    ax1 = line_plot(ax1, sh_qc.dataset.time, sh_qc.dataset.sea_level.squeeze(), 'm', 1, "QC")
    ax1.set_ylabel('water level (m)', color='k')

    ax2 = line_plot(ax2, sh_qc.dataset.time, (sh_qc.dataset.sea_level-sh_nemo.dataset.sea_level).squeeze(), 'b', 1, "QC-NEMO")


    # Now reconstruct possible errors from other years
    dict_cmap = {0: 'k', 1: 'm', 2: 'y', 3: 'g', 4: 'r'}
    for year in range(2012,2019+1):
        yyyy = str(year)

        try:
            # reconstruct error timeseries from given year
            #harmonic_error = tganalysis.reconstruct_tide_utide(sh_qc.dataset.time, dict_ha[yyyy])
            harmonic_error = tganalysis.reconstruct_tide_utide(sh_qc.dataset.time, stored_ha[yyyy])
            ax2 = line_plot(ax2, sh_qc.dataset.time,
                            (sh_qc.dataset.sea_level - sh_nemo.dataset.sea_level - harmonic_error.dataset.reconstructed).squeeze(),
                        size=1, color=dict_cmap[year % 5], label=yyyy)
        except:
            print(f"Problem reconstructing {yyyy}")

    # Finish plot
    #############
    ax2.set_ylabel('Diff (m)', color='b')
    # ax_r.set_ylim([4.8,8.2])
    for tl in ax2.get_yticklabels():
        tl.set_color('b')

    # plot the legend
    ax1.legend(markerscale=6, loc='upper left')
    ax2.legend(markerscale=6, loc='upper right')

    # format the ticks
    myFmt = mdates.DateFormatter('%d-%b')
    days = mdates.DayLocator()
    ax1.xaxis.set_major_locator(days)
    ax1.xaxis.set_minor_locator(mdates.HourLocator([00, 6, 12, 18]))
    ax1.xaxis.set_major_formatter(myFmt)

    ax1.set_xlabel(date_start.astype(datetime.datetime).strftime('%d%b%y') + \
                    '-' + date_end.astype(datetime.datetime).strftime('%d%b%y'))

    if flag_interactive:
        plt.show()
    else:
        plt.savefig('Sheerness_river_levels_modified.png')
