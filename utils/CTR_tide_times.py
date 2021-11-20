"""
Investigate the correlation between CTR HT and Liverpool HT

Author: jpolton
Date: 9 Oct 2021


Conda environment:
    coast + requests,
    (E.g. workshop_env w/ requests)

Example usage:

    python utils/CTR_tide_times.py
    ipython$
    run utils/CTR_tide_times


To Do:
    * fix: --> 296         self.bore[loc+'_height_'+HLW] = xr.DataArray( np.array(HT_h), coords=coords, dims=['time'])
    bore note defined
    * add min search to process()
 """

import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sklearn.metrics as metrics
import pytz
import pickle
import scipy.signal # find_peaks

#from coast.tidegauge import Tidegauge
from shoothill_api.shoothill_api import GAUGE
from coast.general_utils import day_of_week
from coast.stats_util import find_maxima

import logging
logging.basicConfig(filename='ctr.log', filemode='w+')
logging.getLogger().setLevel(logging.DEBUG)

from deebore import GAUGE
from deebore import Controller
#%% ################################################################################

class Databucket():
    """
    This is where the main things happen.
    Where user input is managed and methods are launched
    """
    ############################################################################
    #%% Initialising and Finishing methods
    ############################################################################
    def __init__(self):
        pass


    def process(self, tg:GAUGE=None, HLW:str="HW"):
        """
        Save into an dataset which is indexed against tide table HW times.
        """

        loc = "ctr"
        print(f"loc: {loc}")

        if HLW == 'HW':
            time_var = 'time_highs'
            measure_var = 'sea_level_highs'
        elif HLW == 'LW':
            time_var = 'time_lows'
            measure_var = 'sea_level_lows'
        else:
            print('This should not have happened...')

        # Truncate tide table data is necessary, for speed
        # Iterate of tide table HW times (even for LT analysis)
        HT_h = [] # Extrema - height
        HT_t = [] # Extrema - time
        HT_lag = [] # lag between liv HT and tg_HT
        ind_t = [] # store index times. Input guess_time
        ind_h = [] # store index height. Input height(guess_time)
        winsize = 3 #4h for HW, 6h for LW. +/- search distance for nearest extreme value

        tt =  GAUGE()
        print( tg.dataset.time.min() )
        tt.dataset = self.glad_HLW.dataset.sel( time_highs=slice(tg.dataset.time.min(), tg.dataset.time.max()) )

        for i in range(len(tt.dataset.time_highs)):
            try:
                HH = None
                guess_time = tt.dataset.time_highs[i].values

                # Extracting the highest and lowest value with a cubic spline is
                # very memory costly. Only need to use the cubic method for the
                # bodc and api sources, so compute the high and low waters in a
                # piecewise approach around observations times.
                if(1):
                    # This produces an xr.dataset with sea_level_highs and sea_level_lows
                    # with time variables time_highs and time_lows.
                    win = GAUGE()
                    win.dataset = tg.dataset.sel( time=slice(guess_time - np.timedelta64(winsize, "h"), guess_time + np.timedelta64(winsize, "h"))  )
                    #if HLW == "LW":
                    #    print(f"win.dataset {win.dataset}")
                    #print(i," win.dataset.time.size", win.dataset.time.size)
                    if win.dataset.time.size == 0:
                        tg_HLW = GAUGE()
                        tg_HLW.dataset = xr.Dataset({measure_var: (time_var, [np.NaN])}, coords={time_var: [guess_time]})
                    else:
                        if HLW == "HW" or HLW == "LW":
                            tg_HLW = win.find_high_and_low_water(var_str='sea_level',method='cubic')
                            print(f"max points: {len(tg_HLW.dataset[time_var])}")
                        else:
                            print(f"This should not have happened... HLW:{HLW}")
                # Save the largest
                try:
                    HH = tg_HLW.dataset[measure_var][tg_HLW.dataset[measure_var].argmax()]
                    lag = (tg_HLW.dataset[time_var][tg_HLW.dataset[measure_var].argmax()] - guess_time).astype('timedelta64[m]')
                except:
                    HH = xr.DataArray([np.NaN], dims=(time_var), coords={time_var: [guess_time]})[0]
                    lag = xr.DataArray([np.datetime64('NaT')], dims=(time_var), coords={time_var: [guess_time]})[0]

                #print("time,HH,lag:",i, guess_time, HH.values, lag.values)
                if type(HH) is xr.DataArray: ## Actually I think they are alway xr.DataArray with time, but the height can be nan.
                    #print(f"HW: {HW}")
                    HT_h.append( HH.values )
                    #print('len(HT_h)', len(HT_h))
                    HT_t.append( HH[time_var].values )
                    HT_lag.append( lag.values )
                    ind_t.append( tt.dataset.time_highs[i].values ) # guess_time
                    ind_h.append( tt.dataset.sea_level_highs[i].values )
                    #print('len(HT_t)', len(HT_t))
                    #print(f"i:{i}, {HT_t[-1].astype('M8[ns]').astype('M8[ms]').item()}" )
                    #print(HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))

                    ## Make timeseries plot around the highwater maxima to check
                    # values are being extracted as expected.
                    if (i % 12) == 0:
                        fig = plt.figure()

                    plt.subplot(3,4,(i%12)+1)
                    plt.plot(tg.dataset.time, tg.dataset.sea_level)
                    plt.plot( HT_t[-1], HT_h[-1], 'r+' )
                    plt.plot( [guess_time, guess_time],[0,11],'k')
                    plt.xlim([HT_t[-1] - np.timedelta64(5,'h'),
                              HT_t[-1] + np.timedelta64(5,'h')])
                    plt.ylim([0,11])
                    plt.text( HT_t[-1]-np.timedelta64(5,'h'),1,  HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))
                    # Turn off tick labels
                    plt.gca().axes.get_xaxis().set_visible(False)
                    #plt.xaxis_date()
                    #plt.autoscale_view()
                    if (i%12) == 12-1:
                        plt.savefig('figs/LIV_CTR_get_tidetabletimes_'+str(i//12).zfill(2)+'_'+HLW+'.png')
                        plt.close('all')


                else:
                    logging.info(f"Did not find a high water near this guess")
                    print(f"Did not find a high water near this guess")


            except:
                logging.warning('Issue with appending HLW data')
                print('Issue with appending HLW data')

        try: # Try and print the last observation timeseries
            plt.savefig('figs/LIV_CTR_get_tidetabletimes_'+str(i//12).zfill(2)+'_'+HLW+'.png')
            plt.close('all')
        except:
            logging.info(f"Did not have any extra panels to plot")
            print(f"Did not have any extra panels to plot")


        # Save a xarray objects
        coords = {'time': (('time'), ind_t)}
        #print("length of data:", len(np.array(HT_h)) )
        height_xr = xr.DataArray( np.array(HT_h), coords=coords, dims=['time'])
        time_xr = xr.DataArray( np.array(HT_t), coords=coords, dims=['time'])
        lag_xr = xr.DataArray( np.array(HT_lag), coords=coords, dims=['time'])
        ind_xr = xr.DataArray( np.array(ind_h), coords=coords, dims=['time'])

        #logging.debug(f"len(self.bore[loc+'_time_'{HLW}]): {len(self.bore[loc+'_time_'+HLW])}")
        #logging.info(f'len(self.bore.liv_time)', len(self.bore.liv_time))
        logging.debug(f"type(HT_t): {type(HT_t)}")
        logging.debug(f"type(HT_h): {type(HT_h)}")
        return height_xr, time_xr, lag_xr, ind_xr


    def load_tidetable(self):
        """
        load gladstone data
        save HT values in xarray:
            times_highs
            sea_level_highs
        """
        logging.info("Get Gladstone HLW data")
        # Load tidetable data from files
        filnam1 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2005_2014_HLW.txt'
        filnam2 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2015_2020_HLW.txt'
        filnam3 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2021_2022_HLW.txt'
        tg  = GAUGE()
        tg1 = GAUGE()
        tg2 = GAUGE()
        tg3 = GAUGE()
        tg1.dataset = tg1.read_hlw_to_xarray(filnam1)#, self.bore.time.min().values, self.bore.time.max().values)
        tg2.dataset = tg2.read_hlw_to_xarray(filnam2)#, self.bore.time.min().values, self.bore.time.max().values)
        tg3.dataset = tg3.read_hlw_to_xarray(filnam3)#, self.bore.time.min().values, self.bore.time.max().values)
        tg.dataset = xr.concat([ tg1.dataset, tg2.dataset, tg3.dataset], dim='time')

        # This produces an xr.dataset with sea_level_highs and sea_level_lows
        # with time variables time_highs and time_lows.
        self.glad_HLW = tg.find_high_and_low_water(var_str='sea_level')

    def load_ctr(self):
        """
        load timeseries data.
        store as xr.dataArray
        """

        ctr = GAUGE()
        #ctr.dataset = xr.open_dataset("archive_shoothill/ctr_2021.nc")
        ctr.dataset = xr.open_mfdataset("archive_shoothill/ctr2_202*.nc")

        #ctr_HLW = ctr.find_high_and_low_water(var_str='sea_level', method="cubic")
        self.ctr = ctr
        #self.ctr_HLW = ctr_HLW

    def load_liv(self):
        """
        load timeseries data.
        store as xr.dataArray
        """

        liv = GAUGE()
        liv.dataset = xr.open_dataset("archive_shoothill/liv_2021.nc")
        #liv.dataset = xr.open_mfdataset("archive_shoothill/liv_20*.nc")

        #liv_HLW = liv.find_high_and_low_water(var_str='sea_level', method="cubic")
        self.liv = liv
        #self.liv_HLW = liv_HLW


def histogram_CTR_LIV_lag():

    tt = Databucket()
    tt.load_tidetable()
    tt.load_ctr()

    tt.ctr_height, tt.ctr_time, tt.ctr_lag, tt.liv_height = tt.process(tg = tt.ctr)

    plt.figure()
    plt.plot(  tt.ctr_lag / np.timedelta64(1, 'm'), tt.liv_height, '+')
    plt.xlim([0,100])
    plt.xlabel('Timing CTR HT, minutes after LIV')
    plt.ylabel('Liverpool HT (m)')
    plt.plot([0,100],[8.05, 8.05])  # 13/10/2021  04:39 BST    8.05
    plt.savefig("tt.png")

    lag = tt.ctr_lag.where(tt.liv_height > 7.9).where(tt.liv_height < 8.2) / np.timedelta64(1, 'm')
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(lag, bins = np.linspace(40,100,10))
    plt.xlabel('Timing CTR HT, minutes after LIV')
    plt.ylabel('bin count. Liv HT: 7.9 - 8.2m')
    plt.title('Histogram of CTR HT timing 2020-21')
    plt.savefig('hh.png')

################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":

    #### Initialise logging
    now_str = datetime.datetime.now().strftime("%d%b%y %H:%M")
    logging.info(f"-----{now_str}-----")

    # Plot lag vs Gladstone heights for Chester HT
    # Plot the histogram of CTR lags for a window of Liv heights.
    histogram_CTR_LIV_lag()

    if(0):
        tt = Databucket()
        tt.load_tidetable()
        tt.load_ctr()
        #tt.load_liv()

        #tt.liv_height, tt.liv_time, tt.liv_lag, tt.liv_height = tt.process(tg = tt.liv)

        tt.ctr_height, tt.ctr_time, tt.ctr_lag, tt.liv_height = tt.process(tg = tt.ctr)

        plt.figure()
        tt.glad_HLW.dataset.sea_level_highs[0:10].plot()
        plt.savefig("tt.png")


        plt.figure()
        plt.plot(  tt.ctr_lag / np.timedelta64(1, 'm'), tt.liv_height, '+')
        plt.xlim([0,100])
        plt.xlabel('Timing CTR HT, minutes after LIV')
        plt.ylabel('Liverpool HT (m)')

        plt.plot([0,100],[8.05, 8.05])  # 13/10/2021  04:39 BST    8.05

        #tt.glad_HLW.dataset.sea_level_highs[0:10].plot()
        plt.savefig("tt.png")



        plt.figure()
        plt.plot(  tt.ctr_lag / np.timedelta64(1, 'm'), tt.liv_height-tt.ctr_height, '+')
        plt.xlim([0,100])
        plt.ylim([3,5.5])
        plt.xlabel('Timing CTR HT, minutes after LIV')
        plt.ylabel('Liverpool-Chester HT (m)')

        #plt.plot([0,100],[4.05, 4.05])  # 13/10/2021  04:39 BST    8.05

        #tt.glad_HLW.dataset.sea_level_highs[0:10].plot()
        plt.savefig("dd.png")




        lag = tt.ctr_lag.where(tt.liv_height > 7.9).where(tt.liv_height < 8.2) / np.timedelta64(1, 'm')
        fig, ax = plt.subplots(figsize =(10, 7))
        ax.hist(lag, bins = np.linspace(40,100,10))
        plt.xlabel('Timing CTR HT, minutes after LIV')
        plt.ylabel('bin count. Liv HT: 7.9 - 8.2m')
        plt.title('Histogram of CTR HT timing 2020-21')
        plt.savefig('hh.png')
