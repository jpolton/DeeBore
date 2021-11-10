"""
 Investigate the correlation between CTR HT and Liverpool HT

 Author: jpolton
 Date: 9 Oct 2021

 Currently developing in coast_env
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


coastdir = os.path.dirname('/Users/jeff/GitHub/COAsT/coast')
sys.path.insert(0, coastdir)
from coast.tidegauge import Tidegauge
from coast.general_utils import day_of_week
from coast.stats_util import find_maxima

import scipy.signal # find_peaks

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

    def process_timeseries(self):
        """
        process into HLW values and times
        extract HW/LW
        store as xr.dataArray
        """
        HLW = "HW"
        ## Process the *_highs or *_lows
        #time_var = 'time_highs'
        #measure_var = 'sea_level_highs'
        #ind = [] # list of indices in the obs bore data where gladstone data is found
        if HLW == 'HW':
            time_var = 'time_highs'
            measure_var = 'sea_level_highs'
        elif HLW == 'LW':
            time_var = 'time_lows'
            measure_var = 'sea_level_lows'
        else:
            print('This should not have happened...')

        HT_h = [] # Extrema - height
        HT_t = [] # Extrema - time
        Gl_t = [] # Gladstone times
        Gl_h = [] # Gladstone heights

        for i in range(len(self.glad_HLW.dataset.time_highs)):
            try:
                HW = None
                LW = None
                #HLW = tg.get_tidetabletimes(self.bore.time[i].values)

                HW = self.ctr_HLW.get_tide_table_times(
                                        time_guess=self.glad_HLW.dataset.time_highs[i].values,
                                        time_var=time_var,
                                        measure_var=measure_var,
                                        method='nearest_1',
                                        winsize=6 ) #4h for HW, 6h for LW

                if type(HW) is xr.DataArray and xr.ufuncs.isfinite(HW.data):
                    #print(f"HW: {HW}")
                    HT_h.append( HW.values )
                    #print('len(HT_h)', len(HT_h))
                    HT_t.append( HW[time_var].values - self.glad_HLW.dataset.time_highs[i].values)
                    Gl_t.append( self.glad_HLW.dataset.time_highs[i].values )
                    Gl_h.append( self.glad_HLW.dataset.sea_level_highs[i].values )
                    #print('len(HT_t)', len(HT_t))
                    #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
                    #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
                    #ind.append(i)
                    #print(f"i:{i}, {HT_t[-1].astype('M8[ns]').astype('M8[ms]').item()}" )
                    #print(HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))

                    if(0):
                        ## Make timeseries plot around the highwater maxima to check
                        # values are being extracted as expected.
                        if (i % 12) == 0:
                            fig = plt.figure()

                        plt.subplot(3,4,(i%12)+1)
                        plt.plot(self.tg.dataset.time, self.tg.dataset.sea_level)
                        plt.plot( HT_t[-1], HT_h[-1], 'r+' )
                        plt.plot( [self.bore.time[i].values,self.bore.time[i].values],[0,11],'k')
                        plt.xlim([HT_t[-1] - np.timedelta64(5,'h'),
                                  HT_t[-1] + np.timedelta64(5,'h')])
                        plt.ylim([0,11])
                        plt.text( HT_t[-1]-np.timedelta64(5,'h'),10, self.bore.location[i].values)
                        plt.text( HT_t[-1]-np.timedelta64(5,'h'),1,  HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))
                        # Turn off tick labels
                        plt.gca().axes.get_xaxis().set_visible(False)
                        #plt.xaxis_date()
                        #plt.autoscale_view()
                        if (i%12) == 12-1:
                            plt.savefig('figs/check_get_tidetabletimes_'+str(i//12).zfill(2)+'_'+HLW+'_'+source+'.png')
                            plt.close('all')


                else:
                    logging.info(f"Did not find a high water near this guess")
                    print(f"Did not find a high water near this guess")

            except:
                logging.warning('Issue with appending HLW data')
                print('Issue with appending HLW data')


        # Save a xarray objects
        coords = {'time': (('time'), Gl_t)}
        #self.bore[loc+'_height_'+HLW+'_'+source] = xr.DataArray( np.array(HT_h), coords=coords, dims=['time'])
        #self.bore[loc+'_time_'+HLW+'_'+source] = xr.DataArray( np.array(HT_t), coords=coords, dims=['time'])
        self.height = xr.DataArray( np.array(HT_h), coords=coords, dims=['time'])
        self.lag = xr.DataArray( np.array(HT_t), coords=coords, dims=['time'])
        self.liv = xr.DataArray( np.array(Gl_h), coords=coords, dims=['time'])
        #print('There is a supressed plot.scatter here')
        #self.bore.plot.scatter(x='liv_time', y='liv_height'); plt.show()
        if(0):
            logging.debug(f"len(self.bore[loc+'_time_'{HLW}'_'{source}]): {len(self.bore[loc+'_time_'+HLW+'_'+source])}")
            #logging.info(f'len(self.bore.liv_time)', len(self.bore.liv_time))
            logging.debug(f"type(HT_t): {type(HT_t)}")
            logging.debug(f"type(HT_h): {type(HT_h)}")

            if loc=='liv':
                logging.debug('log time, orig tide table, new tide table lookup')
                for i in range(len(self.bore.time)):
                    logging.debug( f"{self.bore.time[i].values}, {self.bore['Liv (Gladstone Dock) HT time (GMT)'][i].values}, {self.bore['liv_time_'+HLW+'_'+source][i].values}")


        #print('log time, orig tide table, new tide table lookup')
        #for i in range(len(self.bore.time)):
        #    print( self.bore.time[i].values, self.bore['Liv (Gladstone Dock) HT time (GMT)'][i].values, self.bore['liv_time'][i].values)



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
        tg  = Tidegauge()
        tg1 = Tidegauge()
        tg2 = Tidegauge()
        tg3 = Tidegauge()
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
        if(0):
            loc = "ctr"
            ctr = Tidegauge()
            #date_start=np.datetime64('2014-01-01')
            date_start=np.datetime64('2021-01-01')
            date_end=np.datetime64('now','D')
            station_id = 7900 # below weir
            #station_id = 7899 # above weir
            ctr.dataset = ctr.read_shoothill_to_xarray(station_id=station_id ,date_start=date_start, date_end=date_end)

        ctr = Tidegauge()
        #ctr.dataset = xr.open_dataset("archive_shoothill/ctr_2021.nc")
        ctr.dataset = xr.open_mfdataset("archive_shoothill/ctr2_20*.nc")

        ctr_HLW = ctr.find_high_and_low_water(var_str='sea_level')
        self.ctr = ctr
        self.ctr_HLW = ctr_HLW

################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":

    #### Initialise logging
    now_str = datetime.datetime.now().strftime("%d%b%y %H:%M")
    logging.info(f"-----{now_str}-----")

    tt = Databucket()
    tt.load_tidetable()
    tt.load_ctr()
    tt.process_timeseries()

    plt.figure()
    tt.glad_HLW.dataset.sea_level_highs[0:10].plot()
    plt.savefig("tt.png")


    plt.figure()
    plt.plot(  tt.lag / np.timedelta64(1, 'm'), tt.liv, '+')
    plt.xlim([0,100])
    plt.xlabel('Timing CTR HT, minutes after LIV')
    plt.ylabel('Liverpool HT (m)')

    plt.plot([0,100],[8.05, 8.05])  # 13/10/2021  04:39 BST    8.05

    #tt.glad_HLW.dataset.sea_level_highs[0:10].plot()
    plt.savefig("tt.png")





lag = tt.lag.where(tt.liv > 7.9).where(tt.liv < 8.2) / np.timedelta64(1, 'm')
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(lag, bins = np.linspace(40,100,10))
plt.xlabel('Timing CTR HT, minutes after LIV')
plt.ylabel('bin count. Liv HT: 7.9 - 8.2m')
plt.title('Histogram of CTR HT timing 2020-21')
plt.savefig('hh.png')
