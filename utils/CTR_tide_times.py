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
    * add min search to process(). Probably linked to CTR HT search
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

        tg: dataset to process. E.g. full timeseries from chester
        """

        loc = "ctr"
        HLW = "HW"
        print(f"loc: {loc} {HLW}")

        tt =  GAUGE()
        print( tg.dataset.time.min() )

        # TideTable dataset truncated to relevant period for both highs and lows
        tt.dataset = self.glad_HLW.dataset.sel(
            time_highs=slice(tg.dataset.time.min(), tg.dataset.time.max()),
            time_lows =slice(tg.dataset.time.min(), tg.dataset.time.max()) )

        if HLW == 'HW':
            time_var = 'time_highs'
            measure_var = 'sea_level_highs'
            winsize = [3,3] #4h for HW, 6h for LW. +/- search distance for nearest extreme value

        elif HLW == 'LW':
            time_var = 'time_lows'
            measure_var = 'sea_level_lows'
            # TideTable dataset truncated to relevant period
            winsize = [-3,9] #4h for HW, 6h for LW. +/- search distance for nearest extreme value
        else:
            print('This should not have happened...')

        # Truncate tide table data is necessary, for speed
        # Iterate of tide table HW times (even for LT analysis)
        HT_h = [] # Extrema - height
        HT_t = [] # Extrema - time
        HT_lag = [] # lag between liv HT and tg_HT
        LT_h = [] # Extrema low tide - height
        LT_t = [] # Extrema low tide - time
        LT_lag = [] # lag between Liv HT and tg_LT
        ref_HT_t = [] # store index HT times. Input guess_time
        ref_HT_h = [] # store index HT height. Input height(guess_time)
        ref_LT_t = [] # store index LT times. Input guess_time
        ref_LT_h = [] # store index LT height.

        for i in range(len(tt.dataset[time_var])):
            if(1):#try:
                time_var = 'time_highs'
                measure_var = 'sea_level_highs'

                HH = None
                guess_time = tt.dataset[time_var][i].values
                print(f"guess: {guess_time}")

                # Extracting the highest and lowest value with a cubic spline is
                # very memory costly. Only need to use the cubic method for the
                # bodc and api sources, so compute the high and low waters in a
                # piecewise approach around observations times.
                if(1):
                    # This produces an xr.dataset with sea_level_highs and sea_level_lows
                    # with time variables time_highs and time_lows.
                    win = GAUGE()
                    win.dataset = tg.dataset.sel( time=slice(guess_time - np.timedelta64(winsize[0], "h"), guess_time + np.timedelta64(winsize[1], "h"))  )
                    #if HLW == "LW":
                    # print(f"win.dataset {win.dataset}")
                    print(i," win.dataset.time.size", win.dataset.time.size)
                    if win.dataset.time.size <= 3:
                        tg_HW = GAUGE()
                        tg_HW.dataset = xr.Dataset({measure_var: (time_var, [np.NaN])}, coords={time_var: [guess_time]})
                    else:
                        if HLW == "HW" or HLW == "LW":
                            #win.dataset['sea_level_trend'] = win.dataset.sea_level.differentiate("time")
                            tg_HW = win.find_high_and_low_water(var_str='sea_level',method='cubic')
                            #tg_inf = win.find_high_and_low_water(var_str='sea_level_trend',method='cubic')

                            print(f"max points: {len(tg_HW.dataset[time_var])}")
                        else:
                            print(f"This should not have happened... HLW:{HW}")
                # Save the largest
                try:
                    #print("tg_HLW.dataset[measure_var]",i, tg_HLW.dataset[measure_var])
                    HH = tg_HW.dataset[measure_var][tg_HW.dataset[measure_var].argmax()]
                    event_time = tg_HW.dataset[time_var][tg_HW.dataset[measure_var].argmax()]
                    HH_lag = (event_time - guess_time).astype('timedelta64[m]')
                except:
                    HH = xr.DataArray([np.NaN], dims=(time_var), coords={time_var: [guess_time]})[0]
                    HH_lag = xr.DataArray([np.datetime64('NaT').astype('timedelta64[m]')], dims=(time_var), coords={time_var: [guess_time]})[0]

                #print("time,HH,HH_lag:",i, guess_time, HH.values, HH_lag.values)
                if type(HH) is xr.DataArray: ## Actually I think they are alway xr.DataArray with time, but the height can be nan.
                    print(f"HH: {HH}")
                    HT_h.append( HH.values )
                    #print('len(HT_h)', len(HT_h))
                    HT_t.append( HH[time_var].values )
                    HT_lag.append( HH_lag.values )

                    ref_HT_t.append( tt.dataset[time_var][i].values ) # guess_time
                    ref_HT_h.append( tt.dataset[measure_var][i].values )

                ##################
                # Find the turning/shock point before HT.
                # Remove a linear trend from HT-3 : HT. Find minimum.
                time_var = 'time_lows'
                measure_var = 'sea_level_lows'

                win_mod = GAUGE()
                win_mod.dataset = tg.dataset.sel( time=slice(guess_time - np.timedelta64(winsize[0], "h"), HH.time_highs.values)  )
                if win_mod.dataset.time.size == 0:
                    tg_LW = GAUGE()
                    tg_LW.dataset = xr.Dataset({measure_var: (time_var, [np.NaN])}, coords={time_var: [guess_time]})
                else:
                    print(f"win_mod.dataset.time.size : {win_mod.dataset.time.size}")
                    nt = len(win_mod.dataset.sea_level)
                    y0 = win_mod.dataset.sea_level[0].values
                    y1 = win_mod.dataset.sea_level[-1].values
                    win_mod.dataset['sea_level'] = win_mod.dataset.sea_level - [(y0*(nt-1-kk) + y1*kk)/(nt-1) for kk in range(nt)]
                    tg_LW = win_mod.find_high_and_low_water(var_str='sea_level',method='comp')

                    if(0):
                        plt.close('all')
                        plt.figure()
                        plt.plot( win_mod.dataset.time, win_mod.dataset.sea_level, 'g.' )
                        plt.plot( win_mod.dataset.time, win_mod.dataset.sea_level, 'g' )
                        plt.plot( tg_LW.dataset.time_lows, tg_LW.dataset.sea_level_lows, 'r+')
                        plt.plot( tg_LW.dataset.time_lows, tg_LW.dataset.sea_level_lows, 'r')
                        plt.xlim([guess_time - np.timedelta64(winsize[0],'h'),
                                  guess_time + np.timedelta64(winsize[1],'h')])
                        plt.show()



                try:
                    # Find time. Interpolate time onto original timeseries
                    #print(f"tg_LW.dataset:{tg_LW.dataset}")
                    #print(f"---")
                    #print(f"tg_LW.dataset[measure_var].argmin():{tg_LW.dataset[measure_var].argmin().values}")

                    event_time = tg_LW.dataset[time_var][tg_LW.dataset[measure_var].argmin().values]
                    #print(f"event_time: {event_time}")
                    # interpolate back onto original sea_level timeseries (not needed for method="comp")
                    LL = win.dataset.sea_level.interp(time=event_time, method='cubic') # two coords: {time_lows, time} inherited from {event_time, win_mod.dataset}
                    #print(f"LL.values: {LL.values}")
                    #print("tg_LW.dataset[measure_var]",i, tg_LW.dataset[measure_var])
                    #LL = tg_HLW.dataset[measure_var][tg_inf.dataset[measure_trend_var].argmax()] # Units: (m), not (m/s)
                    LL_lag = (event_time - guess_time).astype('timedelta64[m]')
                except:
                    LL = xr.DataArray([np.NaN], dims=(time_var), coords={time_var: [guess_time]})[0]
                    LL_lag = xr.DataArray([np.datetime64('NaT').astype('timedelta64[m]')], dims=(time_var), coords={time_var: [guess_time]})[0]

                # Find the preceeding minima

                #print("time,LL,LL_lag:",i, guess_time, LL.values, LL_lag.values)
                if type(LL) is xr.DataArray: ## Actually I think they are alway xr.DataArray with time, but the height can be nan.
                    LT_h.append( LL.values )
                    #print('len(HT_h)', len(HT_h))
                    LT_t.append( LL[time_var].values )
                    LT_lag.append( LL_lag.values )

                    print(f"Check guess: {tt.dataset.time_highs[i].values}")
                    try: #if(1):
                        if (tt.dataset.time_lows[i].values < tt.dataset.time_highs[i].values) and \
                            (tt.dataset.time_lows[i].values > (tt.dataset.time_highs[i].values - np.timedelta64(12, 'h'))):
                            print('HT_t(i)-12 < LT_t(i) < HT_t(i)')
                            ref_LT_t.append( tt.dataset[time_var][i].values )
                            ref_LT_h.append( tt.dataset[measure_var][i].values )
                        elif (tt.dataset.time_lows[i-1].values < tt.dataset.time_highs[i].values) and \
                            (tt.dataset.time_lows[i-1].values > (tt.dataset.time_highs[i].values - np.timedelta64(12, 'h'))):
                            print('HT_t(i)-12 < LT_t(i-1) < HT_t(i)')
                            ref_LT_t.append( tt.dataset[time_var][i-1].values )
                            ref_LT_h.append( tt.dataset[measure_var][i-1].values )
                        elif (tt.dataset.time_lows[i+1].values < tt.dataset.time_highs[i].values) and \
                            (tt.dataset.time_lows[i+1].values > (tt.dataset.time_highs[i].values - np.timedelta64(12, 'h'))):
                            print('HT_t(i)-12 < LT_t(i+1) < HT_t(i)')
                            ref_LT_t.append( tt.dataset[time_var][i+1].values )
                            ref_LT_h.append( tt.dataset[measure_var][i+1].values )
                        else:
                            #print('LT_t(i) !< HT_t(i)')
                            print(f"LT:{tt.dataset.time_lows[i].values}. HT:{tt.dataset.time_highs[i].values}")
                            ref_LT_t.append( np.datetime64('NaT').astype('timedelta64[m]') )
                            ref_LT_h.append( np.nan )
                    except:
                            ref_LT_t.append( np.datetime64('NaT').astype('timedelta64[m]') )
                            ref_LT_h.append( np.nan )

                    #print('len(HT_t)', len(HT_t))
                    #print(f"i:{i}, {HT_t[-1].astype('M8[ns]').astype('M8[ms]').item()}" )
                    #print(HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))

                    ## Make timeseries plot around the highwater maxima to check
                    # values are being extracted as expected.
                    if (i % 12) == 0:
                        fig = plt.figure()

                    if HLW == "HW":
                        xlim = [HT_t[-1] - np.timedelta64(winsize[0],'h'),
                                  HT_t[-1] + np.timedelta64(winsize[1],'h')]
                    elif HLW == "LW":
                        xlim = [guess_time - np.timedelta64(winsize[0],'h'),
                                  guess_time + np.timedelta64(winsize[1],'h')]
                    else:
                        print(f"Not expecting HLW:{HLW}")
                    if loc == 'ctr':
                        ylim = [2,7]

                    elif loc == 'liv':
                        ylim = [0,11]
                    else:
                        ylim = [0,11]
                    plt.subplot(3,4,(i%12)+1)
                    plt.plot(tg.dataset.time, tg.dataset.sea_level,'b')
                    plt.plot(tg.dataset.time, tg.dataset.sea_level,'b.')
                    #plt.plot(tg.dataset.time, ylim[0]+1e13*tg.dataset.sea_level.differentiate("time"),'g')
                    print(f"LT_h[-1]: {LT_h[-1]}")
                    print(f"LT_t[-1]: {LT_t[-1]}")
                    plt.plot( HT_t[-1], HT_h[-1], 'r+' )
                    plt.plot( LT_t[-1], LT_h[-1], 'g+' )
                    plt.plot( [guess_time, guess_time],[0,11],'k')
                    plt.xlim(xlim)
                    plt.ylim(ylim) #[0,11])
                    plt.text( HT_t[-1]-np.timedelta64(winsize[0],'h'),ylim[0]+ 0.05*(ylim[1]-ylim[0]),  HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))
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


            if(0):#except:
                logging.warning('Issue with appending HLW data')
                print('Issue with appending HLW data')

        try: # Try and print the last observation timeseries
            plt.savefig('figs/LIV_CTR_get_tidetabletimes_'+str(i//12).zfill(2)+'_'+HLW+'.png')
            plt.close('all')
        except:
            logging.info(f"Did not have any extra panels to plot")
            print(f"Did not have any extra panels to plot")


        # Save a xarray objects
        coords = {'time': (('time'), ref_HT_t)}
        #print("length of data:", len(np.array(HT_h)) )
        HT_height_xr = xr.DataArray( np.array(HT_h), coords=coords, dims=['time'])
        HT_time_xr = xr.DataArray( np.array(HT_t), coords=coords, dims=['time'])
        HT_lag_xr = xr.DataArray( np.array(HT_lag), coords=coords, dims=['time'])
        HT_ref_h_xr = xr.DataArray( np.array(ref_HT_h), coords=coords, dims=['time'])


        LT_height_xr = xr.DataArray( np.array(LT_h), coords=coords, dims=['time'])
        LT_time_xr = xr.DataArray( np.array(LT_t), coords=coords, dims=['time'])
        LT_lag_xr = xr.DataArray( np.array(LT_lag), coords=coords, dims=['time'])
        LT_ref_h_xr = xr.DataArray( np.array(ref_LT_h), coords=coords, dims=['time'])
        LT_ref_t_xr = xr.DataArray( np.array(ref_LT_t), coords=coords, dims=['time'])
        #logging.debug(f"len(self.bore[loc+'_time_'{HLW}]): {len(self.bore[loc+'_time_'+HLW])}")
        #logging.info(f'len(self.bore.liv_time)', len(self.bore.liv_time))
        logging.debug(f"type(HT_t): {type(HT_t)}")
        logging.debug(f"type(HT_h): {type(HT_h)}")
        return HT_height_xr, HT_time_xr, HT_lag_xr, HT_ref_h_xr, LT_height_xr, LT_time_xr, LT_lag_xr, LT_ref_h_xr, LT_ref_t_xr


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
        #ctr.dataset = ctr.dataset.sel( time=slice(np.datetime64('2021-03-31T06:00:00'), np.datetime64('2021-03-31T18:00:00')) )

        ctr.dataset = xr.open_mfdataset("archive_shoothill/ctr2_2020.nc")
        #ctr.dataset = ctr.dataset.sel( time=slice(np.datetime64('2020-04-14T04:00:00'), np.datetime64('2020-04-16T18:00:00')) )
        ctr.dataset = ctr.dataset.sel( time=slice(np.datetime64('2020-01-01T04:00:00'), np.datetime64('2020-04-16T18:00:00')) )

        #ctr.dataset = xr.open_mfdataset("archive_shoothill/ctr2_202*.nc")

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
    plt.xlabel(f"Timing CTR {HLW}, minutes after LIV")
    plt.ylabel(f"Liverpool {HLW} (m)")
    plt.plot([0,100],[8.05, 8.05])  # 13/10/2021  04:39 BST    8.05
    plt.savefig("tt.png")

    lag = tt.ctr_lag.where(tt.liv_height > 7.9).where(tt.liv_height < 8.2) / np.timedelta64(1, 'm')
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(lag, bins = np.linspace(40,100,10))
    plt.xlabel(f"Timing CTR {HLW}, minutes after LIV")
    plt.ylabel('bin count. Liv HT: 7.9 - 8.2m')
    plt.title(f"Histogram of CTR {HLW} timing 2020-21")
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

    ## Plot lag vs Gladstone heights for Chester HT
    ## Plot the histogram of CTR lags for a window of Liv heights.
    #histogram_CTR_LIV_lag()

    if(1):
        tt = Databucket()
        tt.load_tidetable()
        tt.load_ctr()

        #tt.ctr_height, tt.ctr_time, tt.ctr_lag, tt.liv_height = tt.process(tg = tt.ctr, HLW="HW")
        HT_height_xr, HT_time_xr, HT_lag_xr, HT_ref_h_xr, LT_height_xr, LT_time_xr, LT_lag_xr, LT_ref_h_xr, LT_ref_t_xr = tt.process(tg = tt.ctr, HLW="HW")

        # Create pandas dataframe
        zipped = list(zip(HT_height_xr.values,
                         HT_time_xr.values,
                         HT_lag_xr.values,
                         HT_ref_h_xr.values,
                         LT_height_xr.values,
                         LT_time_xr.values,
                         LT_lag_xr.values,
                         LT_ref_h_xr.values,
                         LT_ref_t_xr.values))

        columns=['HT_height', 'HT_time', 'HT_lag', 'HT_ref_h',
                 'LT_height', 'LT_time', 'LT_lag',
                 'LT_ref_h', 'LT_ref_t']

        df = pd.DataFrame(zipped, columns=columns)

        # Make some plots
        plt.figure()
        #plt.plot(  tt.ctr_lag / np.timedelta64(1, 'm'), tt.liv_height-tt.ctr_height, '+')
        plt.plot(  HT_lag_xr / np.timedelta64(1, 'm'), HT_ref_h_xr-HT_height_xr, '+')
        plt.xlim([0,100])
        plt.ylim([3,5.5])
        plt.xlabel('Timing CTR HT, minutes after LIV')
        plt.ylabel('Liverpool-Chester HT (m)')
        plt.savefig("dd.png")


        plt.figure()
        #plt.plot(  tt.ctr_lag / np.timedelta64(1, 'm'), tt.liv_height-tt.ctr_height, '+')
        plt.scatter(  (HT_lag_xr - LT_lag_xr) / np.timedelta64(1, 'm'),
            HT_height_xr-LT_height_xr,
            c=HT_ref_h_xr, marker='+')
        #plt.xlim([0,100])
        #plt.ylim([3,5.5])
        #legend
        cbar = plt.colorbar()
        cbar.set_label('High Water at Liverpool (m)', rotation=270)
        plt.xlabel('time(LT:HT) at CTR, mins')
        plt.ylabel('hight(HT-LT) at Chester (m)')
        plt.title('Magnitude and duration of rising tide at CTR')
        plt.savefig("deltaH_deltaT_CTR.png")
