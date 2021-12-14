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
import matplotlib.dates as mdates
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

        return xr.DataSet of tide events and variables indexed by Liv HT time
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
                """
                INPUT:
                        xr.dataset of river data.
                        guess_time : liv_HW time
                        2 part window for time clipping
                RETURNS:
                        xr.dataset single values for river HW height, time and lag, using cubic fit
                        xr.dataset NaN, not enough data
                """
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
                            print(f"This should not have happened... HLW:{HLW}")
                # Save the largest
                try:
                    #print("tg_HLW.dataset[measure_var]",i, tg_HLW.dataset[measure_var])
                    HH = tg_HW.dataset[measure_var][tg_HW.dataset[measure_var].argmax()]
                    event_time = tg_HW.dataset[time_var][tg_HW.dataset[measure_var].argmax()]
                    HH_lag = (event_time - guess_time).astype('timedelta64[m]')
                except:
                    HH = xr.DataArray([np.NaN], dims=(time_var), coords={time_var: [guess_time]})[0]
                    HH_lag = xr.DataArray([np.datetime64('NaT').astype('timedelta64[m]')], dims=(time_var), coords={time_var: [guess_time]})[0]

                """ Append HW event data [floats, np.datetime64] """
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
                """
                INPUT:
                        xr.dataset of river data.
                        guess_time : liv_HW time
                        2 part window for time clipping [window[0] : rivHW_t]
                RETURNS:
                        xr.dataset single values for river LW height, time and lag, using cubic fit
                        xr.dataset NaN, not enough data
                """
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

                """ Append LW event data, being careful to get the appropriate liv LT [floats, np.datetime64] """
                #print("time,LL,LL_lag:",i, guess_time, LL.values, LL_lag.values)
                #if type(LL) is xr.DataArray: ## Actually I think they are alway xr.DataArray with time, but the height can be nan.
                try:
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


                except:
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

        #return HT_height_xr, HT_time_xr, HT_lag_xr, HT_ref_h_xr, LT_height_xr, LT_time_xr, LT_lag_xr, LT_ref_h_xr, LT_ref_t_xr

        # lags are referenced to liv_HT_t, which is also the index variable
        return xr.Dataset(data_vars={
                    "ctr_HT_h": HT_height_xr, "ctr_HT_t": HT_time_xr, "ctr_HT_dt": HT_lag_xr,
                    "liv_HT_h" : HT_ref_h_xr, "liv_HT_t" : HT_ref_h_xr.time,
                    "ctr_LT_h" : LT_height_xr, "ctr_LT_t": LT_time_xr, "ctr_LT_dt": LT_lag_xr,
                    "liv_LT_h" : LT_ref_h_xr, "liv_LT_t" : LT_ref_t_xr
                    })


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

        #ctr.dataset = xr.open_mfdataset("archive_shoothill/ctr2_2020.nc")
        #ctr.dataset = ctr.dataset.sel( time=slice(np.datetime64('2020-04-14T04:00:00'), np.datetime64('2020-04-16T18:00:00')) )
        #ctr.dataset = ctr.dataset.sel( time=slice(np.datetime64('2020-01-01T04:00:00'), np.datetime64('2020-04-16T18:00:00')) )

        #ctr.dataset = xr.open_mfdataset("archive_shoothill/ctr2_202*.nc")
        ctr.dataset = xr.open_mfdataset("archive_shoothill/ctr2_20[12][17890].nc")

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

class PickleJar():
    """ Class to handle pickle methods """
    def __init__(self, pickle_file:str=""):
        print(f"pickle file: {pickle_file}")
        self.pickle_file = pickle_file
        pass

    def load(self):
        """
        Auto load databucket from pickle file if it exists.

        Return:
            self.dataset
            self.load_pickle_flag [True/False]

        """
        print("Add to pickle file, if it exists")
        self.load_pickle_flag = False
        self.dataset = []
        try:
            if os.path.exists(self.pickle_file):
                template = "...Loading (%s)"
                print(template%self.pickle_file)
                with open(self.pickle_file, 'rb') as file_object:
                    self.dataset = pickle.load(file_object)
                    self.load_pickle_flag = True
            else:
                print("... %s does not exist"%pickle_file)
        except KeyError:
            print('ErrorA ')
        except (IOError, RuntimeError):
            print('ErrorB ')


    def to_pickle(self):
        """
        Save copy of self.dataset into pickle file, if requested
        Inputs:
            self.dataset    [xr.dataset]
            pickle_file  [str]
        Returns:
            pkl file
        """
        print('Pickle data.')
        os.system('rm -f '+self.pickle_file)
        try:
            with open(self.pickle_file, 'wb') as file_object:
                pickle.dump(self.dataset, file_object)
        except:
            print(f"Problem saving pickle file {self.pickle_file}")


class PostProcess():
    """
    Test the hypothesis that the data can collapse to a shallow water propagation
    problem, with a reference height to be determined. Ignoring effects of variable
    river depth
    """
    ############################################################################
    #%% Initialising and Finishing methods
    ############################################################################
    def __init__(self):
        pass


    def ref_height_from_ds(self, ds):
        """ Compute a reference height from xr.dataset
        dt_LW = dt(ctr_LW_t:Glad_LW_t) = ctr_t - Glad_HW_t + Glad_HW_t - Glad_LW_t
                = LT_lag + HT_ref_t - LT_ref_t
        """
        dt_LW_sq = ( (ds.ctr_LT_dt + ds.liv_HT_t.time - ds.liv_LT_t)/np.timedelta64(1, 's') )**2
        dt_HW_sq = ( ds.ctr_HT_dt/np.timedelta64(1, 's') )**2
        den = dt_HW_sq - dt_LW_sq

        a = (ds.liv_LT_h*dt_LW_sq - ds.liv_HT_h*dt_HW_sq) / den
        ds['a'] = a
        return ds

    def ref_L_from_ds(self, ds):
        """ Compute hyperthetical distance that linear wave travels, given reference height a"""
        dt_HW_sq = ( ds.ctr_HT_dt/np.timedelta64(1, 's') )**2
        L = np.sqrt( (ds.a + ds.liv_HT_h)*9.81 * dt_HW_sq )/1000. # in km
        ds['L'] = L  # in km
        return ds

############################################################################
## Bespoke methods
############################################################################

def histogram_CTR_LIV_lag():

    tt = Databucket()
    tt.load_tidetable()
    tt.load_ctr()

    HLW = "HW"
    ds = tt.process(tg = tt.ctr, HLW=HLW)

    plt.figure()
    plt.plot(  ds.ctr_HT_dt / np.timedelta64(1, 'm'),ds.liv_HT_h, '+')
    plt.xlim([0,100])
    plt.xlabel(f"Timing CTR {HLW}, minutes after LIV")
    plt.ylabel(f"Liverpool {HLW} (m)")
    plt.plot([0,100],[8.05, 8.05])  # 13/10/2021  04:39 BST    8.05
    plt.savefig("tt.png")

    lag = ds.ctr_HT_dt.where(ds.liv_HT_h > 7.9).where(ds.liv_HT_h < 8.2) / np.timedelta64(1, 'm')
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(lag, bins = np.linspace(40,100,10))
    plt.xlabel(f"Timing CTR {HLW}, minutes after LIV")
    plt.ylabel('bin count. Liv HT: 7.9 - 8.2m')
    plt.title(f"Histogram of CTR {HLW} timing 2020-21")
    plt.savefig('hh.png')

def find_similar_events():
    """
    For a given Liverpool HT height, find such occurances at CTR weir.
    Plot as timeseries relative to Liv HT time
    """
    
    liv_HT_t = np.datetime64('2021-12-19 11:09') # Time of reference Liv HT. Only used to time reference the plot time axis
    liv_HT_h = 8.9 # Height of reference Liv HT (m)
    winsize = 0.1 # +/- increment on HT height (m) over which to search for similar tidal events
   
    data_bucket = Databucket()
    data_bucket.load_tidetable()
    data_bucket.load_ctr()
    

    HLW = "HW"
    ds = data_bucket.process(tg = data_bucket.ctr, HLW=HLW)
    
    time = ds.time.where(ds.liv_HT_h > liv_HT_h - winsize).where(ds.liv_HT_h < liv_HT_h + winsize)
    
    myFmt = mdates.DateFormatter('%H:%M')
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(myFmt)

    for event in time:
        if np.isfinite(event):
            ctr = data_bucket.ctr.dataset.sel(time=slice(event - np.timedelta64(1, "h"), event + np.timedelta64(2, "h")))
            #plt.plot( (ctr.time - event)/ np.timedelta64(1, 'm'), ctr.sea_level, label=time)
            #plt.xlabel('minutes after Liv HT (11:09)')
            plt.plot( liv_HT_t + (ctr.time - event), ctr.sea_level, label=time)
            plt.xlabel('time ')
            plt.ylabel('Chester water level (m)')
            plt.title('Chester water level scenarios, with 8.9m tides at Liverpool')
            
            
            
def main1():
    """ Read and process timeseries. Create xarray dataset. Export and pickly dataframe
    Plot graphs """
    data_bucket = Databucket()
    data_bucket.load_tidetable()
    data_bucket.load_ctr()

    #HT_height_xr, HT_time_xr, HT_lag_xr, HT_ref_h_xr, LT_height_xr, LT_time_xr, LT_lag_xr, LT_ref_h_xr, LT_ref_t_xr = data_bucket.process(tg = tt.ctr, HLW="HW")
    ds = data_bucket.process(tg = data_bucket.ctr, HLW="HW")
    #data_bucket.ds = ds
    #data_bucket.to_pickle()


    pickle_jar = PickleJar(pickle_file="CTR_tide_times.pkl")
    pickle_jar.dataset = ds
    pickle_jar.to_pickle()


    # Make some plots
    plt.figure()
    #plt.plot(  tt.ctr_lag / np.timedelta64(1, 'm'), tt.liv_height-tt.ctr_height, '+')
    plt.plot(  ds.ctr_HT_dt / np.timedelta64(1, 'm'), ds.liv_HT_h-ds.ctr_HT_h, '+')
    plt.xlim([0,100])
    plt.ylim([3,5.5])
    plt.xlabel('Timing CTR HT, minutes after LIV')
    plt.ylabel('Liverpool-Chester HT (m)')
    plt.savefig("dd.png")


    plt.figure()
    #plt.plot(  tt.ctr_lag / np.timedelta64(1, 'm'), tt.liv_height-tt.ctr_height, '+')
    plt.scatter(  (ds.ctr_HT_dt - ds.ctr_LT_dt) / np.timedelta64(1, 'm'),
        ds.ctr_HT_h - ds.ctr_LT_h,
        c=ds.liv_HT_h, marker='+')
    #plt.xlim([0,100])
    #plt.ylim([3,5.5])
    #legend
    cbar = plt.colorbar()
    cbar.set_label('High Water at Liverpool (m)', rotation=270)
    plt.xlabel('time(LT:HT) at CTR, mins')
    plt.ylabel('hight(HT-LT) at Chester (m)')
    plt.title('Magnitude and duration of rising tide at CTR')
    plt.savefig("deltaH_deltaT_CTR.png")



################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":

    #### Constants
    DATABUCKET_FILE = "CTR_tide_times.pkl"


    #### Initialise logging
    now_str = datetime.datetime.now().strftime("%d%b%y %H:%M")
    logging.info(f"-----{now_str}-----")

    ## Plot lag vs Gladstone heights for Chester HT
    ## Plot the histogram of CTR lags for a window of Liv heights.
    #histogram_CTR_LIV_lag()

    ## Read and process timeseries. Create xarray dataset. Export and pickly dataframe
    ##  Plot graphs
    #main1()

    if(0):
        aa = PostProcess()
        #ds = aa.load_databucket()

        pickle_jar = PickleJar(pickle_file="CTR_tide_times.pkl")
        pickle_jar.load()
        ds = pickle_jar.dataset

        ds = aa.ref_height_from_ds(ds)
        # For a river river height (LT_height), is 'a' about constant? Well it does depend on the Glad HT_h...
        #ax1 = df.plot.scatter(x='a', y='LT_height', c='HT_ref_h') #; plt.show()
        plt.scatter( ds.a , ds.ctr_LT_h, c=ds.liv_HT_h )
        plt.xlabel('Estimated displacement depth (m)')
        plt.ylabel('CTR LT waterlevel (m)')
        clb=plt.colorbar()
        clb.ax.set_ylabel('Liv HT (m)')
        plt.show()

        ds = aa.ref_L_from_ds(ds)
        #ax1 = df.plot.scatter(x='L', y='LT_height', c='HT_ref_h'); plt.show()
        plt.scatter( ds.L , ds.ctr_LT_h, c=ds.liv_HT_h )
        plt.xlabel('Estimated separation distance (km)')
        plt.ylabel('CTR LT waterlevel (m)')
        clb=plt.colorbar()
        clb.ax.set_ylabel('Liv HT (m)')
        plt.show()
        

    if(0):    
        main1()
        plt.plot(  ds.ctr_HT_dt / np.timedelta64(1, 'm'), ds.liv_HT_h, '+')
        plt.xlabel('CTR HT timing after Liv (mins)')
        plt.ylabel('Liverpool HT (m)')
        plt.plot( [0, 100], [8.9,8.9], 'r')


## 


