"""
Read in a process Dee Bore data
Author: jpolton
Date: 26 Sept 2020

Conda environment:
    coast + requests,
    (E.g. workshop_env w/ requests)

    ### Build python environment:
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

Example usage:
    python deebore.py


To do:
    * Smooth data before finding flood ebb
    * Workflow for updating all measured data
"""

import os
#import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
#import sklearn.metrics as metrics
import pytz
import pickle

from shoothill_api.shoothill_api import GAUGE
from coast.general_utils import day_of_week
#from coast.stats_util import find_maxima


#import scipy.signal # find_peaks

import logging
logging.basicConfig(filename='bore.log', filemode='w+')
logging.getLogger().setLevel(logging.DEBUG)


class GladstoneTideTable:
    """
    if source == "harmonic": # Load tidetable data from files
    tg = GladstoneTideTable().to_tidegauge()
    """
    def __init__(self):

        logging.info("Get Gladstone HLW data")
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
        self.tg = tg

    def to_tidegauge(self):
        return self.tg


class BODC:
    """
    if source == "bodc": # Load gauge data from BODC files
    tg = BODC().to_tidegauge()
    """
    def __init__(self):
        dir = '/Users/jeff/GitHub/DeeBore/data/BODC_processed/'
        filelist = ['2005LIV.txt',
        '2006LIV.txt', '2007LIV.txt',
        '2008LIV.txt', '2009LIV.txt',
        '2010LIV.txt', '2011LIV.txt',
        '2012LIV.txt', '2013LIV.txt',
        '2014LIV.txt', '2015LIV.txt',
        '2016LIV.txt', '2017LIV.txt',
        '2018LIV.txt', '2019LIV.txt',
        '2020LIV.txt',
        'LIV2101.txt', 'LIV2102.txt',
        'LIV2103.txt', 'LIV2104.txt',
        'LIV2105.txt', 'LIV2106.txt',
        'LIV2107.txt', 'LIV2108.txt',
        'LIV2109.txt', 'LIV2110.txt']
        tg  = GAUGE()
        for file in filelist:
            tg0=GAUGE()
            tg0.dataset = tg0.read_bodc_to_xarray(dir+file)
            if tg.dataset is None:
                tg.dataset = tg0.dataset
            else:
                tg.dataset = xr.concat([ tg.dataset, tg0.dataset], dim='time')
        # Use QC to drop null values
        #tg.dataset['sea_level'] = tg.dataset.sea_level.where( np.logical_or(tg.dataset.qc_flags=='', tg.dataset.qc_flags=='T'), drop=True)
        tg.dataset['sea_level'] = tg.dataset.sea_level.where( tg.dataset.qc_flags!='N', drop=True)
        # Fix some attributes (others might not be correct for all data)
        tg.dataset['start_date'] = tg.dataset.time.min().values
        tg.dataset['end_date'] = tg.dataset.time.max().values
        self.tg = tg

    def to_tidegauge(self):
        return self.tg


class GladstoneAPI:
    """
    if source == "api": # load full tidal signal from shoothill
    tg = GladstoneAPI().to_tidegauge()
    """
    def __init__(self):


        date_start=np.datetime64('2005-04-01')
        date_end=np.datetime64('now','D')
        fn_archive = "liv" # File head for netcdf archive of api call

        # Load timeseries from local file if it exists
        try:
            tg1 = GAUGE()
            tg2 = GAUGE()
            tg = GAUGE()

            # Load local file. Created with archive_shoothill.py
            dir = "archive_shoothill/"
            tg1.dataset = xr.open_mfdataset(dir + fn_archive + "_????.nc") # Tidal port Gladstone Dock, Liverpool
            tg1.dataset = tg1.dataset.sel(time=slice(date_start, date_end))
            print(f"{len(tg1.dataset.time)} pts loaded from netcdf")
            if (tg1.dataset.time[-1].values < date_end):
                tg2 = GAUGE()
                tg2.dataset = tg2.read_shoothill_to_xarray(date_start=tg1.dataset.time[-1].values, date_end=date_end)
                tg.dataset = xr.concat([ tg1.dataset, tg2.dataset], dim='time')
                print(f"{len(tg2.dataset.time)} pts loaded from API")
            else:
                tg = tg1
        except:
            tg.dataset = tg.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)

        self.tg = tg

    def to_tidegauge(self):
        return self.tg


class ChesterAPI:
    """
    if source == "ctr": # use api to load chester weir.
    tg = ChesterAPI().to_tidegauge()
    """
    def __init__(self):

        tg = GAUGE()
        date_start=np.datetime64('2014-01-01')
        date_end=np.datetime64('now','D')
        #station_id = 7900 # below weir
        station_id = 7899 # above weir
        fn_archive = "ctr" # File head for netcdf archive of api call

        # Load timeseries from local file if it exists
        try:
            tg1 = GAUGE()
            tg2 = GAUGE()
            tg = GAUGE()

            # Load local file. Created with archive_shoothill.py
            dir = "archive_shoothill/"
            tg1.dataset = xr.open_mfdataset(dir + fn_archive + "_????.nc") # Tidal port Gladstone Dock, Liverpool
            tg1.dataset = tg1.dataset.sel(time=slice(date_start, date_end))
            print(f"{len(tg1.dataset.time)} pts loaded from netcdf")
            if (tg1.dataset.time[-1].values < date_end):
                tg2 = GAUGE()
                tg2.dataset = tg2.read_shoothill_to_xarray(station_id=station_id, date_start=tg1.dataset.time[-1].values, date_end=date_end)
                tg.dataset = xr.concat([ tg1.dataset, tg2.dataset], dim='time')
                print(f"{len(tg2.dataset.time)} pts loaded from API")
            else:
                tg = tg1
        except:
            tg.dataset = tg.read_shoothill_to_xarray(station_id=station_id ,date_start=date_start, date_end=date_end)

        self.tg = tg

    def to_tidegauge(self):
        return self.tg


class IronbridgeAPI:
    """
    if source == "ctr": # use api to load chester weir
    tg = IronbridgeAPI().to_tidegauge()
    """
    def __init__(self):

        tg = GAUGE()
        date_start=np.datetime64('2014-01-01')
        date_end=np.datetime64('now','D')

        station_id = 968
        fn_archive = "iron"

        # Load timeseries from local file if it exists
        try:
            tg1 = GAUGE()
            tg2 = GAUGE()
            tg = GAUGE()

            # Load local file. Created with archive_shoothill.py
            dir = "archive_shoothill/"
            tg1.dataset = xr.open_mfdataset(dir + fn_archive + "_????.nc") # Tidal port Gladstone Dock, Liverpool
            tg1.dataset = tg1.dataset.sel(time=slice(date_start, date_end))
            print(f"{len(tg1.dataset.time)} pts loaded from netcdf")
            if (tg1.dataset.time[-1].values < date_end):
                tg2 = GAUGE()
                tg2.dataset = tg2.read_shoothill_to_xarray(station_id=station_id, date_start=tg1.dataset.time[-1].values, date_end=date_end)
                tg.dataset = xr.concat([ tg1.dataset, tg2.dataset], dim='time')
                print(f"{len(tg2.dataset.time)} pts loaded from API")
            else:
                tg = tg1
        except:
            tg.dataset = tg.read_shoothill_to_xarray(station_id=station_id ,date_start=date_start, date_end=date_end)

        self.tg = tg

    def to_tidegauge(self):
        return self.tg



class GladstoneHarmonicReconstruction:
    """
    if source == 'harmonic_rec': # load full tidal signal using anyTide code
    tg = GladstoneHarmonicReconstruction().to_tidegauge()
    """
    def __init__(self):
        tg = GAUGE()
        #date_start=np.datetime64('now')
        #ndays = 5
        #tg.dataset = tg.anyTide_to_xarray(date_start=date_start, ndays=5)
        date_start=np.datetime64('2005-04-01')
        date_end=np.datetime64('now','D')
        tg.dataset = tg.anyTide_to_xarray(date_start=date_start, date_end=date_end)
        self.tg = tg

    def to_tidegauge(self):
        return self.tg



class Stats():
    """
    Class to handle RMS and best line fits.

    mask = self.bore['Quality'].values=="A"
    """
    def __init__(cls, X, Y, mask):
        cls.X = X
        cls.Y = Y
        cls.mask = mask

    def linearfit(cls, X , Y):
        """
        Linear regression. Calculates linear fit weights and RMSE

        Is used after computing the lag between Gladstone and Saltney events,
            during load_and_process(), to find a fit between Liverpool heights
            and Saltney arrival lag.

        Returns polynomal function for linear fit that can be used in reconstruction:
        E.g.
        X=range(10)
        np.poly1d(weights)( range(10) )

        Also returns RMSE
        """
        idx = np.isfinite(X).values & np.isfinite(Y).values
        weights = np.polyfit( X[idx], Y[idx], 1)
        logging.debug("weights: {weights}")
        #self.linfit = np.poly1d(weights)
        #self.bore['linfit_lag'] =  self.linfit(X)
        #self.bore.attrs['weights'] = np.poly1d(weights)
        #self.bore.attrs['weights'](range(10))
        Y_fit = np.poly1d(weights)(X)
        rmse = '{:4.1f} mins'.format( np.sqrt(np.nanmean((Y.values - Y_fit)**2)) )
        return np.poly1d(weights), rmse

    def linear_fit_classA(cls):
        """ Only fit to values where mask==1 """
        weights,rmse = cls.linearfit( cls.X.where( cls.mask ),
                    cls.Y.where( cls.mask ) )
        #print(f"{cls.source} class A| {cls.args['label']}: {rmse}")
        return weights, rmse

    def linear_fit_all(cls):
        """ Fit to all values """

        weights,rmse = cls.linearfit( cls.X, cls.Y )
        #print(f"{cls.source} all | {cls.args['label']}: {rmse}")

        return weights, rmse



class marine_gauge():
    """
    Process event for a 'deep water' tide. I.e. harmonic

    ref_time [np.datetime64] - time about which to search. Also called guess_time, or time of observation
    HLW [str] HW/LW/FW/EW - switch controlling parameters related to the tidal state of interest
    source [str] bodc/harmonic/api/ctr/harmonic_rec - data source
    winsize [int] +/- hrs to search around ref_time
    """
    def __init__(self, tg:GAUGE=None, ref_time:np.datetime64=None,
                HLW:str="HW", source:str="bodc", winsize:int=6):
        self.winsize = winsize
        self.ref_time = ref_time
        self.HLW = HLW
        self.source   = source
        self.tg = tg
        if ref_time == None:
            print("Expected an input time: ref_time")
            return
        if tg == None:
            print("Expected an input gauge dataset: tg")
            return

    def get_event(cls):
        """ Get HW, LW, FW, EW event. Return single height and time."""

        HW = None
        LW = None
        tg = cls.tg
        obs_time = cls.ref_time
        HLW = cls.HLW
        source = cls.source
        winsize = cls.winsize

        if HLW == 'HW':
            time_var = 'time_highs'
            measure_var = 'sea_level_highs'
        elif HLW == 'LW':
            time_var = 'time_lows'
            measure_var = 'sea_level_lows'
        elif HLW == 'FW':
            time_var = 'time_flood'
            measure_var = 'sea_level_flood'
        elif HLW == 'EW':
            time_var = 'time_ebb'
            measure_var = 'sea_level_ebb'
        else:
            print('This should not have happened...')

        # Extracting the highest and lowest value with a cubic spline is
        # very memory costly. Only need to use the cubic method for the
        # bodc and api sources, so compute the high and low waters in a
        # piecewise approach around observations times.
        if source == "bodc" or source == "api":
            # This produces an xr.dataset with sea_level_highs and sea_level_lows
            # with time variables time_highs and time_lows.
            win = GAUGE()
            win.dataset = tg.dataset.sel( time=slice(obs_time - np.timedelta64(winsize, "h"), obs_time + np.timedelta64(winsize, "h"))  )
            #if HLW == "LW":
            #    print(f"win.dataset {win.dataset}")
            #print(i," win.dataset.time.size", win.dataset.time.size)
            if win.dataset.time.size == 0:
                tg_HLW = GAUGE()
                tg_HLW.dataset = xr.Dataset({measure_var: (time_var, [np.NaN])}, coords={time_var: [obs_time]})
            else:
                if HLW == "FW" or HLW == "EW":
                    tg_HLW = win.find_flood_and_ebb_water(var_str='sea_level',method='cubic')
                    #print(f"inflection point time: {tg_HLW.dataset[time_var]}")
                    print(f"inflection points: {len(tg_HLW.dataset[time_var])}")
                elif HLW == "HW" or HLW == "LW":
                    tg_HLW = win.find_high_and_low_water(var_str='sea_level',method='cubic')
                    print(f"max points: {len(tg_HLW.dataset[time_var])}")
                else:
                    print(f"This should not have happened... HLW:{HLW}")
        else:
            tg_HLW = tg.find_high_and_low_water(var_str='sea_level')

        HW = tg_HLW.get_tide_table_times(
                                time_guess=obs_time,
                                time_var=time_var,
                                measure_var=measure_var,
                                method='nearest_1',
                                winsize=winsize ) #4h for HW, 6h for LW

        #print("time,HW:",obs_time, HW.values)
        #if type(HW) is xr.DataArray: ## Actually I think they are alway xr.DataArray with time, but the height can be nan.
            #print(f"HW: {HW}")
            #HT_h.append( HW.values )
            #print('len(HT_h)', len(HT_h))
            #HT_t.append( HW[time_var].values )
            #print('len(HT_t)', len(HT_t))
            #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
            #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
            #ind.append(i)
            #print(f"i:{i}, {HT_t[-1].astype('M8[ns]').astype('M8[ms]').item()}" )
            #print(HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))

            ### PLOTTING, plot_event(), WENT HERE

        #else:
        #    logging.info(f"Did not find a high water near this guess")
        #    print(f"Did not find a high water near this guess")

        if len(HW.coords) != 1:
            print(f"Only one event was expected. {len(HW.coords)} events found!")

        return HW





################################################################################
class OpenWeather:
    """
    Class to load in an export OpenWeather history file at Hawarden Airport into
    an xarray dataset.
    """
    def __init__(self):
        self.dataset = None

    #%% Load method
    @classmethod
    def read_openweather_to_xarray(cls, fn_openweather, date_start=None, date_end=None):
        """
        For reading from a single OpenWeather csv history file into an
        xarray dataset.
        If no data lies between the specified dates, a dataset is still created
        containing information on the gauge, but the time dimension will
        be empty.

        The data takes the form:

        dt,dt_iso,timezone,city_name,lat,lon,temp,feels_like,temp_min,temp_max,pressure,sea_level,grnd_level,humidity,wind_speed,wind_deg,rain_1h,rain_3h,snow_1h,snow_3h,clouds_all,weather_id,weather_main,weather_description,weather_icon
        1104537600,2005-01-01 00:00:00 +0000 UTC,0,hawarden airport,53.176908,-2.978784,7.63,6.95,7.54,7.74,1024,,,99,1.5,150,,,,,75,803,Clouds,broken clouds,04n
        1104541200,2005-01-01 01:00:00 +0000 UTC,0,hawarden airport,53.176908,-2.978784,4.83,2.61,4.54,7.54,1023,,,99,2.6,170,,,,,28,802,Clouds,scattered clouds,03n
        ...

        Parameters
        ----------
        fn_openweather (str) : path to OpenWeather location file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data

        Returns
        -------
        xarray.Dataset object.
        E.g.
        Coordinates:
          * time        (time) datetime64[ns] 2005-01-01 ... 2021-11-08T23:00:00
        Data variables:
            pressure    (time) int64 1024 1023 1022 1022 1021 ... 1019 1017 1018 1017
            wind_speed  (time) float64 1.5 2.6 4.6 4.1 5.1 ... 3.6 4.12 0.89 4.02 2.68
            wind_deg    (time) int64 150 170 200 220 210 200 ... 180 190 210 117 239 226
            longitude   float64 53.18
            latitude    float64 -2.979
            site_name   object 'hawarden airport'
        """
        try:
            dataset = cls.read_openweather_data(fn_openweather, date_start, date_end)
        except:
            raise Exception("Problem reading OpenWeather file: " + fn_openweather)
        # Attributes
        dataset["longitude"] = float(dataset["lat"][0])
        dataset["latitude"] = float(dataset["lon"][0])
        dataset["site_name"] = str(dataset["city_name"][0])
        dataset = dataset.drop_vars(["lon", "lat", "city_name"])

        return dataset

    @classmethod
    def read_openweather_data(cls, filnam, date_start=None, date_end=None):
        """
        Reads NRW data from a csv file.

        Parameters
        ----------
        filnam (str) : path to OpenWeather file
        date_start (np.datetime64) : start date for returning data.
        date_end (np.datetime64) : end date for returning data.

        Returns
        -------
        xarray.Dataset containing times, wind_speed, wind_deg, pressure, lat, lon, city_name
        """
        import datetime

        # Initialise empty dataset and lists
        dataset = xr.Dataset()
        # Define custom data parser
        custom_date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S +0000 UTC")
        data = pd.read_csv(filnam, delimiter=',', parse_dates=['dt_iso'], date_parser=custom_date_parser)
        data.rename(columns={'dt_iso':'time'}, inplace=True)
        data.set_index('time', inplace=True)
        data.drop(columns=['dt', 'timezone', 'temp',
           'feels_like', 'temp_min', 'temp_max', 'sea_level',
           'grnd_level', 'humidity', 'rain_1h',
           'rain_3h', 'snow_1h', 'snow_3h', 'clouds_all', 'weather_id',
           'weather_main', 'weather_description', 'weather_icon'], inplace=True)
        dataset = data.to_xarray()
        if  date_start != None:
                dataset = dataset.where(dataset.time >= date_start)
        if date_end != None:
                dataset = dataset.where(dataset.time <= date_end)
        # Assign local dataset to object-scope dataset
        return dataset


#%% ############################################################################
class Controller():
    """
    This is where the main things happen.
    Where user input is managed and methods are launched
    """
    ############################################################################
    #%% Initialising and Finishing methods
    ############################################################################
    def __init__(self):
        """
        Look for pickle file. If exists load it.
        Initialise main controller.
        """
        self.load_databucket()
        logging.info("run interface")
        self.load_bore_flag = False
        self.run_interface()


    def load_databucket(self):
        """
        Auto load databucket from pickle file if it exists.
        """
        #global DATABUCKET_FILE
        #databucket = DataBucket()
        logging.info("Auto load databucket from pickle file if it exists")
        print("Add to pickle file, if it exists")
        try:
            if os.path.exists(DATABUCKET_FILE):
                template = "...Loading (%s)"
                print(template%DATABUCKET_FILE)
                with open(DATABUCKET_FILE, 'rb') as file_object:
                    self.bore = pickle.load(file_object)
                    self.load_bore_flag = True

            else:
                print("... %s does not exist"%DATABUCKET_FILE)
        except KeyError:
            print('ErrorA ')
        except (IOError, RuntimeError):
            print('ErrorB ')


    def pickle_bore(self):
        """ save copy of self.bore into pickle file, if requested """
        print('Pickle data.')
        os.system('rm -f '+DATABUCKET_FILE)
        if(1):
            with open(DATABUCKET_FILE, 'wb') as file_object:
                pickle.dump(self.bore, file_object)
        else:
            print("Don't save as pickle file")
        return


    def export_to_csv(self):
        """
        Export the bore xr.Dataset to a CSV file for sharing
        """
        print('Export data to csv. NOT IMPLEMENTED')
        pass


    def run_interface(self):
        """
        Application's main loop
        Get user input and respond
        """

        print(INSTRUCTIONS)
        while True:
            command = input("What do you want to do? ")

            if command == "q":
                print("run_interface: quit")
                logging.info("quit") # Function call.
                ans = input('Save as pickle file?[Y/n]')
                if ans == "n":
                    break
                else:
                    self.pickle_bore()
                    break

            elif command == "i":
                print(INSTRUCTIONS)

            elif command == "all":
                print('load and process all data')
                self.load_csv()
                print('load and process measured (bodc) data')
                self.load_and_process(source="bodc", HLW_list=["FW", "HW", "LW"])
                #self.load_and_process(source="bodc", HLW="LW")
                #self.load_and_process(source="bodc", HLW="FW")
                print('load and process measured (API) data')
                self.load_and_process(source="api", HLW_list=["HW", "LW", "FW"])
                #self.load_and_process(source="api", HLW="LW")
                #self.load_and_process(source="api", HLW="FW")
                print('load and process CTR data. Obs + API')
                self.get_river_data(HLW_list=["LW"])
                print('load and process harmonic data')
                self.load_and_process(source="harmonic", HLW_list=["HW", "LW"])
                #self.load_and_process(source="harmonic", HLW="LW")
                print('load and process harmonic reconstructed data')
                self.load_and_process(source="harmonic_rec", HLW_list=["HW", "LW"])
                #self.load_and_process(source="harmonic_rec", HLW="LW")
                print('Explore combinations of HLW times and heights for best fit')
                self.fits_to_data(qc_flag=True)
                self.fits_to_data(qc_flag=False)

            elif command == "0":
                print('load bore observations')
                self.load_csv()

            elif command == "h":
                print('load and process harmonic data')
                if not self.load_bore_flag: self.load_csv()
                self.load_and_process(source="harmonic")

            elif command == "hrec":
                print('load and process harmonic reconstructed data')
                if not self.load_bore_flag: self.load_csv()
                self.load_and_process(source="harmonic_rec")

            elif command == "b":
                print('load and process measured (bodc) data')
                if not self.load_bore_flag: self.load_csv()
                self.load_and_process(source="bodc")

            elif command == "a":
                print('load and process measured (API) data')
                if not self.load_bore_flag: self.load_csv()
                self.load_and_process(source="api")

            elif command == "r":
                print('load and process measured (API) river data')
                if not self.load_bore_flag: self.load_csv()
                self.get_river_data()

            elif command == "m":
                print("load and process met data")
                if not self.load_bore_flag: self.load_csv()
                self.get_met_data()

            elif command == "2":
                print('show bore dataset')
                self.show()

            elif command == "3":
                print('plot bore data (lag vs tidal height')
                plt.close('all');self.plot_lag_vs_height('bodc')
                plt.close('all');self.plot_lag_vs_height('bodc', HLW="FW")
                plt.close('all');self.plot_lag_vs_height('all')
                plt.close('all');self.plot_lag_vs_height('harmonic')
                plt.close('all');self.plot_lag_vs_height('harmonic_rec')
                plt.close('all');self.plot_lag_vs_height('api')
                plt.close('all');self.plot_lag_vs_height('api', HLW="FW")

            elif command == "4":
                print('plot difference between predicted and measured (lag vs tidal height)')
                plt.close('all');self.plot_surge_effect('api')
                plt.close('all');self.plot_surge_effect('bodc')

            elif command == "5":
                print('Explore combinations of HLW times and heights for best fit')
                self.fits_to_data(qc_flag=True)
                self.fits_to_data(qc_flag=False)

            elif command == "d1":
                print('load and plot HLW data')
                self.load_and_plot_hlw_data()

            elif command == "d2":
                print("shoothill dev")
                self.shoothill()

            elif command == "d4":
                print('Plot combinations of HLW times, heights and rivers')
                self.combinations_lag_hlw_river()

            elif command == "d5":
                print('Explore how rivers affect bore timing')
                self.river_lag_timing()

            elif command == "6":
                self.predict_bore()

            elif command == "x":
                print('Export data')
                self.export_to_csv()

            elif command == "rm":
                print('Remove pickle file)')
                if os.path.exists(DATABUCKET_FILE):
                    os.remove(DATABUCKET_FILE)
                else:
                    print("Can not delete the pickle file as it doesn't exists")
                #self.load_databucket()

            else:
                template = "run_interface: I don't recognise (%s)"
                print(template%command)

    ############################################################################
    #%% Load and process methods
    ############################################################################

    def load_and_process(self, source:str="harmonic", HLW_list=["HW"]):
        """
        Performs sequential steps to build into the bore object.
        1. Load Gladstone Dock data (though this might also be loaded from the obs logs)
        2. Calculate the time lag between Gladstone and Saltney events.
        3. Perform a linear fit to the time lag.

        Inputs:
        source: 'harmonic' [default] - load HLW from harmonic prediction
                'harmonic_rec' - reconstruct time series from harmonic constants
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        print('loading '+source+' tide data')
        self.load_gauge_timeseries(source=source, HLW_list=HLW_list)
        self.process_gauge_timeseries(source=source, HLW_list=HLW_list)
        #self.compare_Glad_HLW()
        print('Calculating the Gladstone to Saltney time difference')
        self.calc_Glad_Saltney_time_lag(source=source, HLW_list=HLW_list)
        print('Process linear fit. Calc and save')
        self.process_fit(source=source, HLW_list=HLW_list)


    def process_fit(self, source:str="harmonic", HLW_list=["HW"], X:xr.DataArray=None, Y:xr.DataArray=None):
        for HLW in HLW_list:
            # Get linear fit with rmse
            if (X is None) or (Y is None):
                Y = self.bore['liv_height_'+HLW+'_'+source]
                X = self.bore['Saltney_lag_'+HLW+'_'+source]
                stats = Stats(Y, X, self.bore['Quality'].values=="A")
                #stats = Stats(self.bore['liv_height_'+HLW+'_'+source],
                #            self.bore['Saltney_lag_'+HLW+'_'+source],
                #            self.bore['Quality'].values=="A")
            else:
                stats = Stats(Y, X,
                            self.bore['Quality'].values=="A")
            self.bore.attrs['weights_A_'+HLW+'_'+source], self.bore.attrs['rmse_A_'+HLW+'_'+source] = stats.linear_fit_classA()
            self.bore.attrs['weights_'+HLW+'_'+source], self.bore.attrs['rmse_'+HLW+'_'+source] = stats.linear_fit_all()
            # Apply linear model
            self.bore['linfit_A_lag_'+HLW+'_'+source] = self.bore.attrs['weights_A_'+HLW+'_'+source](Y)
            self.bore['linfit_lag_'+HLW+'_'+source] = self.bore.attrs['weights_'+HLW+'_'+source](Y)
            #self.bore['rmse_'+HLW+'_'+source] = '{:4.1f} mins'.format(self.stats(source=source, HLW=HLW))

    def load_csv(self):
        """
        Load observed bore data from text file.
        Load as a dataframe and save to bore:xr.DataSet
        """
        logging.info('Load bore data from csv file')
        self.load_bore_flag = True
        df =  pd.read_csv('data/master-Table 1.csv')
        df.drop(columns=['date + logged time','Unnamed: 15', \
                                'Unnamed: 16','Unnamed: 17'], \
                                 inplace=True)
        df.rename(columns={"date + logged time (GMT)":"time"}, inplace=True)
        df.rename(columns={"wind_deg (from)":"wind_deg"}, inplace=True)
        df.rename(columns={"wind_speed (m/s)":"wind_speed"}, inplace=True)
        df.rename(columns={"pressure (hPa)":"pressure"}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], format="%d/%m/%Y %H:%M")
        #df['time'] = pd.to_datetime(df['time'], utc=True, format="%d/%m/%Y %H:%M")
        #df.set_index(['time'], inplace=True)


        for index, row in df.iterrows():
            df.loc[index,'time'] = np.datetime64( df.at[index,'time'] ) # numpy.datetime64 in UTC
        bore = xr.Dataset()
        bore = df.to_xarray()

        # Set the t_dim to be a dimension and 'time' to be a coordinate
        bore = bore.rename_dims( {'index':'t_dim'} ).assign_coords( time=("t_dim", bore.time.data))
        bore = bore.swap_dims( {'t_dim':'time'} )
        self.bore = bore
        logging.info('Bore data loaded')

    def get_river_data(self, HLW_list=["LW"]):
        """
        Get Chester weir data. Consolidate CTR data.
        Data from the table takes precident. Gaps are filled by the API.
        """

        if HLW_list != ["LW"]:
            print('Not expecting that possibility here')
        else:
            # Obtain CTR data for LW for the observations times.
            self.load_gauge_timeseries(source='ctr',HLW_list=["LW"])
            self.process_gauge_timeseries(source='ctr',HLW_list=["LW"])
            alph = self.bore['Chester Weir height: CHESTER WEIR 15 MIN SG'] *np.NaN
            beta = self.bore['ctr_height_LW_ctr']
            #print( self.bore['ctr_height_LW_ctr'][0:10] )
            self.bore['ctr_height_LW'] = alph
            self.bore['ctr_height_LW'].values = [alph[i].values if np.isfinite(alph[i].values) else beta[i].values for i in range(len(alph))]
            # 2015-06-20T12:16:00 has a -ve value. Only keep +ve values
            self.bore['ctr_height_LW'] = self.bore['ctr_height_LW'].where( self.bore['ctr_height_LW'].values>0)
            #plt.plot( ctr_h_csv, 'b+' )
            #plt.plot( self.bore['ctr_height_LW_ctr'], 'ro')
            #plt.plot( self.bore['ctr_height_LW'], 'g.')
            del self.bore['ctr_height_LW_ctr'], self.bore['ctr_time_LW_ctr']

    def get_met_data(self): #, HLW:str="HW"):
        """
        Get the met data time matching the observation.
        Met data from OpenWeather history download.

        This can then be exported into the obs table:
        c.met.to_pandas().to_csv('met.csv')
        """
        fn_openweather = "data/met/openweather_2005-01-01_2021-11-08.csv"
        met = OpenWeather()
        met.dataset = met.read_openweather_to_xarray(fn_openweather)

        winsize = 6 #4h for HW, 6h for LW. +/- search distance for nearest extreme value
        self.met = xr.Dataset()

        for measure_var in ['wind_speed', 'wind_deg', 'pressure']:

            met_var = []
            met_time = []
            for i in range(len(self.bore.time)):
                try:
                    met_ds = None
                    obs_time = self.bore.time[i].values


                    # Find nearest met observation
                    dt = np.abs(met.dataset['time'] - obs_time)
                    index = np.argsort(dt).values
                    if winsize is not None:  # if search window trucation exists
                        if np.timedelta64(dt[index[0]].values, "m").astype("int") <= 60 * winsize:  # compare in minutes
                            #print(f"dt:{np.timedelta64(dt[index[0]].values, 'm').astype('int')}")
                            #print(f"winsize:{winsize}")
                            met_ds = met.dataset[measure_var][index[0]]
                        else:
                            # return a NaN in an xr.Dataset
                            # The rather odd trailing zero is to remove the array layer
                            # on both time and measurement, and to match the other
                            # alternative for a return
                            met_ds = xr.DataArray( [np.NaN], coords={'time': [obs_time]})
                            #met_ds = xr.Dataset({measure_var: ('time', [np.NaN])}, coords={'time': [obs_time]})

                    else:  # give the closest without window search truncation
                        met_ds = met.dataset[measure_var][index[0]]



                    #print("time,HW:",obs_time, HW.values)
                    if type(met_ds) is xr.DataArray:
                        #print(f"met: {met_ds.values}")
                        met_var.append( float(met_ds.values) )
                        #print('len(met_var)', len(met_var))
                        met_time.append( met_ds.time.values )
                        #print('len(met_time)', len(met_time))
                        #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
                        #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
                        #ind.append(i)
                        #print(f"i:{i}, {met_time[-1].astype('M8[ns]').astype('M8[ms]').item()}" )
                        #print(met_time[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))

                        ## Make timeseries plot around the highwater maxima to check
                        # values are being extracted as expected.
                        if (i % 12) == 0:
                            fig = plt.figure()

                        if measure_var == "wind_speed":
                            ymax = 15
                        if measure_var == "wind_deg":
                            ymax = 360
                        plt.subplot(3,4,(i%12)+1)
                        plt.plot(met.dataset.time, met.dataset[measure_var])
                        plt.plot( met_time[-1], met_var[-1], 'r+' )
                        plt.plot( [self.bore.time[i].values,self.bore.time[i].values],[0,ymax],'k')
                        plt.xlim([met_time[-1] - np.timedelta64(5,'h'),
                                  met_time[-1] + np.timedelta64(5,'h')])
                        #plt.ylim([0,11])
                        plt.text( met_time[-1]-np.timedelta64(5,'h'),ymax*0.9, self.bore.location[i].values)
                        plt.text( met_time[-1]-np.timedelta64(5,'h'),ymax*0.1,  met_time[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))
                        # Turn off tick labels
                        plt.gca().axes.get_xaxis().set_visible(False)
                        #plt.xaxis_date()
                        #plt.autoscale_view()
                        if (i%12) == 12-1:
                            plt.savefig('figs/check_get_'+measure_var+'_times_'+str(i//12).zfill(2)+'.png')
                            plt.close('all')


                    else:
                        logging.info(f"Did not find a met time near this guess {obs_time}")
                        print(f"Did not find a met time near this guess {obs_time}")


                except:
                    logging.warning('Issue with appending met data')
                    print('Issue with appending met data')

            try: # Try and print the last observation timeseries
                plt.savefig('figs/check_get_'+measure_var+'_times_'+str(i//12).zfill(2)+'.png')
                plt.close('all')
            except:
                logging.info("Did not have any extra panels to plot")
                print("Did not have any extra panels to plot")


            # Save a xarray objects
            coords = {'time': (('time'), self.bore.time.values)}
            #print("number of obs:",len(self.bore.time))
            #print("length of time", len(self.bore.time.values))
            #print("length of data:", len(np.array(met_var)) )
            self.met[measure_var] = xr.DataArray( np.array(met_var), coords=coords, dims=['time'])




    def check_event_plot(self, HW:xr.Dataset=None, windowsize:int=5,
            source:str="bodc", HLW:str="HW"):
        """
        Plot of event as a time series with key bore points identified
        Uses:
            self.counter / i [int] - event counter
            self.fig, self.axes - carries the plot information
            self.tg.dataset
            self.bore.time
            self.bore.location

        Inputs:
            HW [xr.dataset] - Pass h,t as xr w/o specifying variables names
            #height [float] - processed tidal event height
            #time [np.datetime64] - processed tidal event time

            windowsize [int] - xlim:+/- windowsize
            source [str] -  datasource [ctr/bodc/api/harm/harm_rec]
            HLW [str]  - parameter describing tidal state: "FW", "EW"

        Output:
            creates file:
            figs/check_get_tidetabletimes_'+str(i//12).zfill(2)+'_'+HLW+'_'+source+'.png'
        """

        ins_winsize = 40 # window size (mins) for inset window
        i = self.counter
        # Extract height and time from xr.Dataset
        try:
            time = HW[np.array(HW.coords)[0]].values
            height = HW.values
        except:
            print(f"Problem extracting tide event (h,t) from xr.Dataset: {HW}")

        ## Make timeseries plot around the highwater maxima to check
        # values are being extracted as expected.
        if (i % 12) == 0:
            self.fig, self.axes = plt.subplots(3,4)
        ax = self.axes.flat[i%12]
        #plt.subplot(3,4,(i%12)+1)
        ax.plot(self.tg.dataset.time, self.tg.dataset.sea_level)
        ax.plot( time, height, 'r+' )
        ax.plot( [self.bore.time[i].values,self.bore.time[i].values],[0,11],'k')
        ax.set_xlim([time - np.timedelta64(windowsize,'h'),
                  time + np.timedelta64(windowsize,'h')])
        ax.set_ylim([0,11])
        ax.text( time-np.timedelta64(windowsize,'h'),10, self.bore.location[i].values)
        ax.text( time-np.timedelta64(windowsize,'h'),1,  time.astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))
        # Turn off tick labels
        ax.axes.get_xaxis().set_visible(False)

        ## Add inset zoom at extrema
        if source == "ctr":
            ins = inset_axes(ax,width="30%", height="30%", loc="upper right")
        else:
            if HLW == "EW":
                ins = inset_axes(ax,width="30%", height="30%", loc="center right")
            elif HLW == "FW":
                ins = inset_axes(ax,width="30%", height="30%", loc="center left")
            else:
                ins = inset_axes(ax,width="30%", height="30%", loc="center")


        ins_dataset = self.tg.dataset.sel( time=slice(time - np.timedelta64(ins_winsize,'m'), time + np.timedelta64(ins_winsize,'m'))  )
        ins.plot(ins_dataset.time, ins_dataset.sea_level,'b+')
        ins.plot( time, height, 'r+' )
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_xticklabels([])
        ins.set_yticklabels([])
        ins.patch.set_alpha(0.5)

        if (i%12) == 12-1:
            plt.savefig('figs/check_get_tidetabletimes_'+str(i//12).zfill(2)+'_'+HLW+'_'+source+'.png')
            plt.close('all')





    def load_gauge_timeseries(self, source:str='harmonic', HLW_list=["HW"]):
        #def get_Glad_data(self, source:str='harmonic', HLW:str="HW"):
        """
        Get Gladstone HLW data from external source
        These data are reported in the bore.csv file but not consistently and it
        is laborous to find old values.
        It was considered a good idea to automate this step.

        inputs:
        source: 'harmonic' [default] - load HLW from harmonic prediction
                'harmonic_rec' - reconstruct time series from harmonic constants
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        """

        if source == "harmonic":
            tg = GladstoneTideTable().to_tidegauge()



        elif source == "bodc": # Load gauge data from BODC files
            tg = BODC().to_tidegauge()


        elif source == "api": # load full tidal signal from shoothill
            tg = GladstoneAPI().to_tidegauge()


        elif source == "ctr": # use api to load river data.
            tg = IronbridgeAPI().to_tidegauge()
            #tg = ChesterAPI().to_tidegauge()


        elif source == 'harmonic_rec': # load full tidal signal using anyTide code
            tg = GladstoneHarmonicReconstruction().to_tidegauge()


        else:
            logging.debug("Did not expect this eventuality...")

        self.tg = tg


    def process_gauge_timeseries(self, source:str='harmonic', HLW_list=["HW"]):
        #def get_Glad_data(self, source:str='harmonic', HLW:str="HW"):
        """
        Extract the HW, LW, etc tide events, for each observation, from the gauge timeseries.
        Save to bore object

        inputs:
        source: 'harmonic' [default] - load HLW from harmonic prediction
                'harmonic_rec' - reconstruct time series from harmonic constants
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        HLW_list: ["LW","HW","FW","EW"] - the data is either processed for High or Low water
                events, or Flood or Ebb (inflection) events
        """

        if source == "ctr": # used in event variables names
            loc = "ctr"
        else:
            loc = "liv"

        ## Process the *_highs or *_lows
        for HLW in HLW_list:
            print(f"HLW: {HLW}")
            #time_var = 'time_highs'
            #measure_var = 'sea_level_highs'
            #ind = [] # list of indices in the obs bore data where gladstone data is found
            if HLW == 'HW':
                time_var = 'time_highs'
                measure_var = 'sea_level_highs'
            elif HLW == 'LW':
                time_var = 'time_lows'
                measure_var = 'sea_level_lows'
            elif HLW == 'FW':
                time_var = 'time_flood'
                measure_var = 'sea_level_flood'
            elif HLW == 'EW':
                time_var = 'time_ebb'
                measure_var = 'sea_level_ebb'
            else:
                print('This should not have happened...')

            HT_h = [] # Extrema - height
            HT_t = [] # Extrema - time

            winsize = 6 #4h for HW, 6h for LW. +/- search distance for nearest extreme value

            ## Process events individually
            for i in range(len(self.bore.time)):
                mg = marine_gauge(tg=self.tg, ref_time=self.bore.time[i].values,
                    HLW=HLW, source=source, winsize=winsize)
                HW = mg.get_event()
                HT_h.append( HW.values )
                #print('len(HT_h)', len(HT_h))
                HT_t.append( HW[time_var].values )
                self.counter = i
                #self.check_event_plot(height=HW.values, time=HW[time_var].values,
                #                windowsize=5, source=source, HLW=HLW )
                self.check_event_plot(HW=HW,
                                windowsize=5, source=source, HLW=HLW )



            # Save a xarray objects
            coords = {'time': (('time'), self.bore.time.values)}
            #print("number of obs:",len(self.bore.time))
            #print("length of time", len(self.bore.time.values))
            #print("length of data:", len(np.array(HT_h)) )
            self.bore[loc+'_height_'+HLW+'_'+source] = xr.DataArray( np.array(HT_h), coords=coords, dims=['time'])
            self.bore[loc+'_time_'+HLW+'_'+source] = xr.DataArray( np.array(HT_t), coords=coords, dims=['time'])

            ## Some logging
            logging.debug(f"len(self.bore[loc+'_time_'{HLW}'_'{source}]): {len(self.bore[loc+'_time_'+HLW+'_'+source])}")
            #logging.info(f'len(self.bore.liv_time)', len(self.bore.liv_time))
            logging.debug(f"type(HT_t): {type(HT_t)}")
            logging.debug(f"type(HT_h): {type(HT_h)}")

            #if loc=='liv':
            #    logging.debug('log time, orig tide table, new tide table lookup')
            #    for i in range(len(self.bore.time)):
            #        logging.debug( f"{self.bore.time[i].values}, {self.bore['Liv (Gladstone Dock) HT time (GMT)'][i].values}, {self.bore['liv_time_'+HLW+'_'+source][i].values}")





    def calc_Glad_Saltney_time_lag(self, source:str="harmonic", HLW_list=["HW"]):
        """
        Compute lag (obs - tide) for arrival at Saltney relative to Glastone HT
        Store lags as integer (minutes) since np.datetime64 and
        np.timedelta64 objects are problematic with polyfitting.

        inputs:
        source: 'harmonic' [default] - load HLW from harmonic prediction
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        for HLW in HLW_list:
            logging.info('calc_Glad_Saltney_time_diff')
            nt = len(self.bore.time)
            lag = (self.bore['time'].values - self.bore['liv_time_'+HLW+'_'+source].values).astype('timedelta64[m]')
            # convert to integers so nans can be applied
            lag = [ lag[i].astype('int') if np.isfinite(self.bore['liv_height_'+HLW+'_'+source].values)[i]  else np.NaN for i in range(nt) ]
            # Pick out FB and Blue bridge
            Saltney_lag    = [ lag[i] if self.bore.location.values[i] == 'bridge' else np.NaN for i in range(nt) ]
            bluebridge_lag = [ lag[i] if self.bore.location.values[i] == 'blue bridge' else np.NaN for i in range(nt) ]
            #Saltney_lag    = [ lag[i].astype('int') if self.bore.location.values[i] == 'bridge' else np.NaN for i in range(nt) ]
            #bluebridge_lag = [ lag[i].astype('int') if self.bore.location.values[i] == 'blue bridge' else np.NaN for i in range(nt) ]

            # Save a xarray objects
            coords = {'time': (('time'), self.bore.time.values)}
            self.bore['lag_'+HLW+'_'+source] = xr.DataArray( lag, coords=coords, dims=['time'])
            self.bore['Saltney_lag_'+HLW+'_'+source] = xr.DataArray( Saltney_lag, coords=coords, dims=['time'])
            self.bore['bluebridge_lag_'+HLW+'_'+source] = xr.DataArray( bluebridge_lag, coords=coords, dims=['time'])


    def linearfit_X(self, X, Y):
        """
        Linear regression. Calculates linear fit weights and RMSE

        Is used after computing the lag between Gladstone and Saltney events,
            during load_and_process(), to find a fit between Liverpool heights
            and Saltney arrival lag.

        Returns polynomal function for linear fit that can be used:
        E.g.
        X=range(10)
        np.poly1d(weights)( range(10) )

        Also returns RMSE
        """
        idx = np.isfinite(X).values & np.isfinite(Y).values
        weights = np.polyfit( X[idx], Y[idx], 1)
        logging.debug("weights: {weights}")
        #self.linfit = np.poly1d(weights)
        #self.bore['linfit_lag'] =  self.linfit(X)
        #self.bore.attrs['weights'] = np.poly1d(weights)
        #self.bore.attrs['weights'](range(10))
        Y_fit = np.poly1d(weights)(X)
        rmse = '{:4.1f} mins'.format( np.sqrt(np.nanmean((Y.values - Y_fit)**2)) )
        return np.poly1d(weights), rmse


    ############################################################################
    #%% Presenting data
    ############################################################################

    def show(self):
        """ Show xarray dataset """
        print( self.bore )


    def plot_lag_vs_height(self, source:str="harmonic", HLW:str="HW"):
        """
        Plot bore lag (obs time - Gladstone tide time) against
        Gladstone extreme water water (m).
        Separate colours for Saltney, Bluebridge, Chester.

        inputs:
        source: 'harmonic' [default] - load HLW from harmonic prediction
                'harmonic_rec' - data from harmonic reconstruction
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
                'all' - Use bodc + api data
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        I = self.bore['Quality'] == "A"
        if source == "all":
            Yliv = self.bore['liv_height_'+HLW+'_bodc']
            Xsalt = self.bore['Saltney_lag_'+HLW+'_bodc']
            Xblue = self.bore['bluebridge_lag_'+HLW+'_bodc']
            Yliv_api = self.bore['liv_height_'+HLW+'_api'].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xsalt_api = self.bore['Saltney_lag_'+HLW+'_api'].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xblue_api = self.bore['bluebridge_lag_'+HLW+'_api'].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xfit = self.bore['linfit_lag_'+HLW+'_bodc']
            Xsalt_api_latest = Xsalt_api.where( xr.ufuncs.isfinite(Xsalt_api), drop=True)[0] # NB obs are in reverse time order
            Yliv_api_latest  = Yliv_api.where( xr.ufuncs.isfinite(Xsalt_api), drop=True)[0] # NB obs are in reverse time order
            lab = self.bore.time.where( xr.ufuncs.isfinite(Xsalt_api), drop=True)[0].values.astype('datetime64[D]').astype(object).strftime('%d%b%y')

            plt.plot( Xsalt,Yliv, 'r.', label='Saltney')
            plt.plot( Xsalt[I],Yliv[I], 'k+', label='Class A')
            plt.plot( Xblue,Yliv, 'b.', label='Bluebridge')
            plt.plot( Xfit,Yliv, 'k-', label='bodc: rmse '+'{:4.1f}'.format(self.stats('bodc'))+'mins')
            plt.plot( Xsalt_api,Yliv_api, 'ro', label='Saltney API')
            plt.plot( Xblue_api,Yliv_api, 'bo', label='Bluebridge API')
            plt.plot( Xsalt_api_latest,Yliv_api_latest, 'go', label='Saltney latest: '+lab)
            plt.plot( Xsalt_api[I],Yliv_api[I], 'k+')



        else:
            Yliv = self.bore['liv_height_'+HLW+'_'+source]
            Xsalt = self.bore['Saltney_lag_'+HLW+'_'+source]
            Xblue = self.bore['bluebridge_lag_'+HLW+'_'+source]
            Xfit = self.bore['linfit_lag_'+HLW+'_'+source]
            plt.plot( Xsalt,Yliv, 'r.', label='Saltney')
            plt.plot( Xsalt[I],Yliv[I], 'k+', label='Class A')
            plt.plot( Xblue,Yliv, 'b.', label='Bluebridge')
            plt.plot( Xfit,Yliv, 'k-', label=source+': rmse '+'{:4.1f}'.format(self.stats(source,HLW))+'mins')
            Xsalt_latest = Xsalt.where( xr.ufuncs.isfinite(Xsalt), drop=True)[0]
            Yliv_latest  = Yliv.where( xr.ufuncs.isfinite(Xsalt), drop=True)[0]
            lab = self.bore.time.where( xr.ufuncs.isfinite(Xsalt), drop=True)[0].values.astype('datetime64[D]').astype(object).strftime('%d%b%y')

            # Highlight recent data
            Yliv = self.bore['liv_height_'+HLW+'_'+source].where( self.bore.time > np.datetime64('2021-01-01') )
            Xsalt = self.bore['Saltney_lag_'+HLW+'_'+source].where( self.bore.time > np.datetime64('2021-01-01') )
            Xblue = self.bore['bluebridge_lag_'+HLW+'_'+source].where( self.bore.time > np.datetime64('2021-01-01') )
            #Yliv = self.bore['liv_height_'+HLW+'_'+source].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            #Xsalt = self.bore['Saltney_lag_'+HLW+'_'+source].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            #Xblue = self.bore['bluebridge_lag_'+HLW+'_'+source].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            plt.plot( Xsalt,Yliv, 'ro', label='Saltney 2021+')
            plt.plot( Xblue,Yliv, 'bo', label='Bluebridge 2021+')
            plt.plot( Xsalt_latest,Yliv_latest, 'go', label='Saltney latest: '+lab)
            plt.plot( Xsalt[I],Yliv[I], 'k+')
            #plt.plot( Xblue[0],Yliv[0], 'b+', label='Bluebridge recent')



        plt.ylabel('Liv (Gladstone Dock) '+HLW+' (m)')
        plt.xlabel('Arrival time (mins) relative to Liv '+HLW)
        if source =='harmonic': str='tide table predicted'
        if source =='harmonic_rec': str='harmonic reconstructed'
        if source =='all': str='all measured'
        if source =='bodc': str='measured only QCd'
        if source == 'api': str='measured w/o QC'
        plt.title(f"Bore arrival time at Saltney Ferry ({str} data)")
        if HLW == "HW":
            plt.xlim([-125, -45])   # minutes
            plt.ylim([8.2, 10.9]) # metres
        plt.legend()
        #plt.show()
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight_'+HLW+'_'+source+'.png')


    def plot_surge_effect(self, source:str='bodc', HLW:str="HW"):
        """
        Compare harmonic predicted HLW+lag with measured HLW+lag
        Plot quiver between harmonic and measured values.

        NB should probably have linfit predicted lag instead of
        Saltney_lag_*_harmonic for the predicted value.

        inputs:
        source:
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        # Example plot
        from matplotlib.collections import LineCollection
        #from matplotlib import colors as mcolors
        #import matplotlib.dates as mdates
        if source=='api':
            last_bodc_time = self.bore['liv_time_'+HLW+'_bodc']\
                .where(np.isfinite(self.bore['liv_height_'+HLW+'_bodc'].values))\
                .dropna('time')\
                .max().values
            I = self.bore['liv_time_'+HLW+'_api'] > last_bodc_time + np.timedelta64(1,'D') #np.datetime64('2020-09-01')
            nval = sum(I).values
        else:
            nval = min( len(self.bore['linfit_lag_'+HLW+'_harmonic']), len(self.bore['linfit_lag_'+HLW+'_bodc']) )
            I = np.arange(nval)
        segs_h = np.zeros((nval,2,2)) # line, pointA/B, t/z
        #convert dates to numbers first


        segs_h[:,0,1] = self.bore['liv_height_'+HLW+'_'+source][I]
        segs_h[:,1,1] = self.bore['liv_height_'+HLW+'_harmonic'][I]
        segs_h[:,0,0] = self.bore['Saltney_lag_'+HLW+'_'+source][I]
        segs_h[:,1,0] = self.bore['Saltney_lag_'+HLW+'_harmonic'][I]

        if source=='api':
            print('liv_height_'+HLW+'_'+source, segs_h[:,0,1])
            print('liv_height_'+HLW+'_harmonic', segs_h[:,1,1])
            print('Saltney_lag_'+HLW+'_'+source, segs_h[:,0,0])
            print('Saltney_lag_'+HLW+'_harmonic', segs_h[:,1,0])

        II = self.bore['Quality'][I] == "A"
        #segs_h[:,0,0] = self.bore.liv_height_bodc[:nval]
        #segs_h[:,1,0] = self.bore.liv_height_harmonic[:nval]
        #segs_h[:,0,1] = self.bore.Saltney_lag_bodc[:nval]
        #segs_h[:,1,1] = self.bore.Saltney_lag_harmonic[:nval]

        fig, ax = plt.subplots()
        ax.set_ylim(np.nanmin(segs_h[:,:,1]), np.nanmax(segs_h[:,:,1]))
        line_segments_HW = LineCollection(segs_h, cmap='plasma', linewidth=1)
        ax.add_collection(line_segments_HW)
        ax.scatter(segs_h[:,1,0],segs_h[:,1,1], c='red', s=4, label='predicted') # harmonic predictions
        ax.scatter(segs_h[:,0,0],segs_h[:,0,1], c='green', s=4, label='measured') # harmonic predictions
        ax.scatter(segs_h[II,0,0],segs_h[II,0,1], c='green', s=16) # 1st hand
        ax.set_title('Harmonic prediction with quiver to measured high waters')

        plt.ylabel('Liv (Gladstone Dock) '+HLW+' (m)')
        plt.xlabel('Arrival time (mins relative to LiV '+HLW+')')
        plt.title('Bore arrival time at Saltney Ferry. Harmonic prediction cf measured')
        plt.legend()
        #plt.xlim([-125, -40])   # minutes
        #plt.ylim([8.2, 10.9]) # metres
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight_shift_'+HLW+'_'+source+'.png')
        plt.close('all')


    def plot_scatter(self, source:str='bodc', HLW:str="HW", colour_var_str:str="river"):
        """
        scatter plot the h vs lag observations. Colour by 3rd variable.
        Add linear regression lines to both class A and all data points.
        """

        if colour_var_str  == "river": # LW on river
            col_var =  self.bore["ctr_height_LW"]
            clabel = "River height (m)"
            vmin=4.4
            vmax=5.5 # 4.6
            cmap = "jet"
        elif colour_var_str == "date":
            col_var = self.bore["time"] #     c=self.bore.time, #self.bore['ctr_height_LW'],
            clabel = "date"
            vmin = None
            vmax = None
            cmap = "jet"
        elif colour_var_str == "wind_along":
            col_var = self.bore.wind_speed * np.cos((315 - self.bore.wind_deg)*np.pi/180.)
            clabel = "along estuary wind (m/s), from Hawarden/Connahs Quay"
            vmin = -7
            vmax = 7
            cmap = "Spectral"
        elif colour_var_str == "wind_across":
            col_var = self.bore.wind_speed * np.sin((315 - self.bore.wind_deg)*np.pi/180.)
            clabel = "across estuary wind (m/s), from Hawarden/Connahs Quay"
            vmin = -7
            vmax = 7
            cmap = "Spectral"
        else:
            print(f"Was not expecting that value of colour_var_str: {colour_var_str}")

        plt.close('all')
        fig = plt.figure(figsize=(8, 6), dpi=120)
        if HLW=="dLW":
            X = self.bore['Saltney_lag_LW_'+source]
            Y = self.bore['liv_height_HW_'+source] - self.bore['liv_height_LW_'+source]
        elif HLW=="dHW":
            X = self.bore['Saltney_lag_HW_'+source]
            Y = self.bore['liv_height_HW_'+source] - self.bore['liv_height_LW_'+source]
        elif HLW=="XX":
            X = self.bore['Saltney_lag_HW_'+source]
            Y = self.bore['liv_height_LW_'+source]
        else:
            X = self.bore['Saltney_lag_'+HLW+'_'+source]
            Y = self.bore['liv_height_'+HLW+'_'+source]

        # Get the rmse scores and linear fits
        self.process_fit(source=source, HLW_list=[HLW], X=X,Y=Y)

        ## Plot all data
        try:
            lab_fit = "RMSE:"+self.bore.attrs['rmse_'+HLW+'_'+source]
            lab_dot = "all"
        except:
            lab_fit = ""
            lab_dot = "all"

        JJ = np.isfinite(X.values)
        ss= plt.scatter( X[JJ], Y[JJ], \
            c=col_var[JJ],
            s=5,
            #cmap='magma',
            cmap=cmap,
            vmin=vmin,
            vmax=vmin, # 4.6
            label=lab_dot
            )

        # Regression for all data
        try:
            Xfit = self.bore['linfit_lag_'+HLW+'_'+source]
            plt.plot(Xfit, Y, linewidth=1.0, color='k', label=lab_fit)
        except:
            pass


        ## Plot class A data
        try:
            lab_fit = "(A)RMSE:"+self.bore.attrs['rmse_A_'+HLW+'_'+source]
            lab_dot = "class A"
        except:
            lab_fit = ""
            lab_dot = "class A"


        JJ = (self.bore['Quality'] == "A") & (np.isfinite(X.values))
        ss= plt.scatter( X[JJ], Y[JJ], \
            c=col_var[JJ],
            s=40,
            #cmap='magma',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax, # 4.6
            label=lab_dot
            )
        cbar = plt.colorbar(ss)

        # Regression for Class A
        try:
            Xfit = self.bore['linfit_A_lag_'+HLW+'_'+source]
            plt.plot(Xfit, Y, linewidth=2.0, color='k', label=lab_fit)
        except:
            pass


        # Add dates to class A only
        lab = [ self.bore.time[i].values.astype('datetime64[D]').astype(object).strftime('%d%b%y') if self.bore['Quality'][i] == "A" else "" for i in range(len(self.bore['Quality']))]
        for ind in range(len(self.bore['Quality'])):
        # zip joins x and y coordinates in pairs

            plt.annotate(lab[ind], # this is the text
                         (X[ind],Y[ind]), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,6), # distance from text to points (x,y)
                         ha='center', # horizontal alignment can be left, right or center
                         fontsize=4)
        plt.legend()
        # Linear fit
        #x = self.df['Liv (Gladstone Dock) HT height (m)']
        #plt.plot( x, self.df['linfit_lag'], '-' )
        cbar.set_label(clabel)
        plt.title('Bore arrival time at Saltney Ferry')
        plt.xlabel('Arrival time (mins) relative to Liv '+HLW)
        plt.ylabel('Liv (Gladstone Dock) '+HLW+' height (m)')
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight_'+colour_var_str+'_'+HLW+'_'+source+'.png')

    ############################################################################
    #%% DIAGNOSTICS
    ############################################################################

    def predict_bore(self, source:str='harmonic', HLW:str="HW"):
        """
        Predict the bore timing at Saltney for a request input date (given in
        days relative to now).
        Implements a linear fit model to predicted tides.
        Can select which linear fit model (weights) to use with by specifying
         'source' and 'HLW'

        INPUTS: which define the weights used.
        -------
        source: 'harmonic' [default] - from harmonic prediction
                'bodc' - from measured and processed data
                'api' - from recent, un processed data from shoothill API
        HLW: [LW/HW] - processed from either High or Low water events

        Requested parameters
        --------------------
        day : day
         DESCRIPTION.

        """
        print('Predict bore event for date')
        filnam = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2021_2022_HLW.txt'

        nd = input('Make predictions for N days from hence (int):?')
        day = np.datetime64('now', 'D') + np.timedelta64(int(nd), 'D')
        dayp1 = day + np.timedelta64(24, 'h')

        if(1): # np.datetime64('now', 'Y') < np.datetime64('2021'): # year 2020
            print("predict_bore(): should check is table data is available. If not use harm reconstructed data")
            tg = GAUGE()
            tg.dataset = tg.read_hlw_to_xarray(filnam, day, dayp1)

            HT = tg.dataset['sea_level'].where(tg.dataset['sea_level']\
                                    .values > 7).dropna('time') #, drop=True)
        else: # year 2021 (no tide table data)
            source = 'harmonic_rec'
            print('source=',source)
            tg = GAUGE()
            tg_tmp = GAUGE()
            tg_tmp.dataset = tg_tmp.anyTide_to_xarray(date_start=day, date_end=dayp1)
            tg = tg_tmp.find_high_and_low_water(var_str='sea_level')
            #tg.dataset = tg.get_Glad_data(source='harmonic_rec',date_start=day, date_end=dayp1)

            HT = tg.dataset['sea_level_highs'].where(tg.dataset['sea_level_highs']\
                                    .values > 7).dropna('time_highs')\
                                    .rename({'time_highs':'time'})

        #plt.plot( HT.time, HT,'.' );plt.show()
        #lag_pred = self.linfit(HT)
        lag_pred = self.bore.attrs['weights_'+HLW+'_'+source](HT)
        #lag_pred = lag_pred[np.isfinite(lag_pred)] # drop nans

        Saltney_time_pred = [HT.time[i].values
                             + np.timedelta64(int(round(lag_pred[i])), 'm')
                             for i in range(len(lag_pred))]
        # Iterate over high tide events to print useful information
        print(f"Predictions based on fit to {source} {HLW} data")
        for i in range(len(lag_pred)):
            #print( "Gladstone HT", np.datetime_as_string(HT.time[i], unit='m',timezone=pytz.timezone('UTC')),"(GMT). Height: {:.2f} m".format(  HT.values[i]))
            #print(" Saltney arrival", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')),"(GMT/BST). Lag: {:.0f} mins".format( lag_pred[i] ))
            print("Predictions for ", day_of_week(Saltney_time_pred[i]), Saltney_time_pred[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y/%m/%d') )
            print("Saltney FB:", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')) )
            try:
                Glad_HLW = tg.get_tide_table_times( Saltney_time_pred[i], method='nearest_2' )
                # Extract the High Tide value
                print('Liv HT:    ', np.datetime_as_string(Glad_HLW[ np.argmax(Glad_HLW.values) ].time.values, unit='m', timezone=pytz.timezone('Europe/London')), Glad_HLW[ np.argmax(Glad_HLW.values) ].values, 'm' )
                # Extract the Low Tide value
                print('Liv LT:    ', np.datetime_as_string(Glad_HLW[ np.argmin(Glad_HLW.values) ].time.values, unit='m', timezone=pytz.timezone('Europe/London')), Glad_HLW[ np.argmin(Glad_HLW.values) ].values, 'm' )
            except:
                pass
            print("")

        #plt.scatter( Saltney_time_pred, HT ,'.');plt.show()
        # problem with time stamp

    def stats(self, source:str='harmonic', HLW:str="HW"):
        """
        root mean square error
        """
        rmse = np.sqrt(np.nanmean((self.bore['Saltney_lag_'+HLW+'_'+source].values - self.bore['linfit_lag_'+HLW+'_'+source].values)**2))
        print(f"{source}: Root mean square error = {rmse}")
        return rmse
    ############################################################################
    #%% SECTION
    ############################################################################

    def load_timeseries(self):
        fn_tidegauge = '../COAsT/example_files/tide_gauges/lowestoft-p024-uk-bodc'
        date0 = datetime.datetime(2007,1,10)
        date1 = datetime.datetime(2007,1,12)
        tidegauge = GAUGE(fn_tidegauge, date_start = date0, date_end = date1)
        print(tidegauge.dataset)

    ############################################################################
    #%% Development / Misc methods
    ############################################################################

    def load_and_plot_hlw_data(self):
        """ Simply load HLW file and plot """
        filnam = 'data/Liverpool_2015_2020_HLW.txt'
        date_start = datetime.datetime(2020, 1, 1)
        date_end = datetime.datetime(2020, 12, 31)
        tg = GAUGE()
        tg.dataset = tg.read_hlw_to_xarray(filnam, date_start, date_end)
        # Exaple plot
        plt.figure()
        tg.dataset.plot.scatter(x="time", y="sea_level")
        plt.savefig('figs/Liverpool_HLW.png')
        plt.close('all')

        print(f"stats: mean {tg.time_mean('sea_level')}")
        print(f"stats: std {tg.time_std('sea_level')}")

    def shoothill(self):

        """
        Extract the timeseries for a period.
        Extract the extrema.
        Plot timeseries. Overlay highs and lows
        """
        date_start = np.datetime64('2020-09-01')
        date_end = np.datetime64('2020-09-30')

        # E.g  Liverpool (Gladstone Dock station_id="13482", which is read by default.
        # Load in data from the Shoothill API
        sg = GAUGE()
        sg.dataset = sg.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)

        #sg = GAUGE(startday=date_start, endday=date_end) # create modified Tidegauge object
        sg_HLW = sg.find_high_and_low_water(var_str='sea_level', method='cubic')
        #g.dataset
        #g_HLW.dataset

        plt.figure()
        sg.dataset.plot.scatter(x="time", y="sea_level")
        sg_HLW.dataset.plot.scatter(x="time_highs", y="sea_level_highs")
        sg_HLW.dataset.plot.scatter(x="time_lows", y="sea_level_lows")
        plt.savefig('figs/Liverpool_shoothill.png')
        plt.close('all')

        """
        Compare harmonic predicted highs with measured highs
        """
        # Compare tide predictions with measured HLW
        filnam = 'data/Liverpool_2015_2020_HLW.txt'
        tg = GAUGE()
        tg.dataset = tg.read_hlw_to_xarray(filnam, date_start, date_end)
        tg_HLW = tg.find_high_and_low_water(var_str='sea_level')

        sg = GAUGE()
        sg.dataset = sg.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)
        sg_HW = sg.find_nearby_high_and_low_water(var_str='sea_level', target_times=tg_HLW.dataset.time_highs, method='cubic', extrema="max")

        # Example plot
        from matplotlib.collections import LineCollection
        #from matplotlib import colors as mcolors
        import matplotlib.dates as mdates

        nval = min( len(sg_HLW.dataset.time_highs), len(tg_HLW.dataset.time_highs) )
        segs_h = np.zeros((nval,2,2)) # line, pointA/B, t/z
        #convert dates to numbers first
        segs_h[:,0,0] = mdates.date2num( tg_HLW.dataset.time_highs[:nval].astype('M8[ns]').astype('M8[ms]') )
        segs_h[:,1,0] = mdates.date2num( sg_HW.dataset.time_highs[:nval].astype('M8[ns]').astype('M8[ms]') )
        segs_h[:,0,1] = tg_HLW.dataset.sea_level_highs[:nval]
        segs_h[:,1,1] = sg_HW.dataset.sea_level_highs[:nval]


        fig, ax = plt.subplots()
        ax.set_ylim(segs_h[:,:,1].min(), segs_h[:,:,1].max())
        line_segments_HW = LineCollection(segs_h, cmap='plasma', linewidth=1)
        ax.add_collection(line_segments_HW)
        ax.scatter(segs_h[:,0,0],segs_h[:,0,1], c='green', s=2) # harmonic predictions
        ax.set_title('Harmonic prediction with quiver to measured high waters')

        ax.xaxis_date()
        ax.autoscale_view()
        plt.savefig('figs/Liverpool_shoothill_vs_table.png')
        plt.close('all')

        """
        Compare QC's BODC measured highs with API highs (check reference levels)
        """
        bg=GAUGE()
        bg.dataset = bg.read_bodc_to_xarray("data/BODC_processed/2020LIV.txt")
        # Use QC to drop null values
        bg.dataset['sea_level'] = bg.dataset.sea_level.where( bg.dataset.qc_flags!='N', drop=True)
        # Trim dataset
        bg.dataset = bg.dataset.sel(time=slice(date_start, date_end))
        # Fix some attributes (others might not be correct for all data)
        bg.dataset['start_date'] = bg.dataset.time.min().values
        bg.dataset['end_date'] = bg.dataset.time.max().values
        # This produces an xr.dataset with sea_level_highs and sea_level_lows
        # with time variables time_highs and time_lows.
        bg_HW = bg.find_nearby_high_and_low_water(var_str='sea_level', target_times=tg_HLW.dataset.time_highs, method='cubic', extrema="max")
        #bg_HLW = bg.find_high_and_low_water(var_str='sea_level',method='cubic') #'cubic')

        nval = min( len(sg_HW.dataset.time_highs), len(bg_HW.dataset.time_highs) )
        segs_h = np.zeros((nval,2,2)) # line, pointA/B, t/z
        #convert dates to numbers first
        segs_h[:,0,0] = mdates.date2num( bg_HW.dataset.time_highs[:nval].astype('M8[ns]').astype('M8[ms]') )
        segs_h[:,1,0] = mdates.date2num( sg_HW.dataset.time_highs[:nval].astype('M8[ns]').astype('M8[ms]') )
        segs_h[:,0,1] = bg_HW.dataset.sea_level_highs[:nval]
        segs_h[:,1,1] = sg_HW.dataset.sea_level_highs[:nval]


        fig, ax = plt.subplots()
        ax.set_ylim(segs_h[:,:,1].min(), segs_h[:,:,1].max())
        line_segments_HW = LineCollection(segs_h, cmap='plasma', linewidth=1)
        ax.add_collection(line_segments_HW)
        ax.scatter(segs_h[:,0,0],segs_h[:,0,1], c='green', s=2) # harmonic predictions
        ax.set_title('BODC QCd quiver to API measured high waters')

        ax.xaxis_date()
        ax.autoscale_view()
        plt.savefig('figs/Liverpool_shoothill_vs_bodc.png')
        plt.close('all')


    def fits_to_data(self, source:str="bodc", qc_flag:bool=False):
        """
        Explore different combinations of HW and LW times and heights to
        find the best fit to the data

        qc_flag: if True, only fit bore['Quality'] == "A" data, else fit all data
        """

        args_list = []

        self.bore.attrs['weights_HW_'+source] = []
        self.bore.attrs['rmse_HW_'+source] = []
        args_list.append( {"HLW":"HW",
                "source":source,
                'xvar':self.bore['liv_height_HW_'+source],
                'yvar':self.bore['Saltney_lag_HW_'+source],
                'label':'height(HW), time(HW)',
                'wvar':'weights_HW'+'_'+source,
                'rvar':'rmse_HW'+'_'+source}
                )

        self.bore.attrs['weights_dHW_'+source] = []
        self.bore.attrs['rmse_dHW_'+source] = []
        args_list.append( {"HLW":"dHW",
                "source":source,
                'xvar':self.bore['liv_height_HW_'+source]-self.bore['liv_height_LW_'+source],
                'yvar':self.bore['Saltney_lag_HW_'+source],
                'label':'height(HW-LW), time(HW)',
                'wvar':'weights_dHW_'+source,
                'rvar':'rmse_dHW'+'_'+source}
                )

        self.bore.attrs['weights_dLW_'+source] = []
        self.bore.attrs['rmse_dLW_'+source] = []
        args_list.append( {"HLW":"dLW",
                "source":source,
                'xvar':self.bore['liv_height_HW_'+source]-self.bore['liv_height_LW_'+source],
                'yvar':self.bore['Saltney_lag_LW_'+source],
                'label':'height(HW-LW), time(LW)',
                'wvar':'weights_dLW'+'_'+source,
                'rvar':'rmse_dLW'+'_'+source}
                )

        self.bore.attrs['weights_LW_'+source] = []
        self.bore.attrs['rmse_LW_'+source] = []
        args_list.append( {"HLW":"LW",
                "source":source,
                'xvar':self.bore['liv_height_LW_'+source],
                'yvar':self.bore['Saltney_lag_LW_'+source],
                'label':'height(LW), time(LW)',
                'wvar':'weights_LW'+'_'+source,
                'rvar':'rmse_LW'+'_'+source}
                )

        #self.bore.attrs['weights_XX_'+source] = []
        #self.bore.attrs['rmse_XX_'+source] = []
        args_list.append( {"HLW":"XX",
                "source":source,
                'xvar':self.bore['liv_height_LW_'+source],
                'yvar':self.bore['Saltney_lag_HW_'+source],
                'label':'height(LW), time(HW)',
                'wvar':'weights_XX'+'_'+source,
                'rvar':'rmse_XX'+'_'+source}
                )

        for args in args_list:
            self.bore.attrs[args['wvar']] = []
            self.bore.attrs[args['rvar']] = []

            stats = Stats(args['xvar'], args['yvar'],
                        self.bore['Quality'].values=="A")
            if qc_flag:
                weights,rmse = stats.linear_fit_classA()
                print(f"{source} class A| {args['label']}: {rmse}")
                self.bore.attrs[args['wvar'].replace('weights_','weights_A_')] = weights
                self.bore.attrs[args['rvar'].replace('rmse_','rmse_A_')] = rmse
            else:
                weights,rmse = stats.linear_fit_all()
                print(f"{source}| {args['label']}: {rmse}")
                self.bore.attrs[args['wvar']] = weights
                self.bore.attrs[args['rvar']] = rmse
        ###



    def combinations_lag_hlw_river(self):
        """
        Plot different combinations of Lag,HLW w/ rivers
        plot_scatter(self, source:str='bodc', HLW:str="HW", colour_var_str:str="river")
        """
        if(0):
            self.plot_scatter(source='harmonic', HLW="HW", colour_var_str="river")
            self.plot_scatter(source='bodc', HLW="HW", colour_var_str="river")
            self.plot_scatter(source='bodc', HLW="LW", colour_var_str="river")
            self.plot_scatter(source='bodc', HLW="dLW", colour_var_str="river")
            self.plot_scatter(source='bodc', HLW="dHW", colour_var_str="river")
            self.plot_scatter(source='bodc', HLW="XX", colour_var_str="river")
            self.plot_scatter(source='bodc', HLW="FW", colour_var_str="river")
            self.plot_scatter(source='api', HLW="HW", colour_var_str="river")
            self.plot_scatter(source='api', HLW="FW", colour_var_str="river")

            self.plot_scatter(source='api', HLW="HW", colour_var_str="date")
            self.plot_scatter(source='bodc', HLW="HW", colour_var_str="date")
            self.plot_scatter(source='bodc', HLW="FW", colour_var_str="date")
            self.plot_scatter(source='harmonic', HLW="HW", colour_var_str="date")

            self.plot_scatter(source='api', HLW="HW", colour_var_str="wind_along")
            self.plot_scatter(source='bodc', HLW="HW", colour_var_str="wind_along")
            self.plot_scatter(source='bodc', HLW="FW", colour_var_str="wind_along")
            self.plot_scatter(source='harmonic', HLW="HW", colour_var_str="wind_along")

            self.plot_scatter(source='api', HLW="HW", colour_var_str="wind_across")
            self.plot_scatter(source='bodc', HLW="HW", colour_var_str="wind_across")
            self.plot_scatter(source='bodc', HLW="FW", colour_var_str="wind_across")
            self.plot_scatter(source='harmonic', HLW="HW", colour_var_str="wind_across")

        for src in ["harmonic", "bodc", "api"]:
            for state in ["HW", "LW", "dLW", "dHW", "XX", "FW", "EW"]:
                for col in ["river", "date", "wind_along", "wind_across"]:
                    try:
                        self.plot_scatter(source=src, HLW=state, colour_var_str=col)
                    except:
                        pass

    def river_lag_timing(self, HLW="HW", source="api"):
        """
        Explore how rivers affect bore timing
        """
        plt.close('all')
        fig = plt.figure(figsize=(8, 6), dpi=120)
        if HLW=="dLW":
            X = self.bore['Saltney_lag_LW_'+source]
            Y = self.bore['liv_height_HW_'+source] - self.bore['liv_height_LW_'+source]
        elif HLW=="dHW":
            X = self.bore['Saltney_lag_HW_'+source]
            Y = self.bore['liv_height_HW_'+source] - self.bore['liv_height_LW_'+source]
        elif HLW=="XX":
            X = self.bore['Saltney_lag_HW_'+source]
            Y = self.bore['liv_height_LW_'+source]
        else:
            Y = self.bore['ctr_height_LW']
            lag_pred = self.bore.attrs['weights_'+HLW+'_'+source](self.bore['liv_height_HW_'+source])
            X = lag_pred - self.bore['Saltney_lag_'+HLW+'_'+source]

        S = [40 if self.bore['Quality'][i] == "A" else 5 for i in range(len(self.bore['Quality']))]
        lab = [ self.bore.time[i].values.astype('datetime64[D]').astype(object).strftime('%d%b%y') if self.bore['Quality'][i] == "A" else "" for i in range(len(self.bore['Quality']))]

        ss= plt.scatter( X, Y, \
            c=self.bore['liv_height_HW_'+source], # - self.bore['liv_height_HW_harmonic'],
            s=S,
            #cmap='magma',
            cmap='jet',
            #vmin=8.5,
            #vmax=10.5,
            label="RMSE:"+self.bore.attrs['rmse_'+HLW+'_'+source]
            )
        cbar = plt.colorbar(ss)

        for ind in range(len(self.bore['Quality'])):
        # zip joins x and y coordinates in pairs


            plt.annotate(lab[ind], # this is the text
                         (X[ind],Y[ind]), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,6), # distance from text to points (x,y)
                         ha='center', # horizontal alignment can be left, right or center
                         fontsize=4)
        plt.legend()
        # Linear fit
        #x = self.df['Liv (Gladstone Dock) HT height (m)']
        #plt.plot( x, self.df['linfit_lag'], '-' )
        cbar.set_label('Liv (Gladstone Dock) '+HLW+' height (m)')
        plt.title('Bore arrival time at Saltney Ferry')
        plt.xlabel('Timing error (mins) on prediction relative to '+HLW)
        plt.ylabel('River height (m)')
        plt.savefig('figs/SaltneyArrivalLag_vs_river_LivHeight'+HLW+'_'+source+'.png')


################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":

    #### Initialise logging
    now_str = datetime.datetime.now().strftime("%d%b%y %H:%M")
    logging.info(f"-----{now_str}-----")

    #### Constants
    DATABUCKET_FILE = "deebore.pkl"

    INSTRUCTIONS = """

    Choose Action:
    all     load and process all data

    0       load bore observations
    h       load and process harmonic data
    hrec    load and process harmonic reconstructed data
    b       load and process measured (bodc) data
    a       load and process measured (API) data
    r       load and process measured (API) river data
    m       load and process met data
    2       show bore dataset
    3       plot bore data (lag vs tidal height)
    4       plot difference between predicted and measured (lag vs tidal height)
    5       Explore different RMSE fits to the data

    6       Predict bore event for date

    x       Export data to csv. NOT IMPLEMENTED
    rm      Remove pickle file

    i       to show these instructions
    q       to quit (and pickle bore)

    ---
    DEV:
    d1     load and plot HLW data
    d2     shoothill dev
    d3     <empty>
    d4     Plot different combinations of Lag,HLW w/ rivers
    d5     Explore how rivers affect bore timing
    """


    ## Do the main program

    c = Controller()
