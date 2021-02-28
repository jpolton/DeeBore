"""
Read in a process Dee Bore data
Author: jpolton
Date: 26 Sept 2020

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
from coast.TIDEGAUGE import TIDEGAUGE
from coast.general_utils import dayoweek
from coast.stats_util import find_maxima

import scipy.signal # find_peaks

import logging
logging.basicConfig(filename='bore.log', filemode='w+')
logging.getLogger().setLevel(logging.DEBUG)


#%% ################################################################################
class GAUGE(TIDEGAUGE):
    """ Inherit from COAsT. Add new methods """
    def __init__(self, ndays: int=5, startday: datetime=None, endday: datetime=None, stationId="7708"):
        try:
            import config_keys # Load secret keys
        except:
            logging.info('Need a Shoothil API Key. Use e.g. create_shoothill_key() having obtained a public key')

        #self.SessionHeaderId=config_keys.SHOOTHILL_KEY #'4b6...snip...a5ea'
        self.ndays=ndays
        self.startday=startday
        self.endday=endday
        self.stationId=stationId # Shoothill id

        #self.dataset = self.read_shoothill_to_xarray(stationId="13482") # Liverpool

        pass

    def get_mean_crossing_time_as_xarray(self, date_start=None, date_end=None):
        """
        Get the height (constant) and times of crossing the mean height as xarray
        """
        pass

    def get_HW_to_xarray(self, date_start=None, date_end=None):
        """ Extract actual HW value and time as an xarray """
        pass


    def find_nearby_high_and_low_water(self, var_str, target_times:xr.DataArray=None, winsize:int=2, method='comp'):
        """
        Finds high and low water for a given variable, in close proximity to
        input xrray of times.
        Returns in a new TIDEGAUGE object with similar data format to
        a TIDETABLE, and same size as target_times.

        winsize: +/- hours search radius
        target_times: xr.DataArray of target times to search around (e.g. harmonic predictions)
        var_str: root of var name for new variable.
        """
        x = self.dataset.time
        y = self.dataset[var_str]

        nt = len(target_times)
        time_max = np.zeros(nt)
        values_max = np.zeros(nt)
        for i in range(nt):
            HLW = self.get_tidetabletimes( target_times[i].values, method='window', winsize=winsize )
            logging.debug(f"{i}: {find_maxima(HLW.time.values, HLW.values, method=method)}")
            time_max[i], values_max[i] = find_maxima(HLW.time.values, HLW.values, method=method)

        new_dataset = xr.Dataset()
        new_dataset.attrs = self.dataset.attrs
        new_dataset['time_highs'] = ('time_highs', time_max)
        print(time_max)
        print(values_max)
        new_dataset[var_str + '_highs'] = (var_str+'_highs', values_max)

        new_object = TIDEGAUGE()
        new_object.dataset = new_dataset

        return new_object
    
############ shoothill gauge methods ##############################################
    @classmethod
    def read_shoothill_to_xarray(cls,
                                ndays: int=5,
                                date_start: np.datetime64=None,
                                date_end: np.datetime64=None,
                                stationId="7708",
                                dataType=3):
        """
        load gauge data via shoothill API
        Either loads last ndays, or from date_start:date_end

        This reqires an API key that is obtained by emailing shoothill.
        They provide a public key. Then SHOOTHILL_KEY can be generated using
        SHOOTHILL_KEY = create_shoothill_key(SHOOTHILL_PublicApiKey)

        To discover the StationId for a particular measurement site check the
         integer id in the url or its twitter page having identified it via
          https://www.gaugemap.co.uk/#!Map
         E.g  Liverpool (Gladstone Dock stationId="13482".
        Liverpool, or stationId="13482", is assumed by default.

        INPUTS:
            ndays : int
            date_start : datetime. UTC format string "yyyy-MM-ddThh:mm:ssZ" E.g 2020-01-05T08:20:01.5011423+00:00
            date_end : datetime
            stationId : str (station id)
            dataType: int (3 level, 15 flow)
        OUTPUT:
            sea_level, time : xr.Dataset
        """
        import requests,json

        try:
            import config_keys # Load secret keys
        except:
            info('Need a Shoothil API Key. Use e.g. create_shoothill_key(SHOOTHILL_PublicApiKey) having obtained a public key')
            print('Expected a config_keys.py file of the form:')
            print('')
            print('# API keys excluded from github repo')
            print('SHOOTHILL_KEY = "4b6...5ea"')
            print('SHOOTHILL_PublicApiKey = "9a1...414"')

        cls.SessionHeaderId=config_keys.SHOOTHILL_KEY #'4b6...snip...a5ea'
        cls.ndays=ndays
        cls.date_start=date_start
        cls.date_end=date_end
        cls.stationId=stationId # Shoothill id
        cls.dataType=dataType

        info("load gauge")

        if cls.stationId == "7708":
            id_ref = "Gladston Dock"
        elif cls.stationId == "7899":
            id_ref = "Chester weir"
        elif cls.stationId == "972":
            id_ref = "Farndon"
        elif cls.stationId == "968":
            id_ref = "Ironbridge (Dee)"
        else:
            id_ref = "No label"
            debug(f"Not ready for that station id. {cls.stationId}")

        headers = {'content-type': 'application/json', 'SessionHeaderId': cls.SessionHeaderId}

        #%% Construct station info API request
        # Obtain and process header information
        info("load station info")
        htmlcall_stationId = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesStationById/?stationId='
        url  = htmlcall_stationId+str(stationId)
        try:
            request_raw = requests.get(url, headers=headers)
            header_dict = json.loads(request_raw.content)
        except ValueError:
            print(f"Failed request for station {cls.stationId}")
            return

        # Assign expected header_dict information
        try: # header_dict['latitude'] and header_dict['longitude'] are present
            header_dict['site_name'] = header_dict['name']
            #header_dict['latitude'] = header_dict['items']['lat']
            #header_dict['longitude'] = header_dict['items']['long']
        except:
            info(f"possible missing some header info: site_name,latitude,longitude")

        #%% Construct data API request
        if (cls.date_start == None) & (cls.date_end == None):
            info(f"GETting ndays= {cls.ndays} of data")

            htmlcall_stationId = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesRecentDatapoints/?stationId='
            url  = htmlcall_stationId+str(cls.stationId)+'&dataType='+str(int(cls.dataType))+'&numberDays='+str(int(cls.ndays))
        else:
            # Check date_start and date_end are timetime objects
            if (type(cls.date_start) is np.datetime64) & (type(cls.date_end) is np.datetime64):
                info(f"GETting data from {cls.date_start} to {cls.date_end}")
                startTime = cls.date_start.item().strftime('%Y-%m-%dT%H:%M:%SZ')
                endTime = cls.date_end.item().strftime('%Y-%m-%dT%H:%M:%SZ')

                htmlcall_stationId = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesDatapointsDateTime/?stationId='
                url   = htmlcall_stationId+str(cls.stationId)+'&dataType='+str(int(cls.dataType))+'&endTime='+endTime+'&startTime='+startTime

            else:
                debug('Expecting date_start and date_end as datetime objects')

        #%% Get the data
        request_raw = requests.get(url, headers=headers)
        request = json.loads(request_raw.content)
        debug(f"Shoothil API request: {request_raw.text}")
        # Check the output
        info(f"Gauge id is {request['gauge']['geoEntityId']}")
        info(f"timestamp and value of the zero index is {[ str(request['values'][0]['time']), request['values'][0]['value'] ]}")

        #print(request)
        #%% Process header information
        #header_dict = request['gauge']
        #header_dict['site_name'] = id_ref

        #%% Process timeseries data
        dataset = xr.Dataset()
        time = []
        sea_level = []
        nvals = len(request['values'])
        time = np.array([np.datetime64(request['values'][i]['time']) for i in range(nvals)])
        sea_level = np.array([request['values'][i]['value'] for i in range(nvals)])

        #%% Assign arrays to Dataset
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['time'])
        dataset = dataset.assign_coords(time = ('time', time))
        dataset.attrs = header_dict
        debug(f"Shoothil API request headers: {header_dict}")
        debug(f"Shoothil API request 1st time: {time[0]} and value: {sea_level[0]}")

        # Assign local dataset to object-scope dataset
        return dataset


############ anyTide harmonic reconstruction method ###########################
    @classmethod
    def anyTide_to_xarray(cls,
                                ndays: int=5,
                                date_start: np.datetime64=None,
                                date_end: np.datetime64=None,
                                loc="Glad",
                                plot_flag=False):
        """
        Construct harmonic timeseries using anyTide code. 
        Either loads last ndays, or from date_start:date_end

        INPUTS:
            ndays : int
            date_start : datetime. UTC format string "yyyy-MM-ddThh:mm:ssZ" E.g 2020-01-05T08:20:01.5011423+00:00
            date_end : datetime
            loc : str (name of harmonics file). ONLY GLADSTONE AT PRESENT
            plot_flag : bool
        OUTPUT:
            sea_level, time : xr.Dataset
        """
    
        anytidedir = os.path.dirname('/Users/jeff/GitHub/anyTide/')
        sys.path.insert(0, anytidedir)
    
        from NOCtidepred import get_port
        from NOCtidepred import test_port
    
        #from NOCtidepred import UtcNow
        from NOCtidepred import date2mjd
        from NOCtidepred import phamp0fast
        from NOCtidepred import set_names_phases
    
        if loc != "Glad":
            print("Can only process Gladstone Dock at present. Proceeding...")
            info("Can only process Gladstone Dock at present. Proceeding...")
            
        if date_start == None:
            date_start = np.datetime64('now')
        if date_end == None:
            date_end = date_start + np.timedelta64(ndays,"D")
            
            
        cls.ndays=ndays
        cls.date_start=date_start
        cls.date_end=date_end
        cls.loc=loc # harmonics file
    
        #info("load gauge")
    
        # Settings
        rad    = np.pi/180
        deg    = 1.0 / rad
    
    
    
        # Set the dates
        # Create a vector of predictions times. Assume 5 min increments
        nvals = round((date_end - date_start)/np.timedelta64(5,"m"))
        dates = [date_start + np.timedelta64(5*mm,"m") for mm in range(0, nvals)]
        if type(dates[1]) != datetime.datetime: 
            mjd = date2mjd( [dates[i].astype(datetime.datetime)for i in range(nvals)] )
        else:
            mjd = date2mjd( dates ) # convert to modified julian dates
    
    
        ## Compute reconstuction on port data.
        #####################################
        ssh = test_port(mjd) # reconstuct ssh for the time vector
        print('plot time series reconstruction of port data')
    
        ssh = np.ma.masked_where( ssh > 1E6, ssh) # get rid of nasties        
    
        # Plot time series
        if plot_flag:
            # Plot sea level time series
            fig, ax = plt.subplots()
            ax.plot(np.array(dates),[ssh[i] for i in range(len(dates))],'+-')
            ax.set_ylabel('Height (m)')
            ax.set_xlabel('Hours since '+dates[0].strftime("%Y-%m-%d"))
            ax.set_title('Harmonic tide prediction')
            
            # Pain plotting time on the x-axis
            myFmt = mdates.DateFormatter('%H')
            ax.xaxis.set_major_formatter(myFmt)
        
            plt.show()
        
        
    
        #%% Process timeseries data
        dataset = xr.Dataset()
        time = []
        sea_level = []
        time = dates #np.array([np.datetime64(request['values'][i]['time']) for i in range(nvals)])
        sea_level = ssh #np.array([request['values'][i]['value'] for i in range(nvals)])
    
        #%% Assign arrays to Dataset
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['time'])
        dataset = dataset.assign_coords(time = ('time', time))
        #dataset.attrs = header_dict
        #debug(f"NOCpredict API request 1st time: {time[0]} and value: {sea_level[0]}")
    
        # Assign local dataset to object-scope dataset
        return dataset

        

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
                print('load and process harmonic data')
                self.load_and_process(source="harmonic", HLW="HW")
                self.load_and_process(source="harmonic", HLW="LW")
                print('load and process measured (bodc) data')
                self.load_and_process(source="bodc", HLW="HW")
                self.load_and_process(source="bodc", HLW="LW")
                print('load and process measured (API) data')
                self.load_and_process(source="api", HLW="HW")
                self.load_and_process(source="api", HLW="LW")
                print('load and process CTR data. Obs + API')
                self.get_CTR_data(HLW="LW")


            elif command == "0":
                print('load bore observations')
                self.load_csv()

            elif command == "h":
                print('load and process harmonic data')
                if not self.load_bore_flag: self.load_csv()
                self.load_and_process(source="harmonic")

            elif command == "b":
                print('load and process measured (bodc) data')
                if not self.load_bore_flag: self.load_csv()
                self.load_and_process(source="bodc")

            elif command == "a":
                print('load and process measured (API) data')
                if not self.load_bore_flag: self.load_csv()
                self.load_and_process(source="api")

            elif command == "2":
                print('show bore dataset')
                self.show()

            elif command == "3":
                print('plot bore data (lag vs tidal height')
                plt.close('all');self.plot_lag_vs_height('bodc')
                plt.close('all');self.plot_lag_vs_height('all')
                plt.close('all');self.plot_lag_vs_height('harmonic')
                plt.close('all');self.plot_lag_vs_height('api')

            elif command == "4":
                print('plot difference between predicted and measured (lag vs tidal height)')
                plt.close('all');self.plot_surge_effect('api')
                plt.close('all');self.plot_surge_effect('bodc')

            elif command == "d1":
                print('load and plot HLW data')
                self.load_and_plot_HLW_data()

            elif command == "d2":
                print("shoothill dev")
                self.shoothill()

            elif command == "d3":
                print('Explore combinations of HLW times and heights for best fit')
                self.fits_to_data()

            elif command == "d4":
                print('Plot combinations of HLW times, heights and rivers')
                self.combinations_lag_HLW_river()

            elif command == "6":
                self.predict_bore()

            elif command == "x":
                print('Export data')
                self.export_to_csv()

            elif command == "r":
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

    def load_and_process(self, source:str="harmonic", HLW:str="HW"):
        """
        Performs sequential steps to build into the bore object.
        1. Load Gladstone Dock data (though this might also be loaded from the obs logs)
        2. Calculate the time lag between Gladstone and Saltney events.
        3. Perform a linear fit to the time lag.

        Inputs:
        source: 'harmonic' [default] - load HLW from harmonic prediction
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        print('loading '+source+' tide data')
        self.get_Glad_data(source=source, HLW=HLW)
        #self.compare_Glad_HLW()
        print('Calculating the Gladstone to Saltney time difference')
        self.calc_Glad_Saltney_time_diff(source=source, HLW=HLW)
        print('Process linear fit. Calc and save')
        self.process_fit(source=source, HLW=HLW)


    def process_fit(self, source:str="harmonic", HLW:str="HW"):
        # Get linear fit with rmse
        self.bore.attrs['weights_'+HLW+'_'+source], self.bore.attrs['rmse_'+HLW+'_'+source] = self.linearfit(
                self.bore['liv_height_'+HLW+'_'+source],
                self.bore['Saltney_lag_'+HLW+'_'+source]
                )
        # Apply linear model
        self.bore['linfit_lag_'+HLW+'_'+source] = self.bore.attrs['weights_'+HLW+'_'+source](self.bore['liv_height_'+HLW+'_'+source])
        #self.bore['rmse_'+HLW+'_'+source] = '{:4.1f} mins'.format(self.stats(source=source, HLW=HLW))

    def load_csv(self):
        """
        Load observed bore data from text file.
        Load as a dataframe and save to bore:xr.DataSet
        """
        logging.info('Load bore data from csv file')
        self.load_bore_flag = True
        df =  pd.read_csv('data/master-Table 1.csv')
        df.drop(columns=['date + logged time','Unnamed: 2','Unnamed: 11', \
                                'Unnamed: 12','Unnamed: 13'], \
                                 inplace=True)
        df.rename(columns={"date + logged time (GMT)":"time"}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], format="%d/%m/%Y %H:%M")
        #df['time'] = pd.to_datetime(df['time'], utc=True, format="%d/%m/%Y %H:%M")
        #df.set_index(['time'], inplace=True)


        for index, row in df.iterrows():
            df.loc[index,'time'] = np.datetime64( df.at[index,'time'] ) # numpy.datetime64 in UTC
        bore = xr.Dataset()
        bore = df.to_xarray()

        # Set the t_dim to be a dimension and 'time' to be a coordinate
        bore = bore.rename_dims( {'index':'t_dim'} ).assign_coords( time=("t_dim", bore.time))
        bore = bore.swap_dims( {'t_dim':'time'} )
        self.bore = bore
        logging.info('Bore data loaded')

    def get_CTR_data(self, HLW:str="LW"):
        """
        Get Chester weir data. Consolidate CTR data.
        Data from the table takes precident. Gaps are filled by the API.
        """

        if HLW != "LW":
            print('Not expecting that possibility here')
        else:
            # Obtain CTR data for LW for the observations times.
            self.get_Glad_data(source='ctr',HLW="LW")
            alph = self.bore['Chester Weir height: CHESTER WEIR 15 MIN SG']
            beta = self.bore['ctr_height_LW_ctr']
            self.bore['ctr_height_LW'] = alph
            self.bore['ctr_height_LW'].values = [alph[i].values if np.isfinite(alph[i].values) else beta[i].values for i in range(len(alph))]
            # 2015-06-20T12:16:00 has a -ve value. Only keep +ve values
            self.bore['ctr_height_LW'] = self.bore['ctr_height_LW'].where( self.bore['ctr_height_LW'].values>0)
            #plt.plot( ctr_h_csv, 'b+' )
            #plt.plot( self.bore['ctr_height_LW_ctr'], 'ro')
            #plt.plot( self.bore['ctr_height_LW'], 'g.')
            del self.bore['ctr_height_LW_ctr'], self.bore['ctr_time_LW_ctr']


    def get_Glad_data(self, source:str='harmonic', HLW:str="HW"):
        """
        Get Gladstone HLW data from external source
        These data are reported in the bore.csv file but not consistently and it
        is laborous to find old values.
        It was considered a good idea to automate this step.

        inputs:
        source: 'harmonic' [default] - load HLW from harmonic prediction
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        loc = "liv" # default location - Liverpool

        logging.info("Get Gladstone HLW data")
        if source == "harmonic": # Load tidetable data from files
            filnam1 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2005_2014_HLW.txt'
            filnam2 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2015_2020_HLW.txt'
            filnam3 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2021_2021_HLW.txt'
            tg  = TIDEGAUGE()
            tg1 = TIDEGAUGE()
            tg2 = TIDEGAUGE()
            tg3 = TIDEGAUGE()
            tg1.dataset = tg1.read_HLW_to_xarray(filnam1)#, self.bore.time.min().values, self.bore.time.max().values)
            tg2.dataset = tg2.read_HLW_to_xarray(filnam2)#, self.bore.time.min().values, self.bore.time.max().values)
            tg3.dataset = tg3.read_HLW_to_xarray(filnam3)#, self.bore.time.min().values, self.bore.time.max().values)
            tg.dataset = xr.concat([ tg1.dataset, tg2.dataset, tg3.dataset], dim='time')

            # This produces an xr.dataset with sea_level_highs and sea_level_lows
            # with time variables time_highs and time_lows.
            tg_HLW = tg.find_high_and_low_water(var_str='sea_level')

        elif source == "bodc": # load full 15min data from BODC files, extract HLW
            dir = '/Users/jeff/GitHub/DeeBore/data/BODC_processed/'
            filelist = ['2005LIV.txt',
            '2006LIV.txt', '2007LIV.txt',
            '2008LIV.txt', '2009LIV.txt',
            '2010LIV.txt', '2011LIV.txt',
            '2012LIV.txt', '2013LIV.txt',
            '2014LIV.txt', '2015LIV.txt',
            '2016LIV.txt', '2017LIV.txt',
            '2018LIV.txt', '2019LIV.txt']
            tg  = TIDEGAUGE()
            for file in filelist:
                tg0=TIDEGAUGE()
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

            # This produces an xr.dataset with sea_level_highs and sea_level_lows
            # with time variables time_highs and time_lows.
            tg_HLW = tg.find_high_and_low_water(var_str='sea_level')

        elif source == "api": # load full tidal signal from shoothill, extract HLW
            tg = TIDEGAUGE()
            date_start=np.datetime64('2005-04-01')
            date_end=np.datetime64('now','D')
            tg.dataset = tg.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)
            # This produces an xr.dataset with sea_level_highs and sea_level_lows
            # with time variables time_highs and time_lows.
            tg_HLW = tg.find_high_and_low_water(var_str='sea_level')

        elif source == "ctr": # use api to load chester weir. Reset loc variable
            loc = "ctr"
            tg = TIDEGAUGE()
            date_start=np.datetime64('2014-01-01')
            date_end=np.datetime64('now','D')
            tg.dataset = tg.read_shoothill_to_xarray(stationId="7899" ,date_start=date_start, date_end=date_end)

            # This produces an xr.dataset with sea_level_highs and sea_level_lows
            # with time variables time_highs and time_lows.
            tg_HLW = tg.find_high_and_low_water(var_str='sea_level')
        
        elif source == 'anyTide': # load full tidal signal using anyTide code, extract HLW
            tg = GAUGE()
            date_start=np.datetime64('now')
            ndays = 5
            tg.dataset = tg.anyTide_to_xarray(date_start=date_start, ndays=5)
            # This produces an xr.dataset with sea_level_highs and sea_level_lows
            # with time variables time_highs and time_lows.
            tg_HLW = tg.find_high_and_low_water(var_str='sea_level')            
        else:
            logging.debug(f"Did not expect this eventuality...")

        self.tg = tg

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

        for i in range(len(self.bore.time)):
            try:
                HW = None
                LW = None
                #HLW = tg.get_tidetabletimes(self.bore.time[i].values)

                HW = tg_HLW.get_tidetabletimes(
                                        time_guess=self.bore.time[i].values,
                                        time_var=time_var,
                                        measure_var=measure_var,
                                        method='nearest_1',
                                        winsize=6 ) #4h for HW, 6h for LW
                #LW = tg_HLW.get_tidetabletimes(
                #                        time_guess=self.bore.time[i].values,
                #                        time_var='time_lows',
                #                        measure_var='sea_level_lows',
                #                        method='nearest_1',
                #                        winsize=6 ) #4

                #HW = HW - LW.values
                if type(HW) is xr.DataArray:
                    #print(f"HW: {HW}")
                    HT_h.append( HW.values )
                    #print('len(HT_h)', len(HT_h))
                    HT_t.append( HW[time_var].values )
                    #print('len(HT_t)', len(HT_t))
                    #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
                    #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
                    #ind.append(i)
                    #print(f"i:{i}, {HT_t[-1].astype('M8[ns]').astype('M8[ms]').item()}" )
                    #print(HT_t[-1].astype('M8[ns]').astype('M8[ms]').item().strftime('%Y-%m-%d'))

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

        try: # Try and print the last observation timeseries
            plt.savefig('figs/check_get_tidetabletimes_'+str(i//12).zfill(2)+'_'+HLW+'_'+source+'.png')
            plt.close('all')
        except:
            logging.info(f"Did not have any extra panels to plot")
            print(f"Did not have any extra panels to plot")


        # Save a xarray objects
        coords = {'time': (('time'), self.bore.time.values)}
        self.bore[loc+'_height_'+HLW+'_'+source] = xr.DataArray( np.array(HT_h), coords=coords, dims=['time'])
        self.bore[loc+'_time_'+HLW+'_'+source] = xr.DataArray( np.array(HT_t), coords=coords, dims=['time'])

        print('There is a supressed plot.scatter here')
        #self.bore.plot.scatter(x='liv_time', y='liv_height'); plt.show()

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


    def calc_Glad_Saltney_time_diff(self, source:str="harmonic", HLW:str="HW"):
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
        logging.info('calc_Glad_Saltney_time_diff')
        nt = len(self.bore.time)
        lag = (self.bore['time'].values - self.bore['liv_time_'+HLW+'_'+source].values).astype('timedelta64[m]')
        Saltney_lag    = [ lag[i].astype('int') if self.bore.location.values[i] == 'bridge' else np.NaN for i in range(nt) ]
        bluebridge_lag = [ lag[i].astype('int') if self.bore.location.values[i] == 'blue bridge' else np.NaN for i in range(nt) ]

        # Save a xarray objects
        coords = {'time': (('time'), self.bore.time.values)}
        self.bore['lag_'+HLW+'_'+source] = xr.DataArray( lag, coords=coords, dims=['time'])
        self.bore['Saltney_lag_'+HLW+'_'+source] = xr.DataArray( Saltney_lag, coords=coords, dims=['time'])
        self.bore['bluebridge_lag_'+HLW+'_'+source] = xr.DataArray( bluebridge_lag, coords=coords, dims=['time'])


    def linearfit(self, X, Y):
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
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
                'all' - Use bodc + api data
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        if source == 'all':
            Yliv = self.bore['liv_height_'+HLW+'_bodc']
            Xsalt = self.bore['Saltney_lag_'+HLW+'_bodc']
            Xblue = self.bore['bluebridge_lag_'+HLW+'_bodc']
            Yliv_api = self.bore['liv_height_'+HLW+'_api'].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xsalt_api = self.bore['Saltney_lag_'+HLW+'_api'].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xblue_api = self.bore['bluebridge_lag_'+HLW+'_api'].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xfit = self.bore['linfit_lag_'+HLW+'_bodc']
            plt.plot( Xsalt,Yliv, 'r.', label='Saltney: rmse '+'{:4.1f}'.format(self.stats('bodc'))+'mins')
            plt.plot( Xblue,Yliv, 'b.', label='Bluebridge')
            plt.plot( Xfit,Yliv, 'k-')
            plt.plot( Xsalt_api,Yliv_api, 'ro', label='Saltney 2020')
            plt.plot( Xblue_api,Yliv_api, 'bo', label='Bluebridge 2020')
        else:
            Yliv = self.bore['liv_height_'+HLW+'_'+source]
            Xsalt = self.bore['Saltney_lag_'+HLW+'_'+source]
            Xblue = self.bore['bluebridge_lag_'+HLW+'_'+source]
            Xfit = self.bore['linfit_lag_'+HLW+'_'+source]
            plt.plot( Xsalt,Yliv, 'r.', label='Saltney: rmse '+'{:4.1f}'.format(self.stats(source,HLW))+'mins')
            plt.plot( Xblue,Yliv, 'b.', label='Bluebridge')
            plt.plot( Xfit,Yliv, 'k-')
            Yliv = self.bore['liv_height_'+HLW+'_'+source].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xsalt = self.bore['Saltney_lag_'+HLW+'_'+source].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            Xblue = self.bore['bluebridge_lag_'+HLW+'_'+source].where( np.isnan(self.bore['liv_height_'+HLW+'_bodc']))
            plt.plot( Xsalt,Yliv, 'ro', label='Saltney 2020')
            plt.plot( Xblue,Yliv, 'bo', label='Bluebridge 2020')

        plt.ylabel('Liv (Gladstone Dock) '+HLW+' (m)')
        plt.xlabel('Arrival time (mins) relative to Liv '+HLW)
        if source =='harmonic': str='predicted'
        if source =='all': str='all measured'
        if source =='bodc': str='measured only QCd'
        if source == 'api': str='measured w/o QC'
        plt.title(f"Bore arrival time at Saltney Ferry ({str} data)")
        #plt.xlim([-125, -40])   # minutes
        #plt.ylim([8.2, 10.9]) # metres
        plt.legend()
        #plt.show()
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight_'+HLW+'_'+source+'.png')

        if(0):
            #plt.show()

            s = plt.scatter( self.bore['Saltney_lag_HW_bodc'], \
                self.bore['liv_height_HW_bodc'], \
                c=self.bore['Chester Weir height: CHESTER WEIR 15 MIN SG'],
                cmap='magma',
                vmin=4.4,
                vmax=4.6 )
            cbar = plt.colorbar(s)
            # Linear fit
            #x = self.df['Liv (Gladstone Dock) HT height (m)']
            #plt.plot( x, self.df['linfit_lag'], '-' )
            cbar.set_label('River height at weir (m)')
            plt.title('Bore arrival time at Saltney Ferry')
            plt.xlabel('Arrival time (mins before Liv HT)')
            plt.ylabel('Liv (Gladstone Dock) HT height (m)')
            plt.show()


    def plot_surge_effect(self, source:str='bodc', HLW:str="HW"):
        """
        Compare harmonic predicted HLW+lag with measured HLW+lag
        Plot quiver between harmonic and measured values.

        inputs:
        source:
                'bodc' - measured and processed data
                'api' - load recent, un processed data from shoothill API
        HLW: [LW/HW] - the data is either processed for High or Low water events
        """
        # Example plot
        from matplotlib.collections import LineCollection
        from matplotlib import colors as mcolors
        import matplotlib.dates as mdates
        if source=='api':
            I = self.bore['liv_time_'+HLW+'_api'] > np.datetime64('2020-09-01')
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
        ax.set_title('Harmonic prediction with quiver to measured high waters')

        plt.ylabel('Liv (Gladstone Dock) '+HLW+' (m)')
        plt.xlabel('Arrival time (mins relative to LiV '+HLW+')')
        plt.title('Bore arrival time at Saltney Ferry. Harmonic prediction cf measured')
        plt.legend()
        #plt.xlim([-125, -40])   # minutes
        #plt.ylim([8.2, 10.9]) # metres
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight_shift_'+HLW+'_'+source+'.png')
        plt.close('all')


    def plot_scatter_river(self, source:str='bodc', HLW:str="HW"):
        """
        """
        plt.close('all')
        fig = plt.figure()
        if HLW=="dLW":
            X = self.bore['Saltney_lag_LW_'+source]
            Y = self.bore['liv_height_HW_'+source] - self.bore['liv_height_LW_'+source]
        elif HLW=="dHW":
            X = self.bore['Saltney_lag_HW_'+source]
            Y = self.bore['liv_height_HW_'+source] - self.bore['liv_height_LW_'+source]
        else:
            X = self.bore['Saltney_lag_'+HLW+'_'+source]
            Y = self.bore['liv_height_'+HLW+'_'+source]
        s = plt.scatter( X, Y, \
            c=self.bore['ctr_height_LW'],
            cmap='magma',
            vmin=4.4,
            vmax=4.6,
            label="RMSE:"+self.bore.attrs['rmse_'+HLW+'_'+source]
            )
        cbar = plt.colorbar(s)
        plt.legend()
        # Linear fit
        #x = self.df['Liv (Gladstone Dock) HT height (m)']
        #plt.plot( x, self.df['linfit_lag'], '-' )
        cbar.set_label('River height at weir (m)')
        plt.title('Bore arrival time at Saltney Ferry')
        plt.xlabel('Arrival time (mins) relative to Liv '+HLW)
        plt.ylabel('Liv (Gladstone Dock) '+HLW+' height (m)')
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight_river_'+HLW+'_'+source+'.png')

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
        #filnam = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2015_2020_HLW.txt'
        filnam = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2021_2021_HLW.txt'

        nd = input('Make predictions for N days from hence (int):?')
        day = np.datetime64('now', 'D') + np.timedelta64(int(nd), 'D')
        dayp1 = day + np.timedelta64(24, 'h')
        
        if(1): # np.datetime64('now', 'Y') < np.datetime64('2021'): # year 2020
            tg = TIDEGAUGE()
            tg.dataset = tg.read_HLW_to_xarray(filnam, day, dayp1)
            
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
            print("Predictions for ", dayoweek(Saltney_time_pred[i]), Saltney_time_pred[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y/%m/%d') )
            print("Saltney FB:", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')) )
            Glad_HLW = tg.get_tidetabletimes( Saltney_time_pred[i], method='nearest_2' )
            # Extract the High Tide value
            print('Liv HT:    ', np.datetime_as_string(Glad_HLW[ np.argmax(Glad_HLW.values) ].time.values, unit='m', timezone=pytz.timezone('Europe/London')), Glad_HLW[ np.argmax(Glad_HLW.values) ].values, 'm' )
            # Extract the Low Tide value
            print('Liv LT:    ', np.datetime_as_string(Glad_HLW[ np.argmin(Glad_HLW.values) ].time.values, unit='m', timezone=pytz.timezone('Europe/London')), Glad_HLW[ np.argmin(Glad_HLW.values) ].values, 'm' )
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

    def load_and_plot_HLW_data(self):
        """ Simply load HLW file and plot """
        filnam = 'data/Liverpool_2015_2020_HLW.txt'
        date_start = datetime.datetime(2020, 1, 1)
        date_end = datetime.datetime(2020, 12, 31)
        tg = TIDEGAUGE()
        tg.dataset = tg.read_HLW_to_xarray(filnam, date_start, date_end)
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

        # E.g  Liverpool (Gladstone Dock stationId="13482", which is read by default.
        # Load in data from the Shoothill API
        sg = TIDEGAUGE()
        sg.dataset = sg.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)

        #sg = GAUGE(startday=date_start, endday=date_end) # create modified TIDEGAUGE object
        sg_HLW = sg.find_high_and_low_water(var_str='sea_level')
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
        tg.dataset = tg.read_HLW_to_xarray(filnam, date_start, date_end)
        tg_HLW = tg.find_high_and_low_water(var_str='sea_level')

        sg = GAUGE()
        sg.dataset = sg.read_shoothill_to_xarray(date_start=date_start, date_end=date_end)
        sg_HW = sg.find_nearby_high_and_low_water(var_str='sea_level', target_times=tg_HLW.dataset.time_highs, method='comp')

        # Example plot
        from matplotlib.collections import LineCollection
        from matplotlib import colors as mcolors
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


    def fits_to_data(self, source:str="bodc"):
        """
        Explore different combinations of HW and LW times and heights to
        find the best fit to the data
        """
        HLW="HW"
        weights,rmse = self.linearfit(
            self.bore['liv_height_HW_'+source],
            self.bore['Saltney_lag_HW_'+source]
            )
        print(f"{source}| height(HW), time(HW): {rmse}")
        #Out[45]: (poly1d([-12.26700862,  45.96440818]), ' 6.6 mins')
        self.bore.attrs['weights_'+HLW+'_'+source] = weights
        self.bore.attrs['rmse_'+HLW+'_'+source] = rmse

        HLW="dHW"
        weights,rmse = self.linearfit(
            self.bore['liv_height_HW_'+source]-self.bore['liv_height_LW_'+source],
            self.bore['Saltney_lag_HW_'+source]
            )
        print(f"{source}| height(HW-LW), time(HW): {rmse}")
        #Out[44]: (poly1d([ -6.56953332, -15.68423086]), ' 6.9 mins')
        self.bore.attrs['weights_'+HLW+'_'+source] = weights
        self.bore.attrs['rmse_'+HLW+'_'+source] = rmse

        HLW="dLW"
        weights,rmse = self.linearfit(
            self.bore['liv_height_HW_'+source]-self.bore['liv_height_LW_'+source],
            self.bore['Saltney_lag_LW_'+source]
            )
        print(f"{source}| height(HW-LW), time(LW): {rmse}")
        #Out[46]: (poly1d([-15.34697352, 379.18885683]), ' 9.0 mins')
        self.bore.attrs['weights_'+HLW+'_'+source] = weights
        self.bore.attrs['rmse_'+HLW+'_'+source] = rmse

        HLW="LW"
        weights,rmse = self.linearfit(
            self.bore['liv_height_LW_'+source],
            self.bore['Saltney_lag_LW_'+source]
            )
        print(f"{source}| height(LW), time(LW): {rmse}")
        #Out[47]: (poly1d([ 23.95624428, 222.70884297]), '12.1 mins')
        self.bore.attrs['weights_'+HLW+'_'+source] = weights
        self.bore.attrs['rmse_'+HLW+'_'+source] = rmse

    def combinations_lag_HLW_river(self):
        """
        Plot different combinations of Lag,HLW w/ rivers
        """
        self.plot_scatter_river(source='bodc', HLW="HW")
        self.plot_scatter_river(source='bodc', HLW="LW")
        self.plot_scatter_river(source='bodc', HLW="dLW")
        self.plot_scatter_river(source='bodc', HLW="dHW")


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
    b       load and process measured (bodc) data
    a       load and process measured (API) data
    2       show bore dataset
    3       plot bore data (lag vs tidal height)
    4       plot difference between predicted and measured (lag vs tidal height)

    6       Predict bore event for date

    x       Export data to csv. NOT IMPLEMENTED
    r       Remove pickle file

    i       to show these instructions
    q       to quit (and pickle bore)

    ---
    DEV:
    d1     load and plot HLW data
    d2     shoothill dev
    d3     Explore different RMSE fits to the data
    d4     Plot different combinations of Lag,HLW w/ rivers
    """


    ## Do the main program

    c = Controller()
