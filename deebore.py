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

import scipy.signal # find_peaks

import logging
logging.basicConfig(filename='bore.log', filemode='w+')
logging.getLogger().setLevel(logging.DEBUG)


#%% ################################################################################
def create_shoothill_key():
    """ Create API key - Only do if you want a new API key
    (which has to be cut and pasted into the latter cells as `SessionHeaderId`)
    """
    api_url = 'http://riverlevelsapi.shoothill.com/ApiAccount/ApiLogin'
    PublicApiKey = config_keys.SHOOTHILL_PublicApiKey #e.g. '9a1...snip...5e414'
    ApiVersion = '2'
    postdata = { 'PublicApiKey': PublicApiKey, 'ApiVersion': ApiVersion}
    headers = {'content-type': 'application/json'}
    response = requests.post(api_url, data=json.dumps(postdata), headers=headers)
    print(response.text)
    return response.text['SessionHeaderId']

def find_maxima(x, y, method='comp', **kwargs):
    '''
    Finds maxima of a time series y. Returns maximum values of y (e.g heights)
    and corresponding values of x (e.g. times).
    **kwargs are dependent on method.

    Methods:
        'comp' :: Find maxima by comparison with neighbouring values.
                  Uses scipy.signal.find_peaks. **kwargs passed to this routine
                  will be passed to scipy.signal.find_peaks.
        DB NOTE: Currently only the 'comp' method is implemented. Future
                 methods include linear interpolation and cublic splines.
    '''

    if method == 'comp':
        peaks, props = scipy.signal.find_peaks(y, **kwargs)
        return x[peaks], y[peaks]

class GAUGE(TIDEGAUGE):
    """ Inherit from COAsT. Add new methods """
    def __init__(self, ndays: int=5, startday: datetime=None, endday: datetime=None, geoId=7708):
        try:
            import config_keys # Load secret keys
        except:
            logging.info('Need a Shoothil API Key. Use e.g. create_shoothill_key() having obtained a public key')

        self.SessionHeaderId=config_keys.SHOOTHILL_KEY #'4b6...snip...a5ea'
        self.ndays=ndays
        self.startday=startday
        self.endday=endday
        self.geoId=geoId # Shoothill id

        self.dataset = self.load_gauge_xr()
        pass

    def get_mean_crossing_time_as_xarray(self, date_start=None, date_end=None):
        """
        Get the height (constant) and times of crossing the mean height as xarray
        """
        pass

    def get_HW_to_xarray(self, date_start=None, date_end=None):
        """ Extract actual HW value and time as an xarray """
        pass

    def load_gauge_xr(self):
        """
        load EA gauge data
        Either loads last ndays, or from startday:endday

        INPUTS:
            ndays : int
            startday : datetime. UTC format string "yyyy-MM-ddThh:mm:ssZ" E.g 2020-01-05T08:20:01.5011423+00:00
            endday : datetime
            SessionHeaderId : string
            id : int (station id)
        OUTPUT:
            sea_level, time : xr.Dataset
        """
        import requests,json

        logging.info("load gauge")

        if self.geoId == 7708:
            id_ref = "Gladston Dock"
        elif self.geoId == 7899:
            id_ref = "Chester weir"
        else:
            logging.debug(f"Not ready for that station id. {self.geoId}")

        #%% Construct API request
        headers = {'content-type': 'application/json', 'SessionHeaderId': self.SessionHeaderId}

        if (self.startday == None) & (self.endday == None):
            logging.info(f"GETting ndays= {self.ndays} of data")

            htmlcall_stationId = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesRecentDatapoints/?stationId='
            url  = htmlcall_stationId+str(self.geoId)+'&dataType=3&numberDays='+str(int(self.ndays))
        else:
            # Check startday and endday are timetime objects
            if (type(self.startday) is datetime.datetime) & (type(self.endday) is datetime.datetime):
                logging.info(f"GETting data from {self.startday} to {self.endday}")
                startTime = self.startday.replace(tzinfo=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                endTime = self.endday.replace(tzinfo=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

                htmlcall_stationId = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesDatapointsDateTime/?stationId='
                url   = htmlcall_stationId+str(self.geoId)+'&dataType=3&endTime='+endTime+'&startTime='+startTime

            else:
                logging.debug('Expecting startday and endday as datetime objects')

        #%% Get the data
        request_raw = requests.get(url, headers=headers)
        request = json.loads(request_raw.content)
        logging.debug(f"Shoothil API request: {request_raw.text}")
        # Check the output
        logging.info(f"Gauge id is {request['gauge']['geoEntityId']}")
        logging.info(f"timestamp and value of the zero index is {[ str(request['values'][0]['time']), request['values'][0]['value'] ]}")

        #%% Process header information
        header_dict = request['gauge']
        header_dict['name'] = id_ref

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
        logging.debug(f"Shoothil API request headers: {header_dict}")
        logging.debug(f"Shoothil API request 1st time: {time[0]} and value: {sea_level[0]}")


        # Assign local dataset to object-scope dataset
        return dataset


    def find_high_and_low_water(self, var_str, method='comp'):
        '''
        Finds high and low water for a given variable.
        Returns in a new TIDEGAUGE object with similar data format to
        a TIDETABLE.

        Methods:
        'comp' :: Find maxima by comparison with neighbouring values.
                  Uses scipy.signal.find_peaks. **kwargs passed to this routine
                  will be passed to scipy.signal.find_peaks.
        DB NOTE: Currently only the 'comp' method is implemented. Future
                 methods include linear interpolation and cublic splines.
        '''

        x = self.dataset.time
        y = self.dataset[var_str]

        time_max, values_max = find_maxima(x, y, method=method)
        time_min, values_min = find_maxima(x,-y, method=method)
        values_min = -1*values_min

        new_dataset = xr.Dataset()
        new_dataset.attrs = self.dataset.attrs
        new_dataset[var_str + '_highs'] = ('time_highs', values_max)
        new_dataset[var_str + '_lows'] = ('time_lows', values_min)
        new_dataset['time_highs'] = ('time_highs', time_max)
        new_dataset['time_lows'] = ('time_lows', time_min)

        new_object = TIDEGAUGE()
        new_object.dataset = new_dataset

        return new_object


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
        new_dataset[var_str + '_highs'] = ('time_highs', values_max)

        new_object = TIDEGAUGE()
        new_object.dataset = new_dataset

        return new_object


############ BODC tide gauge methods ##############################################
    @classmethod
    def read_bodc_to_xarray(cls, fn_bodc, date_start=None, date_end=None):
        '''
        For reading from a single BODC (processed) file into an
        xarray dataset.
        If no data lies between the specified dates, a dataset is still created
        containing information on the tide gauge, but the time dimension will
        be empty.

        Data name: UK Tide Gauge Network, processed data.
        Source: https://www.bodc.ac.uk/
        See data notes from source for description of QC flags.

        The data takes the form:
            Port:              P234
            Site:              Liverpool, Gladstone Dock
            Latitude:          53.44969
            Longitude:         -3.01800
            Start Date:        01AUG2020-00.00.00
            End Date:          31AUG2020-23.45.00
            Contributor:       National Oceanography Centre, Liverpool
            Datum information: The data refer to Admiralty Chart Datum (ACD)
            Parameter code:    ASLVBG02 = Surface elevation (unspecified datum) of the water body by bubbler tide gauge (second sensor)
              Cycle    Date      Time    ASLVBG02   Residual
             Number yyyy mm dd hh mi ssf         f          f
                 1) 2020/08/01 00:00:00     5.354M     0.265M
                 2) 2020/08/01 00:15:00     5.016M     0.243M
                 3) 2020/08/01 00:30:00     4.704M     0.241M
                 4) 2020/08/01 00:45:00     4.418M     0.255M
                 5) 2020/08/01 01:00:00     4.133      0.257
                 ...

        Parameters
        ----------
        fn_bodc (str) : path to bodc tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data

        Returns
        -------
        xarray.Dataset object.
        '''
        logging.debug(f"Reading \"{fn_bodc}\" as a BODC file")  # TODO Maybe include start/end dates
        try:
            header_dict = cls.read_bodc_header(fn_bodc)
            dataset = cls.read_bodc_data(fn_bodc, date_start, date_end)
        except:
            raise Exception('Problem reading BODC file: ' + fn_bodc)
        # Attributes
        dataset['longitude'] = header_dict['longitude']
        dataset['latitude'] = header_dict['latitude']
        del header_dict['longitude']
        del header_dict['latitude']

        dataset.attrs = header_dict

        return dataset

    @staticmethod
    def read_bodc_header(fn_bodc):
        '''
        Reads header from a BODC file (format version 3.0).

        Parameters
        ----------
        fn_bodc (str) : path to bodc tide gauge file

        Returns
        -------
        dictionary of attributes
        '''
        logging.debug(f"Reading BODC header from \"{fn_bodc}\"")
        fid = open(fn_bodc)

        # Read lines one by one (hopefully formatting is consistent)
        # Geographical stuff
        header_dict = {}
        header = True
        for line in fid:
            if ':' in line and header == True:
                (key, val) = line.split(':')
                key = key.lower().strip().replace(' ','_')
                val = val.lower().strip().replace(' ','_')
                header_dict[key] = val
                logging.debug(f"Header key: {key} and value: {val}")
            else:
                #print('No colon')
                header = False
        logging.debug(f"Read done, close file \"{fn_bodc}\"")
        fid.close()

        header_dict['latitude'] = float( header_dict['latitude'] )
        header_dict['longitude'] = float( header_dict['longitude'] )

        return header_dict

    @staticmethod
    def read_bodc_data(fn_bodc, date_start=None, date_end=None,
                           header_length:int=11):
        '''
        Reads observation data from a BODC file.

        Parameters
        ----------
        fn_bodc (str) : path to bodc tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data
        header_length (int) : number of lines in header (to skip when reading)

        Returns
        -------
        xarray.Dataset containing times, sealevel and quality control flags
        '''
        # Initialise empty dataset and lists
        logging.debug(f"Reading BODC data from \"{fn_bodc}\"")
        dataset = xr.Dataset()
        time = []
        sea_level = []
        qc_flags = []
        residual = []
        # Open file and loop until EOF
        with open(fn_bodc) as file:
            line_count = 1
            for line in file:
                # Read all data. Date boundaries are set later.
                if line_count>header_length:
                    working_line = line.split()
                    time_str = working_line[1] + ' ' + working_line[2] # Empty lines cause trouble
                    sea_level_str = working_line[3]
                    residual_str = working_line[4]
                    if sea_level_str[-1].isalpha():
                        qc_flag_str = sea_level_str[-1]
                        sea_level_str = sea_level_str.replace(qc_flag_str,'')
                        residual_str = residual_str.replace(qc_flag_str,'')
                    else:
                        qc_flag_str = ''
                    print(working_line, sea_level_str, qc_flag_str)
                    time.append(time_str)
                    qc_flags.append(qc_flag_str)
                    sea_level.append(float(sea_level_str))
                    residual.append(float(residual_str))

                line_count = line_count + 1
            logging.debug(f"Read done, close file \"{fn_bodc}\"")

        # Convert time list to datetimes using pandas
        time = np.array(pd.to_datetime(time))

        # Return only values between stated dates
        start_index = 0
        end_index = len(time)
        if date_start is not None:
            date_start = np.datetime64(date_start)
            start_index = np.argmax(time>=date_start)
        if date_end is not None:
            date_end = np.datetime64(date_end)
            end_index = np.argmax(time>date_end)
        time = time[start_index:end_index]
        sea_level = sea_level[start_index:end_index]
        qc_flags=qc_flags[start_index:end_index]

        # Set null values to nan
        sea_level = np.array(sea_level)
        qc_flags = np.array(qc_flags)
        #sea_level[qc_flags==5] = np.nan

        # Assign arrays to Dataset
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['time'])
        dataset['qc_flags'] = xr.DataArray(qc_flags, dims=['time'])
        dataset = dataset.assign_coords(time = ('time', time))

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
        self.load_flag = False
        self.run_interface()


    def load_databucket(self):
        """
        Auto load databucket from pickle file if it exists, otherwise create it
        If databucket is loaded. Also perform linear fit to data (couldn't pickle it into bore:xr.DataArray)
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
                    print('Calculating linear fit')
                    self.linearfit( self.bore.glad_height, self.bore.Saltney_lag )
            else:
                print("... %s does not exist"%DATABUCKET_FILE)
        except KeyError:
            print('ErrorA ')
        except (IOError, RuntimeError):
            print('ErrorB ')


    def pickle_bore(self):
        """ save copy of self.bore into pickle file """
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
                self.pickle_bore()
                break

            elif command == "i":
                print(INSTRUCTIONS)

            elif command == "1":
                # Load and plot raw data
                print('load and process bore dataset')
                self.load_and_process()

            elif command == "2":
                print('show bore dataset')
                self.show()

            elif command == "3":
                print('plot bore data (lag vs tidal height')
                self.plot_lag_vs_height()

            elif command == "d1":
                print('load and plot HLW data')
                self.load_and_plot_HLW_data()

            elif command == "d2":
                print("shoothill dev")
                self.shoothill()

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

    def load_and_process(self):
        """
        Performs sequential steps to build the bore object.
        1. Load bore data (Essential elements are obs times and locations)
        2. Load Gladstone Dock data (though this might also be loaded from the obs logs)
        3. Calculate the time lag between Gladstone and Saltney events.
        4. Perform a linear fit to the time lag.
        """
        self.load_csv()
        print('loading tide data')
        self.get_Glad_data()
        #self.compare_Glad_HLW()
        print('Calculating the Gladstone to Saltney time difference')
        self.calc_Glad_Saltney_time_diff()
        print('Calculating linear fit')
        self.linearfit( self.bore.glad_height, self.bore.Saltney_lag )


    def load_csv(self):
        """
        Load observed bore data from text file.
        Load as a dataframe and save to bore:xr.DataSet
        """
        logging.info('Load bore data from csv file')
        self.load_flag = True
        df =  pd.read_csv('data/master-Table 1.csv')
        df.drop(columns=['date + logged time','Unnamed: 2','Unnamed: 11', \
                                'Unnamed: 12','Unnamed: 13', 'Unnamed: 15'], \
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
        self.bore = bore
        logging.info('Bore data loaded')


    def get_Glad_data(self):
        """
        Get Gladstone HLW data from external
        These data are reported in the bore.csv file but not consistently and it
        is laborous to find old values.
        It was considered a good idea to automate this step.
        """
        logging.info("Get Gladstone HLW data from external file")
        HT_h = []
        HT_t = []
        # load tidetable
        option = None
        while (option != 1) and (option != 2):
            try:
                option = int(input('Load tide table (1) or measured data (2)?'))
            except:
                logging.debug(f"Expected an integer, got {option}")
        if option == 1: # Load tidetable data from files
            filnam1 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2005_2014_HLW.txt'
            filnam2 = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2015_2020_HLW.txt'
            tg  = TIDEGAUGE()
            tg1 = TIDEGAUGE()
            tg2 = TIDEGAUGE()
            tg1.dataset = tg1.read_HLW_to_xarray(filnam1)#, self.bore.time.min().values, self.bore.time.max().values)
            tg2.dataset = tg2.read_HLW_to_xarray(filnam2)#, self.bore.time.min().values, self.bore.time.max().values)
            tg.dataset = xr.concat([ tg1.dataset, tg2.dataset], dim='t_dim')


            self.tg = tg
            for i in range(len(self.bore.time)):
                try:
                    HW = None
                    #HLW = tg.get_tidetabletimes(self.bore.time[i].values)
                    HW = tg.get_tidetabletimes( self.bore.time[i].values, method='nearest_HW' )
                    #print(f"HLW: {HLW}")
                    HT_h.append( HW.values )
                    #print('len(HT_h)', len(HT_h))
                    HT_t.append( HW.time.values )
                    #print('len(HT_t)', len(HT_t))
                    #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
                    #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
                except:
                    logging.warning('Issue with appening HLW data')

        elif option == 2: # Load measured height from files


            # Load and plot BODC processed data
            fn_bodc = '/Users/jeff/GitHub/DeeBore/data/LIV2008.txt'

            # Set the start and end dates
            date_start = np.datetime64('2020-08-12 23:59')
            date_end = np.datetime64('2020-08-14 00:01')

            # Initiate a TIDEGAUGE object, if a filename is passed it assumes it is a GESLA type object
            tg = GAUGE()
            # specify the data read as a High Low Water dataset
            tg.dataset = tg.read_bodc_to_xarray(fn_bodc)#, date_start, date_end)

            self.tg = tg
            for i in range(len(self.bore.time)):
                try:
                    HW = None
                    #HLW = tg.get_tidetabletimes(self.bore.time[i].values)
                    HW = tg.find_nearby_high_and_low_water(var_str='sea_level', target_times=self.bore.time[i].values, method='comp')

                    #print(f"HLW: {HLW}")
                    HT_h.append( HW.values )
                    #print('len(HT_h)', len(HT_h))
                    HT_t.append( HW.time.values )
                    #print('len(HT_t)', len(HT_t))
                    #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
                    #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
                except:
                    logging.warning('Issue with appening HLW data')


        else:
            logging.debug(f"Did not expect this eventuality...")




        # Save a xarray objects
        coords = {'time': (('t_dim'), self.bore.time.values)}
        self.bore['glad_height'] = xr.DataArray( np.array(HT_h), coords=coords, dims=['t_dim'])
        self.bore['glad_time'] = xr.DataArray( np.array(HT_t), coords=coords, dims=['t_dim'])

        #self.bore['glad_height'] = np.array(HT_h)
        #self.bore['glad_time'] = np.array(HT_t)
        print('There is a supressed plot.scatter here')
        #self.bore.plot.scatter(x='glad_time', y='glad_height'); plt.show()

        logging.debug(f"len(self.bore.glad_time): {len(self.bore.glad_time)}")
        #logging.info(f'len(self.bore.glad_time)', len(self.bore.glad_time))
        logging.debug(f"type(HT_t): {type(HT_t)}")
        logging.debug(f"type(HT_h): {type(HT_h)}")

        logging.debug('log time, orig tide table, new tide table lookup')
        for i in range(len(self.bore.time)):
            logging.debug( f"{self.bore.time[i].values}, {self.bore['Liv (Gladstone Dock) HT time (GMT)'][i].values}, {self.bore['glad_time'][i].values}")

        #print('log time, orig tide table, new tide table lookup')
        #for i in range(len(self.bore.time)):
        #    print( self.bore.time[i].values, self.bore['Liv (Gladstone Dock) HT time (GMT)'][i].values, self.bore['glad_time'][i].values)


    def calc_Glad_Saltney_time_diff(self):
        """
        Compute lag (-ve) for arrival at Saltney relative to Glastone HT
        Store lags as integer (minutes). Messing with np.datetime64 and
        np.timedelta64 is problematic with polyfitting.
        """
        logging.info('calc_Glad_Saltney_time_diff')
        nt = len(self.bore.time)
        lag = (self.bore['glad_time'].values - self.bore['time'].values).astype('timedelta64[m]')
        Saltney_lag    = [ lag[i].astype('int') if self.bore.location.values[i] == 'bridge' else np.NaN for i in range(nt) ]
        bluebridge_lag = [ lag[i].astype('int') if self.bore.location.values[i] == 'blue bridge' else np.NaN for i in range(nt) ]

        # Save a xarray objects
        coords = {'time': (('t_dim'), self.bore.time.values)}
        self.bore['lag'] = xr.DataArray( lag, coords=coords, dims=['t_dim'])
        self.bore['Saltney_lag'] = xr.DataArray( Saltney_lag, coords=coords, dims=['t_dim'])
        self.bore['bluebridge_lag'] = xr.DataArray( bluebridge_lag, coords=coords, dims=['t_dim'])


    def linearfit(self, X, Y):
        """
        Linear regression
        Is used if pickle file is loaded to get the fit between
            self.bore.glad_height and self.bore.Saltney_lag
        Is used after computing the lag between Gladstone and Saltney events,
            during load_and_process(), to find a fit between Liverpool heights
            and Saltney arrival lag.
        """
        idx = np.isfinite(Y).values
        weights = np.polyfit( X[idx], Y[idx], 1)
        logging.debug("weights: {weights}")
        self.linfit = np.poly1d(weights)
        self.bore['linfit_lag'] =  self.linfit(X)

    ############################################################################
    #%% Presenting data
    ############################################################################

    def show(self):
        """ Show xarray dataset """
        print( self.bore )


    def plot_lag_vs_height(self):
        """
        Plot bore lag (as time difference before Gladstone HW) against
        Gladstone high water (m).
        Separate colours for Saltney, Bluebridge, Chester.
        """
        Xglad = self.bore.glad_height
        Ysalt = self.bore.Saltney_lag
        Yblue = self.bore.bluebridge_lag
        Yfit = self.bore.linfit_lag
        plt.plot( Xglad,Ysalt, 'r+', label='Saltney')
        plt.plot( Xglad,Yblue, 'b.', label='Bluebridge')
        plt.plot( Xglad,Yfit, 'k-')

        plt.xlabel('Liv (Gladstone Dock) HT (m)')
        plt.ylabel('Arrival time (mins before LiV HT)')
        plt.title('Bore arrival time at Saltney Ferry')
        plt.legend()
        #plt.show()
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight.png')

        if(0):
            #plt.show()

            s = plt.scatter( self.bore['glad_height'], \
                self.bore['Saltney_lag']) #, \
                #c=self.bore['Chester Weir height: CHESTER WEIR 15 MIN SG'] )
            cbar = plt.colorbar(s)
            # Linear fit
            #x = self.df['Liv (Gladstone Dock) HT height (m)']
            #plt.plot( x, self.df['linfit_lag'], '-' )
            cbar.set_label('River height at weir (m)')
            plt.title('Bore arrival time at Saltney Ferry')
            plt.ylabel('Arrival time (mins before Liv HT)')
            plt.xlabel('Liv (Gladstone Dock) HT height (m)')
            plt.show()


    ############################################################################
    #%% DIAGNOSTICS
    ############################################################################

    def predict_bore(self):
        """
        Glad_HW - float
        Glad_time - datetime64
        Saltney_time - datetime64
        Saltney_lag - int

        Predict the bore timing at Saltney for a input date
        Parameters
        ----------
        day : day
         DESCRIPTION.

        Returns
        -------
        Glad_HW - float
        Glad_time - datetime64
        Saltney_time - datetime64
        Saltney_lag - int

        """
        print('Predict bore event for date')
        filnam = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2015_2020_HLW.txt'

        nd = input('Make predictions for N days from hence (int):?')
        day = np.datetime64('now', 'D') + np.timedelta64(int(nd), 'D')
        dayp1 = day + np.timedelta64(24, 'h')
        tg = TIDEGAUGE()
        tg.dataset = tg.read_HLW_to_xarray(filnam, day, dayp1)
        HT = tg.dataset['sea_level'].where(tg.dataset['sea_level']\
                                    .values > 7).dropna('t_dim') #, drop=True)

        #plt.plot( HT.time, HT,'.' );plt.show()
        lag_pred = self.linfit(HT)
        #lag_pred = lag_pred[np.isfinite(lag_pred)] # drop nans

        Saltney_time_pred = [HT.time[i].values
                             - np.timedelta64(int(round(lag_pred[i])), 'm')
                             for i in range(len(lag_pred))]
        # Iterate over high tide events to print useful information
        for i in range(len(lag_pred)):
            #print( "Gladstone HT", np.datetime_as_string(HT.time[i], unit='m',timezone=pytz.timezone('UTC')),"(GMT). Height: {:.2f} m".format(  HT.values[i]))
            #print(" Saltney arrival", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')),"(GMT/BST). Lag: {:.0f} mins".format( lag_pred[i] ))
            print("Predictions for ", dayoweek(Saltney_time_pred[i]), Saltney_time_pred[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y/%m/%d') )
            print("Saltney FB:", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')) )
            Glad_HLW = tg.get_tidetabletimes( Saltney_time_pred[i], method='nearest_2' )
            # Extract the High Tide value
            print('Liv HT:    ', np.datetime_as_string(Glad_HLW[ np.argmax(Glad_HLW) ].time.values, unit='m', timezone=pytz.timezone('Europe/London')), Glad_HLW[ np.argmax(Glad_HLW) ].values, 'm' )
            # Extract the Low Tide value
            print('Liv LT:    ', np.datetime_as_string(Glad_HLW[ np.argmin(Glad_HLW) ].time.values, unit='m', timezone=pytz.timezone('Europe/London')), Glad_HLW[ np.argmin(Glad_HLW) ].values, 'm' )
            print("")

        #plt.scatter( Saltney_time_pred, HT ,'.');plt.show()
        # problem with time stamp

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
        date_start = datetime.datetime(2020, 9, 1)
        date_end = datetime.datetime(2020, 9, 30)
        sg = GAUGE(startday=date_start, endday=date_end) # create modified TIDEGAUGE object
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

        sg = GAUGE(startday=date_start, endday=date_end) # create modified TIDEGAUGE object
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
    1       load and process bore dataset
    2       show bore dataset
    3       plot bore data (lag vs tidal height)

    6       Predict bore event for date

    x       Export data to csv. NOT IMPLEMENTED
    r       Remove pickle file

    i       to show these instructions
    q       to quit (and pickle bore)

    ---
    DEV:
    d1     load and plot HLW data
    d2     shoothill dev
    """


    ## Do the main program



    c = Controller()
