"""
COAsT add on with shoothill api wrapper

Created on 2021-11-04
@author: jelt

This package augements the COAsT package acting as a wrapper for the Shoothill
API. This does require a key to be setup. It is assumed that the key is
privately stored in
 config_keys.py

The shoothill API aggregates data across the country for a variety of instruments but,
 requiring a key, is trickier to set up than the EA API.

To discover the StationId for a particular measurement site check the
 integer id in the url or its twitter page having identified it via
  https://www.gaugemap.co.uk/#!Map
E.g  Liverpool (Gladstone Dock stationId="13482", which is read by default.

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
    from shoothill_api.shoothill_api import GAUGE
    liv = GAUGE()
    liv.dataset = liv.read_shoothill_to_xarray(ndays=5)
    liv.plot_timeseries()


To do:
    * logging doesn't work
"""

import coast
import datetime
import numpy as np
import xarray as xr

import logging
logging.basicConfig(filename='shoothill2.log', filemode='w+')
logging.getLogger().setLevel(logging.INFO)


#%% ################################################################################
class GAUGE(coast.Tidegauge):
    """ Inherit from COAsT. Add new methods """
    def __init__(self, ndays: int=5, startday: datetime=None, endday: datetime=None, station_id="7708"):
        try:
            import config_keys # Load secret keys
        except:
            logging.info('Need a Shoothil API Key. Use e.g. create_shoothill_key() having obtained a public key')

        #self.SessionHeaderId=config_keys.SHOOTHILL_KEY #'4b6...snip...a5ea'
        self.ndays=ndays
        self.startday=startday
        self.endday=endday
        self.station_id=station_id # Shoothill id
        self.dataset = None

        #self.dataset = self.read_shoothill_to_xarray(station_id="13482") # Liverpool

        pass

    def get_mean_crossing_time_as_xarray(self, date_start=None, date_end=None):
        """
        Get the height (constant) and times of crossing the mean height as xarray
        """
        pass

    def get_HW_to_xarray(self, date_start=None, date_end=None):
        """ Extract actual HW value and time as an xarray """
        pass


    def find_nearby_high_and_low_water(self, var_str, target_times:xr.DataArray=None, winsize:int=2, method='comp', extrema:str="both"):
        """
        Finds high and low water for a given variable, in close proximity to
        input xrray of times.
        Returns in a new Tidegauge object with similar data format to
        a TIDETABLE, and same size as target_times.

        winsize: +/- hours search radius
        target_times: xr.DataArray of target times to search around (e.g. harmonic predictions)
        var_str: root of var name for new variable.
        extrema (str):  "both". extract max and min (default)
                    :   "max". Extract only max
                    :   "min". Extract only min
        """

        #x = self.dataset.time
        #y = self.dataset[var_str]

        nt = len(target_times)

        if extrema == "min":
            time_min = np.zeros(nt)
            values_min = np.zeros(nt)
            for i in range(nt):
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                logging.debug(f"{i}: {coast.stats_util.find_maxima(HLW.time.values, HLW.values, method=method)}")
                time_min[i], values_min[i] = coast.stats_util.find_maxima(HLW.time.values, -HLW.values, method=method)

            new_dataset = xr.Dataset()
            new_dataset.attrs = self.dataset.attrs
            new_dataset[var_str + "_lows"]  = (var_str + "_lows", -values_min.data)
            new_dataset["time_lows"] = ("time_lows", time_min.data)

        elif extrema == "max":
            time_max = np.zeros(nt)
            values_max = np.zeros(nt)
            for i in range(nt):
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                logging.debug(f"{i}: {coast.stats_util.find_maxima(HLW.time.values, HLW.values, method=method)}")
                time_max[i], values_max[i] = coast.stats_util.find_maxima(HLW.time.values,  HLW.values, method=method)

            new_dataset = xr.Dataset()
            new_dataset.attrs = self.dataset.attrs
            new_dataset[var_str + "_highs"] = (var_str + "_highs", values_max.data)
            new_dataset["time_highs"] = ("time_highs", time_max.data)

        elif extrema == "both":
            time_max = np.zeros(nt)
            values_max = np.zeros(nt)
            time_min = np.zeros(nt)
            values_min = np.zeros(nt)
            for i in range(nt):
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                logging.debug(f"{i}: {coast.stats_util.find_maxima(HLW.time.values, HLW.values, method=method)}")
                time_max[i], values_max[i] = coast.stats_util.find_maxima(HLW.time.values,  HLW.values, method=method)
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                logging.debug(f"{i}: {coast.stats_util.find_maxima(HLW.time.values, HLW.values, method=method)}")
                time_min[i], values_min[i] = coast.stats_util.find_maxima(HLW.time.values, -HLW.values, method=method)

            new_dataset = xr.Dataset()
            new_dataset.attrs = self.dataset.attrs
            new_dataset[var_str + "_highs"] = (var_str + "_highs", values_max.data)
            new_dataset["time_highs"] = ("time_highs", time_max.data)
            new_dataset[var_str + "_lows"]  = (var_str + "_lows", -values_min.data)
            new_dataset["time_lows"] = ("time_lows", time_min.data)

        else:
            print("Not expecting that extrema case")
            pass


        #print(time_max)
        #print(values_max)

        new_object = coast.Tidegauge()
        new_object.dataset = new_dataset

        return new_object

############ shoothill gauge methods ##############################################
    @classmethod
    def read_shoothill_to_xarray(cls,
                                ndays: int=5,
                                date_start: np.datetime64=None,
                                date_end: np.datetime64=None,
                                station_id="7708",
                                dataType=3):
        """
        load gauge data via shoothill API
        Either loads last ndays, or from date_start:date_end

        This reqires an API key that is obtained by emailing shoothill.
        They provide a public key. Then SHOOTHILL_KEY can be generated using
        SHOOTHILL_KEY = create_shoothill_key(SHOOTHILL_PublicApiKey)

        To discover the station_id for a particular measurement site check the
         integer id in the url or its twitter page having identified it via
          https://www.gaugemap.co.uk/#!Map
         E.g  Liverpool (Gladstone Dock station_id="13482".
        Liverpool, or station_id="13482", is assumed by default.

        INPUTS:
            ndays : int
            date_start : datetime. UTC format string "yyyy-MM-ddThh:mm:ssZ" E.g 2020-01-05T08:20:01.5011423+00:00
            date_end : datetime
            station_id : str (station id)
            dataType: int (3 level, 15 flow)
        OUTPUT:
            sea_level, time : xr.Dataset
        """
        import requests,json

        try:
            import config_keys # Load secret keys
        except:
            logging.info('Need a Shoothil API Key. Use e.g. create_shoothill_key(SHOOTHILL_PublicApiKey) having obtained a public key')
            print('Expected a config_keys.py file of the form:')
            print('')
            print('# API keys excluded from github repo')
            print('SHOOTHILL_KEY = "4b6...5ea"')
            print('SHOOTHILL_PublicApiKey = "9a1...414"')

        cls.SessionHeaderId=config_keys.SHOOTHILL_KEY #'4b6...snip...a5ea'
        cls.ndays=ndays
        cls.date_start=date_start
        cls.date_end=date_end
        cls.station_id=station_id # Shoothill id
        cls.dataType=dataType

        logging.info("load gauge")

        if cls.station_id == "7708":
            id_ref = "Gladston Dock"
        elif cls.station_id == "7899":
            id_ref = "Chester weir"
        elif cls.station_id == "972":
            id_ref = "Farndon"
        elif cls.station_id == "968":
            id_ref = "Ironbridge (Dee)"
        else:
            id_ref = "No label"
            logging.debug(f"Not ready for that station id. {cls.station_id}")

        headers = {'content-type': 'application/json', 'SessionHeaderId': cls.SessionHeaderId}

        #%% Construct station info API request
        # Obtain and process header information
        logging.info("load station info")
        htmlcall_station_id = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesStationById/?stationId='
        url  = htmlcall_station_id+str(station_id)
        try:
            request_raw = requests.get(url, headers=headers)
            header_dict = json.loads(request_raw.content)
            # NB dataType is empty from the header request. Fill now
            header_dict['dataType'] = cls.dataType
            # convert attrs to str so that can be saved to netCDF
            header_dict['gaugeList'] = str(header_dict['gaugeList'])
            header_dict['additionalDataObject'] = str(header_dict['additionalDataObject'])
        except ValueError:
            print(f"Failed request for station {cls.station_id}")
            return

        # Assign expected header_dict information
        try: # header_dict['latitude'] and header_dict['longitude'] are present
            header_dict['site_name'] = header_dict['name']
            #header_dict['latitude'] = header_dict['items']['lat']
            #header_dict['longitude'] = header_dict['items']['long']
        except:
            logging.info(f"possible missing some header info: site_name,latitude,longitude")

        #%% Construct data API request
        if (cls.date_start == None) & (cls.date_end == None):
            logging.info(f"GETting ndays= {cls.ndays} of data")

            htmlcall_station_id = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesRecentDatapoints/?stationId='
            url  = htmlcall_station_id+str(cls.station_id)+'&dataType='+str(int(cls.dataType))+'&numberDays='+str(int(cls.ndays))
        else:
            # Check date_start and date_end are timetime objects
            if (type(cls.date_start) is np.datetime64) & (type(cls.date_end) is np.datetime64):
                logging.info(f"GETting data from {cls.date_start} to {cls.date_end}")
                startTime = cls.date_start.item().strftime('%Y-%m-%dT%H:%M:%SZ')
                endTime = cls.date_end.item().strftime('%Y-%m-%dT%H:%M:%SZ')

                htmlcall_station_id = 'http://riverlevelsapi.shoothill.com/TimeSeries/GetTimeSeriesDatapointsDateTime/?stationId='
                url   = htmlcall_station_id+str(cls.station_id)+'&dataType='+str(int(cls.dataType))+'&endTime='+endTime+'&startTime='+startTime

            else:
                logging.debug('Expecting date_start and date_end as datetime objects')

        #%% Get the data
        request_raw = requests.get(url, headers=headers)
        request = json.loads(request_raw.content)
        logging.debug(f"Shoothil API request: {request_raw.text}")
        # Check the output
        logging.info(f"Gauge id is {request['gauge']['geoEntityId']}")
        try:
            logging.info(f"timestamp and value of the zero index is {[ str(request['values'][0]['time']), request['values'][0]['value'] ]}")
        except:
            logging.info(f"timestamp and value of the zero index: problem")
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
        logging.debug(f"Shoothil API request headers: {header_dict}")
        try:
            logging.debug(f"Shoothil API request 1st time: {time[0]} and value: {sea_level[0]}")
        except:
            logging.debug(f"Shoothil API request 1st time: problem")
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
        import os, sys

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
