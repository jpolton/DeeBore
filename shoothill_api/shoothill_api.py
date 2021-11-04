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
            print('Need a Shoothil API Key. Use e.g. create_shoothill_key() having obtained a public key')

        #self.SessionHeaderId=config_keys.SHOOTHILL_KEY #'4b6...snip...a5ea'
        self.ndays=ndays
        self.startday=startday
        self.endday=endday
        self.station_id=station_id # Shoothill id

        #self.dataset = self.read_shoothill_to_xarray(station_id="13482") # Liverpool


        pass


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

        ## Construct data API request
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

        ## Get the data
        request_raw = requests.get(url, headers=headers)
        request = json.loads(request_raw.content)
        logging.debug(f"Shoothil API request: {request_raw.text}")

        # Check the output
        #logging.info(f"Gauge id is {request['gauge']['geoEntityId']}")
        #logging.info(f"timestamp and value of the zero index is {[ str(request['values'][0]['time']), request['values'][0]['value'] ]}")

        #print(request)
        ## Process header information
        #header_dict = request['gauge']
        #header_dict['site_name'] = id_ref

        ## Process timeseries data
        dataset = xr.Dataset()
        time = []
        sea_level = []
        nvals = len(request['values'])
        time = np.array([np.datetime64(request['values'][i]['time']) for i in range(nvals)])
        sea_level = np.array([request['values'][i]['value'] for i in range(nvals)])

        ## Assign arrays to Dataset
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['time'])
        dataset = dataset.assign_coords(time = ('time', time))
        dataset.attrs = header_dict
        #logging.debug(f"Shoothil API request headers: {header_dict}")
        #logging.debug(f"Shoothil API request 1st time: {time[0]} and value: {sea_level[0]}")

        # Assign local dataset to object-scope dataset
        return dataset
