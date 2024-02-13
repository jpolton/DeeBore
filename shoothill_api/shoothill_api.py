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
# Either local use of COAsT: pip install .
import os, sys
from socket import gethostname

#if "LJOB" in gethostname().upper():
coastdir = os.path.dirname(os.environ["HOME"]+'/GitHub/COAsT/coast')
#elif "LIVMAZ" in gethostname().upper():  # Debugging on local machine
#    coastdir = os.path.dirname('/Users/jelt/GitHub/COAsT/coast')
#else:
#    print(f"Unfamiliar machine: {gethostname()}")
sys.path.insert(0, coastdir)
# Or a just pre-built
import coast
import datetime
import numpy as np
import xarray as xr
import scipy

import logging
logging.basicConfig(filename='shoothill2.log', filemode='w+')
logging.getLogger().setLevel(logging.INFO)

#%% ################################################################################

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


#%% ################################################################################
class GAUGE(coast.Tidegauge):
    """ Inherit from COAsT. Add new methods """
    #def __init__(self, ndays: int=5, startday: datetime=None, endday: datetime=None, station_id="7708"):
    def __init__(self, dataset=None):
        try:
            import config_keys # Load secret keys
        except:
            logging.info('Need a Shoothil API Key. Use e.g. create_shoothill_key() having obtained a public key')

        if(0):
            self.ndays=ndays
            self.startday=startday
            self.endday=endday
            self.station_id=station_id # Shoothill id
            self.dataset = None

        if dataset is not None:
            self.dataset = dataset

        #self.dataset = self.read_shoothill_to_xarray(station_id="13482") # Liverpool

        #self.dataset = None
        #self.chunks = None
        #self.var_mapping = None
        #self.dim_mapping = None
        #self.coord_vars = None
        #self.keep_all_vars = False

        pass

    def get_mean_crossing_time_as_xarray(self, date_start=None, date_end=None):
        """
        Get the height (constant) and times of crossing the mean height as xarray
        """
        pass

    def get_HW_to_xarray(self, date_start=None, date_end=None):
        """ Extract actual HW value and time as an xarray """
        pass


    def find_nearby_high_and_low_water(self, var_str, target_times:xr.DataArray=None, winsize:int=2, method:str='comp', extrema:str="both"):
        """

        WORK IN PROGRESS

        Finds high and low water for a given variable, in close proximity to
        input xarray of times.
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
        print(f"METHOD:{method}")
        if extrema == "min":
            time_min = np.array([np.datetime64('NaT') for i in range(nt)], dtype="datetime64[s]")
            values_min = np.zeros(nt)
            for i in range(nt):
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                if HLW.size >= 3: # extrema methods need 3pts
                    logging.debug(f"{i}: {coast._utils.stats_util.find_maxima(HLW.time, -HLW, method=method)}")
                    tmp_output = coast._utils.stats_util.find_maxima(HLW.time, -HLW, method=method)
                    try:
                        time_min[i], values_min[i] = tmp_output[0][0].values, -tmp_output[1][0].values
                    except:
                        time_min[i] = target_times[i].values
                        values_min[i] = np.NaN
                else:
                    try:
                        time_min[i] = HLW.time.where(HLW == np.min(HLW.values), drop=True).values[0]
                        values_min[i] = HLW.where(HLW == np.min(HLW.values), drop=True).values[0]
                    except: # catch exception if no values found in range
                        time_min[i] = target_times[i].values
                        values_min[i] = np.NaN

            new_dataset = xr.Dataset()
            new_dataset.attrs = self.dataset.attrs
            time = np.array(time_min)
            sea_level = np.array(values_min)
            new_dataset[var_str + "_lows"] = xr.DataArray(sea_level, dims='t_dim')
            new_dataset = new_dataset.assign_coords(time_lows=('t_dim', time))

        elif extrema == "max":
            time_max = np.array([np.datetime64('NaT') for i in range(nt)], dtype="datetime64[s]")
            values_max = np.zeros(nt)
            for i in range(nt):
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                if HLW.size >= 3:  # extrema methods need 3pts
                    logging.debug(f"{i}: {coast._utils.stats_util.find_maxima(HLW.time, HLW, method=method)}")
                    tmp_output = coast._utils.stats_util.find_maxima(HLW.time,  HLW, method=method)
                    try:
                        time_max[i], values_max[i] = tmp_output[0][0].values, tmp_output[1][0].values
                    except:
                        time_max[i] = target_times[i].values
                        values_max[i] = np.NaN
                else:
                    try:
                        time_max[i] = HLW.time.where(HLW == np.max(HLW.values), drop=True).values[0]
                        values_max[i] = HLW.where(HLW == np.max(HLW.values), drop=True).values[0]
                    except: # catch exception if no values found in range
                        time_max[i] = target_times[i].values
                        values_max[i] = np.NaN

            new_dataset = xr.Dataset()
            new_dataset.attrs = self.dataset.attrs
            time = np.array(time_max)
            sea_level = np.array(values_max)
            new_dataset[var_str + "_highs"] = xr.DataArray(sea_level, dims='t_dim')
            new_dataset = new_dataset.assign_coords(time_highs=('t_dim', time))

        elif extrema == "both":
            time_max = np.array([np.datetime64('NaT') for i in range(nt)], dtype="datetime64[s]")
            values_max = np.zeros(nt)
            time_min = np.array([np.datetime64('NaT') for i in range(nt)], dtype="datetime64[s]")
            values_min = np.zeros(nt)
            for i in range(nt):
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                if HLW.size > 1:
                    logging.debug(f"{i}: {coast._utils.stats_util.find_maxima(HLW.time.values, HLW.values, method=method)}")
                    tmp_output = coast._utils.stats_util.find_maxima(HLW.time.values,  HLW.values, method=method)
                    time_max[i], values_max[i] = tmp_output[0][0], tmp_output[1][0]
                else:
                    time_max[i], values_max[i] = HLW.time.values, HLW.values
                HLW = self.get_tide_table_times( time_guess=target_times[i].values, measure_var=var_str, method='window', winsize=winsize )
                if HLW.size > 1:
                    logging.debug(f"{i}: {coast._utils.stats_util.find_maxima(HLW.time.values, -HLW.values, method=method)}")
                    tmp_output = coast._utils.stats_util.find_maxima(HLW.time.values, -HLW.values, method=method)
                    time_min[i], values_min[i] = tmp_output[0][0], tmp_output[1][0]
                else:
                    time_min[i], values_min[i] = HLW.time.values, -HLW.values


            new_dataset = xr.Dataset()
            new_dataset.attrs = self.dataset.attrs
            ##new_dataset[var_str + "_highs"] = (var_str + "_highs", values_max.data)
            ##new_dataset["time_highs"] = ("time_highs", time_max.data)
            ##new_dataset[var_str + "_lows"]  = (var_str + "_lows", [-i for i in values_min.data])
            ##new_dataset["time_lows"] = ("time_lows", time_min.data)
            #new_dataset[var_str + "_highs"] = (var_str + "_highs", values_max)
            #new_dataset["time_highs"] = ("time_highs", time_max)
            #new_dataset = new_dataset.swap_dims({"time_highs":"t_dim"})
            #new_dataset[var_str + "_lows"]  = (var_str + "_lows", [-i for i in values_min])
            #new_dataset["time_lows"] = ("time_lows", time_min)
            #new_dataset = new_dataset.swap_dims({"time_lows":"t_dim"})

            ##
            time = np.array(time_max)
            sea_level = np.array(values_max.data)
            new_dataset[var_str + "_highs"] = xr.DataArray(sea_level, dims='t_dim')
            new_dataset = new_dataset.assign_coords(time_highs=('t_dim', time))

            time = np.array(time_min)
            sea_level = np.array(-values_min.data)
            new_dataset[var_str + "_lows"] = xr.DataArray(sea_level, dims='t_dim')
            new_dataset = new_dataset.assign_coords(time_lows=('t_dim', time))

        else:
            print("Not expecting that extrema case")
            pass


        #print(time_max)
        #print(values_max)

        new_object = coast.Tidegauge()
        new_object.dataset = new_dataset

        return new_object

    def find_flood_and_ebb_water(self, var_str, method="comp", **kwargs):
        """
        Finds the time and values for the inflection points (between high and
        low water) for a given variable. These correspond to the max flood and
        ebb points.
        Returns in a new TIDEGAUGE object with similar data format to
        a TIDETABLE.

        Apply rolling smoother to iron out kinks - only interested in the
        steepest, near linear, part of the timeseries.
        The the derivative is taken (2nd order accurate central difference).
        Then maxima/minima of the derivatives are then found and returned.

        Methods:
        'comp' :: Find inflection by comparison with neighbouring values.
                  Uses scipy.signal.find_peaks. **kwargs passed to this routine
                  will be passed to scipy.signal.find_peaks.
        'cubic':: Find the inflections using the roots of cubic spline.
                  Uses scipy.interpolate.InterpolatedUnivariateSpline
                  and scipy.signal.argrelmax. **kwargs are not activated.
        NOTE: Currently only the 'comp' and 'cubic' methods implemented. Future
                  methods include linear interpolation or refinements.

        See also:
            coast.Tidegauge.find_high_and_low_water()

        Example:
        import coast
        liv= xr.open_mfdataset("archive_shoothill/liv_2021.nc")
        obs_time = np.datetime64('2021-11-01')
        winsize = 6
        win = GAUGE()
        win.dataset = liv.sel( time=slice(obs_time - np.timedelta64(winsize, "h"), obs_time + np.timedelta64(winsize, "h"))  )
        y = win.dataset.sea_level.compute()
        x = win.dataset.time.compute()
        f = y.differentiate("time")
        time_max, values_max = coast.stats_util.find_maxima(x, f, method="comp")
        interp = y.interp(time=time_max)
        plt.plot( win.dataset.time, win.dataset.sea_level); plt.plot(interp.time, interp,'+'); plt.show()
        """
        y = self.dataset[var_str].rolling(t_dim=3, center=True).mean() # Rolling smoothing. Note we are only interested in the steep bit when it is near linear.
        try:
            f = y.differentiate("time")
        except ValueError: # There is a problem if there are chunk sizes of 1. Rechunk
            if np.array(y.chunks[0]).min() == 1:   # assuming y.chunks[0] == (47,1) or similar
                y = y.chunk(chunks={"time":len(y)})
                f = y.differentiate("time")
            else:
                print(f"Problem with differentiate over time")
        x = self.dataset.time

        if(0):
            # Convert x to float64 (assuming f is/similar to np.float64)
            if type(x.values[0]) == np.datetime64:  # convert to decimal sec since 1970
                x_float = ((x.values - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")).astype("float64")
                # x_float = x.values.astype('float64')
                f_float = f.values.astype("float64")
                flag_dt64 = True
            else:
                x_float = x.values.astype("float64")
                f_float = f.values.astype("float64")
                flag_dt64 = False

            if type(f.values[0]) != np.float64:
                print("find_maxima(): type(f)=", type(f))
                print("I was expecting a np.float64")

            ## Fit cubic spline
            #f_smooth = scipy.interpolate.InterpolatedUnivariateSpline(x_float, f_float, k=5)

            #x = np.linspace(0,2*np.pi,100)
            #y = np.sin(x) + np.random.random(100) * 0.8
            #plot(x, y,'o')
            #plot(x, smooth(y,3), 'r-', lw=2)
            #plot(x, smooth(y,19), 'g-', lw=2)
            #f_smooth = smooth(f_float,5)
            #f_smooth = smooth(y,5)

            ## FROM STATS_UTIL.PY
            # Convert back to datetime64 if appropriate (to nearest second)
            if flag_dt64:
                N = len(x_float)
                x_out = [
                    np.datetime64("1970-01-01T00:00:00") + np.timedelta64(int(x_float[i]), "s") for i in range(N)
                ]
            else:
                x_out = x_float

        # Package into xr.DataSet for find_maxima method
        #f_ds = xr.Dataset({"sea_level_dt": ("time", f_smooth(x_float))}, coords={"time": x_out})
        #f_ds = xr.Dataset({"sea_level_dt": ("time", f)}, coords={"time": x_out})

        time_max, values_max = coast._utils.stats_util.find_maxima(x, f, method=method, **kwargs)
        time_min, values_min = coast._utils.stats_util.find_maxima(x, -f, method=method, **kwargs)
        #time_max, values_max = coast.stats_util.find_maxima(f_ds.time, f_ds.sea_level_dt, method=method, **kwargs)
        #time_min, values_min = coast.stats_util.find_maxima(f_ds.time, -f_ds.sea_level_dt, method=method, **kwargs)

        #print(f"values_max {values_max.data}")
        #print(f"test  {self.dataset.sea_level.interp(time=[np.datetime64('2021-11-12T17:40')])}")
        #print(f"time_max {time_max}")
        #print(f"time_max {values_max.time}")
        #print(f"test2  {y.interp(time=[np.datetime64('2021-11-12T17:40')])}")

        #print(f"interpolated: {y.interp(time=time_max)}")

        #inflection_flood = y.interp(time=time_max[values_max.values.argmax()])
        #inflection_ebb = y.interp(time=time_min[values_min.values.argmax()])

        ## Extract the large value (i.e. corresponding to steepest sea level)
        inflection_flood = y.swap_dims({"t_dim": "time"}).interp(time=time_max.swap_dims({"t_dim": "time"}),
                                              kwargs={"fill_value": "extrapolate"})[values_max.values.argmax()].rename({var_str + "_flood"})

        #inflection_flood = y.swap_dims({"t_dim":"time"}).where( values_max == values_max.max(), drop=True ) \
        #                                                .interp(time=time_max \
        #                                                        , kwargs={"fill_value": "extrapolate"})
        inflection_flood = xr.DataArray([inflection_flood.values], dims=("t_dim"), coords={"t_dim": inflection_flood.time}).rename(var_str + "_flood")

        inflection_ebb = (y.swap_dims({"t_dim": "time"}).interp(time=time_min.swap_dims({"t_dim": "time"}),
                                              kwargs={"fill_value": "extrapolate"})[values_min.values.argmax()]
                                                .rename({var_str + "_ebb"}))  # argMAX because values_min is computed by finding MAX(-f)

        inflection_ebb = xr.DataArray([inflection_ebb.values], dims="t_dim", coords={"t_dim": inflection_ebb.time}).rename(var_str + "_ebb")

        #inflection_ebb =  (y.swap_dims({"t_dim":"time"}).where( values_min == values_min.max(), drop=True ) \
        #                                                .interp(time=time_min \
        #                                                      , kwargs={"fill_value": "extrapolate"}))

        #inflection_flood = y.interp(time=time_max)
        #inflection_ebb = y.interp(time=time_min)

        #print(f"interpolated2: {y.interp(time=time_max.data)}")

        new_dataset = xr.Dataset()
        new_dataset.attrs = self.dataset.attrs
        #new_dataset[var_str + "_flood"] = ("time_flood", inflection_flood.values)
        #new_dataset[var_str + "_ebb"]  = ("time_ebb",  inflection_ebb.values)
        #new_dataset["time_flood"] = ("time_flood", inflection_flood.time.values)
        #new_dataset["time_ebb"]  =  ("time_ebb", inflection_ebb.time.values)

        if(0):
            new_dataset = xr.Dataset()
            new_dataset.attrs = self.dataset.attrs
            time = np.array(time_min)
            sea_level = np.array(values_min)
            new_dataset[var_str + "_lows"] = xr.DataArray(sea_level, dims='t_dim')
            new_dataset = new_dataset.assign_coords(time_lows=('t_dim', time))



        new_dataset[var_str + "_flood"] = inflection_flood
        new_dataset[var_str + "_ebb"]   = inflection_ebb


        #new_dataset[var_str + "_flood"] = xr.DataArray([sea_level], dims='t_dim')
        #new_dataset["time_flood"] = xr.DataArray([time], dims='t_dim')
        #new_dataset = new_dataset.assign_coords(time_flood=('t_dim', [time]))

        time = np.array(inflection_ebb.time.values)
        sea_level = np.array(inflection_ebb.data)



        #new_dataset[var_str + "_ebb"] = xr.DataArray([sea_level], dims='t_dim')
        #new_dataset["time_ebb"] = xr.DataArray([time], dims='t_dim')
        #new_dataset = new_dataset.assign_coords(time_ebb=('t_dim', [time]))


        new_object = GAUGE()
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
        elif cls.station_id == "7522":
            id_ref = "Sheerness"
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

        anytidedir = os.path.dirname(os.environ["HOME"]+'/GitHub/anyTide/')
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


############ NOC Innovation gauge methods ##############################################
    @classmethod
    def read_nocinnov_to_xarray(cls,
                                ndays: int=5,
                                date_start: np.datetime64=None,
                                date_end: np.datetime64=None,
                                station_id="Liverpool%20(Gladstone%20Dock)",
                                ):
        """
        load gauge data via NOC Innovation API
        Either loads last ndays, or from date_start:date_end

        This reqires an API key that is obtained by emailing NOC Innovation.

        Check the API documentation:
        https://apps.noc-innovations.co.uk/docs/polpred-api/_contents/available-functions.html

        Check for data availability, port names (here station_id) etc using the portal:
        https://apps.noc-innovations.co.uk/

        for Liverpool Gladstone Dock: station_id="Liverpool%20(Gladstone%20Dock)", is assumed by default.

        INPUTS:
            ndays : int
            date_start : datetime. UTC format string "yyyy-MM-ddThh:mm:ssZ" E.g 2023-12-08T23:59:59Z
            date_end : datetime
            station_id : str (station id)
        OUTPUT:
            sea_level, time : xr.Dataset

        https://apps.noc-innovations.co.uk/api/polpred-api/latest/compute-port?key=ArDVr8ntwFyq5x42&port_name=Liverpool%20(Gladstone%20Dock)&dt_start=2023-12-08T00:00:00Z&dt_end=2023-12-08T23:59:59Z&interval=60
        """
        import requests,json

        try:
            import config_keys # Load secret keys
        except:
            logging.info('Need a NOC API Key. Email to get one')
            print('Expected a config_keys.py file of the form:')
            print('')
            print('# API keys excluded from github repo')
            print('NOC_KEY = "ArD...x42"')

        cls.SessionHeaderId=config_keys.NOC_KEY #'ArD...snip...x42'
        cls.ndays=ndays
        cls.date_start=date_start
        cls.date_end=date_end
        cls.station_id=station_id # Shoothill id

        logging.info("load gauge")

        if cls.station_id == "Liverpool%20(Gladstone%20Dock)":
            id_ref = "Gladstone Dock"
            datum_shift = 5.272  # taken from harm_class_A.txt
        elif cls.station_id == "Sheerness":
            id_ref = "Sheerness"
            datum_shift = 3.047 # taken from harm_class_A.txt
        else:
            id_ref = "No label"
            logging.debug(f"Not ready for that station id. {cls.station_id}")

        if (type(cls.date_start) is np.datetime64) & (type(cls.date_end) is np.datetime64):
            logging.info(f"GETting data from {cls.date_start} to {cls.date_end}")
            startTime = cls.date_start.item().strftime('%Y-%m-%dT%H:%M:%SZ')
            endTime = cls.date_end.item().strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            startTime = str(cls.date_start)
            endTime = str(cls.date_end)

        headers = {'content-type': 'application/json'}

        #%% Construct station info API request
        # Obtain and process header information
        logging.info("load station info")
        htmlcall_station_id = 'https://apps.noc-innovations.co.uk/api/polpred-api/latest/compute-port?key='
        url  = htmlcall_station_id+str(config_keys.NOC_KEY) + "&port_name=" + str(cls.station_id) + "&dt_start=" + startTime + "&dt_end=" + endTime + "&interval=5"


        #%% Get the data
        request_raw = requests.get(url, headers=headers)
        request = json.loads(request_raw.content)
        logging.debug(f"NOC API request: {request_raw.text}")
        # Check the output
        logging.info(f"Gauge id is {request['params']['port_name']}")
        try:
            logging.info(f"timestamp and value of the zero index is {[ request['data']['items'][00]['dt'], request['data']['items'][00]['z'] ]}")
        except:
            logging.info(f"timestamp and value of the zero index: problem")

        header_dict = request['params']
        header_dict['site_name'] = header_dict['port_name']
        header_dict['datum_shift'] = datum_shift

        #%% Process timeseries data
        dataset = xr.Dataset()
        time = []
        sea_level = []
        nvals = request['data']['totalItems']
        time = np.array([np.datetime64( request['data']['items'][i]['dt'] ) for i in range(nvals)])
        sea_level = datum_shift + np.array([ request['data']['items'][i]['z'] for i in range(nvals)])

        #%% Assign arrays to Dataset
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['time'])
        dataset = dataset.assign_coords(time = ('time', time))

        dataset.attrs = header_dict
        logging.debug(f"NOC API request headers: {header_dict}")
        try:
            logging.debug(f"NOC API request 1st time: {time[0]} and value: {sea_level[0]}")
        except:
            logging.debug(f"NOC API request 1st time: problem")
        # Assign local dataset to object-scope dataset
        return dataset

