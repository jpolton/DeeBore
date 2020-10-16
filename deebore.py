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
sys.path.insert(0,coastdir)
from coast.TIDEGAUGE import TIDEGAUGE

import logging
logging.basicConfig(filename='bore.log', filemode='w+')
logging.getLogger().setLevel(logging.DEBUG)

#%% ################################################################################

class Controller(object):
    """
    This is where the main things happen.
    Where user input is managed and methods are launched
    """
    def __init__(self):
        """
        Initialise main controller. Look for file. If exists load it
        """
        logging.info("run interface")
        self.load_flag = False
        self.run_interface()


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
                #self.pickle()
                break
            elif command == "i":
                print(INSTRUCTIONS)

            elif command == "1":
                # Load and plot raw data
                print('loading bore data')
                self.load_databucket()

            elif command == "2":
                print('show dataframe')
                self.show()

            elif command == "3":
                print('plot data')
                self.plot_data()

            elif command == "4":
                print('load and plot HLW data')
                filnam = 'data/Liverpool_2015_2020_HLW.txt'
                date_start = datetime.datetime(2020,1,1)
                date_end = datetime.datetime(2020,12,31)
                tg = TIDEGAUGE()
                tg.dataset = tg.get_tidetabletimes( filnam, date_start, date_end )
                # Exaple plot
                tg.dataset.plot.scatter(x="time", y="sea_level")
                print(f"stats: mean {tg.time_mean('sea_level')}")
                print(f"stats: std {tg.time_std('sea_level')}")

            elif command == "5":
                print('stats')
                tt = TIDEGAUGE()
                y1 = self.df['Time difference: Glad-Saltney (mins)'].values
                y2 = self.df['linfit_lag'].values
                print(f"stats: root mean sq err {np.sqrt(metrics.mean_squared_error(y1,y2 ))}")

            elif command == "6":
                """
                Glad_HT - float
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
                 Glad_HT - float
                 Glad_time - datetime64
                 Saltney_time - datetime64
                 Saltney_lag - int

                 """

                filnam = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2015_2020_HLW.txt'

                nd = input('Make predictions for N days from hence (int):?')
                day = np.datetime64('now','D') + np.timedelta64( int(nd), 'D' )
                dayp1 = day + np.timedelta64( 24, 'h' )
                tg = TIDEGAUGE()
                tg.dataset = tg.read_HLW_to_xarray( filnam, day, dayp1 )
                HT = tg.dataset['sea_level'].where( tg.dataset['sea_level'] > 7, drop=True)


                #plt.plot( HT.time, HT,'.' );plt.show()
                lag_pred = self.linfit(HT)

                Saltney_time_pred = [HT.time[i].values - np.timedelta64( int(round(lag_pred[i])), 'm') for i in range(len(lag_pred))]

                for i in range(len(lag_pred)):
                    #print( "Gladstone HT", np.datetime_as_string(HT.time[i], unit='m',timezone=pytz.timezone('UTC')),"(GMT). Height: {:.2f} m".format(  HT.values[i]))
                    #print(" Saltney arrival", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')),"(GMT/BST). Lag: {:.0f} mins".format( lag_pred[i] )) 
                    print(" Saltney pred", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')),". Height: {:.2f} m".format( HT.values[i] )) 
                #plt.scatter( Saltney_time_pred, HT ,'.');plt.show()
                # problem with time stamp

            elif command == "x":
                print('Export data')
                self.export()

            elif command == "r":
                print('Refresh database (delete pickle file is it exists)')
                if os.path.exists(DATABUCKET_FILE):
                    os.remove(DATABUCKET_FILE)
                else:
                    print("Can not delete the pickle file as it doesn't exists")
                self.load_databucket()

            else:
                template = "run_interface: I don't recognise (%s)"
                print(template%command)



    def load_old(self):
        """
        Load pickle file from the standard file save
        """
        self.df =  pd.read_csv('data/data_26Sep20.csv')
        self.df.drop(columns=['Unnamed: 1','Unnamed: 2'], inplace=True)
        self.df.dropna( inplace=True).set_index(['GMT time'])
        #self.df.dropna( inplace=True).set_index(['GMT time'])
        #self.bore = self.df.to_xarray()

    def load(self):
        """
        Load bore data
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
        if(0):
            for index, row in df.iterrows():
                df.loc[index,'time'] = np.datetime64( df.at[index,'time'] )
            # Create new reduced df with essential variables
            df_new = df[['logger','Chester Weir height: CHESTER WEIR 15 MIN SG']]
            df_new.rename(columns={"Chester Weir height: CHESTER WEIR 15 MIN SG":"weir_height"}, inplace=True)


            bore = xr.Dataset()
            bore = df_new.to_xarray()
            nt = len(bore.weir_height)
            tmp = np.array([datetime.datetime(2000,1,1) for i in range(nt) ])
            #bore['time3'] = bore['time']*np.NaN
            #bore['time3'] = [npdatetime64_2_datetime(bore.time[i].item()) for i in range(nt)]

            for i in range(nt):
                tmp[i] = npdatetime64_2_datetime(df.time[i].item())
                logging.debug( 'output',type(tmp[i] ) )
                #print( 'output', type(bore.time3[i] ) )
            bore['time2'] = tmp
        else:
            for index, row in df.iterrows():
                df.loc[index,'time'] = np.datetime64( df.at[index,'time'] ) # numpy.datetime64 in UTC
            bore = xr.Dataset()
            bore = df.to_xarray()

        # Set the t_dim to be a dimension and 'time' to be a coordinate
        bore = bore.rename_dims( {'index':'t_dim'} ).assign_coords( time=("t_dim", bore.time))
        self.bore = bore
        logging.info('Bore data loaded')

    def add_tidetable_data(self):
        """
        Add tide table HT and LT data to xr.DataSet
        Though these can (and in most historical cases have) be looked up, here
        it is automated.
        WIP
        """
        self.bore['HT_t'] = []
        self.bore['HT_h'] = []
        self.bore['LT_t'] = []
        self.bore['LT_h'] = []
        HT_h = []
        HT_t = []
        for i in range(len(self.bore.time)):
            try:
                HLW = None
                HLW = self.get_tidetabletimes(self.bore.time[i].values)
                print(f"HLW {HLW.dataset['sea_level']}")
                HT_h.append( float( HLW.dataset.sea_level[HLW.dataset['sea_level'].argmax()].values ) )
                HT_t.append( HLW.dataset.time[HLW.dataset['sea_level'].argmax()].values.item() )
                #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
                #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
            except:
                print('Issue with appening HLW data')

            print('HT_h:',HT_h)

            print('THE NEXT STEP IS TO ADD THESE HT_h  and HT_t VALUES TO xr.BORE')

    def get_Glad_data(self):
        """ Get Gladstone HLW data from external file """
        logging.info("Get Gladstone HLW data from external file")
        HT_h = []
        HT_t = []
        # load tidetable
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
                HW = tg.get_tidetabletimes( self.bore.time[i].values, method='nearest_HT' )
                #print(f"HLW: {HLW}")
                HT_h.append( HW.values )
                #print('len(HT_h)', len(HT_h))
                HT_t.append( HW.time.values )
                #print('len(HT_t)', len(HT_t))
                #self.bore['LT_h'][i] = HLW.dataset.sea_level[HLW.dataset['sea_level'].argmin()]
                #self.bore['LT_t'][i] = HLW.dataset.time[HLW.dataset['sea_level'].argmin()]
            except:
                logging.warning('Issue with appening HLW data')

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



    def compare_Glad_HLW(self):
        """ Compare Glad HLW from external file with bore tabilated data"""
        print("WIP: Compare Glad HLW from external file with bore tabilated data")
        print('log time, orig tide table, new tide table lookup')
        for i in range(len(self.bore.time)):
            print( self.bore.time[i].values, self.bore['Liv (Gladstone Dock) HT time (GMT)'][i].values, self.bore['glad_time'][i].values)

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
        """ Linear regression """
        idx = np.isfinite(Y).values
        weights = np.polyfit( X[idx], Y[idx], 1)
        logging.debug("weights: {weights}")
        self.linfit = np.poly1d(weights)
        self.bore['linfit_lag'] =  self.linfit(X)

    def show(self):
        """ Show xarray dataset """
        print( self.bore )


    def plot_data(self):
        """ plot data """


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

    def pickle(self):
        """ save copy of self into pickle file """
        print('Pickle data. NOT IMPLEMENTED')
        os.system('rm -f '+DATABUCKET_FILE)
        if(1):
            with open(DATABUCKET_FILE, 'wb') as file_object:
                pickle.dump(self.bore, file_object)
        else:
            print("Don't save as pickle file")
        return

    def load_databucket(self):
        """
        Auto load databucket from pickle file if it exists, otherwise create it
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
            else:
                print("... %s does not exist"%DATABUCKET_FILE)
                print("Load and process data")

                self.load()
                print('loading tide data')
                self.get_Glad_data()
                #self.compare_Glad_HLW()
                print('Calculating the Gladstone to Saltney time difference')
                self.calc_Glad_Saltney_time_diff()
                print('Calculating linear fit')
                self.linearfit( self.bore.glad_height, self.bore.Saltney_lag )

        except KeyError:
            print('ErrorA ')
        except (IOError, RuntimeError):
            print('ErrorB ')

    def export(self):
        print('Export data to csv. NOT IMPLEMENTED')
        pass

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
    3       plot bore data

    4       load and plot HLW data

    5       polyfit rmse.
    6       Predict bore.

    x       Export data to csv
    r       Refresh database

    i       to show these instructions
    q       to quit
    """


    ## Do the main program



    c = Controller()
