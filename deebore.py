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


coastdir = os.path.dirname('/Users/jeff/GitHub/COAsT/coast')
sys.path.insert(0,coastdir)
from coast.TIDETABLE import TIDETABLE

import logging
logging.basicConfig(filename='bore.log', filemode='w+', level=logging.INFO)

################################################################################

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
        self.run_interface()

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
        Load pickle file from the standard file save
        """
        df =  pd.read_csv('data/master-Table 1.csv')
        df.drop(columns=['date + logged time','Unnamed: 2','Unnamed: 11', \
                                'Unnamed: 12','Unnamed: 13', 'Unnamed: 15'], \
                                 inplace=True)
        df.rename(columns={"date + logged time (GMT)":"time"}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True, format="%d/%m/%Y %H:%M")
        df.set_index(['time'], inplace=True)

        self.bore = xr.Dataset()
        self.bore = df.to_xarray()


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
        print("WIP: Get Gladstone HLW data from external file")
        pass

    def compare_Glad_HLW(self):
        """ Compare Glad HLW from external file with bore tabilated data"""
        print("Compare Glad HLW from external file with bore tabilated data")
        pass

    def calc_Glad_Saltney_time_diff(self):
        """ Compute lag (-ve) for arrival at Saltney relative to Glastone HT """
        print('WIP: calc_Glad_Saltney_time_diff')
        pass

    def linearfit(self):
        """ Linear regression """
        weights = np.polyfit( \
                        self.bore['Liv (Gladstone Dock) HT height (m)'], \
                        self.bore['Time difference: Glad-Saltney (mins)'], 1)
        self.linfit = np.poly1d(weights)
        self.bore['linfit_lag'] = self.linfit(self.bore['Liv (Gladstone Dock) HT height (m)'])

    def show(self):
        """ Show dataframe """
        print( self.bore )


    def plot_data(self):
        """ plot dataframe """
        s = plt.scatter( self.bore['Liv (Gladstone Dock) HT height (m)'], \
            self.bore['Time difference: Glad-Saltney (mins)'], \
            c=self.bore['Chester Weir height: CHESTER WEIR 15 MIN SG'] )
        cbar = plt.colorbar(s)
        # Linear fit
        #x = self.df['Liv (Gladstone Dock) HT height (m)']
        #plt.plot( x, self.df['linfit_lag'], '-' )
        cbar.set_label('River height at weir (m)')
        plt.title('Bore arrival time at Saltney Ferry')
        plt.ylabel('Arrival time (mins before Liv HT)')
        plt.xlabel('Liv (Gladstone Dock) HT height (m)')
        #plt.show()
        plt.savefig('figs/SaltneyArrivalLag_vs_LivHeight.png')


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
                break
            elif command == "i":
                print(INSTRUCTIONS)

            elif command == "1":
                #%% Load and plot raw data
                print('load dataframe')
                self.load()
                self.get_Glad_data()
                self.compare_Glad_HLW()
                self.calc_Glad_Saltney_time_diff()
                #self.linearfit()

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
                tg = TIDETABLE(filnam, date_start, date_end)
                # Exaple plot
                tg.dataset.plot.scatter(x="time", y="sea_level")
                print(f"stats: mean {tg.time_mean('sea_level')}")
                print(f"stats: std {tg.time_std('sea_level')}")

            elif command == "5":
                print('stats')
                tt = TIDETABLE()
                y1 = self.df['Time difference: Glad-Saltney (mins)'].values
                y2 = self.df['linfit_lag'].values
                print(f"stats: root mean sq err {np.sqrt(metrics.mean_squared_error(y1,y2 ))}")


            else:
                template = "run_interface: I don't recognise (%s)"
                print(template%command)

if __name__ == "__main__":

    #### Initialise logging
    now_str = datetime.datetime.now().strftime("%d%b%y %H:%M")
    logging.info(f"-----{now_str}-----")

    #### Constants
    INSTRUCTIONS = """

    Choose Action:
    1       load bore dataframe
    2       show bore dataframe
    3       plot bore data

    4       load and plot HLW data

    5       polyfit rmse

    i       to show these instructions
    q       to quit
    """


    ## Do the main program
    c = Controller()
