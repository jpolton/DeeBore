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

    def load(self):
        """
        Load pickle file from the standard file save
        """
        self.df =  pd.read_csv('data/data.csv')
        self.df.drop(columns=['Unnamed: 1','Unnamed: 2'], inplace=True)
        self.df.dropna( inplace=True)

    def linearfit(self):
        """ Linear regression """
        weights = np.polyfit( \
                        self.df['Liv (Gladstone Dock) HT height (m)'], \
                        self.df['Time difference: Glad-Saltney (mins)'], 1)
        linfit = np.poly1d(weights)
        self.df['linfit_lag'] = linfit(self.df['Liv (Gladstone Dock) HT height (m)'])

    def show(self):
        """ Show dataframe """
        print( self.df )


    def plot_data(self):
        """ plot dataframe """
        s = plt.scatter( self.df['Liv (Gladstone Dock) HT height (m)'], \
            self.df['Time difference: Glad-Saltney (mins)'], \
            c=self.df['Chester Weir height: CHESTER WEIR 15 MIN SG'] )
        cbar = plt.colorbar(s)
        # Linear fit
        x = self.df['Liv (Gladstone Dock) HT height (m)']
        plt.plot( x, self.df['linfit_lag'], '-' )
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
                self.linearfit()

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
