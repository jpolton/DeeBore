"""
Read in a process Dee Bore data
Author: jpolton
Date: 26 Sept 2020
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    def show(self):
        """ Show dataframe """
        print( self.df )

    def plot_data(self):
        """ plot dataframe """
        s = plt.scatter( self.df['Liv (Gladstone Dock) HT height (m)'], \
            self.df['Time difference: Glad-Saltney (mins)'], \
            c=self.df['Chester Weir height: CHESTER WEIR 15 MIN SG'] )
        cbar = plt.colorbar(s)
        cbar.set_label('River height at weir (m)')
        plt.title('Bore arrival time at Saltney Ferry')
        plt.ylabel('Arrival time (mins before Liv HT)')
        plt.xlabel('Liv (Gladstone Dock) HT height (m)')
        plt.show()
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

            elif command == "2":
                print('show dataframe')
                self.show()

            elif command == "3":
                print('plot data')
                self.plot_data()

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
    1       load dataframe
    2       show dataframe
    3       plot data

    i       to show these instructions
    q       to quit
    """

    ## Do the main program
    c = Controller()
