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
                tg = TIDEGAUGE(filnam, date_start, date_end)
                # Exaple plot
                tg.dataset.plot.scatter(x="time", y="sea_level")

            else:
                template = "run_interface: I don't recognise (%s)"
                print(template%command)

class TIDEGAUGE(object):
    """
    This is where the main things happen.
    Where user input is managed and methods are launched
    """
    def __init__(self, file_path = None, date_start=None, date_end=None):
        '''
        Initialise TIDEGAUGE object either as empty (no arguments) or by
        reading GESLA data from a directory between two datetime objects.

        # Read tide gauge data for 2020
        filnam = 'data/Liverpool_2015_2020_HLW.txt'
        date_start = datetime.datetime(2020,1,1)
        date_end = datetime.datetime(2020,12,31)
        tg = TIDEGAUGE(filnam, date_start, date_end)

        # Access the data
        tg.dataset

        # Exaple plot
        tg.dataset.plot.scatter(x="time", y="sea_level")

        '''
        # If file list is supplied, read files from directory
        if file_path is None:
            self.dataset = None
        else:
            self.dataset = self.read_HLW_to_xarray(file_path,
                                                        date_start, date_end)
        return

    @classmethod
    def read_HLW_to_xarray(cls, filnam, date_start=None, date_end=None):
        '''

        Parameters
        ----------
        filnam (str) : path to gesla tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data

        Returns
        -------
        xarray.Dataset object.
        '''
        try:
            header_dict = cls.read_HLW_header(filnam)
            dataset = cls.read_HLW_data(filnam, header_dict, date_start, date_end)
            if header_dict['field'] == 'TZ:UT(GMT)/BST':
                print('Assign BST as timezone')
                #assignBST = lambda t: t.astimezone()
                #dataset.time = [assignBST(i) for i in dataset.time]

            else:
                print("Not expecting that timezone")

        except:
            raise Exception('Problem reading HLW file: ' + filnam)

        dataset.attrs = header_dict

        return dataset


    @staticmethod
    def read_HLW_header(filnam):
        '''
        Reads header from a HWL file.

        Parameters
        ----------
        filnam (str) : path to file

        Returns
        -------
        dictionary of attributes
        '''
        print(f"Reading HLW header from \"{filnam}\"")
        fid = open(filnam)

        # Read lines one by one (hopefully formatting is consistent)
        header = fid.readline().split()
        site_name = header[:3]
        site_name = '_'.join(site_name)

        field = header[3:5]
        field = '_'.join(field).replace(':_',':')

        units = header[5:7]
        units = '_'.join(units).replace(':_',':')

        datum = header[7:10]
        datum = '_'.join(datum).replace(':_',':')

        print(f"Read done, close file \"{filnam}\"")
        fid.close()
        # Put all header info into an attributes dictionary
        header_dict = {'site_name' : site_name, 'field':field,
                       'units':units, 'datum':datum}
        return header_dict

    @staticmethod
    def read_HLW_data(filnam, header_dict, date_start=None, date_end=None,
                           header_length:int=1):
        '''
        Reads observation data from a GESLA file (format version 3.0).

        Parameters
        ----------
        filnam (str) : path to HLW tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data
        header_length (int) : number of lines in header (to skip when reading)

        Returns
        -------
        xarray.Dataset containing times, High and Low water values
        '''
        # Initialise empty dataset and lists
        print(f"Reading HLW data from \"{filnam}\"")
        dataset = xr.Dataset()
        time = []
        sea_level = []

        if header_dict['field'] == 'TZ:UT(GMT)/BST':
            localtime_flag = True
            if date_start is not None: date_start = date_start.astimezone()
            if date_end is not None: date_end = date_end.astimezone()
        else:
            localtime_flag = False

        # Open file and loop until EOF
        with open(filnam) as file:
            line_count = 0
            for line in file:
                # Read all data. Date boundaries are set later.
                if line_count>header_length:
                    working_line = line.split()
                    if working_line[0] != '#':
                        time_str = working_line[0] + ' ' + working_line[1]
                        datetime_obj = datetime.datetime.strptime( time_str , '%d/%m/%Y %H:%M')
                        if localtime_flag == True:
                            time.append( datetime_obj.astimezone() )
                        else:
                            time.append( datetime_obj )
                        sea_level.append(float(working_line[2]))

                line_count = line_count + 1
            print(f"Read done, close file \"{filnam}\"")

        # Return only values between stated dates
        start_index = 0
        end_index = len(time)
        if date_start is not None:
            start_index = nearest_datetime_ind(time, date_start)
        if date_end is not None:
            end_index = nearest_datetime_ind(time, date_end)
        time = time[start_index:end_index]
        sea_level = sea_level[start_index:end_index]

        # Assign arrays to Dataset
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['t_dim'])
        dataset = dataset.assign_coords(time = ('t_dim', time))

        # Assign local dataset to object-scope dataset
        return dataset

def nearest_datetime_ind(items, pivot):
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)

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

    i       to show these instructions
    q       to quit
    """


    ## Do the main program
    c = Controller()
