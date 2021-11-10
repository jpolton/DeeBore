#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05 Nov 2021

@author: jeff



Fit Chester data to Ironbridge data => Chester=fn(Ironbridge)

EA gauge data at Chester is no longer being reported.

Loads data from local shoothill files.

Conda environment: workshop_env with coast and requests installed,
    E.g.
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

Useage:
    xxx
"""

# Begin by importing coast and other packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # There are many ways to read netCDF files, including this one!

################################################################################
#%%  plot functions
def line_plot(ax, time, y, color, size, label=None ):
    ax.plot(time, y, color=color, linewidth=size, label=label)
    return ax

def scatter_plot(ax, time, y, color, size, label=None ):
    ax.scatter(time, y, color=color, s=size, label=label)
    return ax


################################################################################
################################################################################
#%% Main Routine
################################################################################
################################################################################
if __name__ == "__main__":
    #%%
    # Choose some arbitary dates
    start_date = np.datetime64('2019-02-17')
    end_date =  np.datetime64('2019-02-23')

    # location of files
    dir = "archive_shoothill/" #

    # load data by location.
    ctr = xr.open_mfdataset(dir+"ctr_????.nc") # above the Chester weir
    #ctr_dn = xr.open_mfdataset(dir+"ctr2_????.nc") # below the Chester weir
    iron= xr.open_mfdataset(dir+"iron_????.nc") # upstream river at Ironbridge

    merge = xr.merge([ctr.rename({'sea_level':'ctr'}), iron.rename({'sea_level':'iron'})])

    #%% Plot data
    plt.close('all')
    fig, ax = plt.subplots(1)

    fig.suptitle('Chester River Heights from Ironbridge')
    ax.scatter(merge.iron, merge.ctr, color='k', s=1, label=None)
    plt.ylabel('ctr')
    plt.xlabel('iron')


    # Harvest click event coordinates
    coords = []

    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print('x = %f, y = %f'%(ix, iy))

        global coords
        coords.append((ix, iy))

        if len(coords) == 15:
            fig.canvas.mpl_disconnect(cid)

        return coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    #plt.savefig('fit_IRN_CTR.png')
