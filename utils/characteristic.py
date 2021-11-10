"""
characteristic.def

Plot the characteristics of waves propagating from source
"""

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import matplotlib.dates as mdates


def func(start_time, duration, A0=6, tg=None):
    """
    Compute the distance travelled
    inputs:
    start_time - [0, 12.4*3600] seconds as float
    duration - seconds tracking line
    A0 - mean depth at oscillator

    Assume wave evolves as -cos(t). I.e. from low to high to low, if full cycle

    y = duration * speed
    speed = 0.5 * sqrt( g*H(start_time) )
    """
    #A0 = 6 # mean depth
    A1 = 5 # wave amplitude
    g = 9.81

    if tg == None:
        H = A0 - A1*np.cos( 2*np.pi/(12.4*3600) * start_time )
    else:
        H = tg
    speed = 0.5 * np.sqrt(g*H)

    return duration * speed


def hyd_jump(self, tg ):
    """
    Estimate the location of the hydrolic jump for information leaving at LT 
    and with the specified time

    Returns
    -------
    None.

    """
    
    extrema = tg.find_high_and_low_water('sea_level', method="cubic")
    T_HT = extrema.dataset.time_highs.values
    H_HT = extrema.dataset.sea_level_highs
    
    T_LT = extrema.dataset.time_lows.values
    H_LT = extrema.dataset.sea_level_lows
    
    if len(T_HT) > 1 or len(H_HT) > 1 or len(T_LT) > 1 or len(H_LT) > 1:
        print( extrema.dataset )
    
    H = tg.dataset.sea_level.where( t.dataset.time >= T_LT)
    T0 = tg.dataset.time.where( t.dataset.time >= T_LT) - T_LT # Time at source since LT
    
    (H + np.sqrt( H_LT * H)) / (H - H_LT) * T0 # minimise this to get interval between LT and hydrolic jump

    
nval = 20
A0 = 5.5 # mean depth at oscillator
duration = 5*3600 # seconds tracking each line
I = np.arange(nval)
segs_h = np.zeros((nval,2,2)) # line, pointA/B, t/z



#%% load BODC data

coastdir = os.path.dirname('/Users/jeff/GitHub/COAsT/coast')
sys.path.insert(0, coastdir)
from coast import TIDEGAUGE

date_start=np.datetime64('2020-10-20 07:00')
date_end = np.datetime64('2020-10-20 13:00')

fil = '/Users/jeff/GitHub/DeeBore/data/BODC_processed/2020LIV.txt'

tg = TIDEGAUGE()
tg.dataset = tg.read_bodc_to_xarray(fil, date_start=date_start, date_end=date_end)


#%%

#convert dates to numbers first

start_time = np.linspace(0, 3600*12.4 / 2., nval) # Half a M2 cycle
start_time = np.divide(tg.dataset.time - tg.dataset.time[0], np.timedelta64(1, 's'))

start_dist = np.linspace(0, 0, nval)

end_time = start_time + duration
end_dist = [func( start_time[I], duration, A0, tg.dataset['sea_level'].values[I]) for I in range(nval)]


segs_h[:,0,1] = [end_dist[I]/1000. for I in range(nval)] # y-axis.1
segs_h[:,1,1] = [start_dist[I]/1000. for I in range(nval)] # y-axis.2
segs_h[:,0,0] = [end_time[I]/3600. for I in range(nval)]   # x-axis.1
segs_h[:,1,0] = [start_time[I]/3600. for I in range(nval)] # x-axis.2


#plt.close('all')
fig = plt.subplots()
ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
ax0.plot(  tg.dataset['time'], tg.dataset['sea_level'])
ax = plt.subplot2grid((3, 1), (1, 0), rowspan=2)

ax.set_ylim(np.nanmin(segs_h[:,:,1]), np.nanmax(segs_h[:,:,1]))
line_segments_HW = LineCollection(segs_h, cmap='plasma', linewidth=1)
ax.add_collection(line_segments_HW)
ax.scatter(segs_h[:,1,0],segs_h[:,1,1], c='red', s=4, label='predicted') # harmonic predictions
ax.scatter(segs_h[:,0,0],segs_h[:,0,1], c='green', s=4, label='measured') # harmonic predictions
ax.set_title('Harmonic prediction with quiver to measured high waters')

plt.ylabel('distance travelled (km)')
plt.xlabel('time (hrs)')
#plt.title('mean depth: {}m'.format(A0))
plt.title(date_start.astype(object).date().strftime('%d-%b-%y'))
plt.tight_layout()
#plt.legend()
