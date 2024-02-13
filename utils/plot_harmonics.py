"""
## plot_harmonics.py
#
# Simple plotting routine created during investigation of Sheerness issues

Plot a panel of harmonic amplitudes and and a panel of phases for a shared
x-axis of consitutent labels

Loads in a pair of json files of harmonic amplitudes and phases
E.g. amplitude
{
    "2020": {
        "M2": 0.265,
        "M4": 0.148,
        "S2": 0.097,
        "MS4": 0.086,
        "MN4": 0.055,
        "SA": 0.046,
        "M6": 0.041,
        "2MS6": 0.033,
        "N2": 0.032,
        "K2": 0.03
    }
}

"""


import pandas as pd
import matplotlib.pyplot as plt

fa = "/Users/jelt/GitHub/DeeBore/data_amp.json"
fg = "/Users/jelt/GitHub/DeeBore/data_pha.json"

da = pd.read_json(fa).to_numpy()
dg = pd.read_json(fg).to_numpy()

const_list = pd.read_json(fa).T.keys()
year_list = pd.read_json(fa).T.index
plt.figure()

fig, [ax_l, ax_r] = plt.subplots(2, sharex=True)

ax_l.plot( da , 'o', label=year_list)
ax_l.set_xticks( range( len(const_list) ) )
ax_l.set_xticklabels( const_list)
ax_l.set_ylabel("Amplitude (m)")

ax_r.plot( dg, 'o')
ax_r.set_ylabel("Phase (deg)")
ax_r.text(4.5,250, "modelled: ERA5 forced AMM7_surge\nobserved: ClassAObsAfterSurgeQC*.nc")
# plot the legend
lgnd = ax_l.legend(markerscale=1, loc='upper right', ncol=3)
ax_l.set_title("Harmonic character of (observed - modelled) total water at Sheerness")

#plt.show()
plt.savefig('Sheerness_harmonics_obs_model_difference.png')

print(da)



