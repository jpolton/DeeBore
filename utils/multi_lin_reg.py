#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:11:21 2021

@author: jeff
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from deebore import Controller
import pickle

#### Constants
DATABUCKET_FILE = "deebore.pkl"


with open(DATABUCKET_FILE, 'rb') as file_object:
    bore = pickle.load(file_object)

#%% find the inflection point.

#%%

def mlr_fit(X,y):
    """
    Multivariate linear regression

    Returns
    -------
    None.

    """
    # Remove rows (events) with nan entries
    I = np.isfinite( np.c_[X,y] ).all(axis=1) # boolean collapse to a column
    X=X[I]
    y=y[I]

    # RMSE fitting all data
    lin_reg_fit = LinearRegression()
    lin_reg_fit.fit(X, y)
    y_fit = lin_reg_fit.predict(X)
    fit_rmse = mean_squared_error(y, y_fit, squared=False)

    # Fit model with test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=8) # test_size=0.2

    lin_reg_mod = LinearRegression() #normalize=True)

    lin_reg_mod.fit(X_train, y_train)

    pred = lin_reg_mod.predict(X_test)
    y_fit = lin_reg_mod.predict(X_train)

    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

    #if test_set_rmse != fit_rmse:
    #    print(f'Fitted rmse: {fit_rmse}')

    test_set_r2 = r2_score(y_test, pred)

    #print(f'rmse: {test_set_rmse}')
    #print(f'r2 score: {test_set_r2}')
    #print(f'coeffs: {lin_reg_mod.coef_}')

    return fit_rmse, test_set_rmse, test_set_r2, lin_reg_mod.coef_, lin_reg_mod.intercept_


if(0):
    x4 = bore['ctr_height_LW'].where(bore['Quality']=="A")
    x3 = (bore['Saltney_lag_HW_bodc']-bore['Saltney_lag_LW_bodc']).where(bore['Quality']=="A")
    x2 = (bore['liv_height_HW_bodc'] - bore['liv_height_LW_bodc']).where(bore['Quality']=="A")
    x1 = bore['liv_height_LW_bodc'].where(bore['Quality']=="A")
    x0 = bore['liv_height_HW_bodc'].where(bore['Quality']=="A")
    y = bore['Saltney_lag_HW_bodc'].where(bore['Quality']=="A")
else:
    x4 = bore['ctr_height_LW']
    x3 = (bore['Saltney_lag_HW_bodc']-bore['Saltney_lag_LW_bodc'])
    x2 = (bore['liv_height_HW_bodc'] - bore['liv_height_LW_bodc'])
    x1 = bore['liv_height_LW_bodc']
    x0 = bore['liv_height_HW_bodc']
    y = bore['Saltney_lag_HW_bodc']

#xx1 = x1[ (np.isfinite(x1)) & (np.isfinite(x4)) & (np.isfinite(y))]
#xx2 = x2[ (np.isfinite(x1)) & (np.isfinite(x4)) & (np.isfinite(y))]
#xx3 = x3[ (np.isfinite(x1)) & (np.isfinite(x4)) & (np.isfinite(y))]
#xx4 = x4[ (np.isfinite(x1)) & (np.isfinite(x4)) & (np.isfinite(y))]
#yy = y[ (np.isfinite(x1)) & (np.isfinite(x4)) & (np.isfinite(y))]

#XX = np.c_[xx1,xx2,xx3,xx4]
#XX = np.c_[x1.values,x2.values,x3.values,x4.values]

import itertools
flag = [1,0]
# result contains all possible combinations.
combinations = (list(itertools.product(flag,flag,flag,flag,flag)))

plt.close('all')
plt.figure()
for com in combinations:
    XX = np.c_[x0*com[0], x1*com[1], x2*com[2], x3*com[3], x4*com[4]]
    fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
    print('inputs:{}, full fit rmse: {:.1f}, test rmse: {:.1f}, r2: {:.1f}'.format(com, fit_rmse, rmse, r2))
    plt.scatter( fit_rmse, rmse, s=50*(r2+1)**4, label=com)
plt.xlabel('full fit RMSE')
plt.ylabel('test RMSE')
plt.legend()
plt.show()

#%%    Note that the Chester river data has some nans so that the full fit
# RMSE is different if river*0 is used. So we do a full fit with train_size=1
#bodc| height(HW), time(HW):  5.5 mins,
#bodc| height(HW-LW), time(HW):  6.2 mins
#bodc| height(HW-LW), time(LW):  8.0 mins
#bodc| height(LW), time(LW): 11.9 mins
#bodc| height(LW), time(HW):  7.5 mins
XX = np.c_[ bore['liv_height_HW_bodc'], bore['liv_height_HW_bodc']*0.]
y = bore['Saltney_lag_HW_bodc']
fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
print('Only fit Saltney_lag_HW_bodc to liv_height_HW_bodc')
print('full fit rmse: {:.1f}, train rmse: {:.1f}, r2: {:.1f}'.format(fit_rmse, rmse, r2))
print('coefs: {}, intercept:{:.1f}'.format(coefs, interc))

#%% Only fit to the tides > 10m

XX = np.c_[ bore['liv_height_HW_bodc']*0, bore['liv_height_HW_bodc'].where(bore['liv_height_HW_bodc']>=10.)]
y = bore['Saltney_lag_HW_bodc']

fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)

print('Only fit to the tides > 10m')
print('full fit rmse: {:.1f}, train rmse: {:.1f}, r2: {:.1f}'.format(fit_rmse, rmse, r2))
print('coefs: {}, intercept:{:.1f}'.format(coefs, interc))

#%% Multivariate linear regression to >10m


x3 = bore['ctr_height_LW']
#x2 = (bore['liv_time_LW_bodc']-bore['liv_time_LW_harmonic'])/np.timedelta64(60,'s')
x2 = (bore['Saltney_lag_HW_bodc']-bore['Saltney_lag_LW_bodc'])
#x2 = (bore['liv_height_HW_bodc'] - bore['liv_height_LW_bodc'])
x1 = bore['liv_height_LW_bodc']
x0 = bore['liv_height_HW_bodc']
y = bore['Saltney_lag_HW_bodc'].where(bore['liv_height_HW_bodc']>=9.)
print('Multivariate linear regression to >9m')


flag = [1,0]
# result contains all possible combinations.
combinations = (list(itertools.product(flag,flag,flag,flag)))

for com in combinations:
    XX = np.c_[x0*com[0], x1*com[1], x2*com[2], x3*com[3]]
    fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
    print('inputs:{}, full fit rmse: {:.1f}, test rmse: {:.1f}, r2: {:.1f}'.format(com, fit_rmse, rmse, r2))


"""
For 10m tides it really looks like the -82.6 mins before HT is the way to go.
Though the RMSE decrease from 4 to 3 mins by throwing all the data at it, the
r2 score is 0 suggesting it is no better than picking a constant.

More data would be required to improve the r2 score, I think.

For bores on tides < 10m then the regression is better.
THe rivers make a tiny positive contribution to improving RMSE, but the
effect is not really measureable in r2 score.
Height(HT) and time(HT-LT) seem to be the best value.
RMSE 5.1 (5.2 w/ rivers). With CLASS A data RMSE 4.4(4.5) but the r2 goes crazy
with so few data points.

Dropping Height(LW) doesn't matter!
Suggesting the bore does not have an origin with LW being caught up.
"""



#%% Fit to HT, dtiming without rivers
XX = np.c_[x0, x2]
y = bore['Saltney_lag_HW_bodc']
fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
print('Fit to HT,dT WITHOUT rivers')
print('full fit rmse: {:.1f}, test rmse: {:.1f}, r2: {:.1f}'.format(fit_rmse, rmse, r2))
print('coefs: {}, intercept:{:.1f}'.format(coefs, interc))

#%% Fit to HT  without rivers
XX = np.c_[x0]
y = bore['Saltney_lag_HW_bodc']
fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
print('Fit to HT WITHOUT rivers')
print('full fit rmse: {:.1f}, test rmse: {:.1f}, r2: {:.1f}'.format(fit_rmse, rmse, r2))
print('coefs: {}, intercept:{:.1f}'.format(coefs, interc))

#%% Fit to timing  without rivers
XX = np.c_[x2]
y = bore['Saltney_lag_HW_bodc']
fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
print('Fit to timing WITHOUT rivers')
print('full fit rmse: {:.1f}, test rmse: {:.1f}, r2: {:.1f}'.format(fit_rmse, rmse, r2))
print('coefs: {}, intercept:{:.1f}'.format(coefs, interc))

#%% Fit to HT, dtiming with rivers
XX = np.c_[x0, x2, x3]
y = bore['Saltney_lag_HW_bodc']
fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
print('Fit to HT,dT WITH rivers')
print('full fit rmse: {:.1f}, test rmse: {:.1f}, r2: {:.1f}'.format(fit_rmse, rmse, r2))
print('coefs: {}, intercept:{:.1f}'.format(coefs, interc))

#%% Fit to HT with rivers
XX = np.c_[x0, x3]
y = bore['Saltney_lag_HW_bodc']
fit_rmse, rmse, r2, coefs, interc = mlr_fit(XX,y)
print('Fit to HT WITH rivers')
print('full fit rmse: {:.1f}, test rmse: {:.1f}, r2: {:.1f}'.format(fit_rmse, rmse, r2))
print('coefs: {}, intercept:{:.1f}'.format(coefs, interc))

if(0):
    #Attempt to condense multivariate regression onto a line plot. Not sure this is it yet...
    plt.figure()
    plt.scatter(y, x0 + coefs[1]/coefs[0]*x2 + interc/coefs[0], c=x4)
    plt.xlabel('Time before HT (mins)')
    plt.ylabel('Transformed input data {Height(HT), dT}')
    plt.colorbar()
    plt.show()
