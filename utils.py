'''
Created on Aug 20, 2014

@author: SSethuraman
'''

from pyxll import xl_func, xl_menu, get_active_object, xlfCaller
import pyximport
import numpy as np

import datetime
from scipy.stats import norm

try:
    import win32com.client
    _have_win32com = True
except ImportError:
    _log.warning("*** win32com.client could not be imported           ***")
    _log.warning("*** some of the object cache examples will not work  ***")
    _log.warning("*** to fix this, install the pywin32 extensions     ***")
    _have_win32com = False

pyximport.install(setup_args={"script_args":["--compiler=mingw32"]}, reload_support=True)

@xl_func("float x:float")
def util_normsinv(x):
    return norm.cdf(x)


@xl_func("float F, float K, float T, float sigma, float r, string value_type: float")
def util_BSFutCall(F, K, T, sigma, r, value_type):
    # update the cache and return the cached object id
    d1      =   (np.log(F/K) + T*np.square(sigma)/2)/(sigma*np.sqrt(T))
    d2      =   (np.log(F/K) - T*np.square(sigma)/2)/(sigma*np.sqrt(T))
    Nd1     =   norm.cdf(d1)
    Nd2     =   norm.cdf(d2)
    Pd1     =   norm.pdf(d1)
    if value_type   ==  'px':
        Call        =   F*Nd1   -   K*Nd2
        Call        =   np.exp(-r*T)*Call
        return Call
    elif value_type ==  'delta':
        CallDelta   =  np.exp(-r*T)*Nd1
        return CallDelta
    elif value_type ==  'gamma':
        CallGamma   =   np.exp(-r*T)*Pd1/(F*sigma*np.sqrt(T))
        return CallGamma
    elif value_type ==  'vega':
        CallVega    =   F*np.exp(-r*T)*Pd1*np.sqrt(T)
        return CallVega
    elif value_type ==  'theta':
        theta1  =   -F*np.exp(-r*T)*Pd1*sigma/(2*np.sqrt(T))
        theta2  =   r*F*np.exp(-r*T)*Nd1
        theta3  =   -r*K*np.exp(-r*T)*Nd2
        return theta1 + theta2 + theta3
    elif value_type ==  'rho':
        CallRho =   T*K*np.exp(-r*T)*Nd2
        return CallRho          
    pass

@xl_func("float F, float K, float T, float sigma, float r, string value_type: float")
def util_BSFutPut(F, K, T, sigma, r, value_type):
    # update the cache and return the cached object id
    d1      =   (np.log(F/K) + T*np.square(sigma)/2)/(sigma*np.sqrt(T))
    d2      =   (np.log(F/K) - T*np.square(sigma)/2)/(sigma*np.sqrt(T))
    Nd1         =   norm.cdf(d1)
    Nd1_minus   =   norm.cdf(-d1)
    Nd2_minus   =   norm.cdf(-d2)
    Pd1         =   norm.pdf(d1)
    if value_type   ==  'px':
        Put        =   K*Nd2_minus   -   F*Nd1_minus
        Put        =   np.exp(-r*T)*Put
        return Put
    elif value_type ==  'delta':
        PutDelta   =  np.exp(-r*T)*(Nd1-1)
        return PutDelta
    elif value_type ==  'gamma':
        g1         =   np.exp(-r*T)*Pd1
        g2         =   F*sigma
        g3         =   np.sqrt(T)
        PutGamma   =   g1/(g2*g3)
        return PutGamma
    elif value_type ==  'vega':
        PutVega    =   F*np.exp(-r*T)*Pd1*np.sqrt(T)
        return PutVega
    elif value_type ==  'theta':
        theta1  =   -F*np.exp(-r*T)*Pd1*sigma/(2*np.sqrt(T))
        theta2  =   -r*F*np.exp(-r*T)*Nd1_minus
        theta3  =   r*K*np.exp(-r*T)*Nd2_minus
        return theta1 + theta2 + theta3
    elif value_type ==  'rho':
        PutRho =   -T*K*np.exp(-r*T)*Nd2_minus
        return PutRho          
    pass

@xl_func("float F, float K, float T, float sigma, float r, string callput, string value_type: float")
def util_BSFut(F, K, T, sigma, r, callput, value_type):
    if (callput == 'C') or (callput == 'c'):
        return util_BSFutCall(F, K, T, sigma, r, value_type)
    elif (callput == 'P') or (callput == 'p'):
        return util_BSFutPut(F, K, T, sigma, r, value_type)
    else:
        return 'Err'
        

@xl_func("float F1, float F2, float K, float T, float sigma1, float sigma2, float correlation, float r, string value_type: float")
def util_SpreadOptionCall(F1, F2, K, T, sigma1, sigma2, correlation, r, value_type):
    # update the cache and return the cached object id
    a       =   F2 + K
    b       =   F2/(F2 + K)
    sigma   =   np.sqrt(np.square(sigma1) + np.square(b)*np.square(sigma2) - 2*correlation*b*sigma1*sigma2)
    d1      =   (np.log(F1/a) + (0.5*np.square(sigma1) - b*correlation*sigma1*sigma2 + 0.5*np.square(b)*np.square(sigma2))*T)/(sigma*np.sqrt(T))
    d2      =   (np.log(F1/a) + (-0.5*np.square(sigma1) + correlation*sigma1*sigma2 + 0.5*np.square(b)*np.square(sigma2) -b*np.square(sigma2))*T)/(sigma*np.sqrt(T))
    d3      =   (np.log(F1/a) + (-0.5*np.square(sigma1) + 0.5*np.square(b)*np.square(sigma2))*T)/(sigma*np.sqrt(T))
    Nd1     =   norm.cdf(d1)
    Nd2     =   norm.cdf(d2)
    Nd3     =   norm.cdf(d3)
    
    if value_type   ==  'px':
        Call       =   F1*Nd1 - F2*Nd2 - K*Nd3
        Call       =   np.exp(-r*T)*Call
        return Call
    else:
        return 0
    pass
