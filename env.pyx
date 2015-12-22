'''
Created on May 19, 2014

@author: SSethuraman
'''
# import ircb
from pyxll import xl_func
import numpy as np
import ir_fixings

class MktEnv:
    def __init__(self, today, builddate, datadate, pvdate):
        self.version        = 1
        #self.rand_string    = np.random()
        self.observers      = {}
        self.today          = today
        self.builddate      = builddate
        self.datadate       = datadate
        self.pvdate         = pvdate
        self.fixings        =  {
            'usd_libor_3m'      :  [(np.datetime64('2015-04-13'), 0.002753), (np.datetime64('2015-04-14'), 0.002755), (np.datetime64('2015-04-15'), 0.002755)],
            'usd_fedfunds_1d'   :  [(np.datetime64('2015-04-13'), 0.001000), (np.datetime64('2014-04-14'), 0.001000), (np.datetime64('2014-04-15'), 0.001000)]
        }
        #self.fixings        = {
        #'    'usd_libor_3m'      : ir_fixings.usd_libor_3m_fixing,
        #'    'usd_fedfunds_1d'   : ir_fixings.usd_fed_funds_1d_fixing
        #'}
        
    def attach(self, obs_string, obs):
        #if not obs_string in self.observers.keys():
            observer = obs
            self.observers.update({obs_string : observer})
    # def attach(self, obs_string, obs_build):
    #    if not obs_string in self.observers.keys():
    #        observer = obs_build.curvebuild(obs_string)
    #        self.observers.update({obs_string : observer})

