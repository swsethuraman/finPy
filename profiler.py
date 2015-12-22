'''
Created on May 29, 2014

@author: SSethuraman
'''
import numpy as np
import ircb_data
import ircb_dataset
import pyximport
import cProfile
import ircb
import env 

pyximport.install(setup_args={"script_args":["--compiler=mingw32"]}, reload_support=True)
tradedate   = np.datetime64('2014-07-07')
menv = env.MktEnv(tradedate, tradedate, tradedate, tradedate)
data_set = ircb_dataset.DataSet('test_data')
data_set.add('usd_3m_libor', ircb_data.usd_3m_libor)
data_set.add('usd_1d_fedfunds', ircb_data.usd_1d_fedfunds)
curves = ['usd_libor_3m', 'usd_fedfunds_1d']
#sm = ircb.SimpleIrcBuilder(curves, menv, dataset)
#sm = ircb.SimpleIrcBuilder()
cProfile.run("ircb.SimpleIrcBuilder(curves, menv, data_set)")


