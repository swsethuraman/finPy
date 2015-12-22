'''
Created on August 7, 2014

@author: SSethuraman
'''

import numpy as np
import useful
import ircb_inst 
import rate_struct as rs
import env
import holidays as hol 

class EDFuturesBuilder:
    def __init__(self, tradedate, contract, price, futures_spec):
        self.tradedate = tradedate
        
        