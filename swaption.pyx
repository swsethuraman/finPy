'''
Created on May 19, 2014

@author: SSethuraman
'''

import useful
import ircb_init
import ircb_inst 
import rate_struct as rs
import env
import holidays as hol 

class SwaptionBuilder:
    def __init__(self, tradedate, swapspec, notional, payrecv, strike, expiry, tnr, start_0=None, first_reg=None, last_reg=None, end_0=None):
        # consistency checks
        swaptype    = swapspec['inst_type']
        fixed_rate  = strike
        self.swap        = ircb_init.inst_builder[swaptype](tradedate, notional, fixed_rate, payrecv, swapspec, tenor=tnr)
        self.expiry      = expiry
        self.tenor       = tnr
        self.strike      = strike
        self.payrecv     = payrecv
        self.tradedate   = tradedate
        self.notional    = notional
    pass

class MidCurveSwaptionBuilder(SwaptionBuilder):
    def __init__(self, tradedate, swapspec, notional, payrecv, strike, expiry, tnr, start_0=None, first_reg=None, last_reg=None, end_0=None):
        # consistency checks
        swaptype    = swapspec['inst_type']
        fixed_rate  = strike
        
        temp = ''
        tenor_str = []
        count = 0
        for x in tnr:
            if useful.is_number(x):
                temp = temp + x
            else:
                tenor_str.append(temp + x)
                temp = ''
                count = count + 1
                
        t1  = (int)(tenor_str[0][:-1])
        t2  = (int)(tenor_str[1][:-1])
        s1  = tenor_str[0][-1]
        s2  = tenor_str[1][-1]
        
        if s1 == 'M':
            t2 = t1 + 12*t2
            ss = 'M'
        elif s2 == 'M':
            t2 = 12*t1 + t2
            ss = 'M'
        else:
            t2 = t1+t2
            ss = 'Y'
        
        tenor_str[1] = str(t2) + ss   
        
        print(tenor_str[0])
        print(tenor_str[1]) 
        
        self.swap1          = ircb_init.inst_builder[swaptype](tradedate, notional, fixed_rate, payrecv, swapspec, tenor=tenor_str[0])
        self.swap2          = ircb_init.inst_builder[swaptype](tradedate, notional, fixed_rate, payrecv, swapspec, tenor=tenor_str[1])
        self.expiry         = expiry
        self.tenor1         = tenor_str[0]
        self.tenor2         = tenor_str[1]
        self.strike         = strike
        self.payrecv        = payrecv
        self.tradedate      = tradedate
        self.notional       = notional
    pass
