'''
Created on Jul 1, 2014

@author: SSethuraman
'''
import numpy as np
import useful

class LiborDepoStruct:
    def __init__(self, tenor, hcal, adj, dcf, settle_lag):
        self.tenor      = tenor
        self.hcal       = hcal
        self.adj        = useful.adj[adj]
        self.dcf        = dcf
        self.settle_lag = settle_lag
        pass
    
    def value_to_settle(self, value_dates):
        if self.settle_lag >=0:
            adj = useful.adj['P']
            lag = -self.settle_lag
        else:
            adj = useful.adj['F']
            lag = -self.settle_lag
        
        settle_dates = np.busday_offset(value_dates, [lag]*len(value_dates), roll=adj, busdaycal=self.hcal)
        return settle_dates
    
    def get_rate(self, settle_dates, irc):
        if self.settle_lag >=0:
            adj = useful.adj['F']
        else:
            adj = useful.adj['P']
        value_dates = np.busday_offset(settle_dates, [self.settle_lag]*len(settle_dates), roll=adj, busdaycal=self.hcal)
        end_dates   = useful.date_offset(value_dates, self.tenor)
        end_dates   = np.busday_offset(end_dates, 0, roll=self.adj, busdaycal=self.hcal)
        if end_dates[-1] > irc.maturity:
            end_dates[-1] = irc.maturity
        fwd_rate    = irc.fwd(value_dates, end_dates, self.dcf)
        #fwd_sensi   = irc.fwd_sensitivity(value_dates, end_dates, self.dcf)
        return fwd_rate

class OISDepoStruct(LiborDepoStruct):
    def __init__(self, tenor, hcal, adj, dcf, settle_lag):
        super(OISDepoStruct, self).__init__(tenor, hcal, adj, dcf, settle_lag)
        pass
    
    def value_to_settle(self, value_dates):
        if self.settle_lag >=0:
            adj = useful.adj['P']
            lag = -self.settle_lag
        else:
            adj = useful.adj['F']
            lag = -self.settle_lag
        
        settle_dates = np.busday_offset(value_dates, [lag]*len(value_dates), roll=adj, busdaycal=self.hcal)
        return settle_dates
    
    def get_rate(self, start_dates, irc, end_dates=None):
        if end_dates == None:
            if self.settle_lag >=0:
                adj = useful.adj['F']
            else:
                adj = useful.adj['P']
            value_dates = np.busday_offset(start_dates, [self.settle_lag]*len(start_dates), roll=adj, busdaycal=self.hcal)
            end_dates   = useful.date_offset(value_dates, self.tenor)
            end_dates   = np.busday_offset(end_dates, 0, roll=self.adj, busdaycal=self.hcal)
            fwd_rate    = irc.fwd(value_dates, end_dates, self.dcf)
        else:
            fwd_rate    = irc.fwd(start_dates, end_dates, self.dcf)
        return fwd_rate
        
rate_struct_map = {
        'usd_libor_3m'      : LiborDepoStruct('3m', useful.hcal['US'], 'MF', useful.daycount['Act360'], 2),
        'usd_fedfunds_1d'   : OISDepoStruct('1y', useful.hcal['US'], 'MF', useful.daycount['Act360'], 0)
}