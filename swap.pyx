'''
Created on May 19, 2014

@author: SSethuraman
'''

import numpy as np
import useful
import ircb_inst 
import rate_struct as rs
import env
import holidays as hol 

class VanillaSwapBuilder:
    def __init__(self, tradedate, notional, fixed_rate, payrecv, swapspec, tenor=None, start0=None, firstreg=None, lastreg=None, end0=None):
        # consistency checks
        if swapspec['legA_type'] == 'float' and swapspec['legA_index_curve'] == None:
            raise ValueError('Float leg should have a non null index')
        elif swapspec['legA_type'] == 'float' and swapspec['legA_multiplier'] == None:
            raise ValueError('Float leg should have non null multiplier')
        elif swapspec['legA_type'] == 'float' and swapspec['legA_fixing_index'] == None:
            raise ValueError('Float leg should have non null fixing index')
        
        if swapspec['legB_type'] == 'float' and swapspec['legB_index_curve'] == None:
            raise ValueError('Float leg should have a non null index')
        elif swapspec['legB_type'] == 'float' and swapspec['legB_multiplier'] == None:
            raise ValueError('Float leg should have non null multiplier')
        elif swapspec['legB_type'] == 'float' and swapspec['legB_fixing_index'] == None:
            raise ValueError('Float leg should have non null fixing index')
        
        if swapspec['legA_spread'] != None:
            raise ValueError('LegA should have no spread')
        elif swapspec['legB_spread'] == None:
            raise ValueError('LegB should have a non-null spread')
        
        if tenor == None and (start0 == None and end0 == None and firstreg == None and lastreg == None):
            raise ValueError('If tenor is not specified, then the swap is irregular and the four day swap schedule must be specified')
        if tenor != None and (start0 != None or end0 != None or firstreg != None or lastreg != None):
            raise ValueError('If tenor is specified, then the swap is regular and the four day swap schedule will be ignored')
        # end of consistency checks
        
        # If tenor is specified, generate start and end dates from the trade date and the calendar
        self.settle_days = swapspec['settle_days']
        self.settle_hcal = swapspec['settle_hcal']
        self.tradedate   = tradedate
        self.collateral  = swapspec['collateral']
        self.inst_type   = swapspec['inst_type']
        self.tenor       = tenor
        self.notional     = notional
        self.payrecv     = payrecv
        self.fixed_rate  = fixed_rate
        self.swapspec    = swapspec
        
        if tenor != None:
            self.isregular  = True
            startdate       = self.tradedate #+ np.timedelta64(self.settle_days, 'D')
            #settle day adjustment is always following, modified following doesn't make sense
            self.startdate  = np.busday_offset(startdate, self.settle_days, roll='following', busdaycal=useful.hcal[self.settle_hcal])
            self.start0     = self.startdate
            startmonth      = np.datetime64(startdate, 'M')
            ndays           = startdate - startmonth
        
            if tenor[-1:]   == 'Y':
                enddate     = startmonth + np.timedelta64(int(tenor[:-1]), 'Y')
                enddate     = enddate + ndays
            elif tenor[-1:] == 'M':
                enddate     = startmonth + np.timedelta64(int(tenor[:-1]), 'M')
                enddate     = enddate + ndays
            elif tenor[-1:] == 'D':
                enddate     = startdate + np.timedelta64(int(tenor[:-1]), 'D')
            elif tenor[-1:] == 'W':
                enddate     = startdate + np.timedelta64(int(tenor[:-1]), 'W')
            else:
                raise ValueError('Unknown code')
            self.end0       = enddate
            self.enddate    = np.busday_offset(self.end0, 0, roll='modifiedfollowing',busdaycal=useful.hcal[swapspec['legA_hcal']])
            self.firstreg   = self.start0
            self.lastreg    = self.end0
        else:
            self.isregular  = False
            self.start0     = start0
            self.startdate  = start0
            self.end0       = end0
            self.firstreg   = firstreg
            self.lastreg    = lastreg
        
        self.legs = {}    
        # Set up Leg A
        if swapspec['quote_leg'] == 'legA':
            legA_payrecv = payrecv
        else:
            if payrecv == 'pay':
                legA_payrecv = 'recv'
            else:
                legA_payrecv = 'pay'
        self.make_SwapLegDetails(swapspec, 'legA', notional, legA_payrecv, fixed_rate)
        if swapspec['legA_type'] == 'fixed':
            self.make_FixedLegSchedule('legA')
        elif swapspec['legA_type'] == 'floating':
            self.make_FloatingLegSchedule('legA')
        else:
            raise ValueError('Unrecognized swap leg type - needs to be either fixed or floating')
            
        # Set up Leg B
        if swapspec['quote_leg'] == 'legB':
            legB_payrecv = payrecv
        else:
            if payrecv == 'pay':
                legB_payrecv = 'recv'
            else:
                legB_payrecv = 'pay'
        self.make_SwapLegDetails(swapspec, 'legB', notional, legB_payrecv, fixed_rate)
        if swapspec['legB_type'] == 'fixed':
            self.make_FixedLegSchedule('legB')
        elif swapspec['legB_type'] == 'floating':
            self.make_FloatingLegSchedule('legB')
        else:
            raise ValueError('Unrecognized swap leg type - needs to be either fixed or floating')     

        if self.legs['legA']['pay_dates'][-1] >= self.legs['legB']['pay_dates'][-1]:
            self.maturity       = self.legs['legA']['pay_dates'][-1]
        else:
            self.maturity       = self.legs['legB']['pay_dates'][-1]
    pass
    
    def make_SwapLegDetails(self, swapspec, leg, notional, payrecv, fixed_rate):                           
        self.legs[leg] = {
            'ccy'          : swapspec[leg + '_ccy'],
            'freq'         : swapspec[leg + '_freq'],
            'dcf'          : useful.daycount[swapspec[leg + '_dcf']],
            'accrual_hcal' : useful.hcal[swapspec[leg + '_accrual_hcal']],
            'adjust'       : swapspec[leg + '_adjust'],
            'pay_hcal'     : useful.hcal[swapspec[leg + '_hcal']],
            'pay_lag'      : swapspec[leg + '_paylag'],
            'type'         : swapspec[leg + '_type']
        }
        if self.legs[leg]['type'] == 'floating':
            float_leg_details = {
                'index'             : swapspec[leg + '_index_curve'],
                'ratestruct_spec'   : swapspec[leg + '_ratestruct'],
                'ratestruct'        : rs.rate_struct_map[swapspec[leg+'_ratestruct']],
                'fixingfreq'        : swapspec[leg + '_fixingfreq'],
                'fixing_hcal'       : useful.hcal[swapspec[leg + '_fixinghcal']],
                'fixing_index'      : swapspec[leg + '_fixing_index'],
                'fixinglag'         : swapspec[leg + '_fixinglag'],
                'multiplier'        : swapspec[leg + '_multiplier']
            }
            self.legs[leg].update(float_leg_details)
        if swapspec[leg + '_spread'] != None:
            spread              = fixed_rate
        else:
            spread              = 0
        spread_details = {
            'spread'            : spread
        }
        self.legs[leg].update(spread_details)
        if payrecv  == 'pay':
            notnl   = -notional
        elif payrecv == 'recv':
            notnl   = notional
        else:
            raise ValueError('Choose either pay or recv')
        notional_details = {
            'notional'  : notnl
        }  
        self.legs[leg].update(notional_details)
        return
    
    def make_FixedLegSchedule(self, leg):
        accrual_dates       = date_schedule(self.firstreg, self.lastreg, self.legs[leg]['freq'], self.legs[leg]['accrual_hcal'], self.legs[leg]['adjust'], self.start0, self.end0)
        pay_dates           = accrual_dates[1:]
        pay_dates           = np.busday_offset(pay_dates, self.legs[leg]['pay_lag'], roll=useful.adj['F'], busdaycal=self.legs[leg]['pay_hcal'])
        leg_add = {        
            'accrual_dates' : accrual_dates,
            'pay_dates'     : pay_dates
        }
        self.legs[leg].update(leg_add)
        return
    
    def make_FloatingLegSchedule(self, leg):
        if self.legs[leg]['freq'] == self.legs[leg]['fixingfreq']: 
            accrual_dates       = date_schedule(self.firstreg, self.lastreg, self.legs[leg]['freq'], self.legs[leg]['accrual_hcal'], self.legs[leg]['adjust'], self.start0, self.end0)
            pay_dates           = accrual_dates[1:]
            pay_dates           = np.busday_offset(pay_dates, self.legs[leg]['pay_lag'], roll=useful.adj['F'], busdaycal=self.legs[leg]['pay_hcal'])
            accrual_start       = accrual_dates[:-1]
            accrual_end         = accrual_dates[1:]
            # check this logic
            fixing_settledates = self.legs[leg]['ratestruct'].value_to_settle(accrual_start)                                                                                                                      
            leg_add = {        
                'accrual_dates'         : accrual_dates,
                'pay_dates'             : pay_dates,
                'fixing_settledates'    : fixing_settledates
            }
            self.legs[leg].update(leg_add)
        else:
            raise ValueError('Vanilla swap builder doesnt handle fixing frequency different from payment frequency')                                                                                                                        
        return

class OISwapBuilder(VanillaSwapBuilder):
    def __init__(self, tradedate, notional, fixed_rate, payrecv, swapspec, tenor='5Y', start0=None, firstreg=None, lastreg=None, end0=None):
        super(OISwapBuilder, self).__init__(tradedate, notional, fixed_rate, payrecv, swapspec, tenor, start0=None, firstreg=None, lastreg=None, end0=None)
        pass
    
    def make_FloatingLegSchedule(self, leg):
        if self.legs[leg]['fixingfreq'] == '1d':
            accrual_dates       = date_schedule(self.firstreg, self.lastreg, self.legs[leg]['freq'], self.legs[leg]['accrual_hcal'], self.legs[leg]['adjust'], self.start0, self.end0)
            pay_dates           = accrual_dates[1:]
            pay_dates           = np.busday_offset(pay_dates, self.legs[leg]['pay_lag'], roll=useful.adj['F'], busdaycal=self.legs[leg]['pay_hcal'])
            accrual_start       = accrual_dates[:-1]
            accrual_end         = accrual_dates[1:]
            accrual_sub_periods = [date_schedule(accrual_start[i], accrual_end[i], '1D', self.legs[leg]['fixing_hcal'], 'F') for i in np.arange(0,len(accrual_start))]
            sub_period_start    = [accrual_sub_periods[i][:-1] for i in np.arange(len(accrual_sub_periods))]
            sub_period_end      = [accrual_sub_periods[i][1:] for i in np.arange(len(accrual_sub_periods))]
            #sub_period_start    = 
            if self.legs[leg]['fixinglag'] > 0:
                lag     = -self.legs[leg]['fixinglag']
                adj     = useful.adj['P']
            else:
                lag     = self.legs[leg]['fixinglag']
                adj     = useful.adj['F']
            fixing_settledates  = [np.busday_offset(accrual_sub_periods[i][:-1], lag, roll=adj, busdaycal=self.legs[leg]['fixing_hcal'])  for i in np.arange(0,len(accrual_sub_periods))] 
            
            leg_add = {        
                'accrual_dates'         : accrual_dates,
                'pay_dates'             : pay_dates,
                'fixing_settledates'    : fixing_settledates,
                'sub_period_start'      : sub_period_start,
                'sub_period_end'        : sub_period_end
            }
            self.legs[leg].update(leg_add)
        else:
            raise ValueError('OIS swap builder handles only 1D fixing frequency')
        return


def date_schedule(startdate, enddate, freq, hcal, adjust, start0=None, end0=None):
    sched = []
    startmonth  = np.datetime64(startdate, 'M')
    endmonth    = np.datetime64(enddate,'M')
    #hcal        = useful.hcal[holiday]
    
    ndays       = int(round((startdate - startmonth)/np.timedelta64(1,'D')))
    n           = int(freq[:-1])
    
    if freq[-1]     == 'm' or freq[-1] == 'M':
        dstring     = 'M'
        sched       = [ date + np.timedelta64(ndays, 'D') for date in np.arange(startmonth, endmonth, np.timedelta64(n, dstring))]       
    elif freq[-1]   == 'y' or freq[-1] == 'Y':
        dstring     = 'Y'
        sched       = [ date + np.timedelta64(ndays, 'D') for date in np.arange(startmonth, endmonth, np.timedelta64(n, dstring))]
    elif freq[-1]   == 'd' or freq[-1] == 'D':
        dstring     = 'D'
        sched       = np.arange(startdate, enddate, np.timedelta64(n, dstring))
    elif freq[-1]   == 'w' or freq[-1] == 'W':
        dstring     = 'W'
        sched       = [ date + np.timedelta64(ndays, 'D') for date in np.arange(startmonth, endmonth, np.timedelta64(n, dstring))]
    else:
        raise ValueError('Unknown code')
    
    if sched == []: sched = [startdate]
        
    sched   = np.append(sched, [enddate])
    if start0 != None and start0 != startdate:
        sched.insert(0, start0)
    if end0 != None and end0 != enddate:
        sched.append(end0)
        
    n_sched = len(sched)
    sched   = np.busday_offset(sched, n_sched*[0], roll=useful.adj[adjust], busdaycal=hcal)
    sched   = np.array(sched, dtype='datetime64[D]')
    sched   = np.unique(sched)
    return sched
                                