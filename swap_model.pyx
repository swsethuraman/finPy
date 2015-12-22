'''
Created on May 26, 2014

@author: SSethuraman
'''
import numpy as np

cdef class VanillaSwapModel:
    cdef public swap
    cdef menv
    cdef public pv
    cdef public settled_cash
    cdef public results
    def __init__(self, swap, menv):
        self.swap           = swap
        self.menv           = menv
        self.pv             = {}
        self.settled_cash   = {}
        self.results        = {}
    pass    
        
    cpdef calculate(self):    
        self.pv             = {}
        self.settled_cash   = {}            
        self.results = {}

        #value legA
        self.value_swapleg('legA')
        #value leg B
        self.value_swapleg('legB')
        
        #append results to the swap
        self.results.update({'pv'           : self.pv['legA'] + self.pv['legB']})
        self.results.update({'settled_cash' : self.settled_cash['legA'] + self.settled_cash['legB']})
        
        if self.swap.legs['legA']['type'] == 'fixed':
            dv01    = self.pv['legA']/(self.swap.legs['legA']['spread']*10000)
            dv01    = np.fabs(dv01)
            self.results.update({'dv01' : dv01})
        else:
            dv01    = self.pv['legB']/(self.swap.legs['legB']['spread']*10000)
            dv01    = np.fabs(dv01)
            self.results.update({'dv01' : dv01})
        
        if self.swap.legs['legA']['type'] == 'fixed':
            par_rate    = np.fabs(self.pv['legB']/dv01)
            self.results.update({'par_rate' : par_rate})
        else:
            par_rate    = np.fabs(self.pv['legA']/dv01)
            self.results.update({'par_rate' : par_rate})
    pass
    
    cpdef value_swapleg(self, leg):
        ncashflows              = len(self.swap.legs[leg]['pay_dates'])
        future_cashflow         = []
        future_cashflow_pv      = []
        past_cashflow           = []
        
        print(self.swap.tenor)
        accrual_start           = self.swap.legs[leg]['accrual_dates'][:-1]
        accrual_end             = self.swap.legs[leg]['accrual_dates'][1:]
        pay_dates               = self.swap.legs[leg]['pay_dates']
        discount                = self.swap.legs[leg]['ccy'] + '_discount_' + self.swap.collateral
        #identify past and future cashflows
        past_pay_dates_index    = [index for index,dates in enumerate(pay_dates) if dates<=self.menv.today]
        future_pay_dates_index  = [index for index,dates in enumerate(pay_dates) if dates>self.menv.today]   
        past_pay_dates          = pay_dates[past_pay_dates_index]
        future_pay_dates        = pay_dates[future_pay_dates_index]
        past_accrual_start      = accrual_start[past_pay_dates_index]
        past_accrual_end        = accrual_end[past_pay_dates_index]
        future_accrual_start    = accrual_start[future_pay_dates_index]
        future_accrual_end      = accrual_end[future_pay_dates_index]    
        
        #calculate settled cash on past cashflows
        if self.swap.legs[leg]['type'] == 'floating':
            fixing_index                = self.swap.legs[leg]['index']
            mult                        = self.swap.legs[leg]['multiplier']
            spread                      = self.swap.legs[leg]['spread']    
            past_fixing_settledates     = self.swap.legs[leg]['fixing_settledates'][past_pay_dates_index]
            past_fixing_rates           = [fixing_rates for (fixing_dates,fixing_rates) in self.menv.fixings[fixing_index] if fixing_dates in past_fixing_settledates]
            past_rate                   = mult*past_fixing_rates + spread*np.ones(len(past_fixing_rates))
        elif self.swap.legs[leg]['type'] == 'fixed':
            spread                      = self.swap.legs[leg]['spread']
            past_rate                   = spread*np.ones(len(past_pay_dates))
        else:
            raise ValueError('Type should be fixed or floating')
        past_cashflow                   = self.swap.legs[leg]['notional']*past_rate*self.swap.legs[leg]['dcf'].dcf(past_accrual_start, past_accrual_end)
        leg_settled_cash                = sum(past_cashflow)
        
        #calculate pv on future cash flows
        if self.swap.legs[leg]['type'] == 'floating':
            float_index                 = self.swap.legs[leg]['index']
            rate_struct                 = self.swap.legs[leg]['ratestruct']
            mult                        = self.swap.legs[leg]['multiplier']
            spread                      = self.swap.legs[leg]['spread']  
            future_fixing_settledates   = self.swap.legs[leg]['fixing_settledates'][future_pay_dates_index]
            is_fixed_fixing_index       = [index for index,dates in enumerate(future_fixing_settledates) if dates < self.menv.today]
            to_fix_fixing_index         = [index for index,dates in enumerate(future_fixing_settledates) if dates >= self.menv.today]

            is_fixed_fixing_dates       = future_fixing_settledates[is_fixed_fixing_index]
            is_fixed_accrual_start      = future_accrual_start[is_fixed_fixing_index]
            is_fixed_accrual_end        = future_accrual_end[is_fixed_fixing_index]
            
            to_fix_fixing_dates         = future_fixing_settledates[to_fix_fixing_index]
            to_fix_accrual_start        = future_accrual_start[to_fix_fixing_index]
            to_fix_accrual_end          = future_accrual_end[to_fix_fixing_index]
            
            is_fixed_fixing_rates       = [fixing_rates for (fixing_dates,fixing_rates) in self.menv.fixings[fixing_index] if fixing_dates in is_fixed_fixing_dates]
            to_fix_fixing_rates         = rate_struct.get_rate(to_fix_fixing_dates,self.menv.observers[float_index])
            future_fixing_rates         = np.append(is_fixed_fixing_rates, to_fix_fixing_rates)
            fwd_rate                    = mult*future_fixing_rates + spread*np.ones(len(future_fixing_rates))

        elif self.swap.legs[leg]['type'] == 'fixed':
            spread                      = self.swap.legs[leg]['spread']
            fwd_rate                    = spread*np.ones(len(future_pay_dates))
        else:
            raise ValueError('Type should be fixed or floating')
        
        future_cashflow                 = self.swap.legs[leg]['notional']*fwd_rate*self.swap.legs[leg]['dcf'].dcf(future_accrual_start, future_accrual_end)
        df                              = self.menv.observers[discount].df(self.menv.today, future_pay_dates)
        future_cashflow_pv              = future_cashflow*df
        leg_pv                          = sum(future_cashflow_pv)
        
        #append results to the swap
        self.pv.update({leg : leg_pv})
        self.settled_cash.update({leg   : leg_settled_cash})
        return
        
cdef class OISwapModel(VanillaSwapModel):
    def __init__(self, swap, menv):   
        super(OISwapModel, self).__init__(swap, menv)
        pass
    cpdef value_swapleg(self, leg):
        ncashflows              = len(self.swap.legs[leg]['pay_dates'])
        future_cashflow         = []
        future_cashflow_pv      = []
        past_cashflow           = []
        
        accrual_start           = self.swap.legs[leg]['accrual_dates'][:-1]
        accrual_end             = self.swap.legs[leg]['accrual_dates'][1:]
        pay_dates               = self.swap.legs[leg]['pay_dates']
        discount                = self.swap.legs[leg]['ccy'] + '_discount_' + self.swap.collateral
        #identify past and future cashflows
        past_pay_dates_index    = [index for index,dates in enumerate(pay_dates) if dates<=self.menv.today]
        future_pay_dates_index  = [index for index,dates in enumerate(pay_dates) if dates>self.menv.today]   
        past_pay_dates          = pay_dates[past_pay_dates_index] 
        future_pay_dates        = pay_dates[future_pay_dates_index]
        past_accrual_start      = accrual_start[past_pay_dates_index]
        past_accrual_end        = accrual_end[past_pay_dates_index]
        future_accrual_start    = accrual_start[future_pay_dates_index]
        future_accrual_end      = accrual_end[future_pay_dates_index]
        
        #calculate settled cash on past cashflows
        if self.swap.legs[leg]['type'] == 'floating':
            fixing_index                = self.swap.legs[leg]['fixing_index']
            mult                        = self.swap.legs[leg]['multiplier']
            spread                      = self.swap.legs[leg]['spread']    
            past_fixing_settledates     = [self.swap.legs[leg]['fixing_settledates'][i] for i in past_pay_dates_index]
            past_subperiod_start        = [self.swap.legs[leg]['sub_period_start'][i] for i in past_pay_dates_index]
            past_subperiod_end          = [self.swap.legs[leg]['sub_period_end'][i] for i in past_pay_dates_index]
            past_fixing_rates           = [[fixing_rates for (fixing_dates,fixing_rates) in self.menv.fixings[fixing_index] if fixing_dates in past_fixing_i] for past_fixing_i in past_fixing_settledates]
            past_rate                   = [[mult*rates + spread*np.ones(len(rates))] for rates in past_fixing_rates]
            past_accrued                = [[past_rate[i]*self.swap.legs[leg]['dcf'].dcf(past_subperiod_start[i], past_subperiod_end[i]) + np.ones(len(past_rate[i]))] for i in past_pay_dates_index]
            past_accrued                = [np.product(past_accrued[i])-1 for i in past_pay_dates_index]
        elif self.swap.legs[leg]['type'] == 'fixed':
            spread                      = self.swap.legs[leg]['spread']
            past_rate                   = spread*np.ones(len(past_pay_dates))
            past_accrued                = past_rate*self.swap.legs[leg]['dcf'].dcf(past_accrual_start, past_accrual_end)
        else:
            raise ValueError('Type should be fixed or floating')
        past_cashflow                   = self.swap.legs[leg]['notional']*past_accrued
        leg_settled_cash                = sum(past_cashflow)
        
        #calculate pv on future cash flows
        if self.swap.legs[leg]['type'] == 'floating':
            float_index                 = self.swap.legs[leg]['fixing_index']
            rate_struct                 = self.swap.legs[leg]['ratestruct']
            mult                        = self.swap.legs[leg]['multiplier']
            spread                      = self.swap.legs[leg]['spread'] 
            future_fixing_settledates   = [self.swap.legs[leg]['fixing_settledates'][i] for i in future_pay_dates_index]
            is_fixed_fixing_index       = [index for index,dates in enumerate(future_accrual_start) if dates < self.menv.today]
            to_fix_fixing_index         = [index for index,dates in enumerate(future_accrual_start) if dates >= self.menv.today]
                      
            is_fixed_fixing_dates       = [future_fixing_settledates[i] for i in is_fixed_fixing_index]

            is_fixed_accrual_start      = future_accrual_start[is_fixed_fixing_index]
            is_fixed_accrual_end        = future_accrual_end[is_fixed_fixing_index]
            #print('to fix')
            to_fix_fixing_dates         = [future_fixing_settledates[i] for i in to_fix_fixing_index]
            to_fix_accrual_start        = future_accrual_start[to_fix_fixing_index]
            to_fix_accrual_end          = future_accrual_end[to_fix_fixing_index]
            
            is_fixed_fixing_rates       = [fixing_rates for (fixing_dates,fixing_rates) in self.menv.fixings[fixing_index] if fixing_dates in is_fixed_fixing_dates]
            to_fix_fixing_rates         = rate_struct.get_rate(to_fix_accrual_start,self.menv.observers[float_index], end_dates=to_fix_accrual_end)
            future_fixing_rates         = np.append(is_fixed_fixing_rates, to_fix_fixing_rates)
            fwd_rate                    = mult*future_fixing_rates + spread*np.ones(len(future_fixing_rates))
        elif self.swap.legs[leg]['type'] == 'fixed':
            spread                      = self.swap.legs[leg]['spread']
            fwd_rate                    = spread*np.ones(len(future_pay_dates))
        else:
            raise ValueError('Type should be fixed or floating')
        
        future_cashflow                 = self.swap.legs[leg]['notional']*fwd_rate*self.swap.legs[leg]['dcf'].dcf(future_accrual_start, future_accrual_end)
        df                              = self.menv.observers[discount].df(self.menv.today, future_pay_dates)
        
        future_cashflow_pv              = future_cashflow*df
        leg_pv                          = sum(future_cashflow_pv)
        
        #append results to the swap
        self.pv.update({leg : leg_pv})
        self.settled_cash.update({leg   : leg_settled_cash})
        return 
