'''
Created on May 19, 2014

@author: SSethuraman
'''
import ircb_curvedef
import ircb_init
import ircb_data
import scipy.interpolate
import scipy.optimize as opt
import datetime
import env
import swap
import swap_model
import sys
import numpy as np
cimport numpy as cnp
import cython
cimport cython
import xll
import ircb_interp

inst_model = {
    'swap'      : swap_model.VanillaSwapModel,
    'ois_swap'  : swap_model.OISwapModel
}

cdef class Irc:
    cpdef df(self, pvdate, date):
        pass
    cpdef fwd(self, startdate, enddate, dcf):
        pass
    
cdef class DummyIrc(Irc):
    cpdef df(self, pvdate,date):    
        return 1
    cpdef fwd(self,startdate,enddate, dcf):
        return 0

@cython.boundscheck(False)
cdef class SimpleIrc(Irc):
    cdef public unicode name
    cdef public int ndates
    cdef public int version
    cdef unsigned long today
    cdef public object maturity
    cdef public object tday
    cdef cnp.int64_t [:] dates
    cdef cnp.double_t [:] zero_dates
    cdef public object zero_calc
    def __init__(self, dates, zero_dates, today, name):
        self.name = name
        self.version    = 1
        self.tday       = today
        self.today      = today.view('i8')
        self.dates      = dates.view('i8')
        self.ndates     = len(self.dates)
        self.maturity   = dates[-1]
        self.zero_dates = zero_dates
        diff = np.subtract(self.dates, [self.dates[0]]*len(self.zero_dates))

        timefrac        = 360.0
        dates_diff      = diff/timefrac
        self.zero_calc = ircb_interp.cubic_monotone(dates_diff, self.zero_dates)
    @cython.boundscheck(False)           
    cpdef df(self, pvdate, date):    
        delta1 = date.view('i8')   - self.today 
        delta2 = pvdate.view('i8') - self.today
        dd1    = delta1/360.0#/np.timedelta64(360,'D')
        dd2    = delta2/360.0#/np.timedelta64(360,'D')
        #fac1 = self.zero_calc(dd1) * dd1
        #fac2 = self.zero_calc(dd2) * dd2
        fac1 = self.zero_calc.interp(dd1) * dd1
        fac2 = self.zero_calc.interp(dd2) * dd2
        return np.exp(-fac1 + fac2)
    
    @cython.boundscheck(False)
    cpdef fwd(self, startdate, enddate, dcf):
        d = self.df(startdate, enddate)
        return (1/d-1)/dcf.dcf(startdate, enddate)
    
    @cython.boundscheck(False)
    cpdef fwd_sensitivity(self, startdate, enddate, dcf):
        length                  = len(startdate)
        today_array             = np.ndarray([1, length], dtype='datetime64[D]')
        today_array.fill(self.tday)
        d1                      = self.df(today_array[0], startdate)
        d2                      = self.df(today_array[0], enddate)
        sensitivity_startdate   = (1/d1)/dcf.dcf(startdate, enddate)
        sensitivity_enddate     = (-d1/(d2*d2))/dcf.dcf(startdate, enddate)
        result                  = [startdate, enddate, sensitivity_startdate, sensitivity_enddate]
        return result
    
    @cython.boundscheck(False)
    cpdef update_zeros(self, name, zero_dates):
        if self.name == name and len(self.dates)==len(self.zero_dates):
            self.zero_calc.update(zero_dates)
        else:
            raise ValueError('Curves are no matching or lengths of zeros doesnt match the dates')
    
def inst_dependcurve(inst, curves):
    collateral              = inst['collateral']
    payleg_ccy              = inst['payleg_ccy']
    payleg_discount_curve   = payleg_ccy + '_discount_' + collateral
    recvleg_ccy             = inst['recvleg_ccy']
    recvleg_discount_curve  = recvleg_ccy + '_discount_' + collateral
    if inst['payleg_type'] == 'float':
        payleg_index_curve  = inst['payleg_index_curve']
    else:
        payleg_index_curve = None
    if inst['recvleg_type'] == 'float':
        recvleg_index_curve = inst['recvleg_index_curve']
    else:
        recvleg_index_curve = None 
    
    curvelist = [payleg_discount_curve, recvleg_discount_curve, payleg_index_curve, recvleg_index_curve]
    curvelist = [curve for curve in curvelist if curve is not None]    
    curvelist = [curve for curve in map(ircb_curvedef.curve_map, curvelist)]
    curvelist = list(set(curvelist))
    
    new_curves = []
    for curve in curvelist:
        if not curve in curves:
            new_curves.append(curve)
    new_curves = list(set(new_curves) - set(curves)) 
    all_curves = curves + new_curves
    
    if not new_curves:
        return [] 
    
    final_curves = []
    for curve in new_curves:
        inst_list = ircb_curvedef.curvedef_map[curve]
        aggregate_curves = []
        for inst_new in inst_list:
            add_curves = inst_dependcurve(inst_new['inst'], all_curves)
            aggregate_curves = aggregate_curves + list(set(add_curves) - set(aggregate_curves)) 
        final_curves = final_curves + list(set(aggregate_curves) - set(final_curves))
    return final_curves

def curve_dependcurve(curve_list):
    temp_list = []
    for curve in curve_list:
        mapped_curve =  ircb_curvedef.curve_map(curve)
        if not mapped_curve in temp_list: temp_list = temp_list + [mapped_curve]
    curve_list = temp_list
    
    curve_set = set(curve_list)
    for curve in curve_list:
        new_curves = ircb_curvedef.curve_depends[curve]
        curve_set.update(new_curves) 
        
    add_curves = list(curve_set - set(curve_list))
    return add_curves + curve_list

cdef class IrcBuilder:
    def curvebuild(self,curve_string, menv):
        pass

cdef class DummyIrcBuilder(IrcBuilder):
        def curvebuild(self, curve_string):
            dummy = DummyIrc()
            return dummy

cdef class SimpleIrcBuilder(IrcBuilder):
        cdef public object curvelist
        cdef public object menv
        cdef public object dataset
        cdef public int version
        def __init__(self, curvelist, menv, dataset):
            self.version    = 1
            self.curvelist  = curvelist
            self.menv       = menv
            self.dataset    = dataset
            self.dataset.attach(self)
            self.curvebuild(curvelist, menv, dataset)
        def update(self, dataset):
            self.curvebuild(self.curvelist, self.menv, self.dataset)
            self.menv.version = self.menv.version + 1
            #xll.update_menv(self.menv)
        def curvebuild(self, curve_list, menv, dataset):
            # obtain a list of all required curves
            curves = curve_dependcurve(curve_list)
            curves = sorted(curves, key= lambda x:ircb_curvedef.curve_key[x])
            today = menv.today
            #figure out the geometric points of the curve
            curve_inst = {}
            curve_df_dates = {}
            curve_list = []
            #curve_df = {}
            irc = []
            curve_dict = {}
            num_df = 0
            ncurves = len(curves)
            ninst   = np.empty(ncurves, dtype=int)
            count = 0
            for curve in curves:
                curve_list.append(curve)
                inst_block = ircb_curvedef.curvedef_map[curve]
                curve_dates = []
                for block in inst_block:
                    inst_spec   = block['inst']
                    inst_tenors = block['tenors']
                    inst_type   = inst_spec['inst_type']
                    inst_name   = inst_spec['inst_name']
                    inst_list = []
                    for tenor in inst_tenors:
                        inst_tenor          = tenor
                        inst_rate           = dataset.data[inst_name][tenor]
                        inst_string         = inst_type+ "_" + inst_name + "_" + inst_tenor
                        inst_list.append((inst_string, ircb_init.inst_initializer[inst_type](today, inst_rate, inst_spec, inst_tenor)))
                        curve_dates.append(inst_list[-1][1].maturity)
                curve_inst[curve] = inst_list
                ninst[count] = len(inst_list)
                curve_dict[curve]=count
                curve_df_dates[curve] = np.asarray([today] + curve_dates)  
                num_df = num_df + len(curve_df_dates[curve])
                irc.append(SimpleIrc(curve_df_dates[curve], np.zeros(len(curve_df_dates[curve])), today, curve))
                menv.attach(curve, irc[count])
                count = count + 1
            menv.attach('usd_discount_usd',irc[curve_dict['usd_fedfunds_1d']]) #need to streamline this part. Need to attach curves that were mapped to this curve as well.
            max_inst = np.max(ninst)
            inst_obj = np.empty([ncurves, max_inst], dtype=object)
                
            i = 0
            for curve in curves:
                j = 0
                for inst in curve_inst[curve]:
                    inst_obj[i][j] = inst_model[inst[1].inst_type](inst[1], menv)
                    j = j + 1
                i = i + 1
                
            for i in np.arange(ncurves):
                init = np.empty(irc[i].ndates-1)
                init.fill(0.0015)
                res = residual(1, ninst[i], inst_obj[i,:], irc[i], menv)
                #sol = opt.root(res.calculate, init, method='lm')
                sol = opt.fsolve(res.calculate, init)
                #print(sol)
                #print(sol1)
            d_irc = {}
            for i in np.arange(ncurves):
                d_irc[curves[i]] = irc[i]
            #return d_irc
                
            
cdef class residual:
        cdef int ncurves
        cdef int ninst
        cdef int n_err
        cdef inst_array
        cdef menv
        cdef irc
        def __init__(self, ncurves, ninst, inst_array, irc, menv ):
                      
            self.ncurves            = ncurves
            self.ninst              = ninst
            self.inst_array         = inst_array
            self.irc                = irc
            self.menv               = menv
            self.n_err              = self.ninst
            pass
        cpdef cnp.ndarray[cnp.double_t, ndim=1] calculate(self, cnp.ndarray[cnp.double_t, ndim=1] zeros):
            cdef int low_index   = 0
            cdef int high_index  = 0
            cdef int n_df
            cdef cnp.ndarray[cnp.double_t, ndim =1] zeros_temp
            cdef cnp.ndarray[cnp.double_t, ndim=1] err = np.zeros(self.n_err)
            cdef cnp.ndarray[cnp.double_t, ndim=1] err_temp
            
            for curve in np.arange(self.ncurves):
                    n_df          = self.irc.ndates-1
                    high_index    = high_index + n_df
                    zeros_temp    = zeros[low_index:high_index]
                    self.irc.update_zeros(self.irc.name, np.concatenate(([0.0005],zeros_temp),axis=0)) # need to generazlie this part
                    low_index     = high_index
            for i in np.arange(self.ncurves):
                err_temp = np.zeros(self.ninst)
                for j in np.arange(self.ninst):
                    self.inst_array[j].calculate()
                    err_temp[j] = self.inst_array[j].results['pv']
                err = err_temp
            return err
            pass
