'''
Created on Jun 25, 2014

@author: SSethuraman
'''
from pyxll import xl_func, xl_menu, get_active_object, xlfCaller, xl_macro, xlfGetDocument
import pyximport
import env
import numpy as np
import scipy.misc as sc
import objectcache
import ircb
import random
import ircb_dataset
import ircb_inst
import ircb_init
import useful
import datetime
import ircb_interp
import irvc
import swaption
import swaption_model
import holidays as hol
from functools import reduce
from scipy.optimize import root

try:
    import win32com.client
    _have_win32com = True
except ImportError:
    _log.warning("*** win32com.client could not be imported           ***")
    _log.warning("*** some of the object cache examples will not work  ***")
    _log.warning("*** to fix this, install the pywin32 extensions     ***")
    _have_win32com = False

pyximport.install(setup_args={"script_args":["--compiler=mingw32"]}, reload_support=True)

def xl_app():
    """returns a Dispatch object for the current Excel instance"""
    # get the Excel application object from PyXLL and wrap it
    xl_window = get_active_object()
    xl_app = win32com.client.Dispatch(xl_window).Application
    # it's helpful to make sure the gen_py wrapper has been created
    # as otherwise things like constants and event handlers won't work.
    win32com.client.gencache.EnsureDispatch(xl_app)
    return xl_app

@xl_func("date[]: cached_object")
def build_menv(x):
    """returns a Market Env object"""
    today         = np.datetime64(x[0][0], 'D')
    builddate     = np.datetime64(x[0][1], 'D')
    datadate      = np.datetime64(x[0][2], 'D')
    pvdate        = np.datetime64(x[0][3], 'D')
    menv          = env.MktEnv(today, builddate, datadate, pvdate)
    return menv

@xl_func("cached_object menv:cached_object", volatile=True)
def update_menv(menv):
    # update the cache and return the cached object id
    return menv

@xl_func("string[] curves, cached_object menv, cached_object dataset: cached_object")
def build_curves(curves, menv, dataset):
    """returns a a list of interest rate curve objects"""
    #curves = curves[0]
    sm = ircb.SimpleIrcBuilder(curves, menv, dataset)
    return sm

@xl_func("string[] curves, cached_object menv, cached_object dataset: cached_object")
def build_curves_excel(curves, menv, dataset):
    """returns a a list of interest rate curve objects"""
    curves = curves[0]
    sm = ircb.SimpleIrcBuilder(curves, menv, dataset)
    return sm

@xl_func("string curve, cached_object menv: cached_object")
def get_ircurve(curve, menv):
    return menv.observers[curve]

@xl_func("date pvdate, date enddate, cached_object curve: float")
def ircurve_df(pvdate, enddate, curve):
    pvdate  = np.datetime64(pvdate, 'D')
    enddate = np.datetime64(enddate, 'D')
    return curve.df(pvdate, enddate)

@xl_func("string name: cached_object")
def ircb_init_dataset(name):
    return ircb_dataset.DataSet(name)

@xl_func("string name: cached_object")
def irvc_init_dataset(name):
    return ircb_dataset.DataSet(name)

@xl_func("string[] tenors, float[] rate, string curve, cached_object dataset, cached_object menv: cached_object")
def ircb_setdata(tenors, rates, curves, dataset):
    i = 0
    rate_set = []
    for tenor in tenors:
        rate_set.append(dict(zip(tenor, rates[i])))
        i = i + 1
    dataset.add(curves, rate_set)
    return dataset

@xl_func("string[] tenors, float[] moneyness, float[][] normal_vol, string vol_cube, cached_object dataset: cached_object")
def irvc_set_voldata(tenors, moneyness, normal_vol, vol_cube, dataset):
    i = 0
    rate_set = {}
    for tenor in tenors[0]:
        rate_set[tenor] = dict(zip(moneyness[0], normal_vol[i]))
        i = i + 1
    dataset.add([vol_cube], [rate_set])
    return dataset

@xl_func("float[] array_2d, string corr_cube, string array_desc, cached_object dataset: cached_object")
def irvc_set_corrdata(array_2d, corr_cube, array_desc, dataset):
    array_add = {}
    array_add[array_desc] = array_2d
    dataset.add([corr_cube], [array_add])
    return dataset


@xl_func("string[] tenor, float[] factor, string corr_cube, string array_desc, cached_object dataset: cached_object")
def irvc_set_corrfactor(tenor, factor, corr_cube, array_desc, dataset):
    array_add = {}
    array_add[array_desc] = dict(zip(tenor[0], factor[0]))
    dataset.add([corr_cube], [array_add])
    return dataset

@xl_func("string[] tenor1, string[] tenor2, float[][] array_2d, string corr_cube, cached_object dataset: cached_object")
def irvc_set_corrgrid(tenor1, tenor2, array_2d, corr_cube, dataset):
    array_np = np.array(array_2d)
    array_add = {}
    i = 0
    for tenor in tenor1[0]:
        array_add[tenor] = dict(zip(tenor2[0], array_np[:,[i]]))
        i = i + 1
    dataset.add([corr_cube], [array_add])
    return dataset

@xl_func("string[] expiry, string[] tenors, string param_string, float[][] param_grid, cached_object dataset: cached_object")
def irvc_set_SABRparams(expiry, tenors, param_string, param_grid, dataset):
    i = 0
    value_set = {}
    for exp in expiry[0]:
        value_set[exp] = dict(zip(tenors[0], param_grid[i]))
        i = i + 1
    dataset.add([param_string], [value_set])
    return dataset

@xl_func("string[] expiry, string[] corr_params, string param_string, float[][] param_grid, cached_object dataset: cached_object")
def irvc_set_SwaptionCorrParams(expiry, corr_params, param_string, param_grid, dataset):
    i = 0
    value_set = {}
    for exp in expiry[0]:
        value_set[exp] = dict(zip(corr_params[0], param_grid[i]))
        i = i + 1
    dataset.add([param_string], [value_set])
    return dataset

@xl_func("string param_string, cached_object corr_grid, cached_object dataset: cached_object")
def irvc_set_SwaptionCorrGrid(param_string, corr_grid, dataset):
    value_set = corr_grid.data
    dataset.add([param_string], [value_set])
    return dataset

@xl_func("date today, string[] expiry, string[] tenors, cached_object alpha, cached_object beta, cached_object rho, cached_object sigma0, cached_object corr: cached_object")
def make_VolCube(today, expiry, tenors, alpha, beta, rho, sigma0, corr):
    volcube = irvc.VolCube(today, expiry[0], tenors[0], alpha, beta, rho, sigma0, corr)
    return volcube
        
@xl_func("string[] expiry, string[] tenors, float[][] atm_rate, string swap_data, cached_object dataset: cached_object")
def irvc_set_swapdata(expiry, tenors, atm_rate, swap_data, dataset):
    i = 0
    rate_set = {}
    for exp in expiry[0]:
        rate_set[exp] = dict(zip(tenors[0], atm_rate[i]))
        i = i + 1
    dataset.add([swap_data], [rate_set])
    return dataset

@xl_func("cached_object vol_dataset, cached_object swap_dataset, cached_object menv: cached_object")
def irvc_calibrate_volcube(vol_dataset, swap_dataset, menv):
    return irvc.SABR_Calibration_VolSlice(vol_dataset, swap_dataset, menv)

@xl_func("cached_object corr_dataset, cached_object weights_dataset, cached_object factor_dataset: cached_object")
def irvc_calibrate_corrcube(corr_dataset, weights_dataset, factor_dataset):
    return irvc.Corr5P_Calibration_CorrSlice(corr_dataset, weights_dataset, factor_dataset)


@xl_func("cached_object SABR_Calib_Params, string expiry, string tenor, string param: float")
def irvc_results(SABR_Calib_Params, expiry, tenor, param):
    if param == 'alpha':
        return SABR_Calib_Params[expiry][tenor]['x'][0]
    elif param == 'rho':
        return SABR_Calib_Params[expiry][tenor]['x'][1]
    elif param == 'sigma0':
        return SABR_Calib_Params[expiry][tenor]['x'][2]

@xl_func("cached_object Corr_Calib_Params, string expiry, string param: float")
def irvc_corr_results(Corr_Calib_Params, expiry, param):
    if param == 'p_inf':
        return Corr_Calib_Params[expiry]['x'][0]
    elif param == 'p_beta':
        return Corr_Calib_Params[expiry]['x'][1]
    elif param == 'p_alpha':
        return Corr_Calib_Params[expiry]['x'][2]
    elif param == 'p_gamma':
        return Corr_Calib_Params[expiry]['x'][3]
    elif param == 'p_delta':
        return Corr_Calib_Params[expiry]['x'][4]


@xl_func("float F0, float T, float beta: cached_object")
def util_SABR_CalibK(F0, T, beta):
    return irvc.SABRCalibrate_K(F0, T, beta)

@xl_func("float alpha, float rho, float sigma0, float K, cached_object f: float")
def util_SABR_CalibNVol(alpha, rho, sigma0, K, f):
    x = [alpha, rho, sigma0, K]
    return f(x)

@xl_func("float alpha, float beta, float rho, float F0, float K, float T, float sigB: float")
def util_SABR_initVol(alpha, beta, rho, F0, K, T, sigB):
    sol = irvc.SABR_initVol(alpha, beta, rho, F0, K, T, sigB)
    return sol.x[0]

@xl_func("string spec: cached_object")
def get_swapspec(spec):
    return ircb_inst.inst_speclist[spec]

@xl_func("date tradedate, float notional, float fixed_rate, string payrecv, cached_object swapspec, string tenor: cached_object")
def make_swap(tradedate, notional, fixed_rate, payrecv, swapspec, swap_tenor):
    swaptype    = swapspec['inst_type']
    tradedate   = np.datetime64(tradedate, 'D')
    swap        = ircb_init.inst_builder[swaptype](tradedate, notional, fixed_rate, payrecv, swapspec, tenor=swap_tenor)
    return  swap

@xl_func("date tradedate, cached_object swapspec, float notional, string payrecv, float strike, date expiry, string tenor: cached_object")
def make_swaption(tradedate, swapspec, notional, payrecv, strike, expiry, tenor):
    tradedate   = np.datetime64(tradedate, 'D')
    expiry      = np.datetime64(expiry, 'D')
    swpt        = swaption.SwaptionBuilder(tradedate, swapspec, notional, payrecv, strike, expiry, tenor)
    return  swpt

@xl_func("date tradedate, cached_object swapspec, float notional, string payrecv, float strike, date expiry, string tenor: cached_object")
def make_midcurve_swaption(tradedate, swapspec, notional, payrecv, strike, expiry, tenor):
    tradedate   = np.datetime64(tradedate, 'D')
    expiry      = np.datetime64(expiry, 'D')
    swpt        = swaption.MidCurveSwaptionBuilder(tradedate, swapspec, notional, payrecv, strike, expiry, tenor)
    return  swpt

@xl_func("date tradedate, float notional, float fixed_rate, string payrecv, cached_object swapspec, date start0, date first_reg, date last_reg, date end0: cached_object")
def make_irregular_swap(tradedate, notional, fixed_rate, payrecv, swapspec, start0, first_reg, last_reg, end0):
    swaptype        = swapspec['inst_type']
    tradedate       = np.datetime64(tradedate, 'D')
    sw_start0       = np.datetime64(start0, 'D')
    sw_first_reg    = np.datetime64(first_reg, 'D')
    sw_last_reg     = np.datetime64(last_reg, 'D')
    sw_end0         = np.datetime64(end0, 'D')
    swap            = ircb_init.inst_builder[swaptype](tradedate, notional, fixed_rate, payrecv, swapspec, start0=sw_start0, firstreg=sw_first_reg, lastreg=sw_last_reg, end0=sw_end0)
    return  swap

@xl_func("cached_object swap_model, float target_dv01: float")
def swap_notional_from_dv01(swap_model, target_dv01):
    swap_model.calculate()
    notional        = swap_model.swap.legs['legA']['notional']
    dv01            = swap_model.results['dv01']
    target_notional = notional*target_dv01/dv01
    return target_notional

@xl_macro("cached_object swap_model, float target_dv01: float")
def swp_notional_from_dv01(swap_model, target_dv01):
    swap_model.calculate()
    notional        = swap_model.swap.legs['legA']['notional']
    dv01            = swap_model.results['dv01']
    target_notional = notional*target_dv01/dv01
    return target_notional

@xl_func("cached_object swap, cached_object menv: cached_object")
def px_swap(swap, menv):
    inst_type  = swap.inst_type
    swap_model = ircb.inst_model[inst_type]
    return swap_model(swap, menv)

@xl_func("cached_object swaption, cached_object menv, cached_object volcube: cached_object")
def px_swaption(swaption, menv, volcube):
    swapt_model = swaption_model.SwaptionModel(swaption, menv, volcube)
    return swapt_model

@xl_func("cached_object swaption, cached_object menv, cached_object volcube, cached_object model_params: cached_object")
def px_midcurve_swaption(swaption, menv, volcube, model_params):
    swapt_model = swaption_model.MidCurveSwaptionModel(swaption, menv, volcube, model_params)
    return swapt_model

@xl_func("cached_object swap_model, string result_str: float")
def px_result(swap_model, result_str):
    swap_model.calculate()
    if result_str == 'pv':
        return swap_model.results['pv']
    elif result_str == 'dv01':
        return swap_model.results['dv01']
    elif result_str == 'par_rate':
        return swap_model.results['par_rate']
    else:
        return -1000000000

@xl_func("cached_object swaption_model, string result_str: float")
def px_swaption_result(swaption_model, result_str):
    #swap_model.calculate()
    if result_str == 'pv':
        return swaption_model.results['pv']
    elif result_str == 'delta':
        return swaption_model.results['delta']
    elif result_str == 'delta_adj':
        return swaption_model.results['delta_adj']
    elif result_str == 'delta_lnadj':
        return swaption_model.results['delta_lnadj']
    elif result_str == 'theta':
        return swaption_model.results['theta']
    elif result_str == 'gamma':
        return swaption_model.results['gamma']
    elif result_str == 'vega':
        return swaption_model.results['vega']
    elif result_str == 'nvega':
        return swaption_model.results['nvega']
    elif result_str == 'dv01':
        return swaption_model.results['dv01']
    elif result_str == 'atm_rate':
        return swaption_model.results['atm_rate']
    elif result_str == 'theta_atm_rate':
        return swaption_model.results['theta_atm_rate']
    elif result_str == 'ln_vol':
        return swaption_model.results['ln_vol']
    elif result_str == 'sigma_atm':
        return swaption_model.results['sigma_atm']
    elif result_str == 'n_vol':
        return swaption_model.results['n_vol']
    elif result_str == 'corr':
        return swaption_model.results['corr']
    elif result_str == 'hedge1':
        return swaption_model.results['hedge1']
    elif result_str == 'hedge2':
        return swaption_model.results['hedge2']
    elif result_str == 'tenor1':
        return swaption_model.results['tenor1']
    elif result_str == 'tenor2':
        return swaption_model.results['tenor2']
    elif result_str == 'alpha1':
        return swaption_model.results['alpha1']
    elif result_str == 'alpha2':
        return swaption_model.results['alpha2']
    else:
        return -1000000000
    
@xl_func("cached_object swaption_model, string result_str: str")
def px_swaption_str_result(swaption_model, result_str):
    #swap_model.calculate()
    if result_str == 'tenor1':
        return swaption_model.results['tenor1']
    elif result_str == 'tenor2':
        return swaption_model.results['tenor2']

@xl_func("float alpha, float beta, float rho, float sigma0, float F0, float K, float T: float")
def util_SABR_BlackVol(alpha, beta, rho, sigma0, F0, K, T):
    return irvc.SABRBlackVol(alpha, beta, rho, sigma0, F0, K, T)

@xl_func("float F0, float K, float T, float lnvol: float")
def util_SABRLnVolToNVol(F0, K, T, lnvol):
    return irvc.SABRLnVolToNVol(F0, K, T, lnvol)

@xl_func("float F0, float K, float T, float nvol: float")
def util_SABRNVolToLnVol(F0, K, T, nvol):
    lnvol_0 = nvol/F0
    f = lambda x: irvc.SABRLnVolToNVol(F0, K, T, x) - nvol
    lnvol = root(f, lnvol_0, method='lm')
    print(lnvol)
    return lnvol.x[0]

@xl_func("float p_inf, float p_beta, float p_alpha, string tnr1, string tnr2: float")
def util_SwaptionCorr(p_inf, p_beta, p_alpha, tnr1, tnr2):
    return irvc.Swaption_Corr(p_inf, p_beta, p_alpha, tnr1, tnr2)

@xl_func("float p_inf, float p_beta, float p_alpha, float p_gamma, float p_delta, float factor, string tnr1, string tnr2: float")
def util_Swaption_Corr5P(p_inf, p_beta, p_alpha, p_gamma, p_delta, factor, tnr1, tnr2):
    return irvc.Swaption_Corr_5P(p_inf, p_beta, p_alpha, p_gamma, p_delta, factor, tnr1, tnr2)

@xl_func("string code: string")
def util_futures_month_to_code(month):    
    return useful.futures_month_to_code[month]

@xl_func("string code: string")
def util_futures_code_to_month(code):    
    return useful.futures_code_to_month[code]

@xl_func("string code: string")
def util_edfutures_next_std_contract(code):    
    return useful.edfutures_next_standard_contract(code)

@xl_func("date dt, int n, int weekday: date")
def nth_weekday(dt, n, weekday):
    dt1 = useful.n_th_weekday(dt, n, weekday)
    dt1 = dt1.astype(datetime.date)
    return dt1

@xl_macro("cached_object dataset, cached_object menv")
def update_range(dataset, menv):
    xl = xl_app()
    tenors = []
    rates = [] 
    curves = []
    curve_names = []
    
    rng1 = xl.Range("C67:S67")
    tenors.append(list(rng1.Value[0]))
    rng2 = xl.Range("C74:Y74")
    tenors.append(list(rng2.Value[0]))
    
    rng1 = xl.Range("C68:S68")
    rates.append(list(rng1.Value[0]))
    rng2 = xl.Range("C75:Y75")
    rates.append(list(rng2.Value[0]))
    
    rng1 = xl.Range("C64")
    curves.append(rng1.Value)
    rng2 = xl.Range("C71")
    curves.append(rng2.Value)
    
    rng1 = xl.Range("C98")
    curve_names.append(rng1.Value)
    rng2 = xl.Range("C99")
    curve_names.append(rng2.Value)
    
    #print(menv.observers.keys())
    if 'usd_libor_3m' in menv.observers.keys() and 'usd_discount_usd' in menv.observers.keys():
        ircb_setdata(tenors, rates, curves, dataset)
    else:
        ircb_setdata(tenors, rates, curves, dataset)
        build_curves(curve_names, menv, dataset)
        
@xl_func("cached_object dataset, cached_object menv:cached_object")
def update_dataset(dataset, menv, volatile=True):
    xl = xl_app()
    tenors = []
    rates = [] 
    curves = []
    curve_names = []
    
    rng1 = xl.Range("rng_libor_tenors")
    tenors.append(list(rng1.Value[0]))
    rng2 = xl.Range("rng_ois_tenors")
    tenors.append(list(rng2.Value[0]))
    
    rng1 = xl.Range("rng_libor_rates")
    rates.append(list(rng1.Value[0]))
    rng2 = xl.Range("rng_ois_rates")
    rates.append(list(rng2.Value[0]))
    
    rng1 = xl.Range("rng_libor_curve")
    curves.append(rng1.Value)
    rng2 = xl.Range("rng_ois_curve")
    curves.append(rng2.Value)
    
    rng1 = xl.Range("C98")
    curve_names.append(rng1.Value)
    rng2 = xl.Range("C99")
    curve_names.append(rng2.Value)
    
    #print(menv.observers.keys())
    #if 'usd_libor_3m' in menv.observers.keys() and 'usd_discount_usd' in menv.observers.keys():
    return ircb_setdata(tenors, rates, curves, dataset)
    #print(menv.observers.keys())

@xl_macro()
def write_mktenv():
    xl                      = xl_app()
    wb                      = xl.ActiveWorkbook
    ws                      = wb.Worksheets("MarketData")
    dd                      = datetime.date.today()
    today                   = np.datetime64(dd, 'D')
    mkt_env                 = env.MktEnv(today, today, today, today)
    workbook                = 'RatesModel_v2.2.xlsm'
    sheet                   = 'MarketData'
    cell                    = '$C$52'
    obj_str                 = objectcache._global_cache.update(workbook, sheet, cell, mkt_env)
    ws.Range("C52").Value   = obj_str 
    dataset                 = ircb_init_dataset("MktData")
    cell                    = '$C$61'
    obj_str                 = objectcache._global_cache.update(workbook, sheet, cell, dataset)
    ws.Range("C61").Value   = obj_str
    swp_spec                = get_swapspec('usd_3m_libor')
    cell                    = '$C$86'
    obj_str                 = objectcache._global_cache.update(workbook, sheet, cell, swp_spec)
    ws.Range("C86").Value   = obj_str
    curves                  = ['usd_libor_3m', 'usd_fedfunds_1d']
    dataset                 = update_dataset(dataset, mkt_env)
    sm                      = ircb.SimpleIrcBuilder(curves, mkt_env, dataset)
    #print(mkt_env.today)
    #objectcache._global_cache.get(mkt_env) 
    pass

@xl_func("float[] x, float[] y: cached_object")
def util_cubic_interp(x,y):
    x1 = reduce(lambda x,y:x+y, x)
    y1 = reduce(lambda x,y:x+y, y)
    return ircb_interp.cubic_monotone(x1,y1)

@xl_func("float[] x, float[] y: cached_object")
def util_exp_interp(x,y):
    x1 = reduce(lambda x,y:x+y, x)
    y1 = reduce(lambda x,y:x+y, y)
    return ircb_interp.exponential_fit(x1,y1)

@xl_func("float[] x, float[] y, string tenor1, float factor: cached_object")
def util_corr5P_tenor1_interp(x,y, tenor1, factor):
    x1 = reduce(lambda x,y:x+y, x)
    y1 = reduce(lambda x,y:x+y, y)
    return ircb_interp.corr5P_tenor1(x1,y1, tenor1, factor)

@xl_func("float[] x, float[] y, string tenor1, float factor: cached_object")
def util_corr5P_tenor2_interp(x,y, tenor2, factor):
    x1 = reduce(lambda x,y:x+y, x)
    y1 = reduce(lambda x,y:x+y, y)
    return ircb_interp.corr5P_tenor2(x1,y1, tenor2, factor)

@xl_func("float[] x, float[] y, string tenor1, float factor: cached_object")
def util_corr3P_tenor1_interp(x,y, tenor1, factor):
    x1 = reduce(lambda x,y:x+y, x)
    y1 = reduce(lambda x,y:x+y, y)
    return ircb_interp.corr3P_tenor1(x1,y1, tenor1, factor)

@xl_func("float[] x, float[] y, string tenor1, float factor: cached_object")
def util_corr3P_tenor2_interp(x,y, tenor2, factor):
    x1 = reduce(lambda x,y:x+y, x)
    y1 = reduce(lambda x,y:x+y, y)
    return ircb_interp.corr3P_tenor2(x1,y1, tenor2, factor)


@xl_func("float x, float upper, float lower: float")
def util_boundfunc(x, upper, lower):
    if(x >= upper):
        return upper
    elif(x <= lower):
        return lower
    else:
        return x

@xl_func("float[] x, float[] y, int order, int extrapol: cached_object")
def util_spline_interp(x,y, order, extrapol):
    x1 = reduce(lambda x,y:x+y, x)
    y1 = reduce(lambda x,y:x+y, y)
    return ircb_interp.cubic_spline(x1,y1, order, extrapol)

@xl_func("cached_object intp, float[] ynew: float")
def util_interp(intp, xnew):
    y_n = np.asarray(xnew[0], dtype=np.float64)
    value = intp.interp(y_n)
    return value

@xl_func("cached_object intp, cached_object extp, cached_object fallback, float[] ynew: float")
def util_combo_interp(intp, extp, fallback, xnew):
    xmin = intp.get_xbound('min')
    xmax = intp.get_xbound('max')    
    x = xnew[0]
    x_n = np.asarray(x, dtype=np.float64)
    if x < xmin:
        if float(extp.interp(x_n)):
            return extp.interp(x_n)
        else:
            return fallback.interp(x_n)
    elif x > xmax:
        if float(extp.interp(x_n)):
            return extp.interp(x_n)
        else:
            return fallback.interp(x_n)
    else:
        return intp.interp(x_n)
    

@xl_func("date effective, string dstring, string adjust: date")
def util_nBday(effective, dstring, adjust):
    eff_date = np.array([effective], dtype='datetime64[D]')
    end_date = useful.date_offset(eff_date, dstring)
    end_date = np.busday_offset(end_date, [0], roll=useful.adj[adjust], busdaycal=useful.hcal['US'])
    end_date = end_date[0]
    end_month = np.datetime64(end_date, 'M')
    end_year = np.datetime64(end_date, 'Y')
    ndays = (end_date - end_month)/np.timedelta64(1, 'D')
    ndays = int(ndays+1)
    nmonths = (end_month - end_year)/np.timedelta64(1, 'M')
    nmonths = int(nmonths) + 1
    nyears = (end_year - np.datetime64('0000'))/np.timedelta64(1, 'Y')
    nyears = int(nyears)
    final_date = datetime.date(nyears, nmonths, ndays)
    return final_date

@xl_func("date effective, string dstring, string adjust: date")
def util_spot_nBday(effective, dstring, adjust):
    spot_date = util_nBday(effective, '2D', 'F')
    final_date = util_nBday(spot_date, dstring, adjust)
    return final_date

@xl_func("string switch_str, string r_l: string") 
def util_switch_to_tenor(switch_str, r_l):
    tenor_list = switch_str.split('s')
    if r_l == 'l':
        return tenor_list[0] + 'Y'
    elif r_l == 'r':
        return tenor_list[1] + 'Y'
    pass

@xl_func("string fly_str, string r_l: string") 
def util_fly_to_tenor(fly_str, r_l):
    tenor_list = fly_str.split('s')
    if r_l == 'l':
        return tenor_list[0] + 'Y'
    elif r_l == 'm':
        return tenor_list[1] + 'Y'
    elif r_l == 'r':
        return tenor_list[2] + 'Y'
    pass


@xl_func("float n, float k: float")
def util_nCk(n,k):
    return sc.comb(n,k)

@xl_func("string[] keys, string[] values: cached_object")
def util_create_dict(keys, values):
    print(keys)
    print(values)
    return dict(zip(keys[0], values[0]))