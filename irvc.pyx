'''
Created on Sept 10, 2014

@author: SSethuraman
'''
import numpy as np
import useful
import irvc_cubedef
from scipy.optimize import minimize 
from scipy.optimize import basinhopping
from scipy.optimize import root

def SABRNormalVol(alpha, beta, rho, sigma0, F0, Fmid, K, T):
    gamma1 = beta / Fmid
    gamma2 = -beta * (1 - beta) / Fmid ** 2
    eta = alpha * (F0 ** (1 - beta) - K ** (1 - beta)) / (sigma0 * (1 - beta))
    D = np.log((np.sqrt(1 - 2 * rho * eta + eta ** 2) + eta - rho) / (1 - rho))
    C = Fmid ** beta
    epsilon = alpha * T ** 2
    
    if np.abs(F0 - K) > 0.0002 :
        sigmaN = alpha * (F0 - K) * (1 + ((2 * gamma2 - gamma1 ** 2) * (sigma0 * C / alpha) ** 2 + rho * gamma1 * sigma0 * C / (4 * alpha) + (2 - 3 * rho ** 2) / 24) * epsilon) / D
    else:
        sigmaN = alpha * (1 + ((2 * gamma2 - gamma1 ** 2) * (sigma0 * C / alpha) ** 2 + rho * gamma1 * sigma0 * C / (4 * alpha) + (2 - 3 * rho ** 2) / 24) * epsilon)
    pass
    return sigmaN
pass

def Swaption_Corr(p_inf, p_beta, p_alpha, tenor1, tenor2):
    i = (int)(tenor1[:-1])
    j = (int)(tenor2[:-1])
    
    if tenor1[-1] == 'M':
        i = i / 12.0
    if tenor2[-1] == 'M':
        j = j / 12.0
    return p_inf + (1-p_inf)*np.exp(-np.abs(i-j)*(p_beta - p_alpha*(np.maximum(i,j)-1)))

def Swaption_Corr_RebExp(p_inf, p_beta, p_alpha, tenor1, tenor2):
    i = (int)(tenor1[:-1])
    j = (int)(tenor2[:-1])
    
    if tenor1[-1] == 'M':
        i = i / 12.0
    if tenor2[-1] == 'M':
        j = j / 12.0
    return p_inf + (1-p_inf)*np.exp(-np.abs(i-j)*p_beta*np.exp(-p_alpha*(np.maximum(i,j))))

def Swaption_Corr_SC3(p_inf, p_eta1, p_eta2, tenor1, tenor2):
    if tenor1 == 'y':
        tenor1 = 'Y'
    if tenor2 == 'm':
        tenor2 = 'M'
    
    i = (float)(tenor1[:-1])
    j = (float)(tenor2[:-1])
    
    if tenor1[-1] == 'M':
        i = i / 12.0
    if tenor2[-1] == 'M':
        j = j / 12.0
    m = 40.0
    
    return p_inf + (1-p_inf)*np.exp(-np.abs(i-j)*(p_eta1 - p_eta2*(np.maximum(i,j)-1)))

def Swaption_Corr_5P(p_inf, p_beta, p_alpha, p_gamma, p_delta, factor, tenor1, tenor2):
    if tenor1 == 'y':
        tenor1 = 'Y'
    if tenor2 == 'm':
        tenor2 = 'M'
    
    i = float(tenor1[:-1])
    j = float(tenor2[:-1])
    
    if tenor1[-1] == 'M':
        i = i / 12.0
    if tenor2[-1] == 'M':
        j = j / 12.0
    
    
    #factor = (1/i)
    
    i = i*factor
    j = j*factor
    N = 40.0*factor
    
    v_ij = Helper_5P(i, j, N, p_gamma, p_delta)
    v_ii = Helper_5P(i, i, N, p_gamma, p_delta)
    v_jj = Helper_5P(j, j, N, p_gamma, p_delta)
    
    t1 = np.sqrt((1 - np.exp(-2*p_beta*(i-1)**p_alpha))*(1 - np.exp(-2*p_beta*(j-1)**p_alpha)))
    t2 = ((v_ij)/np.sqrt(v_ii*v_jj))*t1
    t3 = np.exp(-p_beta*((i-1)**p_alpha + (j-1)**p_alpha))
    corr = p_inf + (1 - p_inf)*(t3 + t2)

    return corr

def Swaption_Corr_NP(corr_grid, tenor1, tenor2):
    if tenor1 == 'y':
        tenor1 = 'Y'
    if tenor2 == 'm':
        tenor2 = 'M'
    
    i = float(tenor1[:-1])
    j = float(tenor2[:-1])
    
    tenors_1 = corr_grid.keys()
    rand_tenor1 = list(tenors_1)
    tenors_2 = corr_grid[rand_tenor1[0]].keys()
    
    
    if tenor1[-1] == 'M':
        i = i / 12.0
    if tenor2[-1] == 'M':
        j = j / 12.0
    
    lowest = 0
    highest = 1000
    count = 0
    for t in tenors_1:
        if t[-1] == 'M' or t[-1] == 'm':
            temp = ((int)(t[:-1]))/12.0
        elif t[-1] == 'Y' or t[-1] == 'y':
            temp = ((int)(t[:-1]))/1.0
        if temp <= i and temp >= lowest:
            lowest = temp
            lowest_high = t
            low_count = count
        if temp >= i and temp <= highest:
            highest = temp
            highest_low = t
            high_count = count
        count = count + 1
    
    lower_tenor1 = lowest_high
    higher_tenor1 = highest_low
    lowest1 = lowest
    highest1 = highest
    
    if highest == 1000:
        higher_tenor1 = lower_tenor1
        highest1 = lowest
    if lowest == 0:
        lower_tenor1 = higher_tenor1
        lowest1 = highest
    
    lowest = 0
    highest = 1000
    count = 0
    for t in tenors_2:
        if t[-1] == 'M' or t[-1] == 'm':
            temp = ((int)(t[:-1]))/12.0
        elif t[-1] == 'Y' or t[-1] == 'y':
            temp = ((int)(t[:-1]))/1.0
        if temp <= j and temp >= lowest:
            lowest = temp
            lowest_high = t
            low_count = count
        if temp >= j and temp <= highest:
            highest = temp
            highest_low = t
            high_count = count
        count = count + 1
        
    lower_tenor2    = lowest_high
    higher_tenor2   = highest_low
    lowest2         = lowest
    highest2        = highest
    
    if highest == 1000:
        higher_tenor2 = lower_tenor2
        highest2 = lowest
    if lowest == 0:
        lower_tenor2 = higher_tenor2
        lowest2 = highest

    
    corr1_low       =   corr_grid[lower_tenor1][lower_tenor2]
    corr1_high      =   corr_grid[lower_tenor1][higher_tenor2]
    corr2_low       =   corr_grid[higher_tenor1][lower_tenor2]
    corr2_high      =   corr_grid[higher_tenor1][higher_tenor2]
    
    if highest2 == lowest2:
        w1 = 1
        w2 = 0
    else:
        w1              =   (j - lowest2)/(highest2 - lowest2)
        w2              =   (highest2 - j)/(highest2 - lowest2)
    corr1           =   w1*corr1_low + w2*corr1_high
    corr2           =   w1*corr2_low + w2*corr2_high
    
    if highest1 == lowest1:
        ww1 = 1
        ww2 = 0
    else:
        ww1              =   (j - lowest1)/(highest1 - lowest1)
        ww2              =   (highest1 - j)/(highest1 - lowest1)
    corr             =   ww1*corr1 + ww2*corr2
    
    return corr
    
def Helper_5P(i, j, N, gamma, delta):
    if min(i, j) <> 1:
        psi_1 = np.exp(-(1/(i-1))*((i-2)*gamma/(N-2) + (N-i)*delta/(N-2)))
        psi_2 = np.exp(-(1/(j-1))*((j-2)*gamma/(N-2) + (N-j)*delta/(N-2)))
    if min(i,j) == 1:
        v_ij = 1
    elif min(i, j) <> 1 and psi_1*psi_2 == 1:
        v_ij = min(i-1, j-1)
    elif min(i,j) <> 1 and psi_1*psi_2 <> 1:
        v_ij = ((psi_1*psi_2)**min(i-1, j-1) - 1)/(1 - 1/(psi_1*psi_2))
    return v_ij
    
def SABRBlackVol(alpha, beta, rho, sigma0, F0, K, T):
    Fmid = (F0 * K)
    if np.abs(F0 - K) > 0.0002:
        logFK = np.log(F0 / K)
        Z = (alpha / sigma0) * ((Fmid) ** ((1 - beta) / 2)) * logFK
        xz = np.log((np.sqrt(1 - 2 * rho * Z + Z ** 2) + Z - rho) / (1 - rho))
        vol1 = sigma0 / ((Fmid ** ((1 - beta) / 2)) * (1 + ((1 - beta) ** 2) * logFK ** 2 / 24 + ((1 - beta) ** 4) * logFK ** 4 / 1920))
        
        vol2_1 = (1 - beta) ** 2 * sigma0 ** 2 / (24 * Fmid ** (1 - beta))
        vol2_2 = 0.25 * (rho * beta * sigma0 * alpha) / (Fmid ** ((1 - beta) / 2))
        vol2_3 = (2 - 3 * rho ** 2) * alpha ** 2 / 24
        
        vol2_test = 1 + (vol2_1 + vol2_2 + vol2_3)*T
        vol2 = 1 + ((1 - beta) ** 2 * sigma0 ** 2 / (24 * Fmid ** (1 - beta)) + 0.25 * (rho * beta * sigma0 * alpha) / (Fmid ** ((1 - beta) / 2)) + (2 - 3 * rho ** 2) * alpha ** 2 / 24) * T

        SABRBlackVol = vol1 * Z * vol2 / xz
    else:
        vol1 = sigma0 / (F0 ** (1 - beta))
        vol2 = (((1 - beta) ** 2 )* (sigma0 ** 2 )/ (24 * (F0 **(2 - 2 * beta))) + 0.25 * (rho * beta * alpha * sigma0) / (F0 ** (1 - beta)) + (2 - 3 * (rho ** 2)) * (alpha ** 2 )/ 24) * T
        SABRBlackVol = vol1 * (1 + vol2)
    pass
    return SABRBlackVol

def SABRLnVolToNVol(F0, K, T, lnVol):
    if np.abs(F0 - K) > 0.0002:
        v1 = (F0 - K) / np.log(F0 / K)
        v2 = lnVol / (1 + lnVol ** 2 * T / 24)
        SABRNVol = v1 * v2
    else:
        v2 = lnVol / (1 + lnVol ** 2 * T / 24)
        SABRNVol = K*v2
    pass
    return SABRNVol

def SABRNVolToLnVol(F0, K, T, nvol):
    lnvol_0 = nvol/F0
    f = lambda x: SABRLnVolToNVol(F0, K, T, x) - nvol
    lnvol = root(f, lnvol_0, method='lm')
    return lnvol.x[0]   

def SABR_Calibration_VolSlice(vol_dataset, swap_dataset, menv):
    vol_cube_def = irvc_cubedef.usd_3m_libor_swaption_calib_def
    today = menv.today
    settle = np.busday_offset(today, 2, roll=useful.adj['F'], busdaycal=useful.hcal['US'])
    SABR_Calib_Params = {}
    SABR_tenor = {}
    
    for expiry in vol_cube_def['expiry'].keys():
        for tenor in vol_cube_def['expiry'][expiry].keys():
            F0                  = swap_dataset.data['atm_rate'][expiry][tenor]
            date_str            = expiry[-1]
     
            if date_str == 'y':
                date_str = 'Y'
            elif date_str == 'm':
                date_str = 'M'
            elif date_str == 'd':
                date_str = 'D'
            pass
            
            settle_month        = np.datetime64(settle, 'M')
            n_days              = settle - settle_month
            #n_days              = n_days.astype(int)
            T                   = settle_month + np.timedelta64(tenor[:-1], date_str)
            T                   = settle_month + n_days
            T_settle            = np.busday_offset(T, 0, roll=useful.adj['F'], busdaycal=useful.hcal['US'])
            
            T_settlemonth       = np.datetime64(T_settle, 'M')
            n_settledays        = T_settle - T_settlemonth
            T_expiry            = T_settlemonth + np.timedelta64(expiry[:-1], date_str)
            T_expiry            = T_expiry + n_settledays
            T_expiry            = np.busday_offset(T_expiry, 0, roll=useful.adj['MF'], busdaycal=useful.hcal['US'])
            
            T_SABR              = (T_expiry - today)/np.timedelta64(1, 'D')
            T_SABR              = T_SABR/365
            
            sabr_calib_k        = SABRCalibrate_K(F0, T_SABR, 0.5)
            strikes             = vol_cube_def['expiry'][expiry][tenor]
            nvol                = vol_dataset.data[expiry][tenor]

            x0                  = [0.5, -0.2, 0.04]
            bnds                = ((0, None), (-1,1), (0, None))
            min_kwargs          = {"method" : "BFGS"}
            #options={'xtol': 10, 'disp': True} method='BFGS'
            SABR_params         = minimize(SABR_solve(sabr_calib_k, F0, strikes, nvol), x0, method='Nelder-Mead', tol = 1e-4) 
            print(SABR_params)
            SABR_tenor.update({tenor : SABR_params})
        SABR_Calib_Params.update({expiry : SABR_tenor})
        SABR_tenor = {}
    return SABR_Calib_Params


def SABR_solve(f, F0, strikes, nvol):
    return lambda x: SABR_MMSE(f, F0, x, strikes, nvol)

def SABR_nvol_solve(F0, K, T, nvol):
    return lambda x:SABRLnVolToNVol(F0, K, T, x) - nvol

def SABR_init_vol_solve(alpha, beta, rho, F0, K, T, lnvol):
    return lambda x:SABRBlackVol(alpha, beta, rho, x, F0, K, T) - lnvol

def SABR_initVol(alpha, beta, rho, F0, K, T, lnvol):
    f = SABR_init_vol_solve(alpha, beta, rho, F0, K, T, lnvol)
    s0 = lnvol/(F0 ** (1-beta))
    initVol = root(f, s0, method='lm')
    return initVol
    

def SABR_MMSE(f, F0, x, strikes, nvol):
        err = 0
        for k in strikes:
            if F0+k/10000 > 0.0:
                abs_k = F0 + k/10000
                err = err + (f(np.append(x,[abs_k],1))*10000 - nvol[k])**2
                #print(np.append(x, [abs_k],1))
                #print(f(np.append(x,[abs_k],1))*10000)
                #print(nvol[k])
                #print err
        return err 
        
        
def SABRCalibrate(F0, K, T, beta): 
    return lambda x: SABRLnVolToNVol(F0, K, T, SABRBlackVol(x[0], beta, x[1], x[2], F0, K, T))

def SABRCalibrate_K(F0, T, beta): 
    return lambda x: SABRLnVolToNVol(F0, x[3], T, SABRBlackVol(x[0], beta, x[1], x[2], F0, x[3], T))  

def Corr5PCalibrate(tenor1, tenor2, f):
    return lambda x: Swaption_Corr_5P(x[0], x[1], x[2], x[3], x[4], f, tenor1, tenor2)

def Corr5P_MMSE(x, tenor1, tenor2, ncorr, w, f):
    err     = 0
    i       = 0
    for t1 in tenor1:
        j = 0
        for t2 in tenor2:
            t3 = float(t1[:-1])
            t4 = float(t2[:-1])
            if t1[-1] == 'M' or t1[-1] == 'm':
                t3 = t3/12.0
            if t2[-1] == 'M' or t2[-1] == 'm':
                t4 = t4/12.0    
            t5 = t3 + t4
            t6 = str(t5) + 'Y'
            factor = f[t2]
            f_0 = Corr5PCalibrate(t2, t6, factor)
            err = err + ((f_0(x) - ncorr[i][j])**2)*(w[i][j]**2)
            j = j + 1
        i = i + 1
    return err

def Corr5P_Solve(tenor1, tenor2, ncorr, w, f):
    return lambda x : Corr5P_MMSE(x, tenor1, tenor2, ncorr, w, f)

class VolCube:
    def __init__(self, today, expiry, tenors, alpha, beta, rho, sigma0, corr):
        self.today          = today
        self.expiry         = expiry
        self.tenors         = tenors
        self.alpha          = alpha
        self.beta           = beta
        self.rho            = rho
        self.sigma0         = sigma0
        self.corr           = corr
    pass

def Corr5P_Calibration_CorrSlice(corr_dataset, weights_dataset, factor_dataset):
    corr_surf_def = irvc_cubedef.usd_3m_libor_corr_calib_def
    Corr_Calib_Params = {}
    
    for expiry in corr_surf_def['expiry'].keys():
        tenor1 = corr_surf_def['expiry'][expiry]['tenor1']
        tenor2 = corr_surf_def['expiry'][expiry]['tenor2']
        
        ncorr               = corr_dataset.data[expiry]['ncorr']
        w                   = weights_dataset.data[expiry]['weights']
        factor              = factor_dataset.data[expiry]['factor']
        x0                  = [0.6, 0.11, 1.2, 5, 11]
        bnds                = ((-1, 1), (None, None), (None, None), (None, None), (None, None))
        min_kwargs          = {"method" : "BFGS"}
        #options={'xtol': 10, 'disp': True} method='BFGS'
    
        f = Corr5P_Solve(tenor1, tenor2, ncorr, w, factor)
        Corr_params         = minimize(Corr5P_Solve(tenor1, tenor2, ncorr, w, factor), x0, method='Nelder-Mead', options = {'xtol' : 1e-2}) 
        Corr_Calib_Params.update({expiry : Corr_params})
        
    return Corr_Calib_Params
