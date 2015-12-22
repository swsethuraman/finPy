'''
Created on May 19, 2014

@author: SSethuraman
'''

import useful
import ircb
import ircb_init
import ircb_inst 
import rate_struct as rs
import env
import xll
import holidays as hol 
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import irvc
import swap
from itertools import count

class SwaptionModel:
    def __init__(self, swaption, menv, volcube):
        # consistency checks
        self.swaption    = swaption
        self.swap        = self.swaption.swap
        self.menv        = menv
        self.volcube     = volcube
        self.results     = {}
        self.expiry      = self.swaption.expiry
        self.tenor       = self.swaption.tenor
        self.strike      = self.swaption.strike
        self.payrecv     = self.swaption.payrecv
        self.tradedate   = self.swaption.tradedate
        inst_type        = self.swaption.swap.inst_type
        model            = ircb.inst_model[inst_type]
        self.swap_model  = model(self.swaption.swap, menv)
        self.swap_model.calculate()
        self.par_rate    = self.swap_model.results['par_rate']
        self.dv01        = self.swap_model.results['dv01']
        sigma            = self.get_swaption_vol()

        T                = (self.expiry - self.menv.today)/np.timedelta64(1, 'D')
        T                = T/365
        F                = self.par_rate/10000
        K                = self.swaption.strike
        d1               = (np.log(F/K) + T*np.square(sigma)/2)/(sigma*np.sqrt(T))
        d2               = (np.log(F/K) - T*np.square(sigma)/2)/(sigma*np.sqrt(T))
        Nd1              = norm.cdf(d1)
        Nd2              = norm.cdf(d2)
        Pd1              = norm.pdf(d1)
        Pd2              = norm.pdf(d2)
        Nd1m             = norm.cdf(-d1)
        Nd2m             = norm.cdf(-d2)
    
        self.payer_pv    = (F*Nd1 - K*Nd2)*self.dv01*10000
        self.recv_pv     = (-F*Nd1m + K*Nd2m)*self.dv01*10000
        self.lnvol       = sigma
        self.nvol        = irvc.SABRLnVolToNVol(F, K, T, sigma)
        #self.dnvol_df    = (irvc.SABRLnVolToNVol(F+0.00001, K, T, sigma) - self.nvol)/0.00001
        self.payer_delta = Nd1*self.dv01
        self.recv_delta  = (Nd1-1)*self.dv01
        self.payer_theta = -F*self.payer_pv + self.dv01*F*sigma*Pd1/(2*np.sqrt(T))
        self.recv_theta  = -F*self.recv_pv + self.dv01*F*sigma*Pd1/(2*np.sqrt(T))
        self.gamma       = self.dv01*Pd1/(np.sqrt(T)*F*self.lnvol)
        self.vega        = self.dv01*F*np.sqrt(T)*Pd1
        self.nvega        = self.dv01*np.sqrt(T)*Pd1
        self.payer_delta_adj = self.payer_delta + self.dnvol_df*self.nvega
        self.recv_delta_adj  = self.recv_delta + self.dnvol_df*self.nvega
        self.payer_delta_lnadj = self.payer_delta + self.dlnvol_df*self.vega
        self.recv_delta_lnadj  = self.recv_delta + self.dlnvol_df*self.vega
        
        
        # Create a second swap for theta - get current swap details 
        current_swap_type   = self.swap.inst_type
        current_payrecv     = self.swap.payrecv
        current_notional    = self.swap.notional
        current_fixed_rate  = self.swap.fixed_rate
        current_swapspec    = self.swap.swapspec
        current_tenor       = self.swap.tenor
        current_trade_date  = self.swap.tradedate
        new_trade_date      = xll.util_nBday(current_trade_date,"-1D", "P")
        new_trade_date      = np.datetime64(new_trade_date, 'D')
        print(new_trade_date)
        theta_swap          = ircb_init.inst_builder[current_swap_type](new_trade_date, current_notional, current_fixed_rate, current_payrecv, current_swapspec, current_tenor)
        theta_swap_model    = model(theta_swap, menv)
        theta_swap_model.calculate()
        self.theta_swap_par_rate = theta_swap_model.results['par_rate']
        
        if self.payrecv == 'pay':    
            self.results.update({'pv'           : self.payer_pv})
            self.results.update({'delta'        : self.payer_delta})
            self.results.update({'theta'        : self.payer_theta})
            self.results.update({'delta_adj'    : self.payer_delta_adj})
            self.results.update({'delta_lnadj'    : self.payer_delta_lnadj})
        elif self.payrecv == 'recv':
            self.results.update({'pv'           : self.recv_pv})
            self.results.update({'delta'        : self.recv_delta})
            self.results.update({'theta'        : self.recv_theta})
            self.results.update({'delta_adj'    : self.recv_delta_adj})
            self.results.update({'delta_lnadj'    : self.recv_delta_lnadj})
        pass
        
        
        self.results.update({'gamma'            : self.gamma/10000})
        self.results.update({'vega'             : self.vega})
        self.results.update({'nvega'            : self.nvega})
        self.results.update({'dv01'             : self.dv01})
        self.results.update({'atm_rate'         : self.par_rate})
        self.results.update({'ln_vol'           : self.lnvol})
        self.results.update({'n_vol'            : self.nvol})
        self.results.update({'theta_atm_rate'   : self.theta_swap_par_rate})
            
    pass

    def get_swaption_vol(self):
        exp_date                    = self.expiry
        today                       = self.menv.today
        tenor                       = self.tenor
        
        exp_t                       = (exp_date - today)/np.timedelta64(1, 'D')
        exp_t                       = exp_t / 365.0
        
        volcube_expiry              = self.volcube.expiry
        volcube_expiry_dates        = []
        count = 0
        exp_index = len(volcube_expiry) - 1
        max_exp   = len(volcube_expiry) - 1
        for exp in volcube_expiry:
            actual_exp_date         = xll.util_nBday(today, exp, "MF")
            actual_exp_date         = np.datetime64(actual_exp_date, 'D')
            temp                    = (actual_exp_date - today)/np.timedelta64(1, 'D')
            volcube_expiry_dates.append(temp/365.0) 
            if temp/365.0 <= exp_t:
                exp_index = count
            count = count + 1
        
        low_exp = volcube_expiry[exp_index]
        if exp_index < max_exp:
            high_exp = volcube_expiry[exp_index + 1]
        else:
            high_exp = volcube_expiry[exp_index]
        
        actual_exp_low              = xll.util_nBday(today, low_exp, "MF")
        actual_exp_low              = np.datetime64(actual_exp_low, 'D')
        actual_exp_low              = (actual_exp_low - today)/np.timedelta64(1, 'D')
        actual_exp_low              = actual_exp_low/365.0
        
        actual_exp_high              = xll.util_nBday(today, high_exp, "MF")
        actual_exp_high              = np.datetime64(actual_exp_high, 'D')
        actual_exp_high              = (actual_exp_high - today)/np.timedelta64(1, 'D')
        actual_exp_high              = actual_exp_high/365.0
        
        volcube_tenors               = self.volcube.tenors
        count                        = 0
        tenor_index                  = len(volcube_tenors) - 1
        max_tenor                    = len(volcube_tenors) - 1
        
        if tenor[-1] == 'Y':
            tenor_tt = int(tenor[:-1])
        elif tenor[-1] == 'M':
            tenor_tt = int(tenor[:-1])/12.0
            
        for t in volcube_tenors:
            num_t = int(t[:-1])
            if num_t <= tenor_tt:
                tenor_index = count
            count = count + 1
        
        
        
        low_tenor = volcube_tenors[tenor_index]
        if tenor_index < max_tenor:
            high_tenor = volcube_tenors[tenor_index + 1]
        else:
            high_tenor = volcube_tenors[tenor_index]
            
            
        SABR_params_low_low     = [self.volcube.alpha.data['alpha'][low_exp][low_tenor], self.volcube.beta.data['beta'][low_exp][low_tenor], self.volcube.rho.data['rho'][low_exp][low_tenor], self.volcube.sigma0.data['sigma0'][low_exp][low_tenor]]    
        SABR_params_low_high    = [self.volcube.alpha.data['alpha'][low_exp][high_tenor], self.volcube.beta.data['beta'][low_exp][high_tenor], self.volcube.rho.data['rho'][low_exp][high_tenor], self.volcube.sigma0.data['sigma0'][low_exp][high_tenor]]
        SABR_params_high_low    = [self.volcube.alpha.data['alpha'][high_exp][low_tenor], self.volcube.beta.data['beta'][high_exp][low_tenor], self.volcube.rho.data['rho'][high_exp][low_tenor], self.volcube.sigma0.data['sigma0'][high_exp][low_tenor]]          
        SABR_params_high_high   = [self.volcube.alpha.data['alpha'][high_exp][high_tenor], self.volcube.beta.data['beta'][high_exp][high_tenor], self.volcube.rho.data['rho'][high_exp][high_tenor], self.volcube.sigma0.data['sigma0'][high_exp][high_tenor]]
  
        F = self.swap_model.results['par_rate']/10000
        K = self.swaption.strike     
        
        sabr_lnvol_low_low                      = irvc.SABRBlackVol(SABR_params_low_low[0], SABR_params_low_low[1], SABR_params_low_low[2], SABR_params_low_low[3], F, K, actual_exp_low)
        sabr_lnvol_low_high                     = irvc.SABRBlackVol(SABR_params_low_high[0], SABR_params_low_high[1], SABR_params_low_high[2], SABR_params_low_high[3], F, K, actual_exp_high)
        sabr_lnvol_high_low                     = irvc.SABRBlackVol(SABR_params_high_low[0], SABR_params_high_low[1], SABR_params_high_low[2], SABR_params_high_low[3], F, K, actual_exp_low)
        sabr_lnvol_high_high                    = irvc.SABRBlackVol(SABR_params_high_high[0], SABR_params_high_high[1], SABR_params_high_high[2], SABR_params_high_high[3], F, K, actual_exp_high)
        
        sabr_nvol_low_low                       = irvc.SABRLnVolToNVol(F, K, actual_exp_low,sabr_lnvol_low_low)
        sabr_nvol_low_high                      = irvc.SABRLnVolToNVol(F, K, actual_exp_low,sabr_lnvol_low_high)
        sabr_nvol_high_low                      = irvc.SABRLnVolToNVol(F, K, actual_exp_high,sabr_lnvol_high_low)
        sabr_nvol_high_high                     = irvc.SABRLnVolToNVol(F, K, actual_exp_high,sabr_lnvol_high_high)
        
        tenor_low             = (int)(low_tenor[:-1])
        tenor_high            = (int)(high_tenor[:-1])
        tenor_actual          = (int)(tenor[:-1])
        
        if tenor_high <> tenor_low:
            lambda_tenor          = (tenor_actual - tenor_low)/(tenor_high - tenor_low)
            sabr_nvol_low           = (1-lambda_tenor)*sabr_nvol_low_low + lambda_tenor*sabr_nvol_low_high
            sabr_nvol_high          = (1-lambda_tenor)*sabr_nvol_high_low + lambda_tenor*sabr_nvol_high_high
        else:
            sabr_nvol_low           = sabr_nvol_low_low
            sabr_nvol_high          = sabr_nvol_high_low
            
        if actual_exp_high <> actual_exp_low:
            lambda_exp              = (exp_t - actual_exp_low)/(actual_exp_high - actual_exp_low)
            sabr_nvol               = (1-lambda_exp)*sabr_nvol_low + lambda_exp*sabr_nvol_high
        else:
            sabr_nvol               = sabr_nvol_low
        
        sabr_lnvol                  = fsolve(irvc.SABR_nvol_solve(F, K, exp_t, sabr_nvol), sabr_lnvol_low_low, xtol=1e-08)
        
        # Bumping F
        bump_delta = 0.00001
        F = self.swap_model.results['par_rate']/10000 + bump_delta
        K = self.swaption.strike     
        
        sabr_lnvol_low_low                      = irvc.SABRBlackVol(SABR_params_low_low[0], SABR_params_low_low[1], SABR_params_low_low[2], SABR_params_low_low[3], F, K, actual_exp_low)
        sabr_lnvol_low_high                     = irvc.SABRBlackVol(SABR_params_low_high[0], SABR_params_low_high[1], SABR_params_low_high[2], SABR_params_low_high[3], F, K, actual_exp_high)
        sabr_lnvol_high_low                     = irvc.SABRBlackVol(SABR_params_high_low[0], SABR_params_high_low[1], SABR_params_high_low[2], SABR_params_high_low[3], F, K, actual_exp_low)
        sabr_lnvol_high_high                    = irvc.SABRBlackVol(SABR_params_high_high[0], SABR_params_high_high[1], SABR_params_high_high[2], SABR_params_high_high[3], F, K, actual_exp_high)
        
        sabr_nvol_low_low                       = irvc.SABRLnVolToNVol(F, K, actual_exp_low,sabr_lnvol_low_low)
        sabr_nvol_low_high                      = irvc.SABRLnVolToNVol(F, K, actual_exp_low,sabr_lnvol_low_high)
        sabr_nvol_high_low                      = irvc.SABRLnVolToNVol(F, K, actual_exp_high,sabr_lnvol_high_low)
        sabr_nvol_high_high                     = irvc.SABRLnVolToNVol(F, K, actual_exp_high,sabr_lnvol_high_high)
        
        tenor_low             = (int)(low_tenor[:-1])
        tenor_high            = (int)(high_tenor[:-1])
        tenor_actual          = (int)(tenor[:-1])
        
        if tenor_high <> tenor_low:
            lambda_tenor          = (tenor_actual - tenor_low)/(tenor_high - tenor_low)
            sabr_nvol_low           = (1-lambda_tenor)*sabr_nvol_low_low + lambda_tenor*sabr_nvol_low_high
            sabr_nvol_high          = (1-lambda_tenor)*sabr_nvol_high_low + lambda_tenor*sabr_nvol_high_high
        else:
            sabr_nvol_low           = sabr_nvol_low_low
            sabr_nvol_high          = sabr_nvol_high_low
            
        if actual_exp_high <> actual_exp_low:
            lambda_exp              = (exp_t - actual_exp_low)/(actual_exp_high - actual_exp_low)
            sabr_nvol_bump               = (1-lambda_exp)*sabr_nvol_low + lambda_exp*sabr_nvol_high
        else:
            sabr_nvol_bump               = sabr_nvol_low
        
        #func_nvol               = irvc.SABR_nvol_solve(F, K, exp_t, sabr_nvol)
        sabr_lnvol_bump              = fsolve(irvc.SABR_nvol_solve(F, K, exp_t, sabr_nvol_bump), sabr_lnvol_low_low, xtol=1e-08)
        
        
        self.dnvol_df = (sabr_nvol_bump - sabr_nvol)/bump_delta
        self.dlnvol_df = (sabr_lnvol_bump - sabr_lnvol)/bump_delta
        
        return sabr_lnvol

class MidCurveSwaptionModel:
    def __init__(self, swaption, menv, volcube, model_params):
        # consistency checks
        self.swaption     = swaption
        self.swap1        = self.swaption.swap1
        self.swap2        = self.swaption.swap2
        self.menv         = menv
        self.volcube      = volcube
        self.model_params = model_params
        self.results      = {}
        self.expiry       = self.swaption.expiry
        self.tenor1       = self.swaption.tenor1
        self.tenor2       = self.swaption.tenor2
        self.strike       = self.swaption.strike
        self.payrecv      = self.swaption.payrecv
        self.tradedate    = self.swaption.tradedate
        inst_type         = self.swaption.swap1.inst_type
        swap_spec         = self.swap1.swapspec
        model             = ircb.inst_model[inst_type] 
        
        self.swap_model1  = model(self.swaption.swap1, menv)
        self.swap_model1.calculate()
        self.swap_model2  = model(self.swaption.swap2, menv)
        self.swap_model2.calculate()
                                                                                                                                                            
        self.par_rate1    = self.swap_model1.results['par_rate']
        self.par_rate2    = self.swap_model2.results['par_rate']
        
        self.dv01_1           = self.swap_model1.results['dv01']
        self.dv01_2           = self.swap_model2.results['dv01']
        dv01_diff             = self.dv01_2 - self.dv01_1
        self.dv01             = dv01_diff

        T                     = (self.expiry - self.menv.today)/np.timedelta64(1, 'D')
        T                     = T/365
        F1                    = self.par_rate1/10000
        F2                    = self.par_rate2/10000
        self.alpha1           = self.dv01_1/dv01_diff
        self.alpha2           = self.dv01_2/dv01_diff
        self.hedge1           = -self.alpha1*self.dv01_1
        self.hedge2           = self.alpha2*self.dv01_2

        F                     = -self.alpha1*F1 + self.alpha2*F2
        K                     = self.swaption.strike
        skew_scale            = F/K
        self.par_rate         = F*10000
        
        today                       = self.menv.today
        exp_t                       = (self.expiry - today)/np.timedelta64(1, 'D')
        exp_t                       = exp_t / 365.0
        
        volcube_expiry              = self.volcube.expiry
        volcube_expiry_dates        = []
        exp_array                   = []
        count = 0
        exp_index = len(volcube_expiry) - 1
        max_exp   = len(volcube_expiry) - 1
        for exp in volcube_expiry:
            actual_exp_date         = xll.util_nBday(today, exp, "MF")
            actual_exp_date         = np.datetime64(actual_exp_date, 'D')
            temp                    = (actual_exp_date - today)/np.timedelta64(1, 'D')
            volcube_expiry_dates.append(temp/365.0) 
            if temp/365.0 <= exp_t:
                exp_index = count
                exp_array.append(actual_exp_date) 
            count = count + 1
        
        low_exp = volcube_expiry[exp_index]
        if exp_index < max_exp:
            high_exp = volcube_expiry[exp_index + 1]
        else:
            high_exp = volcube_expiry[exp_index]
        high_exp_date         = xll.util_nBday(today, high_exp, "MF")
        high_exp_date         = np.datetime64(high_exp_date, 'D')
        exp_array.append(high_exp_date)
        
        
        exp_atm_vol1 = []
        exp_atm_s1 = []
        exp_atm_vol2 = []
        exp_atm_s2 = []
        count = 0
        total_sig_1 = 0
        total_sig_2 = 0
        total_sig_12 = 0
        #for exp in exp_array:
        #    swp1 = swap.VanillaSwapBuilder(exp, 1000000, 0.03, 'recv', swap_spec, self.tenor1)
        #    swp1_model = model(swp1, self.menv)
        #    swp1_model.calculate()
        #    par_swp1_rate = swp1_model.results['par_rate']/10000.0
        #    atm_swapt1_vol = self.get_swaption_vol(self.tenor1, exp, par_swp1_rate, par_swp1_rate,'lnvol')
        #    swp2 = swap.VanillaSwapBuilder(exp, 1000000, 0.03, 'recv', swap_spec, self.tenor2)
        #    swp2_model = model(swp2, self.menv)
        #    swp2_model.calculate()
        #    par_swp2_rate = swp2_model.results['par_rate']/10000.0
        #    atm_swapt2_vol = self.get_swaption_vol(self.tenor2, exp, par_swp2_rate, par_swp2_rate, 'lnvol')
        #    if len(exp_atm_vol1) > 0:
        #        t_j = (exp - today)/np.timedelta64(1, 'D')
        #        t_j = t_j/365.0
        #        t_i = (exp_array[count-1] - today)/np.timedelta64(1, 'D')
        #        t_i = t_i/365.0
        #        sig_1 = np.sqrt(((atm_swapt1_vol[0]**2)*t_j  - (exp_atm_vol1[-1]**2)*t_i)/(t_j - t_i))
        #        sig_2 = np.sqrt(((atm_swapt2_vol[0]**2)*t_j  - (exp_atm_vol2[-1]**2)*t_i)/(t_j - t_i))
        #    else:
        #        sig_1 = atm_swapt1_vol[0]
        #        sig_2 = atm_swapt2_vol[0]
        #        t_j = (exp - today)/np.timedelta64(1, 'D')
        #        t_j = t_j/365.0
        #        t_i = 0.0
        #    total_sig_1 = total_sig_1 + (sig_1**2)*(t_j - t_i)
        #    total_sig_2 = total_sig_2 + (sig_2**2)*(t_j - t_i)
        #    total_sig_12 = total_sig_12 + (sig_1*sig_2)*(t_j - t_i) 
        #    
        #    exp_atm_vol1.append(atm_swapt1_vol[0])
        #    exp_atm_vol2.append(atm_swapt2_vol[0])
        #    exp_atm_s1.append(sig_1)
        #    exp_atm_s2.append(sig_2)
        #    count = count + 1
        #corr_adjust = total_sig_12/np.sqrt(total_sig_1*total_sig_2)

            
        sigma1                = self.get_swaption_vol(self.tenor1, self.expiry, F1, F1, 'lnvol')
        sigma1_params         = self.get_swaption_vol(self.tenor1, self.expiry, F1, F1, 'params')
        sigma1                = sigma1[0]
        sigma1_skew           = self.get_swaption_vol(self.tenor1, self.expiry, F1, F1*skew_scale, 'lnvol')
        sigma1_skew           = sigma1_skew[0]
        skew_ratio1           = sigma1_skew/sigma1
        
        sigma2                = self.get_swaption_vol(self.tenor2, self.expiry, F2, F2, 'lnvol')
        sigma2_params         = self.get_swaption_vol(self.tenor2, self.expiry, F2, F2, 'params')
        sigma2                = sigma2[0]
        sigma2_skew           = self.get_swaption_vol(self.tenor2, self.expiry, F2, F2*skew_scale, 'lnvol')
        sigma2_skew           = sigma2_skew[0]
        skew_ratio2           = sigma2_skew/sigma2
        
        w1                    = self.alpha1/(self.alpha1 + self.alpha2)
        w2                    = self.alpha2/(self.alpha1 + self.alpha2)
        sigma_skew            = w1*skew_ratio1 + w2*skew_ratio2
        sigma_skew_params     = w1*sigma1_params + w2*sigma2_params
        print(self.alpha1)
        print(self.alpha2)
        print(sigma1_params)
        print(sigma2_params)
        print(sigma_skew_params)
        swaption_corr         = self.get_corr()
       
        self.corr             = swaption_corr
        #self.corr             = swaption_corr
        sigma_atm             = (self.alpha1**2)*(sigma1**2) + (self.alpha2**2)*(sigma2**2) - 2*self.alpha1*sigma1*self.alpha2*sigma2*swaption_corr
        sigma_atm             = np.sqrt(sigma_atm)
 
        sigma_nv1             = irvc.SABRLnVolToNVol(F1, F1, T, sigma1)
        sigma_nv2             = irvc.SABRLnVolToNVol(F2, F2, T, sigma2)

        sigma_nv              = (self.alpha1**2)*(sigma_nv1**2) + (self.alpha2**2)*(sigma_nv2**2) - 2*self.alpha1*sigma_nv1*self.alpha2*sigma_nv2*swaption_corr
        sigma_nv              = np.sqrt(sigma_nv) 
        sigma_nv_skew         = sigma_nv*sigma_skew
        #self.nvol             = sigma_nv
        
        self.sigma_atm        = irvc.SABRNVolToLnVol(F, F, T, sigma_nv)
        #self.sigma            = irvc.SABRNVolToLnVol(F, K, T, sigma_nv_skew)
        #self.nvol             = self.nvol
        #sigma                 = self.sigma
        
        mc_alpha              = sigma_skew_params[0]
        mc_beta               = sigma_skew_params[1]
        mc_rho                = sigma_skew_params[2]
        mc_sigma0             = irvc.SABR_initVol(mc_alpha, mc_beta, mc_rho, F, F, T, self.sigma_atm)
        self.sigma            = irvc.SABRBlackVol(mc_alpha, mc_beta, mc_rho, mc_sigma0.x[0], F, K, T)
        print(mc_alpha)
        print(mc_beta)
        print(mc_rho)
        print(mc_sigma0.x[0])
        print(self.sigma_atm)
        print(self.sigma)
        #print(self.nvol)
        sigma                 = self.sigma
        self.nvol             = irvc.SABRLnVolToNVol(F, K, T, self.sigma)
        print(irvc.SABRLnVolToNVol(F, F, T, self.sigma_atm))
        print(self.nvol)
        #sigma                 = sigma_atm*sigma_skew
        
        d1                    = (np.log(F/K) + T*np.square(sigma)/2)/(sigma*np.sqrt(T))
        d2                    = (np.log(F/K) - T*np.square(sigma)/2)/(sigma*np.sqrt(T))
        Nd1                   = norm.cdf(d1)
        Nd2                   = norm.cdf(d2)
        Pd1                   = norm.pdf(d1)
        Pd2                   = norm.pdf(d2)
        Nd1m                  = norm.cdf(-d1)
        Nd2m                  = norm.cdf(-d2)
        self.payer_pv         = (F*Nd1 - K*Nd2)*self.dv01*10000
        self.recv_pv          = (-F*Nd1m + K*Nd2m)*self.dv01*10000
        self.lnvol            = sigma
        #self.nvol             = irvc.SABRLnVolToNVol(F, K, T, sigma)
        self.payer_delta      = Nd1*self.dv01
        self.recv_delta       = (Nd1-1)*self.dv01
        self.payer_theta      = -F*self.payer_pv + self.dv01*F*sigma*Pd1/(2*np.sqrt(T))
        self.recv_theta       = -F*self.recv_pv + self.dv01*F*sigma*Pd1/(2*np.sqrt(T))
        self.gamma            = self.dv01*Pd1/(np.sqrt(T)*F*self.lnvol)
        self.vega             = self.dv01*F*np.sqrt(T)*Pd1
        self.nvega             = self.dv01*np.sqrt(T)*Pd1
        
        if self.payrecv == 'pay':    
            self.results.update({'pv'       : self.payer_pv})
            self.results.update({'delta'    : self.payer_delta})
            self.results.update({'theta'    : self.payer_theta})
        elif self.payrecv == 'recv':
            self.results.update({'pv'       : self.recv_pv})
            self.results.update({'delta'    : self.recv_delta})
            self.results.update({'theta'    : self.recv_theta})
        pass
        
        
        self.results.update({'gamma'    : self.gamma/10000})
        self.results.update({'vega'     : self.vega})
        self.results.update({'nvega'     : self.nvega})
        self.results.update({'dv01'     : self.dv01})
        self.results.update({'atm_rate' : self.par_rate})
        self.results.update({'ln_vol'   : self.lnvol})
        self.results.update({'sigma_atm'   : self.sigma_atm})
        self.results.update({'n_vol'    : self.nvol})
        self.results.update({'corr'     : self.corr})
        self.results.update({'hedge1'     : self.hedge1})
        self.results.update({'hedge2'     : self.hedge2})
        self.results.update({'tenor1'     : self.tenor1})
        self.results.update({'tenor2'     : self.tenor2})
        self.results.update({'alpha1'     : self.alpha1})
        self.results.update({'alpha2'     : self.alpha2})              
    pass

    def get_swaption_vol(self, tnr, expiry, atm_rate, strike, ret_string):
        exp_date                    = expiry
        today                       = self.menv.today
        tenor                       = tnr
        F                           = atm_rate
        K                           = strike
        
        exp_t                       = (exp_date - today)/np.timedelta64(1, 'D')
        exp_t                       = exp_t / 365.0
        
        volcube_expiry              = self.volcube.expiry
        volcube_expiry_dates        = []
        count = 0
        exp_index = len(volcube_expiry) - 1
        max_exp   = len(volcube_expiry) - 1
        for exp in volcube_expiry:
            actual_exp_date         = xll.util_nBday(today, exp, "MF")
            actual_exp_date         = np.datetime64(actual_exp_date, 'D')
            temp                    = (actual_exp_date - today)/np.timedelta64(1, 'D')
            volcube_expiry_dates.append(temp/365.0) 
            if temp/365.0 <= exp_t:
                exp_index = count
            count = count + 1

        low_exp = volcube_expiry[exp_index]
        if exp_index < max_exp:
            high_exp = volcube_expiry[exp_index + 1]
        else:
            high_exp = volcube_expiry[exp_index]
        
        actual_exp_low              = xll.util_nBday(today, low_exp, "MF")
        actual_exp_low              = np.datetime64(actual_exp_low, 'D')
        actual_exp_low              = (actual_exp_low - today)/np.timedelta64(1, 'D')
        actual_exp_low              = actual_exp_low/365.0
        
        actual_exp_high              = xll.util_nBday(today, high_exp, "MF")
        actual_exp_high              = np.datetime64(actual_exp_high, 'D')
        actual_exp_high              = (actual_exp_high - today)/np.timedelta64(1, 'D')
        actual_exp_high              = actual_exp_high/365.0
        
        volcube_tenors               = self.volcube.tenors
        count                        = 0
        tenor_index                  = len(volcube_tenors) - 1
        max_tenor                    = len(volcube_tenors) - 1
        
        tenor_tt = useful.tenor_to_float(tenor)
            
        for t in volcube_tenors:
            num_t = useful.tenor_to_float(t)
            if num_t <= tenor_tt:
                tenor_index = count
            count = count + 1
        
        low_tenor = volcube_tenors[tenor_index]
        if tenor_index < max_tenor:
            high_tenor = volcube_tenors[tenor_index + 1]
        else:
            high_tenor = volcube_tenors[tenor_index]
        
                
        SABR_params_low_low     = [self.volcube.alpha.data['alpha'][low_exp][low_tenor], self.volcube.beta.data['beta'][low_exp][low_tenor], self.volcube.rho.data['rho'][low_exp][low_tenor], self.volcube.sigma0.data['sigma0'][low_exp][low_tenor]]    
        SABR_params_low_high    = [self.volcube.alpha.data['alpha'][low_exp][high_tenor], self.volcube.beta.data['beta'][low_exp][high_tenor], self.volcube.rho.data['rho'][low_exp][high_tenor], self.volcube.sigma0.data['sigma0'][low_exp][high_tenor]]
        SABR_params_high_low    = [self.volcube.alpha.data['alpha'][high_exp][low_tenor], self.volcube.beta.data['beta'][high_exp][low_tenor], self.volcube.rho.data['rho'][high_exp][low_tenor], self.volcube.sigma0.data['sigma0'][high_exp][low_tenor]]          
        SABR_params_high_high   = [self.volcube.alpha.data['alpha'][high_exp][high_tenor], self.volcube.beta.data['beta'][high_exp][high_tenor], self.volcube.rho.data['rho'][high_exp][high_tenor], self.volcube.sigma0.data['sigma0'][high_exp][high_tenor]]
        
        tenor_low             = (float)(low_tenor[:-1])
        tenor_high            = (float)(high_tenor[:-1])
        tenor_actual          = (float)(tenor[:-1])
        
        if tenor_low <> tenor_high:
            lambda_tenor            = (tenor_actual - tenor_low)/(tenor_high - tenor_low)
        else:
            lambda_tenor = 0
            
        if actual_exp_high <> actual_exp_low:
            lambda_exp              = (exp_t - actual_exp_low)/(actual_exp_high - actual_exp_low)
        else:
            lambda_exp = 0
        
        SABR_params_low         = (1-lambda_tenor)*np.asarray(SABR_params_low_low, dtype=np.float64) + lambda_tenor*np.asarray(SABR_params_low_high, dtype=np.float64)
        SABR_params_high        = (1-lambda_tenor)*np.asarray(SABR_params_low_low, dtype=np.float64) + lambda_tenor*np.asarray(SABR_params_low_high, dtype=np.float64)
        SABR_params_final       = (1-lambda_exp)*SABR_params_low + lambda_exp*SABR_params_high
        
        #F1 = self.swap_model1.results['par_rate']/10000
        #F2 = self.swap_model2.results['par_rate']/10000
        #K = self.swaption.strike     
        sabr_lnvol_low_low                      = irvc.SABRBlackVol(SABR_params_low_low[0], SABR_params_low_low[1], SABR_params_low_low[2], SABR_params_low_low[3], F, K, actual_exp_low)
        sabr_lnvol_low_high                     = irvc.SABRBlackVol(SABR_params_low_high[0], SABR_params_low_high[1], SABR_params_low_high[2], SABR_params_low_high[3], F, K, actual_exp_high)
        sabr_lnvol_high_low                     = irvc.SABRBlackVol(SABR_params_high_low[0], SABR_params_high_low[1], SABR_params_high_low[2], SABR_params_high_low[3], F, K, actual_exp_low)
        sabr_lnvol_high_high                    = irvc.SABRBlackVol(SABR_params_high_high[0], SABR_params_high_high[1], SABR_params_high_high[2], SABR_params_high_high[3], F, K, actual_exp_high)
        
        sabr_nvol_low_low                       = irvc.SABRLnVolToNVol(F, K, actual_exp_low,sabr_lnvol_low_low)
        sabr_nvol_low_high                      = irvc.SABRLnVolToNVol(F, K, actual_exp_low,sabr_lnvol_low_high)
        sabr_nvol_high_low                      = irvc.SABRLnVolToNVol(F, K, actual_exp_high,sabr_lnvol_high_low)
        sabr_nvol_high_high                     = irvc.SABRLnVolToNVol(F, K, actual_exp_high,sabr_lnvol_high_high)
        

        if tenor_high <> tenor_low:       
            sabr_nvol_low           = (1-lambda_tenor)*sabr_nvol_low_low + lambda_tenor*sabr_nvol_low_high
            sabr_nvol_high          = (1-lambda_tenor)*sabr_nvol_high_low + lambda_tenor*sabr_nvol_high_high
        else:
            sabr_nvol_low           = sabr_nvol_low_low
            sabr_nvol_high          = sabr_nvol_high_low
            
        if actual_exp_high <> actual_exp_low: 
            sabr_nvol               = (1-lambda_exp)*sabr_nvol_low + lambda_exp*sabr_nvol_high
        else:
            sabr_nvol               = sabr_nvol_low
            
        func_nvol               = irvc.SABR_nvol_solve(F, K, exp_t, sabr_nvol)
        sabr_lnvol              = fsolve(irvc.SABR_nvol_solve(F, K, exp_t, sabr_nvol), sabr_lnvol_low_low, xtol=1e-08)
        
        
        if ret_string == 'lnvol':
            return sabr_lnvol
        elif ret_string == 'params':
            return SABR_params_final
        else:
            return 0.0
    
    def get_corr(self):
        exp_date                    = self.expiry
        today                       = self.menv.today
        
        exp_t                       = (exp_date - today)/np.timedelta64(1, 'D')
        exp_t                       = exp_t / 365.0
        
        volcube_expiry              = self.volcube.expiry
        
        volcube_expiry_dates        = []
        count = 0
        exp_index = len(volcube_expiry) - 1
        max_exp   = len(volcube_expiry) - 1
        for exp in volcube_expiry:
            actual_exp_date         = xll.util_nBday(today, exp, "MF")
            actual_exp_date         = np.datetime64(actual_exp_date, 'D')
            temp                    = (actual_exp_date - today)/np.timedelta64(1, 'D')
            volcube_expiry_dates.append(temp/365.0) 
            if temp/365.0 <= exp_t:
                exp_index = count
            count = count + 1
        
        low_exp = volcube_expiry[exp_index]
        if exp_index < max_exp:
            high_exp = volcube_expiry[exp_index + 1]
        else:
            high_exp = volcube_expiry[exp_index]
        
        actual_exp_low              = xll.util_nBday(today, low_exp, "MF")
        actual_exp_low              = np.datetime64(actual_exp_low, 'D')
        actual_exp_low              = (actual_exp_low - today)/np.timedelta64(1, 'D')
        actual_exp_low              = actual_exp_low/365.0
        
        actual_exp_high              = xll.util_nBday(today, high_exp, "MF")
        actual_exp_high              = np.datetime64(actual_exp_high, 'D')
        actual_exp_high              = (actual_exp_high - today)/np.timedelta64(1, 'D')
        actual_exp_high              = actual_exp_high/365.0
        
        if self.model_params['corr_model'] == '5P':
            corr_params1                 = self.volcube.corr.data['corr'][low_exp]
            corr_params2                 = self.volcube.corr.data['corr'][high_exp]
            corr1                        = irvc.Swaption_Corr_5P(corr_params1['p_inf'], corr_params1['p_beta'], corr_params1['p_alpha'], corr_params1['p_gamma'], corr_params1['p_delta'], corr_params1['factor'], self.tenor1, self.tenor2)
            corr2                        = irvc.Swaption_Corr_5P(corr_params2['p_inf'], corr_params2['p_beta'], corr_params2['p_alpha'], corr_params2['p_gamma'], corr_params2['p_delta'], corr_params1['factor'], self.tenor1, self.tenor2)
        elif self.model_params['corr_model']  == 'NP':
            corr_params1                = self.volcube.corr.data['corr_grid'][low_exp]
            corr_params2                = self.volcube.corr.data['corr_grid'][high_exp]
            corr1                       = irvc.Swaption_Corr_NP(corr_params1, self.tenor1, self.tenor2)
            corr2                       = irvc.Swaption_Corr_NP(corr_params2, self.tenor1, self.tenor2)
            
        print('inside get corr')
        
        if actual_exp_high <> actual_exp_low:
            lambda_exp                   = (exp_t - actual_exp_low)/(actual_exp_high - actual_exp_low)
            corr                         = (1-lambda_exp)*corr1 + lambda_exp*corr2
        else:
            corr                         = corr1
        
        return corr
        