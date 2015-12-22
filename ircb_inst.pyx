'''
Created on May 19, 2014

@author: SSethuraman
'''

usd_swap_3m_libor = {
  'inst_name'         : 'usd_3m_libor',
  'inst_type'         : 'swap',
  'settle_days'       :  2,
  'settle_hcal'       : 'US',
  'collateral'        : 'usd',
  'quote_leg'         : 'legB',
  'legA_type'         : 'floating', 
  'legA_index_curve'  : 'usd_libor_3m',
  'legA_ratestruct'   : 'usd_libor_3m',
  'legA_ccy'          : 'usd',
  'legA_freq'         : '3m',
  'legA_dcf'          : 'Act360',
  'legA_hcal'         : 'US',
  'legA_accrual_hcal' : 'US',
  'legA_adjust'       : 'MF',
  'legA_paylag'       :  0,
  'legA_fixingfreq'   : '3m',
  'legA_fixinghcal'   : 'US',  
  'legA_fixing_index' : 'usd_libor_3m',
  'legA_fixinglag'    : 2,
  'legA_multiplier'   : 1,
  'legA_spread'       : None,
  'legB_type'         : 'fixed',
  'legB_index_curve'  : None,
  'legB_ratestruct'   : None,
  'legB_ccy'          : 'usd',
  'legB_freq'         : '6m',
  'legB_dcf'          : '30360E',
  'legB_hcal'         : 'US',
  'legB_accrual_hcal' : 'US',
  'legB_adjust'       : 'MF',
  'legB_paylag'       :  0,
  'legB_fixingfreq'   : None,
  'legB_fixinghcal'   : None,
  'legB_fixing_index' : None,
  'legB_fixinglag'    : None,
  'legB_multiplier'   : None,
  'legB_spread'       : 0.01
}

usd_swap_1d_fedfunds = {
  'inst_name'           : 'usd_1d_fedfunds',
  'inst_type'           : 'ois_swap',
  'settle_days'         :  2,
  'settle_hcal'         : 'US',
  'collateral'          : 'usd',
  'quote_leg'           : 'legB',
  'legA_type'           : 'floating', 
  'legA_index_curve'    : 'usd_fedfunds_1d',
  'legA_ratestruct'     : 'usd_fedfunds_1d',
  'legA_ccy'            : 'usd',
  'legA_freq'           : '1y',
  'legA_dcf'            : 'Act360',
  'legA_hcal'           : 'US',
  'legA_accrual_hcal'   : 'US',
  'legA_adjust'         : 'MF',
  'legA_paylag'         :  0,
  'legA_fixingfreq'     : '1d',
  'legA_fixinghcal'     : 'US',  
  'legA_fixing_index'   : 'usd_fedfunds_1d',
  'legA_fixinglag'      : 0,
  'legA_multiplier'     : 1,
  'legA_spread'         : None,
  'legB_type'           : 'fixed',
  'legB_index_curve'    : None,
  'legB_ratestruct'     : None,
  'legB_ccy'            : 'usd',
  'legB_freq'           : '1y',
  'legB_dcf'            : 'Act360',
  'legB_hcal'           : 'US',
  'legB_accrual_hcal'   : 'US',
  'legB_adjust'         : 'MF',
  'legB_paylag'         :  0,
  'legB_fixingfreq'     : None,
  'legB_fixinghcal'     : None,
  'legB_fixing_index'   : None,
  'legB_fixinglag'      : None,
  'legB_multiplier'     : None,
  'legB_spread'         : 0.01
  }

ed_future           = {
  'inst_name'           : 'ed_future',
  'inst_type'           : 'ir_future',
  'currency'            : 'usd',
  'collateral'          : 'usd',
  'index'               : 'usd_3m_libor',
  'notional'            : 100000000.0,
  'dcf'                 : 'ED90360',                         
}

inst_speclist = {
    'usd_3m_libor'      : usd_swap_3m_libor,
    'usd_1d_fedfunds'   : usd_swap_1d_fedfunds,
    'ed_future'         : ed_future           
}