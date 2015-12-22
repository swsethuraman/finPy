'''
Created on May 21, 2014

@author: SSethuraman
'''
import ircb_inst

curve_mapping           = {
    'usd_discount_usd'  : 'usd_fedfunds_1d'
}

curve_key               = { 
    'usd_fedfunds_1d'   : 1 , 
    'usd_libor_3m'      : 2
}

def curve_map(curve_string): 
    if not curve_string in curve_mapping:
        return curve_string
    else:
        return curve_mapping[curve_string]

curve_depends           = {
    'usd_fedfunds_1d'   : {'usd_fedfunds_1d'},
    'usd_libor_3m'      : {'usd_fedfunds_1d', 'usd_libor_3m'}
}

usd_libor_3m_def        = [
    { 
        'inst'          : ircb_inst.usd_swap_3m_libor ,
        'tenors'        : ['3M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '15Y', '20Y', '25Y', '30Y', '35Y', '40Y']   
    }
]

usd_fedfunds_1d_def    = [
    { 
        'inst'          : ircb_inst.usd_swap_1d_fedfunds ,
        'tenors'        : ['1W', '1M', '2M', '3M', '4M', '5M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y','10Y','15Y', '20Y', '25Y', '30Y', '35Y', '40Y']   
    }
]

curvedef_map           = {
    'usd_libor_3m'     :    usd_libor_3m_def,
    'usd_fedfunds_1d'  :    usd_fedfunds_1d_def
}
