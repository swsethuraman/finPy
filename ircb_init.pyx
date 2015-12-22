'''
Created on Jul 10, 2014

@author: SSethuraman
'''
import swap

inst_builder = {
    'swap'      : swap.VanillaSwapBuilder,
    'ois_swap'  : swap.OISwapBuilder                   
} 


def SwapInitializer(today, fixed_rate, swapspec, swap_tenor):
        notional    =  10000
        payrecv     = 'recv'
        swap_type   = swapspec['inst_type']
        swap        =  inst_builder[swap_type](today, notional, fixed_rate, payrecv, swapspec, tenor=swap_tenor)
        return swap

inst_initializer = {
    'swap'          : SwapInitializer,
    'ois_swap'      : SwapInitializer
}    