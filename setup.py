'''
Created on Jun 3, 2014

@author: SSethuraman
'''
'''
Created on Jun 3, 2014

@author: SSethuraman
'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import scipy 
#from Cython.Build import cythonize

#extensions=[
#    Extension("*",
#              ["*.pyx"])]

ext_modules = [
  Extension("env",            ["env.pyx"]),
  Extension("ir_fixings",     ["ir_fixings.pyx"]),
  Extension("useful",         ["useful.pyx"]),
  Extension("ircb_inst",      ["ircb_inst.pyx"]),
  Extension("ircb_data",      ["ircb_data.pyx"]),
  Extension("ircb_curvedef",  ["ircb_curvedef.pyx"]),
  Extension("swap",           ["swap.pyx"]),
  Extension("swaption",       ["swaption.pyx"]),
  Extension("swap_model",     ["swap_model.pyx"]),
  Extension("swaption_model", ["swaption_model.pyx"]),
  Extension("ircb",           ["ircb.pyx"]),
  Extension("irvc",           ["irvc.pyx"]),
  Extension("irvc_cubedef",   ["irvc_cubedef.pyx"]),
  Extension("rate_struct",    ["rate_struct.pyx"]),
  Extension("ircb_dataset",   ["ircb_dataset.pyx"]),
  Extension("ircb_init",      ["ircb_init.pyx"])
]

setup(
    #cmdclass = {'build_ext': build_ext},
    name = 'ir',
    cmdclass = {'build_ext' : build_ext},
    ext_modules = ext_modules,
    include_dirs=[np.get_include()]
)