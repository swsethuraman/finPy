'''
Created on Jun 24, 2014

@author: SSethuraman
'''


"""
eclipse_debug.py

PyDev can be used to interactively debug Python code running
in Excel via PyXLL.

Before using this script you must have Eclipse and PyDev
installed:

http://www.eclipse.org/
http://pydev.org/

To be able to attach the PyDev debugger to Excel and you
Python code open the PyDev Debug perspective in Eclipse
and start the PyDev server by clicking the toolbar
button with a bug and a small P on it (hover over for the
tooltip).

Any python process can now attach to the PyDev debug
server by importing the 'pydevd' module included as part
of PyDev and calling pydevd.settrace()

This module adds an Excel menu item to attach to the
PyDev debugger, and also an Excel macro so that this
script can be run outside of Excel and call PyXLL to
attach to the PyDev debugger.

See http://pydev.org/manual_adv_remote_debugger.html
for more details about remote debugging using PyDev.

"""
import sys
import os
import logging
import time
import glob

_log = logging.getLogger(__name__)

##
## UPDATE THIS TO MATCH WHERE YOU HAVE ECLIPSE AND PYDEV INSTALLED
##
## The following code tries to guess where Eclipse is installed
eclipse_roots = [r"C:\dev\eclipse"]
if "USERPROFILE" in os.environ:
    eclipse_roots.append(os.path.join(os.environ["USERPROFILE"],
                                      ".eclipse",
                                      "org.eclipse.platform_*"))

for eclipse_root in eclipse_roots:
    pydev_src = os.path.join(eclipse_root, r"plugins\org.python.pydev.debug_*\pysrc")
    paths = glob.glob(pydev_src)
    if paths:
        paths.sort()
        _log.info("Adding PyDev path '%s' to sys.path" % paths[-1])
        sys.path.append(paths[-1])
        break

def main():
    import win32com.client

    # get Excel and call the macro declared below
    import sys
    print ("\n".join(sys.path))
    xl_app = win32com.client.GetActiveObject("Excel.Application")
    xl_app.Run("attach_to_pydev")

#
# PyXLL function for attaching to the debug server
#
try:
    from pyxll import xl_menu, xl_macro, xlcAlert

    # if this doesn't import check the paths above
    try:
        import pydevd
        import pydevd_tracing
    except ImportError:
        _log.warn("pydevd failed to import - eclipse debugging won't work")
        _log.warn("Check the eclipse path in %s" % __file__)
        raise

    try:
        import threading
    except ImportError:
        threading = None

    # this creates a menu item and a macro from the same function
    @xl_menu("Attach to PyDev")
    @xl_macro()
    def attach_to_pydev():
        # remove any redirection from previous debugging
        if getattr(sys, "_pyxll_pydev_orig_stdout", None) is None:
            sys._pyxll_pydev_orig_stdout = sys.stdout
        if getattr(sys, "_pyxll_pydev_orig_stderr", None) is None:
            sys._pyxll_pydev_orig_stderr = sys.stderr

        sys.stdout = sys._pyxll_pydev_orig_stdout
        sys.stderr = sys._pyxll_pydev_orig_stderr

        # stop any existing PyDev debugger
        dbg = pydevd.GetGlobalDebugger()
        if dbg:
            dbg.FinishDebuggingSession()
            time.sleep(0.1)
            pydevd_tracing.SetTrace(None)

        # remove any additional info for the current thread
        if threading:
            try:
                del threading.currentThread().__dict__["additionalInfo"]
            except KeyError:
                pass
        
        pydevd.SetGlobalDebugger(None)
        pydevd.connected = False
        time.sleep(0.1)
        
        _log.info("Attempting to attach to the PyDev debugger")
        try:
            pydevd.settrace(stdoutToServer=True, stderrToServer=True, suspend=False)
        except Exception as e:
            xlcAlert("Failed to connect to PyDev\n"
                     "Check the debug server is running.\n"
                     "Error: %s" % e)
            return

        xlcAlert("Attatched to PyDev")            

except ImportError:
    pass

if __name__ == "__main__":
    sys.exit(main())