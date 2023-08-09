"""
NeuroTools.visual_logging
=========================

Log graphs, rather than text. This is useful when dealing with large data
structures, such as arrays. x-y data is plotted as a PNG file, which is stored
inside a zip archive.

You can specify a logging level such that only graphs with an importance above
that level will be created. e.g., if the logging level is set to WARNING,
log graphs with a level of DEBUG or INFO will not be created.

The interface is a restricted version of that available in the standard
library's logging module.

Functions
---------

basicConfig - specify the zipfile that will be used to store the graphs, and
              the logging level (DEBUG, INFO, WARN, etc)
debug       - plots data with level DEBUG
info        - plots data with level INFO
warning     - plots data with level WARNING
error       - plots data with level ERROR
critical    - plots data with level CRITICAL 
exception   - plots data with level ERROR
log         - plots data with a user-specified level

"""

import zipfile, atexit, os
from NeuroTools import check_dependency
from datetime import datetime
from logging import CRITICAL, DEBUG, ERROR, FATAL, INFO, WARN, WARNING, NOTSET
from time import sleep


if check_dependency('matplotlib'):
    import matplotlib
    matplotlib.use('Agg')


if check_dependency('pylab'):
    import pylab

_filename = 'visual_log.zip'
_zipfile = None
_level = INFO
_last_timestamp = ''

def _remove_if_empty():
    if len(_zipfile.namelist()) == 0 and os.path.exists(_filename):
        os.remove(_filename)

def basicConfig(filename, level=INFO):
    global _zipfile, _filename, _level
    _filename = filename
    _level = level
    #_zipfile.close()
    if os.path.exists(filename) and zipfile.is_zipfile(filename):
        mode = 'a'
    else:
        mode = 'w'
    _zipfile = zipfile.ZipFile(filename, mode=mode, compression=zipfile.ZIP_DEFLATED)
    atexit.register(_zipfile.close)
    atexit.register(_remove_if_empty)

def _reopen():
    global _zipfile
    if (_zipfile.fp is None) or _zipfile.fp.closed:
        _zipfile = zipfile.ZipFile(_filename, mode='a', compression=zipfile.ZIP_DEFLATED)

def flush():
    """Until the zipfile is closed (normally on exit), the zipfile cannot
    be accessed by other tools. Calling flush() closes the zipfile, which 
    will be reopened the next time a log function is called.
    """
    _zipfile.close()

def _get_timestamp():
    """At the moment, it is not possible to create visual
    logs at a rate of more than one/second."""
    global _last_timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    while timestamp == _last_timestamp:
        sleep(0.1)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    _last_timestamp = timestamp
    return timestamp

def _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs):
    _reopen()
    timestamp = _get_timestamp()
    # create figure
    pylab.clf()
    if xdata is not None:
        pylab.plot(xdata, ydata, **kwargs)
    else:
        if hasattr(ydata, 'shape') and len(ydata.shape) > 1:
            pylab.matshow(ydata, **kwargs)
            pylab.colorbar()
        else:
            pylab.plot(ydata)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title)
    # add it to the zipfile
    fig_name = timestamp + '.png'
    pylab.savefig(fig_name)
    _zipfile.write(fig_name,
                   os.path.join(os.path.basename(os.path.splitext(_filename)[0]), fig_name))
    os.remove(timestamp+'.png')

def debug(ydata, xdata=None, xlabel='', ylabel='', title='', **kwargs):
    if _level <= DEBUG:
        _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs)

def info(ydata, xdata=None, xlabel='', ylabel='', title='', **kwargs):
    if _level <= INFO:
        _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs)

def warning(ydata, xdata=None, xlabel='', ylabel='', title='', **kwargs):
    if _level <= WARNING:
        _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs)

def error(ydata, xdata=None, xlabel='', ylabel='', title='', **kwargs):
    if _level <= ERROR:
        _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs)

def critical(ydata, xdata=None, xlabel='', ylabel='', title='', **kwargs):
    if _level <= CRITICAL:
        _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs)

def exception(ydata, xdata=None, xlabel='', ylabel='', title='', **kwargs):
    if _level <= ERROR:
        _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs)

def log(level, ydata, xdata=None, xlabel='', ylabel='', title='', **kwargs):
    if _level <= level:
        _plot_fig(ydata, xdata, xlabel, ylabel, title, **kwargs)

def test():
    test_file = 'visual_logging_test.zip'
    if os.path.exists(test_file):
        os.remove(test_file)
    basicConfig(test_file, level=DEBUG)
    xdata = pylab.arange(0, 2*pylab.pi, 0.02*pylab.pi)
    debug(pylab.sin(xdata), xdata, 'x', 'sin(x)', 'visual_logging test 1')
    flush()
    debug(0.5*pylab.sin(2*xdata-0.3), xdata, 'x', 'sin(2x-0.3)/2')
    debug(pylab.sqrt(xdata), xdata, 'x', 'sqrt(x)')
    flush()
    zf = zipfile.ZipFile(test_file, 'r')
    print zf.namelist()
    assert len(zf.namelist()) == 3, zf.namelist()
    zf.close()
    
# ==============================================================================
if __name__ == '__main__':
    test()
