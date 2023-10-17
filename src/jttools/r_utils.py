import pandas as pd
import rpy2
import rpy2.rinterface
import rpy2.robjects as ro
import rpy2.rinterface as ri
R = ro.r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import os, logging

Pathy = os.PathLike | str

# more convenient than the pandas one
np_convert = ro.numpy2ri.numpy2rpy

def pd_context(*args, **kwargs):
    """Return active context manager for calling R functions with, or
     directly converting, pandas objects."""
    return (ro.default_converter + pandas2ri.converter).context(*args, **kwargs)

def pd_convert(pd_obj):
    """Return an rpy2 conversion of a Pandas object."""
    with pd_context():
        return ro.conversion.get_conversion().py2rpy(pd_obj)

def r_to_pd(r_obj):
    with pd_context():
        res = ro.conversion.get_conversion().rpy2py(r_obj)
    return res


from rpy2.rinterface_lib.embedded import RRuntimeError
def rcatcher(func, verbosity=1):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except RRuntimeError as err:
            logging.critical(f"{func.__name__} failed, R error: \n\t{str(err)}")
            raise err
    return wrapper


NULL = rpy2.rinterface.NULL

pkgdir = os.path.dirname(__file__)

def rpy2_converter():
    """The default converter used with IPython magics, handles common obj sensibly.

    From https://rpy2.github.io/doc/v3.5.x/html/_modules/rpy2/ipython/rmagic.html"""
    from rpy2.robjects.conversion import (Converter,
                                          localconverter,
                                          get_conversion)

    template_converter = get_conversion()

    from rpy2.robjects import numpy2ri
    template_converter += numpy2ri.converter

    from rpy2.robjects import pandas2ri
    template_converter += pandas2ri.converter

    converter = Converter('generic converter', template=template_converter)

    # The default conversion for lists is currently to make them an R list. That
    # has some advantages, but can be inconvenient (and, it's inconsistent with
    # the way python lists are automatically converted by numpy functions), so
    # for interactive use in the rmagic, we call unlist, which converts lists to
    # vectors **if the list was of uniform (atomic) type**.
    @converter.py2rpy.register(list)
    def py2rpy_list(obj):
        # simplify2array is a utility function, but nice for us
        # TODO: use an early binding of the R function
        cv = ro.conversion.get_conversion()
        robj = ri.ListSexpVector(
            [cv.py2rpy(x) for x in obj]
        )
        res = ro.r.simplify2array(robj)
        # The current default converter for the ipython rmagic
        # might make `res` a numpy array. We need to ensure that
        # a rpy2 objects is returned (issue #866).
        res_rpy = cv.py2rpy(res)
        return res_rpy

    return localconverter(converter)