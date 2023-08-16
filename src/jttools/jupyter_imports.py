
import platform, pathlib
import os
from typing import Dict, List, Union, Tuple
import typing
import itertools
from itertools import combinations, combinations_with_replacement
# import attrdict
import collections
from copy import copy, deepcopy

import pandas as pd

pd.options.display.date_yearfirst = True

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import scipy.stats as stats
import yaml
from functools import partial


import logging

from IPython.display import display
from pathlib import Path

from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

OLS = sm.regression.linear_model.OLS

import pickle

from jttools.picklepot import PicklePot
from bioscreen.experiment_classes import *
from bioscreen.PCA import CountPCA

import xlsxwriter

from jttools.statistics import *
from jttools.data_wrangling import *
from jttools.plotting import *

monkey_patch_dataframe(pd)

AMap = AttrMapAC

rpy2_magic_cmd = '%reload_ext rpy2.ipython'

revcomp = lambda s: ''.join([dict(zip('ACTGN', 'TGACN'))[nt] for nt in s[::-1]])
sns.set(style='whitegrid', context='paper')

### USEFUL GLOBALS
computer_name = platform.node()
print(f"running on '{computer_name}'.")
if computer_name == 'Precision-3660':
    ONEDRIVE = pathlib.Path('/home/jthomas/OneDrive')
    MDRIVE = pathlib.Path('/mnt/m')
    TDRIVE = pathlib.Path('/mnt/t')
    GDRIVE = pathlib.Path('/mnt/g')
elif computer_name == 'LT-JTHOMAS':
    ONEDRIVE = pathlib.Path('/home/jthomas/OneDrive')
    MDRIVE = pathlib.Path('/mnt/m')
    TDRIVE = pathlib.Path('/mnt/t')
    GDRIVE = pathlib.Path('/mnt/g')
else:
    print("unknown host, not setting drive paths")

csv_encoding_with_bom = 'utf-8-sig'

DFOrSeries = pd.DataFrame | pd.Series




def windows_to_unix_path(p: str, onedrive='OneDrive - MISSION Therapeutics Ltd'):
    p = p.replace('\\', '/')
    if onedrive in p:
        p = p.split(onedrive)[1]
        path = ONEDRIVE / p
    else:
        drive, path = p.split(':')
        path = f"/mnt/{drive.lower()}{path}"
    return Path(path)


def dir2(obj):
    return [c for c in dir(obj) if c[0] != '_']

class working_dir:
    """Context manager to temporarily alter the working directory.
    Use as part of a `with` statment.
    E.g:
    os.chdir('/path/one')
    with working_dir('/path/two/'):
        print(os.getcwd())
    print(os.getcwd())
    out:
        [1] /path/two
        [2] /path/one"""

    def __init__(self, new_dir):
        self.new_dir = new_dir
        self.old_dir = os.getcwd()

    def __enter__(self):
        os.chdir(self.new_dir)
        return os.getcwd()

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.old_dir)
        if exc_type:
            return False
        return True




def why_FileNotFound(p):
    """Prints the left-most part of a file path that doesn't exist"""
    p = os.path.normpath(p)
    if os.path.isfile(p):
        print(p, 'is a file')
        return p
    elif os.path.isdir(p):
        print(p, 'is a dir')
        return p
    else:
        target = ''
        p = p.split('/')

        for bit in p:
            if not os.path.isdir(target + '/'):
                print(target, 'is not found')
                return
            target = target + '/' + bit

def string_to_df(s, sep='\t'):
    return pd.DataFrame([l.split(sep) for l in s.split('\n')])

def nlprint(things: typing.Collection[str], sort=False):
    if sort:
        things = sorted(things)
    print('\n'.join(things))


def rpy2_set_lib():
    import rpy2.robjects
    rlibscript = """libs = c("/usr/local/lib/R/site-library", "/usr/lib/R/site-library", "/usr/lib/R/library")
# add them to the path
.libPaths(libs)"""
    rpy2.robjects.r(rlibscript)


