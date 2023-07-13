from typing import List, Union, Tuple

import pandas as pd

import logging

import statsmodels.api as sm

OLS = sm.regression.linear_model.OLS

from typing import Dict
import xlsxwriter

from attrdictionary import AttrMap

class AttrMapAC(AttrMap):
    """Tab completable AttrMap"""

    def f(self):
        for thing in self:
            print(thing)

    def __dir__(self):
        super_dir = super().__dir__()
        string_keys = [str(key) for key in self if isinstance(key, str)]
        return super_dir + [key for key in string_keys if key not in super_dir]


def list_not_str(thing):
    """If thing is a string, wrap it in a list."""
    if type(thing) is str:
        return [thing]
    return thing

def read_csv(fn, idx_also_column=False, **kwargs) -> pd.DataFrame:
    """pd.read_csv(fn, index_col=0), checks for dupes in the index.

    idx_also_column != False means that the first column will be part of the columns
    and also the index. Set to a string to rename the index, else it'll be
    'Index'. """
    if not idx_also_column:
        df = pd.read_csv(fn, index_col=0, **kwargs)
    else:
        df = pd.read_csv(fn, **kwargs)
        df.index = df[list(df.columns)[0]]
        if type(idx_also_column) is str:
            df.index.name = idx_also_column
        else:
            df.index.name = 'Index'

    if df.index.duplicated().any():
        logging.warning("Index contains duplicate values")

    return df


def read_tsv(fn, idx_also_column=False, **kwargs) -> pd.DataFrame:
    """See read_csv"""
    return read_csv(fn, sep='\t', idx_also_column=idx_also_column, **kwargs)


def read_txt(fn, sep='\t') -> pd.DataFrame:
    """Read headerless, tab separated table. If it's one column wide, give a Series."""
    df = read_csv(fn, header=None, sep=sep)
    if df.shape[1] == 1:
        return df[1]
    return df


def write_excel_text(sheets: Dict[str, pd.DataFrame],
                     filename: str,
                     text_columns: Union[List, Dict[str, List[str]]] = 'ALL',
                     index_as_text=False,
                     **to_excel_kwargs):
    """Write XLSX with columns optionally formated as text.

    Args:
        sheets: Dataframes to be written as sheets, dict keyed by sheetname,
            values DF
        filename: For the written XLSX
        text_columns: Specify columns (by name) to be formated as text.
            Default 'ALL' formats all columns as string.
            Lists of column names (including index name) can specify columns.
            A single list applies to all sheets, use a dictionary to specify
            different columns per sheet. Can't specify the index
        index_as_text: Since you can't just pass the name of the index (for
            tedious reasons), set this to True to write the index as text.
            Index name will be written as header. Won't work with multiindex
            columns for some reason.
        **to_excel_kwargs: Additional kwargs are passed to pd.DataFrame.to_excel()
    Doesn't work work with multiindex columns.
    """
    # tests 'all', text_col is list or dict

    text_columns = list_not_str(text_columns)

    for df in sheets.values():
        if type(df.columns) == pd.MultiIndex:
            raise RuntimeError('MultiIndex columns not supported')

    writer = pd.ExcelWriter(filename,
                            engine='xlsxwriter')

    # Go through the DF
    for sheet_name, df in sheets.items():
        if index_as_text:
            df = df.reset_index(drop=False)
        df.to_excel(writer, sheet_name=sheet_name, **to_excel_kwargs)

        worksheet = writer.sheets[sheet_name]

        workbook = writer.book
        txt = workbook.add_format({'num_format': '@'})

        # get the columns to be textualised

        if text_columns == ['ALL']:
            txtcols = df.columns
        elif type(text_columns) is Dict:
            txtcols = text_columns[sheet_name]
            if txtcols == 'ALL':
                txtcols = df.columns
        elif (text_columns == None) or (text_columns == False):
            txtcols = []
        else:
            txtcols = list(text_columns)

        # Note: index formats get written over by pd, and any sensible automated
        #   solution to work around this uses index=False - but that's incompatible with
        #   multiindex columns for some fucking reason, so there is no good
        #   solution for dealing with the index until Pandas impliments
        #   a stable way of disabling automatic formating of the index

        index_offset = 1
        if 'index' in to_excel_kwargs:
            if not to_excel_kwargs['index']:
                index_offset = 0

        # get indicies of named columns
        col_i = []
        col_i.extend([list(df.columns).index(c) + index_offset for c in txtcols])

        # set the format of selected columns
        for i in col_i:
            worksheet.set_column(i, i, cell_format=txt)

    writer.save()
    return writer


def underscore_columns(df):
    """replace spaces with underscores in all columns of a pandas DF.
    In place, but returns the DF"""
    df.columns = df.columns.map(lambda x: x.replace(' ', '_'))
    return df


def index_of_true(S: pd.Series) -> pd.Series:
    return S[S].index

def keys_crawler(d, level=0, exclude: set = None):
    """Accepts a dictionary, gets the type of values for each key,
    if it's a list the type of the FIRST item obtained. Lists and dicts
    encountered are explored recursively.
    Output gives the structure of the object as a string, levels of
    indentation show the hierarchy. Lists indicated with [...]

    `exclude` indicates key values that will not be investigated deeper
    """
    out = ''

    for k in d.keys():
        out += '  ' * level + k + ':\n'

        if (exclude is not None) and (k in exclude):
            out = out[:-1] + f'  {type(d[k])}\n'
        else:
            if type(d[k]) is dict:
                out += keys_crawler(d[k], level=level + 1, exclude=exclude)
            elif type(d[k]) is list:
                out += '  ' * (level + 1) + '[\n'
                if type(d[k][0]) is dict:
                    out += keys_crawler(d[k][0], level=level + 2, exclude=exclude)
                out += '  ' * (level + 1) + ']\n'
            else:  # type(d[k]) is str:
                out = out[:-1] + f'  {type(d[k])}\n'
    return out




def write_stats_workbook(
        sheets: Dict[str, pd.DataFrame],
        filename=None,
        columns=None,
        close_workbook=True,
        float_format='0.000',
        workbook: xlsxwriter.Workbook = None,
):
    """Writes DF in sheets as tables to tabs in an excel workbook.

    `columns` is passed to sheet.add_table, see:
        https://xlsxwriter.readthedocs.io/working_with_tables.html"""
    if workbook is None:
        if filename is None and close_workbook:
            raise RuntimeError('Provide a filename, or set close_workbook=False to avoid an Error')
        workbook = xlsxwriter.Workbook(filename, )

    int_format = workbook.add_format({'num_format': '0'})
    float_format = workbook.add_format({'num_format': float_format})
    text_format = workbook.add_format({'num_format': '@', })
    for sheet_name, tab in sheets.items():
        sheet = workbook.add_worksheet(name=sheet_name)
        if not type(tab.index) is pd.RangeIndex:
            tab = tab.reset_index()
        nrows, ncols = tab.shape

        dtypes = tab.dtypes

        if columns:
            # if specifying columns, need to include index as a column
            assert len(columns) == ncols
            tab_columns = columns
        # automatically style columns based on dtype
        # todo add more automatic formatting
        else:
            tab_columns = []
            for col in tab.columns:
                cold = {'header': col}
                try:
                    cold['format'] = {
                        int: int_format,
                        float: float_format,
                        object: text_format,
                    }[dtypes[col]]
                except KeyError:
                    pass

                tab_columns.append(cold)
        # so I think these numbers are are 0-index end-INCLUSIVE, but you need one more for the columns.
        sheet.add_table(
            0, 0, nrows, ncols - 1,
            options={
                'data': tab.values,
                'columns': tab_columns
            }
        )

    if close_workbook:
        workbook.close()
    return workbook


def unify_indicies(*pd_obj: Union[pd.DataFrame, pd.Series]) -> Tuple[Union[pd.DataFrame, pd.Series]]:
    """return DataFrame|Series with identical indexes.

    Any duplicate index values raises error."""

    # get first index
    shared = pd_obj[0].index

    # create index with only shared values
    for obj in pd_obj:
        # duplicates mess this up and need to be dealt with prior
        if obj.index.duplicated().any():
            raise RuntimeError("Duplicate values found in index.")
        # update the shared index
        shared = shared.intersection(obj.index)

    # reindex original objects
    return tuple([tab.reindex(shared) for tab in pd_obj])


def rename_columns(df, newcols=dict, inplace=False) -> pd.Index:
    """Return index with renamed columns:

    Column labels not found as keys in newcols will be left alone"""
    mapper = {k: k for k in df.columns}
    mapper.update(newcols)

    nucols = df.columns.map(mapper)
    if not inplace:
        return nucols
    df.columns = nucols
    return nucols



def format_vertical_headers(styler, css: dict):
    """Display a dataframe with vertical column headers

    Use with df.style.pipe(format_vertical_headers).
    """
    styles = [dict(selector="th", props=[('width', '40px')]),
              dict(selector="th.col_heading",
                   props=[("writing-mode", "vertical-rl"),
                          ('transform', 'rotateZ(180deg)'),
                          ('height', 'auto'),
                          ('vertical-align', 'center'),
                          ('text-align', 'left')])]
    styles.update(css)
    return (styler.set_table_styles(styles))


def write_list(things, filename):
    """for t in things: write(t+newline)"""
    with open(filename, 'w') as f:
        f.write(
            things.join('\n'.join(things))
        )

def read_list(fn):
    with open(fn) as f:
        return f.read().split('\n')


def iterate_groups(series):
    """Yield factor name and indexes from a series of factors."""
    for k, grp in series.groupby(series).groups.items():
        yield k, grp


def tidy_to_map(tidytable: pd.DataFrame, idxkey: str, valuekey: str, report=True) \
        -> pd.Series:
    """Return a 1D Series of unique pairs of values in the specified columns.
    tidytable[idxkey] : tidytable[valuekey].

    Report duplicate/NA status of columns. NAN in idxkey are dropped."""
    k1, k2 = idxkey, valuekey
    mapr = tidytable.dropna(subset=k1, ).drop_duplicates([k1, k2]).set_index(k1)[k2]

    def valsduped(vals):
        m = vals.duplicated()
        return len(vals[m].unique())

    if report:
        idxdupe = valsduped(mapr.index)
        valdupe = valsduped(mapr.dropna())
        valnan = mapr.isna().sum()
        if idxdupe:
            print(f"!! {idxdupe} labels duplicated in the index !!")
        if valdupe:
            print(f"{valdupe} values duplicated.")
        if valnan:
            print(f"Values contain {valnan} NaNs")
    return mapr