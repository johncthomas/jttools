from typing import List, Union, Tuple

import pandas as pd

import logging

import statsmodels.api as sm


import typing
from typing import Callable

from jttools.excel import write_stats_workbook

from attrdictionary import AttrMap

__all__ = [
    'OLS', 'is_numeric', 'AttrMapAC', 'list_not_str', 'read_csv', 'read_txt', 'read_list', 'read_tsv',
    'write_stats_workbook', 'df_underscore_columns', 'index_of_true', 'keys_crawler', 'unify_indicies',
    'df_rename_columns', 'format_vertical_headers', 'format_p_table_for_display', 'write_list', 'iterate_groups',
    'tidy_to_map', 'str_remove_consecutive', 'df_from_str', 'df_set_index_dropping_dupes',
    'monkey_patch_dataframe', 'load_latest_pyensembl_release'
]

OLS = sm.regression.linear_model.OLS
is_numeric = pd.api.types.is_numeric_dtype


def load_latest_pyensembl_release(**kwargs):
    """kwargs passed to pyensembl.EnsembleRelease(i, **kw)"""
    from pyensembl import EnsemblRelease
    ensembl = None
    for i in range(300, 55, -1):
        try:
            ensembl = EnsemblRelease(i, **kwargs)
            ensembl.contigs()
            print('Ensembl release', i)
            break
        except:
            pass
    if ensembl is None:
        print('No ensembl DB found')
        return None

    return ensembl




from typing import Callable

class AttrMapAC(AttrMap):
    """Tab completable AttrMap"""

    # def f(self):
    #     for thing in self:
    #         print(thing)

    def __dir__(self):
        super_dir:list[str] = super().__dir__()
        string_keys = [str(key) for key in self if isinstance(key, str)]
        return super_dir + [key for key in string_keys if key not in super_dir]

    def keys(self):
        return list(self._mapping.keys())

    def items(self, dict2attrmap=True):
        for k, v in super().items():
            if dict2attrmap and (type(v) is dict):
                v = AttrMapAC(v)
            yield k, v




def list_not_str(thing):
    """If thing is a string, wrap it in a list."""
    if type(thing) is str:
        return [thing]
    return thing


def read_csv(fn, idx_also_column=False, **kwargs) -> pd.DataFrame:
    """pd.read_csv(fn, index_col=0), checks for dupes in the index.

    If idx_also_column, the first column will be the index and also
    the first columns."""
    default_kwargs = dict(index_col=0)
    kwargs = default_kwargs | kwargs
    df = pd.read_csv(fn, **kwargs)

    if idx_also_column:
        df = pd.read_csv(fn, **kwargs)
        idx = df.index
        df.insert(0, idx.name, idx)

    if df.index.duplicated().any():
        logging.warning("Index contains duplicate values")

    return df


def read_tsv(fn, idx_also_column=False, **kwargs) -> pd.DataFrame:
    """See read_csv"""
    return read_csv(fn, sep='\t', idx_also_column=idx_also_column, **kwargs)


def read_txt(fn, sep='\t', index_col=None) -> pd.DataFrame:
    """Read headerless, tab separated table. If it's one column wide, give a Series."""
    df = read_csv(fn, header=None, sep=sep, index_col=index_col)
    if df.shape[1] == 1:
        return df[1]
    return df


# def write_excel_text(sheets: Dict[str, pd.DataFrame],
#                      filename: str,
#                      text_columns: Union[List, Dict[str, List[str]]] = 'ALL',
#                      index_as_text=False,
#                      **to_excel_kwargs):
#     """Write XLSX with columns optionally formated as text.
#
#     Args:
#         sheets: Dataframes to be written as sheets, dict keyed by sheetname,
#             values DF
#         filename: For the written XLSX
#         text_columns: Specify columns (by name) to be formated as text.
#             Default 'ALL' formats all columns as string.
#             Lists of column names (including index name) can specify columns.
#             A single list applies to all sheets, use a dictionary to specify
#             different columns per sheet. Can't specify the index
#         index_as_text: Since you can't just pass the name of the index (for
#             tedious reasons), set this to True to write the index as text.
#             Index name will be written as header. Won't work with multiindex
#             columns for some reason.
#         **to_excel_kwargs: Additional kwargs are passed to pd.DataFrame.to_excel()
#     Doesn't work work with multiindex columns.
#     """
#     # tests 'all', text_col is list or dict
#
#     text_columns = list_not_str(text_columns)
#
#     for df in sheets.values():
#         if type(df.columns) == pd.MultiIndex:
#             raise RuntimeError('MultiIndex columns not supported')
#
#     writer = pd.ExcelWriter(filename,
#                             engine='xlsxwriter')
#
#     # Go through the DF
#     for sheet_name, df in sheets.items():
#         if index_as_text:
#             df = df.reset_index(drop=False)
#         df.to_excel(writer, sheet_name=sheet_name, **to_excel_kwargs)
#
#         worksheet = writer.sheets[sheet_name]
#
#         workbook = writer.book
#         txt = workbook.add_format({'num_format': '@'})
#
#         # get the columns to be textualised
#
#         if text_columns == ['ALL']:
#             txtcols = df.columns
#         elif type(text_columns) is Dict:
#             txtcols = text_columns[sheet_name]
#             if txtcols == 'ALL':
#                 txtcols = df.columns
#         elif (text_columns == None) or (text_columns == False):
#             txtcols = []
#         else:
#             txtcols = list(text_columns)
#
#         # Note: index formats get written over by pd, and any sensible automated
#         #   solution to work around this uses index=False - but that's incompatible with
#         #   multiindex columns for some fucking reason, so there is no good
#         #   solution for dealing with the index until Pandas impliments
#         #   a stable way of disabling automatic formating of the index
#
#         index_offset = 1
#         if 'index' in to_excel_kwargs:
#             if not to_excel_kwargs['index']:
#                 index_offset = 0
#
#         # get indicies of named columns
#         col_i = []
#         col_i.extend([list(df.columns).index(c) + index_offset for c in txtcols])
#
#         # set the format of selected columns
#         for i in col_i:
#             worksheet.set_column(i, i, cell_format=txt)
#
#     writer.save()
#     return writer


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



#
# def write_stats_workbook(
#         tables: Dict[str, pd.DataFrame]|pd.DataFrame,
#         filename=None,
#         columns=None,
#         close_workbook=True,
#         float_format='0.000',
#         workbook: xlsxwriter.Workbook = None,
# ):
#
#     """Writes DF in sheets as tables to tabs in an excel workbook.
#
#     `columns` is passed to sheet.add_table, see:
#         https://xlsxwriter.readthedocs.io/working_with_tables.html"""
#     if workbook is None:
#         if filename is None and close_workbook:
#             raise RuntimeError('Provide a filename, or set close_workbook=False to avoid an Error')
#         workbook = xlsxwriter.Workbook(filename, )
#
#     int_format = workbook.add_format({'num_format': '0'})
#     float_format = workbook.add_format({'num_format': float_format})
#     text_format = workbook.add_format({'num_format': '@', })
#
#     if isinstance(tables, pd.DataFrame):
#         if isinstance(tables.columns, pd.MultiIndex):
#             sheet_names = tables.columns.levels[0]
#         else:
#             sheet_names = ['Result']
#             tables = {sheet_names[0]:tables}
#     else:
#         sheet_names = tables.keys()
#
#
#     for sheet_name in sheet_names:
#         tab = tables[sheet_name]
#         sheet = workbook.add_worksheet(name=sheet_name)
#         if not type(tab.index) is pd.RangeIndex:
#             tab = tab.reset_index()
#         nrows, ncols = tab.shape
#
#         dtypes = tab.dtypes
#
#         if columns:
#             # if specifying columns, need to include index as a column
#             assert len(columns) == ncols
#             tab_columns = columns
#         # automatically style columns based on dtype
#         # todo add more automatic formatting
#         else:
#             tab_columns = []
#             for col in tab.columns:
#                 cold = {'header': col}
#                 try:
#                     cold['format'] = {
#                         int: int_format,
#                         float: float_format,
#                         object: text_format,
#                     }[dtypes[col]]
#                 except KeyError:
#                     pass
#
#                 tab_columns.append(cold)
#         # so I think these numbers are are 0-index end-INCLUSIVE, but you need one more for the columns.
#         sheet.add_table(
#             0, 0, nrows, ncols - 1,
#             options={
#                 'data': tab.values,
#                 'columns': tab_columns
#             }
#         )
#
#     if close_workbook:
#         workbook.close()
#     return workbook


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


def df_rename_columns(df:pd.DataFrame, newcols=dict, inplace=False, verbose=False) -> pd.Index:
    """Return index with renamed columns. Columns missing from newcols will be
    unchanged (this is the main difference to using df.columns.map(newcols).

    Args:
        df: DataFrame with columns we want to change
        newcols: Mapping of current column names to new ones. Only those we
            want to change need be included.
        inplace: df.columns updated in place
        verbose: If true, changed cols are printed. Silent if nothing changes.

    Column labels not found as keys in newcols will be retained.

    """
    old_cols = df.columns

    mapper = {k: k for k in df.columns}
    mapper.update(newcols)

    nucols = df.columns.map(mapper)
    if verbose:
        if not (nucols==old_cols).all():
            logging.info("Columns were changed (starting with orginal): "
                         f"{old_cols.symmetric_difference(nucols, sort=False)}")
        else:
            logging.info('No columns changed')

    if not inplace:
        return nucols

    df.columns = nucols
    return nucols


def df_set_index_dropping_dupes(df:pd.DataFrame, col:str):
    df = df.drop_duplicates(col).set_index(col)
    return df

def df_underscore_columns(df):
    """replace spaces with underscores in all columns of a pandas DF.
    In place, but returns the DF"""
    df.columns = df.columns.map(lambda x: x.replace(' ', '_'))
    return df

def df_select_rows_matching(df:pd.DataFrame, value, column=None, not_=False):

    if column is None:
        m = (df == value).any(1)
    else:
        m = df[column] == value
    if not_:
        m = ~m
    return df.loc[m]


def monkey_patch_dataframe(pd):
    """Monkey patch in method:
    - rename_columns
    - set_index_dropping_dupes
    - underscore_columns
    - select_rows_matching
    """
    pd.DataFrame.rename_columns = df_rename_columns
    pd.DataFrame.set_index_dropping_dupes = df_set_index_dropping_dupes
    pd.DataFrame.underscore_columns = df_underscore_columns
    pd.DataFrame.select_rows_matching = df_select_rows_matching
    # add to doc when more are added

def format_vertical_headers(styler, ):
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


    #styles.update(css)

    return (styler.set_table_styles(styles))

DFStyler = type(pd.DataFrame().style)

def format_p_table_for_display(df, start_gradient=0.2,
                   vert_headers=True) -> DFStyler:

    styler = df.style.format(
        precision=2,
    ).background_gradient(
        axis=1, cmap='Greens_r', vmin=0, vmax=start_gradient
    )

    if vert_headers:
        styler = styler.pipe(format_vertical_headers)

    return styler

def write_list(things, filename):
    """for t in things: write(t+newline)"""
    with open(filename, 'w') as f:
        f.write(
            '\n'.join(things)
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

def str_remove_consecutive(string:str, char:str):
    """Remove all occurances of char when it appears more than once

    Example:
        f("aaaa,aaa,,,a,a,,a", ",") -> "aaaa,aaa,a,a,a"
    """
    dchar = char+char
    while dchar in string:
        string = string.replace(dchar, char)
    return string

def df_from_str(string:str, sep:str, header=True, index_col:int=None,
              consume_multiple_sep=True, markdown=False):
    """String is split by \n into rows, and then sep into columns.

    Note that dtypes might all be object even if they're obviously numbers."""

    if string.startswith('|', ) and string.endswith('|') and (sep == '|') and (not markdown):
        logging.warning("This looks like markdown... if it is call with markdown=True "
                        "or you'll end up with some empty unnamed columns.")

    def splitter(line:str) -> list[str]:
        """Reformat and split the line."""
        if consume_multiple_sep:
            line = str_remove_consecutive(line, sep)
        return [s.strip() for s in line.strip().split(sep)]

    splines = list(map(splitter, string.split('\n')))

    def demarkdowner(line):
        # if it's a header/alignment line, skip it
        if set(list(line)).issubset({'-', '|', ':', ' '}):
            pass
        return line[1:-1]

    if markdown:
        splines = list(map(demarkdowner, splines))

    if header:
        df = pd.DataFrame(splines[1:], columns=splines[0])
    else:
        df = pd.DataFrame(splines)

    if index_col is not None:
        df.set_index(df.columns[index_col], inplace=True)

    return df

def _test_write_stats_workbook():
    print("Testing write stats workbook. Just basic.")
    df = pd.read_csv('/home/jthomas/pycharm/JTtools/test_data/df_of_types.csv')
    print(df)
    testfn = '/home/jthomas/pycharm/JTtools/test_data/test.out'
    write_stats_workbook(testfn, {'sheet':df})
    df2 = pd.read_excel(testfn, sheet_name=0)
    print(df2)


if __name__ == '__main__':
    _test_write_stats_workbook()
    #
    # df = pd.read_csv('/mnt/m/tasks/NA327_Proteomics_UbPulldown/data/regression.USP21_vs_all.1.csv')
    # write_stats_workbook(
    #     '/mnt/m/tasks/NA327_Proteomics_UbPulldown/data/regression.USP21_vs_all.1.xslx',
    #     df,
    #
    # )
