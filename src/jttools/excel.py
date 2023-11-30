import logging
import os
import pathlib
import math

import pandas as pd
import xlsxwriter
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet
from xlsxwriter import Workbook
import typing
from typing import Callable, TypedDict, NewType

from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

ColStr = str

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

from icecream import ic
ic.disable()

__all__ = ['add_info_sheet', 'conditional_format_definitions', 'add_stats_worksheet',
           'write_stats_workbook', 'ROWMAX', 'COLMAX']

# useful for selecting an entire row/col (this is the recommended method...)
ROWMAX = 1048575
COLMAX = 16383

def open_wb(wb: Workbook | str | os.PathLike) -> Workbook:
    if isinstance(wb, Workbook):
        return wb
    return Workbook(wb)

# def ignore_nan(worksheet, row, col, number, format=None):
#     """This should be added to a worksheet using:
#     worksheet.add_write_handler(float, ignore_nan)"""
#     if math.isnan(number):
#         return worksheet.write_blank(row, col, None, format)
#     else:
#         # Return control to the calling write() method for any other number.
#         return None
#
# # # Set up the workbook as usual.
# # workbook = xlsxwriter.Workbook("user_types2.xlsx")
# # worksheet = workbook.add_worksheet()
# # worksheet.add_write_handler()

def add_row_title_fmt(wb:Workbook, **format_kwargs) -> Format:
    """Add xlsxwriter Format to workbook and return it."""
    fmt = dict(
        bold=True,
        align='center',
        text_wrap=True,
        num_format='@',
        font_size=14,
        bg_color='black',
        font_color='white'
    )

    title_fmt = wb.add_format(fmt | format_kwargs)
    title_fmt.set_align('vcenter')

    return title_fmt

def add_wrapped_text_fmt(wb:Workbook, **format_kwargs) -> Format:
    text_fmt = wb.add_format(dict(
        align='vcenter',
        text_wrap=True,
        num_format='@'

    ) | format_kwargs)

    return text_fmt

def add_info_sheet(wb:Workbook, info_dict: dict[str, str] | str,
                   sheet_name='Info') -> Worksheet:
    """Write sheet with information arranged with first column giving
    title of info and 2nd col giving text.

    Args:
        wb: Open Workbook. Is left open.
        info_dict: title:info pairs, or just a string that will be written
            to row entitled "Info".
        sheet_name: Name for the sheet that is added.
    """
    ws = wb.add_worksheet(sheet_name)
    ws.set_column(0, 0, 15, cell_format=add_row_title_fmt(wb))
    ws.set_column(1, 1, 100, cell_format=add_wrapped_text_fmt(wb))

    if isinstance(info_dict, str):
        info_dict = {'Info':info_dict}

    for rowi, (title, text) in enumerate(info_dict.items()):
        ws.write_string(rowi, 0, title)
        ws.write_string(rowi, 1, text)

    return ws


class ThreeColorScaleFmt(TypedDict):
    """Dict of specifications for 3_color_scale conditionalformatting

    Colors are #htmlstyle.
    valid values for min_type are:
        min        (for min_type only)
        num
        percent
        percentile
        formula
        max        (for max_type only)

    example:
        fmt = ThreeColorScaleFmt(type='3_color_scale', ...)
        worksheet.conditional_format(loc,fmt )"""
    type:str
    min_value:float
    max_value:float
    mid_value:float
    min_type:str
    mid_type: str
    max_type: str
    min_color:str
    mid_color:str
    max_color:str

class conditional_format_definitions:
    """Static methods returning default values as dicts for conditional formats,
    pass the dict to:

    conditional_format_columns(**kw, format={'column':<formatdict>})

    Methods:
        score: ...
        signifcance: ...

    """

    @staticmethod
    def score(threshold=2) -> ThreeColorScaleFmt:
        """Get 3_color_scale conditional format dict with white=0, orange
        for low values, pale blue for high.
        """
        return ThreeColorScaleFmt(
            type='3_color_scale',
            min_value=-threshold,
            mid_value=0,
            max_value=threshold,
            min_color='#00B0F0',
            mid_color='#FFFFFF',
            max_color='#FFC000',
            min_type='num',
            mid_type='num',
            max_type='num',
        )

    @staticmethod
    def significance() -> ThreeColorScaleFmt:
        """Get 3_color_scale conditional format dict that fades to
        yellow 0.2-0.1 and then to green at lower values.
        """
        return ThreeColorScaleFmt(
            type='3_color_scale',
            min_value=0,
            mid_value=0.1,
            max_value=0.2,
            min_color ='#92D050',
            mid_color ='#FFFF00',
            max_color = '#FFFFFF',
            min_type='num',
            mid_type='num',
            max_type='num',
        )


def format_columns(
        workbook:xlsxwriter.Workbook,
        df:pd.DataFrame,
        col_formats: dict[ColStr, dict[str, typing.Any]] = None,
        auto_formats=True,
) -> dict[ColStr, dict]:
    """Get dict of column definitions.

    use num_formats to specify specific formats. Dictionaries passed to
    xlsxwriter WB.add_format()"""

    col_formats = {} if col_formats is None else col_formats

    int_format = workbook.add_format({'num_format': '0'})
    float_format = workbook.add_format({'num_format': '0.00'})
    text_format = workbook.add_format({'num_format': '@', })

    # The trick here is that we need to call workbook.add_format to get the
    #   proper format obj (can't create one from the class directly that works),
    #   so we'll tuplize the format dictionaries and keep track of the returned
    #   Format objects.

    # This should be safe, the dicts shouldn't contain any unhashables.
    # If not use IDs as keys.
    dict_to_tupes = lambda d: tuple(sorted(list(d.items())))
    added_formats = {}

    tab_columns = {}
    for col, dtype in df.dtypes.items():
        col:ColStr
        cold = {'header': col}

        fmt = False
        if col in col_formats:
            fmt_d = col_formats[col]
            fmt_k = dict_to_tupes(fmt_d)
            if fmt_k not in added_formats:
                fmt = workbook.add_format(fmt_d)
            else:
                fmt = added_formats[fmt_k]
        elif auto_formats:
            if is_float_dtype(dtype):
                fmt = float_format
            elif is_integer_dtype(dtype):
                fmt = int_format
            elif is_string_dtype(dtype):
                fmt = text_format

        if fmt:
            cold['format'] = fmt
        #ic(cold)
        tab_columns[col] = cold
    return tab_columns

def conditional_format_columns(
        worksheet: Worksheet,
        formats:dict[ColStr, dict],
        df_columns:typing.Collection[str]
):
    """Applies conditional formatting directly to the worksheet. Formats should
    give the names of column(s) in the df_columns. Format applied to the whole
    column.

    Worksheet is altered in place."""

    df_columns = list(df_columns)
    for col, fmt in formats.items():
        coli = df_columns.index(col)+1
        ic(fmt)
        worksheet.conditional_format(
            0, coli, ROWMAX, coli, fmt
        )

def add_stats_worksheet(
        workbook: xlsxwriter.Workbook | str | os.PathLike,
        table: pd.DataFrame,
        sheet_name: str,
        column_definitions:dict[ColStr, dict]=None,
        xlsx_table_opts: dict = None,
) -> Worksheet:

    workbook = open_wb(workbook)

    """Score_col and sig_col are used for sorting table if included."""
    if column_definitions is None:
        column_definitions = {}
    if xlsx_table_opts is None:
        xlsx_table_opts = {}

    sheet_name = sheet_name
    worksheet: Worksheet = workbook.add_worksheet(name=sheet_name, )
    worksheet.nan_inf_to_errors = True

    # Because a bunch of things need to be done by getting an integer index, I need
    #   to know if the index will be included or not. So it's always included even if
    #   it's just a bunch of meaningless numbers or a duplicate.

    # excel table does not include df.index by default
    # If it's a range index we don't want to write it,
    # if ((type(table.index) is not pd.RangeIndex)
    #         # If it's already in the columns, we don't want to add it.
    #         and (table.index.name not in table.columns.values)):

    table = table.reset_index()

    nrows, ncols = table.shape

    # by default a column definition is just a header, then we add in any
    # extra per column details and convert it to a list
    col_dict = {c: {'header': c} for c in table.columns}
    for col, d in column_definitions.items():
        if col in col_dict:
            col_dict[col].update(d)
    columns = list(col_dict.values())

    opts = dict(
        data=table.values,
        columns=columns,
    )

    opts = opts | xlsx_table_opts
    worksheet.add_table(
        0, 0, nrows, ncols - 1,
        opts
    )

    return worksheet

def write_stats_workbook(
        workbook:Workbook|str|os.PathLike,
        tables: dict[str, pd.DataFrame]|pd.DataFrame,
        auto_num_formats=True,
        extra_num_formats:dict[ColStr, str]=None,
        other_formats:dict[ColStr, dict]=None,
        conditional_formats:dict[ColStr, dict]=None,
        close_workbook=True,
        xlsx_table_opts:dict=None,
        xlsx_table_opts_per_sheet: dict[str, dict] = None,
):
    """Writes DF in sheets as tables to tabs in an excel workbook.

    Per column options apply to all sheets with that column name.

    Args:
        workbook: xlsxwriter Workbook class or path to where WB should be opened.
        tables: A single table or multiple can be passed as a dictionary,
            keys defining sheet names (so they need to follow Excel rules for that.)
        auto_num_formats: Use DF.dtypes to auto define and apply Excel formatting
            (including formatting `obj` as text)
        extra_num_formats: Number format per column, value should be an Excel format
            string, what you see as "example" when customising formats.
        other_formats: Dict of cell formats per column.
            e.g.: {'LongTextCol':{'num_format':'@', 'text_wrap':True}}
            See: https://xlsxwriter.readthedocs.io/format.html (nice table half way down)
        conditional_formats: Conditional format per column. Some available as
            conditional_format_definitions.<method>(), or define a dict.
            See: https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html
        close_workbook: WB is written upon closing, set to False to continue working on
            the open WB.
        xlsx_table_opts: See https://xlsxwriter.readthedocs.io/working_with_tables.html
        xlsx_table_opts_per_sheet:
            See https://xlsxwriter.readthedocs.io/working_with_tables.html
    """

    workbook = open_wb(workbook)

    # deal with the optional values.
    # extra_num_formats = {} if extra_num_formats is None else extra_num_formats
    # conditional_formats = {} if conditional_formats is None else conditional_formats
    if xlsx_table_opts_per_sheet is None:
        xlsx_table_opts_per_sheet = {}
    if xlsx_table_opts is None:
        xlsx_table_opts = {}

    # figure out how to iterate through the tables
    if isinstance(tables, pd.DataFrame):
        if isinstance(tables.columns, pd.MultiIndex):
            sheet_names = tables.columns.levels[0]
        else:
            sheet_names = ['Result']
            tables = {sheet_names[0]:tables}
    else:
        sheet_names = tables.keys()

    for sheet_name in sheet_names:

        if sheet_name in xlsx_table_opts_per_sheet:
            sheet_opts = xlsx_table_opts | xlsx_table_opts_per_sheet[sheet_name]
        else:
            sheet_opts = xlsx_table_opts
        tab = tables[sheet_name]
        #print(tab.head())

        def filter_to_available_cols(d:dict|None) -> dict:
            if d is None:
                return {}
            return {col: v for col, v in d.items() if col in tab.columns}

        if extra_num_formats is not None:
            num_formats = {col:{'num_format': f}
                           for col, f in extra_num_formats.items()}
        else:
            num_formats = {}

        if other_formats is None:
            other_formats = {}

        # join the formatting dictionaries
        keys = set()

        keys.update(num_formats.keys())
        keys.update(other_formats.keys())

        column_formats = {}
        for col in keys:
            column_formats[col] = num_formats.get(col, {}) | other_formats.get(col, {})

        number_formats = format_columns(
            workbook,
            tab,
            col_formats=column_formats,
            auto_formats=auto_num_formats,
        )

        sheet = add_stats_worksheet(
            workbook,
            tab,
            sheet_name,
            column_definitions=number_formats,
            xlsx_table_opts=sheet_opts,
        )

        if conditional_formats:
            ic(conditional_formats)
            conditional_format_columns(
                worksheet=sheet,
                formats=conditional_formats,
                df_columns=tab.columns
            )

    if close_workbook:
        workbook.close()
    return workbook

def _test():
    ks = ['00nM', '_1uM']
    fn = '/home/jthomas/pycharm/JTtools/test_data/test_data_{}.csv'
    test_tables = {k:pd.read_csv(fn.format(k), index_col=0) for k in ks}

    cnd_fmts = {
        'LFC':conditional_format_definitions.score(2),
        'FDR':conditional_format_definitions.significance(),
    }

    ic(cnd_fmts)

    logging.info('Testing writing multiple sheets, conditional formating, '
                 'auto num formatting, manual numformatting, table options'
                 ' and per sheet table options', )

    sheet_opts = {}
    for k, v in tt.items():
        v.Expr = v.Expr.astype(int)
        test_tables[k] = v
        sheet_opts[k] = {'name':'TAble' + k}

    info = {'thing': 'some text about thing',
            'other': 'hey here is more text wow' * 50}

    wb = xlsxwriter.Workbook('/mnt/m/tmp/formatted.1.xlsx')
    add_info_sheet(wb, info, )
    write_stats_workbook(tables=test_tables,
                         workbook=wb,
                         conditional_formats=cnd_fmts,
                         extra_num_formats={'p':'0.0000'},
                         xlsx_table_opts={'style':'Table Style Medium 16'},
                         xlsx_table_opts_per_sheet=sheet_opts,
                         )

if __name__ == '__main__':
    pass
    #
    # wrapped = dict(text_wrap=True, num_format='@')
    # cond_formats = conditional_format_definitions
    # gsea_table = pd.read_csv('/mnt/m/tmp/x.xlsx', index_col=0)
    # score = 'p10'
    # for trt in gsea_table.Comparison.unique():
    #     t = gsea_table.loc[(gsea_table.Comparison == trt) & (gsea_table.Score == score)].drop('Score', axis=1)
    #     t = t.sort_values('FDR', )
    #     wb = write_stats_workbook(
    #         # f'out/Excel workbooks/GSEA.{trt}.{score}.xlsx',\\
    #         '/mnt/m/tmp/test.xlsx',
    #         {'Results': t},
    #         conditional_formats={
    #             'ES': cond_formats.score(t.ES.abs().max()),
    #             'NES': cond_formats.score(t.NES.abs().max()),
    #             'FDR': cond_formats.significance(),
    #             'FWER': cond_formats.significance(),
    #         },
    #         other_formats={'Term': wrapped},
    #         close_workbook=False,
    #     )
    #
    #     wb: xlsxwriter.Workbook
    #     sheet: Worksheet = wb.get_worksheet_by_name('Results')
    #     # plus 1 cus index
    #     term_i = list(t.columns).index('Term') + 1
    #     sheet.set_column_pixels(term_i, term_i, width=500)
    #     wb.close()