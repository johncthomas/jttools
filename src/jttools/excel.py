import logging
import os
import pathlib

import pandas as pd
import xlsxwriter
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet
from xlsxwriter import Workbook
import typing
from typing import Callable, TypedDict, NewType

from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype



ColStr = str

# useful for selecting an entire row/col (this is the recommended method...)
ROWMAX = 1048575
COLMAX = 16383

from icecream import ic
ic.disable()


__all__ = ['add_vert_info_sheet', 'conditional_format_definitions', 'add_stats_worksheet',
           'write_stats_workbook']

def open_wb(wb: Workbook | str | os.PathLike) -> Workbook:
    if isinstance(wb, Workbook):
        return wb
    return Workbook(wb)

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
    ))
    return text_fmt

def add_vert_info_sheet(wb:Workbook, info_dict: dict[str, str] | str,
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
    """Methods returning default values as dicts for conditional formats,
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
            mid_color ='#FFEB84',
            max_color = '#FFFFFF',
            min_type='num',
            mid_type='num',
            max_type='num',
        )


def number_format_columns(
    workbook:xlsxwriter.Workbook,
    df:pd.DataFrame,
    num_formats: dict[ColStr, str] = None,
    auto_formats=True,
) -> dict[ColStr, dict]:
    """Get dict of column definitions. Should be

    use num_formats to specify specific formats"""
    num_formats = {} if num_formats is None else num_formats

    # this results in the same formats being added to the workbook
    #  multiple times, but I can't find an elegant way to avoid it
    #  (like, creating Format directly and passing it to add_format
    #  doesn't work even if you convert it to a dict first) that definitely
    #  wouldn't cause issues in the future,
    #  but hopefully it doesn't cause issues...
    int_format = workbook.add_format({'num_format': '0'})
    float_format = workbook.add_format({'num_format': '0.00'})
    text_format = workbook.add_format({'num_format': '@', })

    num_formats = {col: workbook.add_format({'num_format': f})
                   for col, f in set(num_formats.items())}

    tab_columns = {}
    for col, dtype in df.dtypes.items():
        col:ColStr
        cold = {'header': col}

        fmt = False
        if col in num_formats:
            fmt = num_formats[col]
        elif auto_formats:
            if is_float_dtype(dtype):
                fmt = float_format
            elif is_integer_dtype(dtype):
                fmt = int_format
            elif is_string_dtype(dtype):
                fmt = text_format

        if fmt:
            cold['format'] = fmt
        ic(cold)
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
    worksheet: Worksheet = workbook.add_worksheet(name=sheet_name)

    # excel table does not include df.index by default
    # If it's a range index we don't want to write it,
    if ((type(table.index) is not pd.RangeIndex)
            # If it's already in the columns, we don't want to add it.
            and (table.index.name not in table.columns.values)):
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
    conditional_formats:dict[ColStr, dict]=None,
    close_workbook=True,
    drop_empty_rows=True,
    xlsx_table_opts:dict=None,
    xlsx_table_opts_per_sheet: dict[str, dict] = None,
):
    """Writes DF in sheets as tables to tabs in an excel workbook."""

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

        number_formats = number_format_columns(
            workbook,
            tab,
            extra_num_formats,
            auto_num_formats
        )

        sheet = add_stats_worksheet(
            workbook,
            tab,
            sheet_name,
            column_definitions=number_formats,
            drop_na_rows=drop_empty_rows,
            xlsx_table_opts=sheet_opts,
        )

        if conditional_formats:
            conditional_format_columns(
                worksheet=sheet,
                formats=conditional_formats,
                df_columns=tab.columns
            )

    if close_workbook:
        workbook.close()
    return workbook

def _test():

    # data
    test_data = {'00nM': {'LFC': {'RLA0_P05388_50': 0.8688471880046018,
       'MFAP3_P55082_219': 0.19358790662551417,
       'PRKDC_P78527_824': -0.21253595333269892,
       'ABCD3_P28288_260': 0.16619306725573324,
       'PA2G4_Q9UQ80_287': 0.0767043168285726,
       'UB2R2_Q712K3_167': -0.08322290529507548,
       'RS15_P62841_65': 0.47915986219714224,
       'RS27A_P62979_89': 0.272159532569356,
       'NAF1_Q96HR8_172': -0.06664983797016344,
       'TPR_P12270_463': -0.19953959571059254},
      'Expr': {'RLA0_P05388_50': 7.915942931537074,
       'MFAP3_P55082_219': 8.040632690275782,
       'PRKDC_P78527_824': 6.308567919439216,
       'ABCD3_P28288_260': 6.935404840129473,
       'PA2G4_Q9UQ80_287': 6.913402237993912,
       'UB2R2_Q712K3_167': 5.956775159225586,
       'RS15_P62841_65': 5.479334777003339,
       'RS27A_P62979_89': 7.086846837017442,
       'NAF1_Q96HR8_172': 6.620867074448822,
       'TPR_P12270_463': 5.647648322628092},
      'p': {'RLA0_P05388_50': 1.2770378903979727e-05,
       'MFAP3_P55082_219': 0.19424579690219598,
       'PRKDC_P78527_824': 0.15829370431753703,
       'ABCD3_P28288_260': 0.19440047290467188,
       'PA2G4_Q9UQ80_287': 0.6165331013683086,
       'UB2R2_Q712K3_167': 0.6129617901639373,
       'RS15_P62841_65': 0.0028088592006240655,
       'RS27A_P62979_89': 0.11290906572421838,
       'NAF1_Q96HR8_172': 0.5812291302533621,
       'TPR_P12270_463': 0.12323413399501641},
      'FDR': {'RLA0_P05388_50': 0.03915398171960184,
       'MFAP3_P55082_219': 0.9992272153953043,
       'PRKDC_P78527_824': 0.9992272153953043,
       'ABCD3_P28288_260': 0.9992272153953043,
       'PA2G4_Q9UQ80_287': 0.9992272153953043,
       'UB2R2_Q712K3_167': 0.9992272153953043,
       'RS15_P62841_65': 0.8898925533983435,
       'RS27A_P62979_89': 0.9992272153953043,
       'NAF1_Q96HR8_172': 0.9992272153953043,
       'TPR_P12270_463': 0.9992272153953043},
      'GeneSymbol': {'RLA0_P05388_50': 'RPLP0',
       'MFAP3_P55082_219': 'MFAP3',
       'PRKDC_P78527_824': 'PRKDC',
       'ABCD3_P28288_260': 'ABCD3',
       'PA2G4_Q9UQ80_287': 'PA2G4',
       'UB2R2_Q712K3_167': 'UBE2R2',
       'RS15_P62841_65': 'RPS15',
       'RS27A_P62979_89': 'RPS27A',
       'NAF1_Q96HR8_172': 'NAF1',
       'TPR_P12270_463': 'TPR'},
      'Description': {'RLA0_P05388_50': '60S acidic ribosomal protein P0 OS=Homo sapiens OX=9606 GN=RPLP0 PE=1 SV=1',
       'MFAP3_P55082_219': 'Microfibril-associated glycoprotein 3 OS=Homo sapiens OX=9606 GN=MFAP3 PE=1 SV=1',
       'PRKDC_P78527_824': 'DNA-dependent protein kinase catalytic subunit OS=Homo sapiens OX=9606 GN=PRKDC PE=1 SV=3',
       'ABCD3_P28288_260': 'ATP-binding cassette sub-family D member 3 OS=Homo sapiens OX=9606 GN=ABCD3 PE=1 SV=1',
       'PA2G4_Q9UQ80_287': 'Proliferation-associated protein 2G4 OS=Homo sapiens OX=9606 GN=PA2G4 PE=1 SV=3',
       'UB2R2_Q712K3_167': 'Ubiquitin-conjugating enzyme E2 R2 OS=Homo sapiens OX=9606 GN=UBE2R2 PE=1 SV=1',
       'RS15_P62841_65': '40S ribosomal protein S15 OS=Homo sapiens OX=9606 GN=RPS15 PE=1 SV=2',
       'RS27A_P62979_89': 'Ubiquitin-40S ribosomal protein S27a OS=Homo sapiens OX=9606 GN=RPS27A PE=1 SV=2',
       'NAF1_Q96HR8_172': 'H/ACA ribonucleoprotein complex non-core subunit NAF1 OS=Homo sapiens OX=9606 GN=NAF1 PE=1 SV=2',
       'TPR_P12270_463': 'Nucleoprotein TPR OS=Homo sapiens OX=9606 GN=TPR PE=1 SV=3'},
      'HGNCID': {'RLA0_P05388_50': 'HGNC:10371',
       'MFAP3_P55082_219': 'HGNC:7034',
       'PRKDC_P78527_824': 'HGNC:9413',
       'ABCD3_P28288_260': 'HGNC:67',
       'PA2G4_Q9UQ80_287': 'HGNC:8550',
       'UB2R2_Q712K3_167': 'HGNC:19907',
       'RS15_P62841_65': 'HGNC:10388',
       'RS27A_P62979_89': 'HGNC:10417',
       'NAF1_Q96HR8_172': 'HGNC:25126',
       'TPR_P12270_463': 'HGNC:12017'},
      'UniprotID': {'RLA0_P05388_50': 'P05388',
       'MFAP3_P55082_219': 'P55082',
       'PRKDC_P78527_824': 'P78527',
       'ABCD3_P28288_260': 'P28288',
       'PA2G4_Q9UQ80_287': 'Q9UQ80',
       'UB2R2_Q712K3_167': 'Q712K3',
       'RS15_P62841_65': 'P62841',
       'RS27A_P62979_89': 'P62979',
       'NAF1_Q96HR8_172': 'Q96HR8',
       'TPR_P12270_463': 'P12270'}},
     '_1uM': {'LFC': {'RLA0_P05388_50': 2.1183969255068646,
       'MFAP3_P55082_219': 0.17095923114560474,
       'PRKDC_P78527_824': -0.04388346209771843,
       'ABCD3_P28288_260': 0.7119694016197817,
       'PA2G4_Q9UQ80_287': -0.1420110383395201,
       'UB2R2_Q712K3_167': -0.3327212863176685,
       'RS15_P62841_65': 2.3834083415659295,
       'RS27A_P62979_89': 0.640275240420082,
       'NAF1_Q96HR8_172': -0.21973203948955788,
       'TPR_P12270_463': -0.19989017522914398},
      'Expr': {'RLA0_P05388_50': 7.915942931537074,
       'MFAP3_P55082_219': 8.040632690275782,
       'PRKDC_P78527_824': 6.308567919439216,
       'ABCD3_P28288_260': 6.935404840129473,
       'PA2G4_Q9UQ80_287': 6.913402237993912,
       'UB2R2_Q712K3_167': 5.956775159225586,
       'RS15_P62841_65': 5.479334777003339,
       'RS27A_P62979_89': 7.086846837017442,
       'NAF1_Q96HR8_172': 6.620867074448822,
       'TPR_P12270_463': 5.647648322628092},
      'p': {'RLA0_P05388_50': 2.0638619148096507e-11,
       'MFAP3_P55082_219': 0.24922784497820819,
       'PRKDC_P78527_824': 0.764747155240887,
       'ABCD3_P28288_260': 1.7186988493746083e-05,
       'PA2G4_Q9UQ80_287': 0.35789189575377145,
       'UB2R2_Q712K3_167': 0.0542165667878174,
       'RS15_P62841_65': 1.0597786162157583e-12,
       'RS27A_P62979_89': 0.0009874403172407393,
       'NAF1_Q96HR8_172': 0.08037061894199579,
       'TPR_P12270_463': 0.12261871188289274},
      'FDR': {'RLA0_P05388_50': 4.2185337538709264e-08,
       'MFAP3_P55082_219': 0.6204580555702286,
       'PRKDC_P78527_824': 0.9189995892779093,
       'ABCD3_P28288_260': 0.0023952412146284316,
       'PA2G4_Q9UQ80_287': 0.710224305748261,
       'UB2R2_Q712K3_167': 0.3303620380518885,
       'RS15_P62841_65': 6.49856247463503e-09,
       'RS27A_P62979_89': 0.036475807381447074,
       'NAF1_Q96HR8_172': 0.3842157357746917,
       'TPR_P12270_463': 0.46005022019901093},
      'GeneSymbol': {'RLA0_P05388_50': 'RPLP0',
       'MFAP3_P55082_219': 'MFAP3',
       'PRKDC_P78527_824': 'PRKDC',
       'ABCD3_P28288_260': 'ABCD3',
       'PA2G4_Q9UQ80_287': 'PA2G4',
       'UB2R2_Q712K3_167': 'UBE2R2',
       'RS15_P62841_65': 'RPS15',
       'RS27A_P62979_89': 'RPS27A',
       'NAF1_Q96HR8_172': 'NAF1',
       'TPR_P12270_463': 'TPR'},
      'Description': {'RLA0_P05388_50': '60S acidic ribosomal protein P0 OS=Homo sapiens OX=9606 GN=RPLP0 PE=1 SV=1',
       'MFAP3_P55082_219': 'Microfibril-associated glycoprotein 3 OS=Homo sapiens OX=9606 GN=MFAP3 PE=1 SV=1',
       'PRKDC_P78527_824': 'DNA-dependent protein kinase catalytic subunit OS=Homo sapiens OX=9606 GN=PRKDC PE=1 SV=3',
       'ABCD3_P28288_260': 'ATP-binding cassette sub-family D member 3 OS=Homo sapiens OX=9606 GN=ABCD3 PE=1 SV=1',
       'PA2G4_Q9UQ80_287': 'Proliferation-associated protein 2G4 OS=Homo sapiens OX=9606 GN=PA2G4 PE=1 SV=3',
       'UB2R2_Q712K3_167': 'Ubiquitin-conjugating enzyme E2 R2 OS=Homo sapiens OX=9606 GN=UBE2R2 PE=1 SV=1',
       'RS15_P62841_65': '40S ribosomal protein S15 OS=Homo sapiens OX=9606 GN=RPS15 PE=1 SV=2',
       'RS27A_P62979_89': 'Ubiquitin-40S ribosomal protein S27a OS=Homo sapiens OX=9606 GN=RPS27A PE=1 SV=2',
       'NAF1_Q96HR8_172': 'H/ACA ribonucleoprotein complex non-core subunit NAF1 OS=Homo sapiens OX=9606 GN=NAF1 PE=1 SV=2',
       'TPR_P12270_463': 'Nucleoprotein TPR OS=Homo sapiens OX=9606 GN=TPR PE=1 SV=3'},
      'HGNCID': {'RLA0_P05388_50': 'HGNC:10371',
       'MFAP3_P55082_219': 'HGNC:7034',
       'PRKDC_P78527_824': 'HGNC:9413',
       'ABCD3_P28288_260': 'HGNC:67',
       'PA2G4_Q9UQ80_287': 'HGNC:8550',
       'UB2R2_Q712K3_167': 'HGNC:19907',
       'RS15_P62841_65': 'HGNC:10388',
       'RS27A_P62979_89': 'HGNC:10417',
       'NAF1_Q96HR8_172': 'HGNC:25126',
       'TPR_P12270_463': 'HGNC:12017'},
      'UniprotID': {'RLA0_P05388_50': 'P05388',
       'MFAP3_P55082_219': 'P55082',
       'PRKDC_P78527_824': 'P78527',
       'ABCD3_P28288_260': 'P28288',
       'PA2G4_Q9UQ80_287': 'Q9UQ80',
       'UB2R2_Q712K3_167': 'Q712K3',
       'RS15_P62841_65': 'P62841',
       'RS27A_P62979_89': 'P62979',
       'NAF1_Q96HR8_172': 'Q96HR8',
       'TPR_P12270_463': 'P12270'}}}



    tt = {k:pd.DataFrame(v) for k, v in test_data.items()}
    test_tables = {}

    cnd_fmts = {
        'LFC':conditional_format_definitions.score(2),
        'FDR':conditional_format_definitions.significance(),
    }

    # ic(cnd_fmts)
    ic.enable()
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
    add_vert_info_sheet(wb, info, )
    write_stats_workbook(tables=test_tables,
                         workbook=wb,
                         conditional_formats=cnd_fmts,
                         extra_num_formats={'p':'0.0000'},
                         xlsx_table_opts={'style':'Table Style Medium 16'},
                         xlsx_table_opts_per_sheet=sheet_opts,
                         )

if __name__ == '__main__':
    _test()