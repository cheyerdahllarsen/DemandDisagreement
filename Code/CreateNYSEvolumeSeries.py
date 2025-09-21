import pandas as pd
from pathlib import Path

def annual_avg_volume(path_1888_2003: str, path_2004_2017: str) -> pd.Series:
    """
    Re‑creates the NYSE annual *average‑daily* volume series found in
    StockVolumeNYSE.mat, using the two original Excel work‑books.

    Returns
    -------
    pandas.Series
        Index = calendar year (int), Values = average daily volume.
        Shares from 1888‑1999; US‑dollars from 2000‑2017.
    """
    #########
    # helper
    #########
    def _first_numeric_row(df):
        """index of first row whose first cell is numeric."""
        return df[pd.to_numeric(df.iloc[:,0], errors='coerce').notna()].index[0]

    def _load_decade_sheet(book: Path, sheet: str) -> pd.DataFrame:
        df = pd.read_excel(book, sheet_name=sheet, header=None)
        df = df.iloc[_first_numeric_row(df):, [0, -1]]      # date col, last col
        df.columns = ['date', 'val']
        df['date'] = pd.to_numeric(df['date'], errors='coerce').astype('Int64')
        df.dropna(inplace=True)

        # YYYYMMDD ➟ datetime
        df['date'] = pd.to_datetime(df['date'].astype(str).str.zfill(8),
                                    format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['date'])
        df['val']  = pd.to_numeric(df['val'], errors='coerce')

        # the 2000‑2003 sheet stores dollar volume in *millions*
        if sheet == '2000-2003':
            df['val'] *= 1_000_000

        return df[['date', 'val']]

    ###########################################################
    # 1) 1888‑2003 workbook  •  one sheet per decade
    ###########################################################
    wb1 = Path(path_1888_2003)
    dfs   = []
    for sheet in pd.ExcelFile(wb1).sheet_names:
        if sheet == 'AllData':            # summary tab – skip
            continue
        dfs.append(_load_decade_sheet(wb1, sheet))

    df_early = pd.concat(dfs, ignore_index=True)

    ###########################################################
    # 2) 2004‑2017 workbook  •  find the real header row
    ###########################################################
    wb2 = Path(path_2004_2017)
    tmp = pd.read_excel(wb2, header=None)
    header_row = tmp[tmp.iloc[:,1].astype(str).str.strip() == 'Trade Date'].index[0]

    df_late = (
        pd.read_excel(wb2, header=header_row)
          .loc[:, ['Trade Date', 'NYSE Group Dollar Volume']]
          .rename(columns={'Trade Date':'date', 'NYSE Group Dollar Volume':'val'})
          .dropna()
    )

    df_late['date'] = pd.to_datetime(df_late['date'])
    df_late['val']  = pd.to_numeric(df_late['val'], errors='coerce')

    ###########################################################
    # 3)  merge, average by calendar year
    ###########################################################
    full = pd.concat([df_early, df_late], ignore_index=True)
    full['year'] = full['date'].dt.year

    return (
        full.groupby('year')['val']
            .mean()
            .sort_index()
            .astype('float64')
    )

# usage -------------------------------------------------------
annual_volume = annual_avg_volume('Data/NYSEvolumeData.xlsx',
                                  'Data/NYSEvolumeData2004to2017.xlsx')

# Save to Excel
annual_volume_df = annual_volume.reset_index()
annual_volume_df.columns = ['Year', 'AverageDailyVolume']
annual_volume_df.to_excel('Data/NYSE_annual_volume_series.xlsx', index=False)

print(annual_volume.head())   # 1888‑1892
print(annual_volume.tail())   # 2013‑2017
