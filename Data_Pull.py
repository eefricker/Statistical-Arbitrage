import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import os
import sys

def yfinance_chunk_pull():
    
    # Read list of S&P500 companies as of 2026-05-11
    ticker_table = pd.read_csv(osp.path.join('data','s_and_p_500_as_of_2026-05-11.csv'))
    symbols = ticker_table['Symbol'].tolist()
    
    # Pull Data by chunks
    chunk_size = 10
    chunks = len(symbols)//chunk_size
    for i in range(chunks+1):
        
        tic= dt.datetime.today()
        print(i,tic)
        
        symbol_chunk = symbols[chunk_size*i:chunk_size*(i+1)]
        symbol_chunk = [symbol.replace('.','-') for symbol in symbol_chunk]
            
        data_chunk = yf.download(symbol_chunk, start="1990-01-01", end="2022-01-01",auto_adjust=True)
    
        os.makedirs("data", exist_ok=True)
        os.makedirs(os.path.join("data",'chunks'), exist_ok=True)
        data_chunk.to_parquet(os.path.join('data','chunks',f'yfinance_selected_chunk_{i:02d}.parquet'))
    
        print('Done',dt.datetime.today()-tic)
        time.sleep(2)

    return

def combine_yfinance_chunks():
    
    # Combine chunked data
    data = []
    folder = os.path.join('data','chunks')
    for file_name in os.listdir(folder):
        data.append(pd.read_parquet(os.path.join(folder,file_name)))
    data = pd.concat(data,axis=1)
    data = data[sorted(data.columns.tolist())]
    data.to_parquet(os.path.join('data','yfinance_selected.parquet'))
    print('Saved Combined yfinance data')

    return

def build_characteristics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:

    close = df["Close"]
    volume = df["Volume"]
    returns = close.pct_change(fill_method=None)
    mkt_ret = returns.mean(axis=1)

    chars = {}

    # --- Past returns ---
    chars["r2_1"]    = (close.shift(2) / close.shift(21)) - 1
    chars["r12_2"]   = (close.shift(21) / close.shift(252)) - 1
    chars["r12_7"]   = (close.shift(126) / close.shift(252)) - 1
    chars["r36_13"]  = (close.shift(252) / close.shift(756)) - 1
    chars["ST_Rev"]  = (close.shift(1) / close.shift(21)) - 1
    chars["Ret_D1"] = returns.shift(1)
    chars["Ret_W1"] = ((close.shift(1) / close.shift(6)) - 1)
    chars["STD_W1"] = returns.rolling(5).std().shift(1)

    # --- Trading frictions ---
    chars["Rel2High"]   = (close / close.rolling(252).max()).shift(1)
    chars["Variance"]   = returns.rolling(60).var().shift(1)
    cov = returns.rolling(252).cov(mkt_ret)
    chars["Beta"]       = cov.div(mkt_ret.rolling(252).var(), axis=0).shift(1)
    chars["LME_proxy"]  = np.log((close * volume).rolling(21).mean()).shift(1)
        
    return returns, chars

def chars_dict_to_tensor(chars: dict[str, pd.DataFrame],
                         returns: pd.DataFrame,
                         char_order: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, 
                                                                       np.ndarray, pd.DatetimeIndex, 
                                                                       list[str], list[str]]:
    
    if char_order is None:
        char_order = sorted(chars.keys())
    
    # Align everything to a common (date × ticker) grid using one char as reference
    ref = chars[char_order[0]]
    dates = ref.index
    tickers = list(ref.columns)
    
    # Sanity: all chars and returns must share index and columns
    for name in char_order:
        assert chars[name].index.equals(dates), f"{name} has mismatched dates"
        assert list(chars[name].columns) == tickers, f"{name} has mismatched tickers"
    assert returns.index.equals(dates), "returns has mismatched dates"
    assert list(returns.columns) == tickers, "returns has mismatched tickers"
    
    # Stack into (T, N, M)
    X = np.stack([chars[name].values for name in char_order], axis=-1)
    R = returns.values
    
    # Mask: True where everything is finite. Pandas + numpy handle inf via np.isfinite.
    char_finite = np.isfinite(X).all(axis=-1)  # (T, N)
    ret_finite  = np.isfinite(R)               # (T, N)
    mask = char_finite & ret_finite

    np.savez(
        os.path.join('data',"data_tensors.npz"),
        R=R,            # (T, N) returns
        mask=mask,      # (T, N) bool
        dates=dates.values.astype("datetime64[D]"),  # (T,) datetime64
        tickers=np.array(tickers),
    )
    
    return X, R, mask, dates, tickers, char_order
