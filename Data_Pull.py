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
    
def build_returns(df: pd.DataFrame):
	
	# This was more involved when the project started (was going to compare linear PCA with Neural Nets)
	# Was originally "build_characteristics" but decided to simplfy
	
    close = df["Close"]
    volume = df["Volume"]
    returns = close.pct_change(fill_method=None)
    mkt_ret = returns.mean(axis=1)

    tickers = returns.columns.values
    dates = returns.index.values
    returns = returns.values

    mask  = np.isfinite(returns)
    
    return returns, mask, tickers, dates
