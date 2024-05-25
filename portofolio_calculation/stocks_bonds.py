import pandas as pd
import yfinance as yf

def get_adjusted_close_prices(stock_symbols, start_date, end_date):
    all_data = pd.DataFrame()
    for symbol in stock_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if 'Adj Close' in stock_data.columns:
            all_data[symbol] = stock_data['Adj Close']
        else:
            print(f"Adjusted Close data not available for {symbol}")

    s_and_p = yf.download('^GSPC', start=start_date, end=end_date)
    all_data['S&P500'] = s_and_p['Adj Close']
    all_data_diff = all_data.diff()

    irx_data = yf.download('^IRX', start=start_date, end=end_date)
    if 'Adj Close' in irx_data.columns:
        all_data_diff['^IRX'] = irx_data['Adj Close']
    else:
        print("Adjusted Close data not available for ^IRX")

    return all_data_diff