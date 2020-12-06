import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web  # fetch stock data
from yfinance import download

# define exception class for unequal stock histories
def histerror(a,b):
    '''
    Checks if stock histories a and b have the same length

    Inputs
    a: stock history (list-like)
    b: stock history (list-like)

    Output
    Exception if histories are not equal length
    '''

    class HistoryError(Exception):
        pass
    raise HistoryError('Stocks do not have equal history')

def checkhist(a,b):
    '''
    Raises exception if stock histories a and b have the same length

    Inputs
    a: stock history (list-like)
    b: stock history (list-like)

    Output
    Raises the xception if histories are not equal length
    '''

    if len(a) != len(b):
        histerror(a,b)


def read_stock(stock, start = '2017-01-01',end = '2019-12-31'):
    '''
    Reads stock table from Yahoo Finance

    Inputs
    stock: symbol of stock (str)

    Output
    Pandas Series of daily stock adj close
    '''
    
    tab = download(stock,start,end)['Adj Close']
    tab.name = stock.upper()
    '''
    PLEASE KEEP THIS HERE IN CASE BEN IS REMOTE
    if stock.lower() == '^fvx':
        tab = pd.read_csv('fvx_test.csv',index_col='Date')['Adj Close']
    elif stock.lower() == '^gspc':
        tab = pd.read_csv('gspc_test.csv',index_col='Date')['Adj Close']
    elif stock.lower() == 'jpm':
        tab = pd.read_csv('test.csv',index_col='Date')['Adj Close']
    tab.index = pd.to_datetime(tab.index)
    '''
    return tab

def returns(stock_table):
    '''
    Calculates daily returns from a series of daily stock or portfolio values

    Inputs
    stock_table: list of daily stock or portfolio values

    Output
    returns: daily multiplier return values (Series)
    '''

    returns = stock_table.copy()
    returns.iloc[1:] = returns.iloc[1:]/returns.values[:-1]
    returns.iloc[0] = 1
    return returns

def get_upper_lower_bands(values, window):

    upper = values.rolling(window=window).mean() + \
        values.rolling(window=window).std() * 2
    lower = values.rolling(window=window).mean() - \
        values.rolling(window=window).std() * 2

    upper = upper.apply(lambda x: round(x, 5))
    lower = lower.apply(lambda x: round(x, 5))

    return upper, lower


def get_stock_data(symbol, start, end):
    '''
    Get stock data in the given date range
    Inputs:
    symbol(str): stock symbol
    start(datetime): start date
    end(datetime): end date
    train_size(float): amount of data used for training
    Outputs:
    train_df, test_df OR df(if train_size=1)
    '''
    df = web.DataReader(symbol, 'yahoo', start, end) 
    '''
    PLEASE KEEP HERE IN CASE BEN REMOTE
    df = pd.read_csv('train.csv',index_col='Date')
    df.index=pd.to_datetime(df.index)
    '''
    return df


def get_bollinger_bands(values, window):
    '''
    Return upper and lower Bollinger Bands.
    INPUTS:
    values(pandas series)
    window(int): time period to consider
    OUTPUS:
    band_width(pandas series)
    '''
    #  rolling mean
    rm = values.rolling(window=window).mean()
    rstd = values.rolling(window=window).std()

    band_width = rm / rstd
    return band_width.apply(lambda x: round(x, 5))


def get_adj_close_sma_ratio(values, window):
    '''
    Return the ratio of adjusted closing value to the simple moving average.
    INPUTS:
    values(pandas series)
    window(int): time period to consider
    OUTPUS:
    ratio(series)
    '''
    rm = values.rolling(window=window).mean()
    ratio = values/rm
    return ratio.apply(lambda x: round(x, 5))


def discretize(values, num_states=4):
    '''
    Convert continuous values to integer state
    Inputs:
    values(pandas series)
    num_states(int): dividing the values in to n blocks
    Output:
    states_value(dict): a dictionary with state_value as key, and the real value as value
    '''
    states_value = dict()
    step_size = 1./num_states
    for i in range(num_states):
        if i == num_states - 1:
            states_value[i] = values.max()
        else:
            states_value[i] = values.quantile((i+1)*step_size)
    states_value[num_states] = float('inf')
    return states_value

def create_cash_and_holdings_quantiles():
    # CASH (State 3)
    cash_list = [*range(1,10)]
    cash_list = [int(180000/9)*each for each in cash_list]

    cash_states_values = {}
    for i in range(len(cash_list)):
        cash_states_values[i] = cash_list[i]
    cash_states_values[9] = float("inf")

    # HOLDINGS = Num Shares (State 4)
    shares_list = [*range(1,10)]
    shares_list = [int(252/9)*each for each in shares_list]

    shares_states_values = {}
    for i in range(len(shares_list)):
        shares_states_values[i] = shares_list[i]
    shares_states_values[9] = float("inf")

    return cash_states_values, shares_states_values

def value_to_state(value, states_value):
    '''
    Convert values to state
    Inputs:
    value(float)
    States_values(dict)
    Output:
    the converted state
    '''
    if np.isnan(value):
        return np.nan
    else:
        for state, v in states_value.items():
            if value <= v:
                return str(state)
        return 'value out of range'


def create_df(df, window=45):
    '''
    Create a dataframe with the normalized predictors
    norm_bb_width, norm_adj_close, norm_close_sma_ratio
    Input:
    df(dataframe)
    window(int): a window to compute rolling mean
    Ouput:
    df(dataframe): a new dataframe with normalized predictors
    '''

    # get the ratio of close price to simple moving average
    close_sma_ratio = get_adj_close_sma_ratio(df['Adj Close'], window)
    # get the upper and lower BB values
    upper, lower = get_upper_lower_bands(df['Adj Close'], window)
    baseline = read_stock('^GSPC','2007-01-01','2016-12-31')

    # create bb measure, close-sma-ratio columns
    df['close_sma_ratio'] = close_sma_ratio
    df['upper_bb'] = upper
    df['lower_bb'] = lower
    
    # drop missing values
    df.dropna(inplace=True)

    # Calculate the Bollinger Percentage
    df['percent_b'] = (df['Adj Close'] - df['lower_bb']) * \
        100 / (df['upper_bb'] - df['lower_bb'])

    # normalize close price
    df['norm_adj_close'] = df['Adj Close']/df.iloc[0, :]['Adj Close']
    df['norm_close_sma_ratio'] = df['close_sma_ratio'] / \
        df.iloc[0, :]['close_sma_ratio']

    return df


def get_states(df):
    '''
    Discretize continous values to intergers
    Input:
    df(dataframe)
    Output:
    the discretized dictionary of norm_bb_width,
    norm_adj_close, norm_close_sma_ratio columns
    '''
    # discretize values
    percent_b_states_values = {
        0: 0,
        1: 25,
        2: 75,
        3: 100,
        4: float('inf')
    }

    close_sma_ratio_states_value = discretize(df['norm_close_sma_ratio'])
    
    return percent_b_states_values, close_sma_ratio_states_value


def create_state_df(df, bb_states_value, close_sma_ratio_states_value):
    '''
    Add a new column to hold the state information to the dataframe
    Inputs:
    df(dataframe)
    price_states_value(dict)
    bb_states_value(dict)
    close_sma_ratio_states_value(dict)
    Output:
    df(dataframe)
    '''
    percent_b_states_values, close_sma_ratio_states_value = get_states(df)

    #df['norm_bb_width_state'] = df['norm_bb_width'].apply(lambda x : value_to_state(x, bb_states_value)) #2 
    df['norm_close_sma_ratio_state'] = df['norm_close_sma_ratio'].apply(lambda x : value_to_state(x, close_sma_ratio_states_value))
    df['percent_b_state'] = df['percent_b'].apply(lambda x : value_to_state(x, percent_b_states_values))
    #df['norm_adj_close_state'] = df['norm_adj_close'].apply(lambda x : value_to_state(x, price_states_value))
    
    #df['state'] = df['norm_close_sma_ratio_state'] + df['norm_bb_width_state']
    df['state'] = df['norm_close_sma_ratio_state'] + df['percent_b_state']
    df.dropna(inplace=True)
    return df

def get_all_states(percent_b_states_values, close_sma_ratio_states_value, cash_states_values, shares_states_values):
    '''
    Combine all the states from the discretized 
    norm_adj_close, norm_close_sma_ratio columns.
    Inputs:
    price_states_value(dict)
    bb_states_value(dict)
    close_sma_ratio_states_value(dict)
    Output:
    states(list): list of strings
    '''
    states = []
    for c, _ in close_sma_ratio_states_value.items():
        for b, _ in percent_b_states_values.items():
          for m, _ in cash_states_values.items():
            for s, _ in shares_states_values.items(): 
              state =  str(c) + str(b) + str(m) + str(s)
              states.append(str(state))
    
    return states

def weighted_average_and_normalize(qtable,state_history,state_num,quantile_length):
    '''
    takes a q table and does a weighted average group by given the input state_number (what digit number it is in the state)
    
    Inputs:
    qtable: the qtable (DataFrame)
    state_history: the state history (Series)
    state_num: the number digit that indicates the state
    quantile_length: the number of quantiles we built this out with
    '''
    qtab_2 = pd.merge(qtable,pd.Series(state_history,name='state_history'),'inner',left_index=True,right_index=True)
    
    sh = qtab_2['state_history']
    qtab_2 = qtab_2.drop(columns=['state_history']).multiply(qtab_2['state_history'],axis=0)
    
    qtab_2 = pd.merge(qtab_2,sh,'inner',left_index=True,right_index=True)
    
    qtab_2['state'] = qtab_2.index.str.slice(state_num,state_num+1)
    
    qtab_3 = qtab_2.groupby('state').sum()
    
    qtab_4 = qtab_3.divide(qtab_3['state_history'],axis=0).drop(columns='state_history')
    
    qtab_5 = qtab_4.reindex([str(i) for i in range(quantile_length)])
    
    #normalize by max
    qtab_6 = qtab_5.divide(qtab_5.max(axis=1),axis=0)
    
    return qtab_6
    
    
                                           


