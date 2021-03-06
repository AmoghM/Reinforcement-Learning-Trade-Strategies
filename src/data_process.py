import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web  # fetch stock data


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


def create_df(df, window=3):
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


def create_state_df(df, percent_b_states_values, close_sma_ratio_states_value):
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
    df['percent_b_state'] = df['percent_b'].apply(
        lambda x: value_to_state(x, percent_b_states_values))
    df['norm_close_sma_ratio_state'] = df['norm_close_sma_ratio'].apply(
        lambda x: value_to_state(x, close_sma_ratio_states_value))

    df['state'] = df['norm_close_sma_ratio_state'] + df['percent_b_state']
    df.dropna(inplace=True)
    return df


def get_all_states(percent_b_states_values, close_sma_ratio_states_value):
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
            state = str(c) + str(b)
            states.append(str(state))

    return states
