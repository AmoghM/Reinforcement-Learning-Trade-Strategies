import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from scipy.stats import ttest_ind, levene
import math as m
from seaborn import heatmap
import data_process as d
from data_process import histerror, checkhist, read_stock, returns, weighted_average_and_normalize
import trainqlearner_util as tu
import time
import pandas_datareader.data as web # fetch stock data

#TODO: make this a single function call
ticker = 'JPM'

np.random.seed(1000)
start = '2017-01-01'
end = '2019-12-31'

start_date = dt.datetime(2007, 1, 1)
end_date = dt.datetime(2016, 12, 31)


q, train_actions_history, train_returns_since_entry, percent_b_states_values, close_sma_ratio_states_value, cash_states_values, shares_states_values = tu.trainqlearner(ticker, start_date, end_date, window = 5, gamma = 0.95, episodes = 500, sh = 30, alp = 0.1, epsi = 0.99)

q.columns = ['HOLD', 'BUY', 'SELL']
nq=q
nq.columns = ['HOLD', 'BUY', 'SELL']

action_list = ['BUY','HOLD','SELL']
nq = nq[action_list]

print(type(nq))



test_df = d.get_stock_data(ticker, start, end)
test_df = d.create_df(test_df, window = 5)
test_df = d.create_state_df(test_df, percent_b_states_values , close_sma_ratio_states_value)
temp = test_df.iloc[:-1, :-1]
test_df = np.array(test_df[['Adj Close', 'state']])





def hold(stock_table,money,inc,original_shares,commission):
    '''
    Enacts hold-every-day strategy

    Inputs
    stock_table: list of daily stock or portfolio values
    money: original cash held
    inc: increment of buy/sell permitted
    original_shares: original number of shares held

    Output
    results: dictionary holding...
        *one Pandas series each (key/Series names are identical) for*
        final_vals: final daily values of portfolio
        actions: daily actions taken ("BUY" "SELL" "HOLD")
        shares: daily number of shares of stock held
        cash: daily amount of cash held

        *additionally*
        qtable: returns None (does not apply to this strategy)
    '''

    # calculate daily returns
    ret = returns(stock_table)
    original_shares = int(money/stock_table.values[0])
    money = money - stock_table.values[0]*original_shares
    print('Hold Strategy')
    print("cash" + str(money))
    print('shares' + str(original_shares))
    print('adj close' + str(stock_table.values[0]))


    # dummy calculations to reset to initialize return calculations
    # what this does is just sets the first entry of the returns Series to total value of stock held originally
    # this way, day 1's return changes from 1 to the original stock value, and the cumprod propogates the daily values accordingly
    ret.iloc[0] = (stock_table.values[0]*original_shares)

    # create actions table
    actions = pd.Series(['HOLD']*len(stock_table),index=stock_table.index)

    # create shares table
    shares = pd.Series([original_shares]*len(stock_table),index=stock_table.index)

    # create cash table
    cash = pd.Series([money]*len(stock_table),index=stock_table.index)

    # calculate daily value of stock held
    final_vals = np.cumprod(ret)
    final_vals.columns = ['value']

    # add original cash to this
    final_vals += money

    # create markov transition matrix
    markov = pd.DataFrame(np.zeros((3,3)),index=action_list,columns=action_list)

    markov.loc['HOLD','HOLD']=1

    results = {'final_vals':final_vals,'actions':actions,'shares':shares,'cash':cash,'qtable':None, 'markov':markov,'state_history': None}
    return results


# function to randomly choose action every day
def random_action(stock_table,money,inc,original_shares,commission):
    '''
    Enacts random-daily-action strategy

    Inputs
    stock_table: list of daily stock or portfolio values
    money: original cash held
    inc: increment of buy/sell permitted
    original_shares: original number of shares held

    Output
    results: dictionary holding...
        *one Pandas series each (key/Series names are identical) for*
        final_vals: final daily values of portfolio
        actions: daily actions taken ("BUY" "SELL" "HOLD")
        shares: daily number of shares of stock held
        cash: daily amount of cash held

        *additionally*
        qtable: returns None (does not apply to this strategy)
    '''

    # record original value
    original_val = money + (stock_table.values[0]*original_shares) # initial cash

    # generate table of returns
    ret = returns(stock_table)

    # create actions table
    actions = ['HOLD']

    # create shares table
    shares = stock_table.copy()
    shares.iloc[0] = original_shares

    # create markov transition matrix
    markov = pd.DataFrame(np.zeros((3,3)),index=action_list,columns=action_list)

    # create cash table
    cash = stock_table.copy()
    cash.iloc[0] = money

    # calculate daily portfolio value
    final_vals = stock_table.copy()
    final_vals.iloc[0] = original_val

    # iterate through days
    for i in range(1,stock_table.shape[0]):
        j = i-1 # last day
        cur_cash = cash.values[j] # current cash
        cur_shares = shares.values[j] # current shares
        final_vals.iloc[i] = cur_cash + (cur_shares*stock_table.values[i]) # end of day portfolio value
        cur_price = stock_table.values[j]

        # if you can't buy or sell, hold
        if cur_shares < inc and cur_cash < (cur_price*inc):
            act = 'HOLD'

        # if you can't sell, but you can buy, buy or hold
        elif cur_shares < inc:
            act = np.random.choice(['BUY','HOLD'])

        # if you can't buy, but you can sell, sell or hold
        elif cur_cash < (cur_price*inc):
            act = np.random.choice(['SELL','HOLD'])

        # otherwise do whatever you want
        else:
            act = np.random.choice(['BUY','SELL','HOLD'])

        # take action
        if act == 'HOLD':
            cash.iloc[i] = cash.values[j]
            shares.iloc[i] = shares.values[j]
        if act == 'BUY':
            cash.iloc[i] = cash.values[j] - (inc*cur_price) - commission
            shares.iloc[i] = shares.values[j] + inc
        if act == 'SELL':
            cash.iloc[i] = cash.values[j] + (inc*cur_price) - commission
            shares.iloc[i] = shares.values[j] - inc

        actions += [act]

        # increment markov
        markov.loc[actions[j],actions[i]] +=1

    actions = pd.Series(actions,index=stock_table.index)

    # normalize markov
    markov = markov.divide(markov.sum(axis=1),axis=0).round(2)

    results = {'final_vals':final_vals,'actions':actions,'shares':shares,'cash':cash,'qtable':None, 'markov':markov,'state_history':None}
    return results


#function to choose action based on yesterday's return
def rule_based(stock_table,money,inc, original_shares,commission):
    '''
    Enacts rule-based (buy/sell/hold based on prior day's return) strategy

    Inputs
    stock_table: list of daily stock or portfolio values
    money: original cash held
    inc: increment of buy/sell permitted
    original_shares: original number of shares held

    Output
    results: dictionary holding...
        *one Pandas series each (key/Series names are identical) for*
        final_vals: final daily values of portfolio
        actions: daily actions taken ("BUY" "SELL" "HOLD")
        shares: daily number of shares of stock held
        cash: daily amount of cash held

        *additionally*
        qtable: returns None (does not apply to this strategy)
    '''

    # record original value
    original_val = money + (stock_table.values[0]*original_shares) # initial cash

    # generate table of returns
    ret = returns(stock_table)

    # create actions table
    actions = ['HOLD']

    # create shares table
    shares = stock_table.copy()
    shares.iloc[0] = original_shares

    # create cash table
    cash = stock_table.copy()
    cash.iloc[0] = money

    # create markov transition matrix
    markov = pd.DataFrame(np.zeros((3,3)),index=action_list,columns=action_list)

    # calculate daily portfolio value
    final_vals = stock_table.copy()
    final_vals.iloc[0] = original_val

    # iterate through days
    for i in range(1,stock_table.shape[0]):
        j = i-1 # last day
        cur_cash = cash.values[j] # current cash
        cur_shares = shares.values[j] # current shares
        final_vals.iloc[i] = cur_cash + (cur_shares*stock_table.values[i]) # end of day portfolio value
        cur_price = stock_table.values[j]

        # calculate last return
        last_ret = ret.values[j]

        # if you can't buy or sell, hold
        if cur_shares < inc and cur_cash < (cur_price*inc):
            act = 'HOLD'

        # if you can't sell, but you can buy... buy if it makes sense, or hold if it doesn't
        elif cur_shares < inc:
            act = 'BUY' if last_ret > 1 else 'HOLD'

        # if you can't buy, but you can sell... sell if it makes sense, or hold if it doesn't
        elif cur_cash < (cur_price*inc):
            act = 'SELL' if last_ret < 1 else 'HOLD'

        # otherwise do whatever makes sense
        else:
            if last_ret > 1:
                act = 'BUY'
            elif last_ret < 1:
                act = 'SELL'
            else:
                act = 'HOLD'

        # take action
        if act == 'HOLD':
            cash.iloc[i] = cash.values[j]
            shares.iloc[i] = shares.values[j]
        if act == 'BUY':
            cash.iloc[i] = cash.values[j] - (inc*cur_price)  - commission
            shares.iloc[i] = shares.values[j] + inc
        if act == 'SELL':
            cash.iloc[i] = cash.values[j] + (inc*cur_price)  - commission
            shares.iloc[i] = shares.values[j] - inc

        actions += [act]

        # increment markov
        markov.loc[actions[j],actions[i]] +=1

    actions = pd.Series(actions,index=stock_table.index)

    # normalize markov
    markov = markov.divide(markov.sum(axis=1),axis=0).round(2)

    results = {'final_vals':final_vals,'actions':actions,'shares':shares,'cash':cash,'qtable':None, 'markov':markov, 'state_history': None}
    return results


# function to buy stock every day
def buy_always(stock_table,money,inc,original_shares,commission):
    '''
    enacts buy_always strategy

    Inputs
    stock_table: list of daily stock or portfolio values
    money: original cash held
    inc: increment of buy/sell permitted
    original_shares: original number of shares held

    Output
    results: dictionary holding...
        *one Pandas series each (key/Series names are identical) for*
        final_vals: final daily values of portfolio
        actions: daily actions taken ("BUY" "SELL" "HOLD")
        shares: daily number of shares of stock held
        cash: daily amount of cash held

        *additionally*
        qtable: returns None (does not apply to this strategy)
    '''

    # record original value
    original_val = money + (stock_table.values[0]*original_shares) # initial cash

    # generate table of returns
    ret = returns(stock_table)

    # create actions table
    actions = ['HOLD']

    # create shares table
    shares = stock_table.copy()
    shares.iloc[0] = original_shares

    # create markov transition matrix
    markov = pd.DataFrame(np.zeros((3,3)),index=action_list,columns=action_list)

    # create cash table
    cash = stock_table.copy()
    cash.iloc[0] = money

    # calculate daily portfolio value
    final_vals = stock_table.copy()
    final_vals.iloc[0] = original_val

    # iterate through days
    for i in range(1,stock_table.shape[0]):
        j = i-1 # last day
        cur_cash = cash.values[j] # current cash
        cur_shares = shares.values[j] # current shares
        final_vals.iloc[i] = cur_cash + (cur_shares*stock_table.values[i]) # end of day portfolio value
        cur_price = stock_table.values[j]

        # if you can't buy, hold
        if cur_cash < (cur_price*inc):
            act = 'HOLD'
        else:
            act = 'BUY'

        # take action
        if act == 'HOLD':
            cash.iloc[i] = cash.values[j]
            shares.iloc[i] = shares.values[j]
        if act == 'BUY':
            cash.iloc[i] = cash.values[j] - (inc*cur_price) - commission
            shares.iloc[i] = shares.values[j] + inc

        actions += [act]

        # increment markov
        markov.loc[actions[j],actions[i]] +=1

    actions = pd.Series(actions,index=stock_table.index)

    # normalize markov
    markov = markov.divide(markov.sum(axis=1),axis=0).round(2)

    results = {'final_vals':final_vals,'actions':actions,'shares':shares,'cash':cash,'qtable':None, 'markov':markov,'state_history':None}
    return results

# function to choose action based on OLS of returns looking back to trading days t-6 to t-1
def ols(stock_table,money,inc, original_shares,commission):
    '''
    Enacts OLS strategy based on last five days

    Inputs
    stock_table: list of daily stock or portfolio values
    money: original cash held
    inc: increment of buy/sell permitted
    original_shares: original number of shares held

    Output
    results: dictionary holding...
        *one Pandas series each (key/Series names are identical) for*
        final_vals: final daily values of portfolio
        actions: daily actions taken ("BUY" "SELL" "HOLD")
        shares: daily number of shares of stock held
        cash: daily amount of cash held

        *additionally*
        qtable: returns None (does not apply to this strategy)
    '''

    # set lookback window to one week
    lookback = 5

    # record original value
    original_val = money + (stock_table.values[0]*original_shares) # initial cash

    # generate table of returns
    ret = returns(stock_table)

    # create actions table
    actions = ['HOLD']

    # create shares table
    shares = stock_table.copy()
    shares.iloc[0] = original_shares

    # create cash table
    cash = stock_table.copy()
    cash.iloc[0] = money

    # calculate daily portfolio value
    final_vals = stock_table.copy()
    final_vals.iloc[0] = original_val

    # create markov transition matrix
    markov = pd.DataFrame(np.zeros((3,3)),index=action_list,columns=action_list)

    # iterate through days
    for i in range(1,stock_table.shape[0]):

        j = i-1 # last day
        cur_cash = cash.values[j] # current cash
        cur_shares = shares.values[j] # current shares
        final_vals.iloc[i] = cur_cash + (cur_shares*stock_table.values[i]) # end of day portfolio value
        cur_price = stock_table.values[j]


        # Perform OLS if past day 3 to define expected return
        if i>2:
            st = ret.values[max(0,j-lookback):j+1] # y
            x = np.arange(len(st)) # x
            exp_ret = sm.OLS(st,add_constant(x)).fit().predict([1,len(st)])
        else:
            exp_ret = 1


        # if you can't buy or sell, hold
        if cur_shares < inc and cur_cash < (cur_price*inc):
            act = 'HOLD'

        # if you can't sell, but you can buy... buy if it makes sense, or hold if it doesn't
        elif cur_shares < inc:
            act = 'BUY' if exp_ret > 1 else 'HOLD'

        # if you can't buy, but you can sell... sell if it makes sense, or hold if it doesn't
        elif cur_cash < (cur_price*inc):
            act = 'SELL' if exp_ret < 1 else 'HOLD'

        # otherwise do whatever makes sense
        else:
            if exp_ret > 1:
                act = 'BUY'
            elif exp_ret < 1:
                act = 'SELL'
            else:
                act = 'HOLD'

        # take action
        if act == 'HOLD':
            cash.iloc[i] = cash.values[j]
            shares.iloc[i] = shares.values[j]
        if act == 'BUY':
            cash.iloc[i] = cash.values[j] - (inc*cur_price)  - commission
            shares.iloc[i] = shares.values[j] + inc
        if act == 'SELL':
            cash.iloc[i] = cash.values[j] + (inc*cur_price)  - commission
            shares.iloc[i] = shares.values[j] - inc

        actions += [act]

        # increment markov
        markov.loc[actions[j],actions[i]] +=1

    actions = pd.Series(actions,index=stock_table.index)

    # normalize markov
    markov = markov.divide(markov.sum(axis=1),axis=0).round(2)

    results = {'final_vals':final_vals,'actions':actions,'shares':shares,'cash':cash,'qtable':None, 'markov':markov,'state_history': None}
    return results

# def qlearner(stock_table,money,inc, original_shares,qtable=ql[0], BB_quantiles=ql[1], SMA_quantiles=ql[2],window=window):

def qlearner(stock_table,money,inc, original_shares, commission, q_table = nq, test_data = test_df,  percent_b_states_values = percent_b_states_values, close_sma_ratio_states_value = close_sma_ratio_states_value, cash_states_values = cash_states_values, shares_states_values = shares_states_values, temp = temp):
    '''
    Evaluate the Q-table
    Inputs:
    test_data(dataframe)
    q(dataframe): trained Q-table
    Output:
    actions_history(dict): has everydays' actions and close price
    returns_since_entry(list): contains every day's return since entry
    '''
    current_portfolio_value = []
    cash = money
    num_shares = original_shares
    curr_cash = []
    curr_shares = []
    curr_cash_s = []
    curr_shares_s = []
    act_list = []
    cash_list = []
    shares_list = []
    final_states = []
    state_history = {}
    actions_history =[]
    q_buy = []
    q_hold = []
    q_sell = []
    for i, val in enumerate(test_data):
        current_adj_close, state = val
        try:
            next_adj_close, next_state = test_data[i + 1]
        except:
            print('End of data! Done!')
            break

        current_cash_state = d.value_to_state(cash, cash_states_values)
        current_share_state = d.value_to_state(num_shares, shares_states_values)
        state = state + current_cash_state + current_share_state

        final_states.append(state)
        curr_cash.append(cash)
        curr_shares.append(num_shares)
        curr_cash_s.append(current_cash_state)
        curr_shares_s.append(current_share_state)

        try:
            state_history[state] += 1
        except KeyError:
            state_history[state] = 1

        q_buy.append(q_table.loc[state].values[0])
        q_hold.append(q_table.loc[state].values[1])
        q_sell.append(q_table.loc[state].values[2])

        action = tu.act(state, q_table, threshold=0, actions_size=3)


        if action == 0:  # buy
            if cash > inc * current_adj_close:
                next_cash = cash - inc * current_adj_close
                num_shares += inc
                cash = next_cash
            else:
                action = 1

        if action == 2:  # sell
            if num_shares > 0:
                next_cash = cash + inc * current_adj_close
                num_shares -= inc
                cash = next_cash
            else:
                action = 1

        if action == 0:
            act_list.append('BUY')
        elif action == 2:
            act_list.append('SELL')
        else:
            act_list.append('HOLD')

        actions_history.append((i, current_adj_close, action))

        cash_list.append(cash)
        shares_list.append(num_shares)
        current_portfolio_value.append(cash + num_shares * next_adj_close)



    markov = pd.DataFrame(np.zeros((3, 3)), index=action_list, columns=action_list)
    for i in range(1,len(act_list)):
        markov.loc[act_list[i-1],act_list[i]] +=1

    temp['cash'] = curr_cash
    temp['cash_state'] = curr_cash_s
    temp['shares'] = curr_shares
    temp['shares_state'] = curr_shares_s
    temp['state'] = final_states
    temp['q_buy'] = q_buy
    temp['q_hold'] = q_hold
    temp['q_sell'] = q_sell
    temp.to_csv('./data/viz_data.csv')


    actions = pd.Series(act_list, index=stock_table.index)
    f_shares = pd.Series(shares_list, index=stock_table.index)
    f_cash = pd.Series(cash_list, index=stock_table.index)
    final_vals = pd.Series(current_portfolio_value, index=stock_table.index)

    results = {'final_vals': final_vals, 'actions': actions, 'shares': f_shares, 'cash': f_cash, 'qtable': q_table,
               'state_history': pd.Series(state_history), 'BB_quantiles': list(percent_b_states_values.values()),
               'SMA_quantiles': list(close_sma_ratio_states_value.values()),
               'CASH_quantiles': list(cash_states_values.values()), 'SHARE_quantiles': list(shares_states_values.values()),
               'markov': markov, 'actions_history' : actions_history}
    return results


# function to return stats and graphs
def return_stats(stock='jpm',
                 commission = 2,
                 money=100000,
                 #inc=10,- can read this argument and change code below if doing absolute share-based
                 #original_shares=100, - can read this argument and change code below if doing absolute share-based
                 policies=[hold,random_action,rule_based,ols,buy_always,qlearner]):

    '''
    Enacts every strategy and provides summary statistics and graphs

    Inputs
    stock:
    money: original cash held
    inc: increment of buy/sell permitted
    original_shares: original number of shares held

    Output
    None

    Provides numerous summary statistics and visualizations
    '''

    original_money = money

    # generate stock table
    stock_table = read_stock(stock,start,end)


    # note stock name
    stock_name = stock.upper()

    # approximate 50/50 split in money-stock
    original_shares = 0

    # recalculate money accordingly

    money = original_money

    # make share increment about 1% of original share holdings
    inc = 30

    stock_table = stock_table[4:]





    # generate results
    results = {policy.__name__:policy(stock_table,
                                      money = money,
                                      inc = inc,
                                      original_shares = original_shares,
                                     commission = commission) for policy in policies}

    actions_history = results['qlearner']['actions_history']

    days, prices, actions = [], [], []
    for d, p, a in actions_history:
        days.append(d)
        prices.append(p)
        actions.append(a)
    hold_d, hold_p, buy_d, buy_p, sell_d, sell_p = [], [], [], [], [], []
    for d, p, a in actions_history:
        if a == 0:
            hold_d.append(d)
            hold_p.append(p)
        if a == 1:
            buy_d.append(d)
            buy_p.append(p)
        if a == 2:
            sell_d.append(d)
            sell_p.append(p)

    buys = pd.DataFrame(list(zip(buy_d, buy_p)), columns =['Date', 'Adj Close'])
    sells = pd.DataFrame(list(zip(sell_d, sell_p)), columns =['Date', 'Adj Close'])
    holds = pd.DataFrame(list(zip(hold_d, hold_p)), columns=['Date', 'Adj Close'])

    buys.to_csv('./data/buy_data.csv')
    sells.to_csv('./data/sell_data.csv')
    holds.to_csv('./data/hold_data.csv')








    # plot qtables only for qlearner (or any other strategies with Q table)
    for policy in policies:
        if results[policy.__name__]['qtable'] is not None: #don't try to plot Q tables for benchmark strategies

            # get state history and quantile length and qtable for normalization and averaging function
            state_history = results[policy.__name__]['state_history']
            quantile_length = len(results[policy.__name__]['BB_quantiles'])
            qtab = results[policy.__name__]['qtable']

            qtab_bb = weighted_average_and_normalize(qtab, state_history, 1, quantile_length)
            qtab_bb = qtab_bb.iloc[::-1] # reverse order of rows for visualization purposes - now biggest value will be on top
            qtab_bb.index = np.round(np.flip(np.array(results[policy.__name__]['BB_quantiles'])),5) # define index as bb quantiles, reversing quantile order in kind so biggest value is first


            # plot BB heatmap
            plt.figure(figsize=(9,7))
            fig = heatmap(qtab_bb,cmap='Blues')
            plt.title('Bollinger Band % Q-Table',size=16)
            plt.gca().hlines([i+1 for i in range(len(qtab_bb.index))],xmin=0,xmax=10,linewidth=10,color='white')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=14,rotation=0)
            plt.gca().tick_params(axis='x',bottom=False,left=False)
            plt.gca().tick_params(axis='y',bottom=False,left=False)
            plt.show(fig)

            # marginalize over SMA
            # TODO - determine if this mean was taken correctly
            quantile_length = len(results[policy.__name__]['SMA_quantiles'])
            qtab_sma = weighted_average_and_normalize(qtab, state_history, 0, quantile_length)
            qtab_sma = qtab_sma.iloc[::-1]
            qtab_sma.index = np.round(np.flip(np.array(results[policy.__name__]['SMA_quantiles'])),10)

            plt.figure(figsize=(9,7))
            fig = heatmap(qtab_sma,cmap='Blues')
            plt.title('SMA Percentage Q-Table',size=16)
            plt.gca().hlines([i+1 for i in range(len(qtab_sma.index))],xmin=0,xmax=10,linewidth=10,color='white')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=14,rotation=0)
            plt.gca().tick_params(axis='x',bottom=False,left=False)
            plt.gca().tick_params(axis='y',bottom=False,left=False)
            plt.show(fig)

            #CASH
            quantile_length = len(results[policy.__name__]['CASH_quantiles'])
            qtab_sma = weighted_average_and_normalize(qtab, state_history, 2, quantile_length)
            qtab_sma = qtab_sma.iloc[::-1]
            qtab_sma.index = np.round(np.flip(np.array(results[policy.__name__]['CASH_quantiles'])), 10)

            plt.figure(figsize=(9, 7))
            fig = heatmap(qtab_sma, cmap='Blues')
            plt.title('CASH Q-Table', size=16)
            plt.gca().hlines([i + 1 for i in range(len(qtab_sma.index))], xmin=0, xmax=10, linewidth=10, color='white')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=14, rotation=0)
            plt.gca().tick_params(axis='x', bottom=False, left=False)
            plt.gca().tick_params(axis='y', bottom=False, left=False)
            plt.show(fig)

            #SHARES
            quantile_length = len(results[policy.__name__]['SHARE_quantiles'])
            qtab_sma = weighted_average_and_normalize(qtab, state_history, 3, quantile_length)
            qtab_sma = qtab_sma.iloc[::-1]
            qtab_sma.index = np.round(np.flip(np.array(results[policy.__name__]['SHARE_quantiles'])), 10)

            plt.figure(figsize=(9, 7))
            fig = heatmap(qtab_sma, cmap='Blues')
            plt.title('SHARE Q-Table', size=16)
            plt.gca().hlines([i + 1 for i in range(len(qtab_sma.index))], xmin=0, xmax=10, linewidth=10, color='white')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=14, rotation=0)
            plt.gca().tick_params(axis='x', bottom=False, left=False)
            plt.gca().tick_params(axis='y', bottom=False, left=False)
            plt.show(fig)



    # get markov transition models
    for policy in policies:
        plt.figure(figsize=(6,3))
        plt.title('Transition Matrix For '+policy.__name__,size=16)
        mkv = results[policy.__name__]['markov']
        fig = heatmap(mkv,annot=True,annot_kws={'size':14},cmap='Greens',cbar=False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14,rotation=0)
        plt.gca().set(xlabel='Current Trading Day', ylabel='Last Trading Day')
        plt.gca().tick_params(axis='x',bottom=False,left=False)
        plt.gca().tick_params(axis='y',bottom=False,left=False)
        plt.gca().hlines([1,2],xmin=0,xmax=10,linewidth=10,color='white')
        plt.show(fig)


    # plot daily portfolio values
    plt.figure(figsize=(14,8))
    for policy in policies:
        plt.plot(results[policy.__name__]['final_vals'],label = policy.__name__)
    plt.legend()
    plt.xlabel("Date",fontsize=20)
    plt.ylabel("Portfolio Value ($)",fontsize=20)
    plt.title("Daily Portfolio Values For Different Trading Strategies: "+stock.upper(),fontsize=25)
    plt.show()

    # plot daily cash values
    plt.figure(figsize=(14,8))
    for policy in policies:
        plt.plot(results[policy.__name__]['cash'],label = policy.__name__)
    plt.legend()
    plt.xlabel("Date",fontsize=20)
    plt.ylabel("Cash Held ($)",fontsize=20)
    plt.title("Daily Cash Held For Different Trading Strategies: "+stock.upper(),fontsize=25)
    plt.show()

    # plot daily shares
    plt.figure(figsize=(14,8))
    for policy in policies:
        plt.plot(results[policy.__name__]['shares'],label = policy.__name__)
    plt.legend()
    plt.xlabel("Date",fontsize=20)
    plt.ylabel("Shares Held",fontsize=20)
    plt.title("Daily Share Holdings For Different Trading Strategies: "+stock_name,fontsize=25)
    plt.show()

    # plot daily portfolio values
    for i, policy in enumerate(policies):
        dic = results[policy.__name__]
        if dic['state_history'] is not None:
            print("States History for " + policy.__name__ + "is: ", dic['state_history'])

        del dic['state_history']
        del dic['qtable']
        del dic['markov']
        try:
            del dic['BB_quantiles']
            del dic['SMA_quantiles']
            del dic['CASH_quantiles']
            del dic['SHARE_quantiles']
        except:
            pass
        df = pd.DataFrame(dic)

        plt.figure(figsize=(14,8))
        plt.plot([], label="BUY", color="orange", marker='o')
        plt.plot([], label="SELL", color="black",marker='o')
        plt.plot([], label="HOLD", color="red",marker='o')
        buy_df = df[df.actions == 'BUY']
        sell_df = df[df.actions == 'SELL']
        hold_df = df[df.actions == 'HOLD']
        plt.plot(results[policy.__name__]['final_vals'],label =policy.__name__)
        plt.scatter(buy_df.index, buy_df['final_vals'], color='orange',marker='^',s=10)
        plt.scatter(sell_df.index, sell_df['final_vals'], color='black',marker='v',s=10)
        plt.scatter(hold_df.index, hold_df['final_vals'], color='red',marker='s',s=10)
        plt.xlabel("Date",fontsize=20)
        plt.ylabel("Portfolio Value ($)",fontsize=20)
        plt.title("Daily Portfolio Values For Trading Strategies of "+ policy.__name__ +" for stock : "+stock.upper(),fontsize=25)
        plt.legend()
        plt.show()

        # plt.figure(figsize=(14, 8))
        # plt.plot([], label="BUY", color="orange", marker='o')
        # plt.plot([], label="SELL", color="black", marker='o')
        # plt.plot([], label="HOLD", color="red", marker='o')
        # buy_df = df[df.actions == 'BUY']
        # sell_df = df[df.actions == 'SELL']
        # hold_df = df[df.actions == 'HOLD']
        # plt.plot(adj_list, label='adj_close')
        # plt.scatter(buy_df.index, buy_df['final_vals'], color='orange', marker='^', s=10)
        # plt.scatter(sell_df.index, sell_df['final_vals'], color='black', marker='v', s=10)
        # plt.scatter(hold_df.index, hold_df['final_vals'], color='red', marker='s', s=10)
        # plt.xlabel("Date", fontsize=20)
        # plt.ylabel("Portfolio Value ($)", fontsize=20)
        # plt.title("Daily Actions on Adj Close For Trading Strategies of " + policy.__name__ + "for stock : " + stock.upper(),
        #     fontsize=25)
        # plt.legend()
        # plt.show()

    # display percentages
    #TODO: display(res) has no display() function. Fix bug.
    for policy in policies:
        print('For '+stock_name+',',policy.__name__,'action proportions were:')
        res = results[policy.__name__]['actions'].value_counts()
        res = res / res.sum()
        print(res)
        print('\n')
        print('For '+stock_name+',',policy.__name__,'average return based on action was:')
        res = returns(results[policy.__name__]['final_vals']).groupby(results[policy.__name__]['actions']).mean()
        print(res)
        print('\n')

    # calculate final returns
    for policy in policies:
        print('Final porfolio value under',policy.__name__,'strategy for '+stock_name+':',round(results[policy.__name__]['final_vals'].values[-1],0))
    print('\n')

    # calculate final percentage of money invested in stock
    for policy in policies:
        print('Final percentage of money invested in stock under',policy.__name__,              'strategy for '+stock_name+':',str(round(100*(1-(results[policy.__name__]['cash'].values[-1]/results[policy.__name__]['final_vals'].values[-1])),1))+'%')
    print('\n')

    # calculate returns
    rets = {policy:returns(results[policy.__name__]['final_vals']) for policy in policies}

    # generate risk_free return for sharpe ratio - five-year treasury yield
    rfs = returns(read_stock('^FVX')[4:])

    # find common indecies between stock tables and treasury yields
    rfn = set(stock_table.index).intersection(set(rfs.index))

    # now reindex
    rfr = rfs.loc[rfn]
    rfi = rfr.index

    # generate baseline return for information ratio - s&p 500
    bls = returns(read_stock('^GSPC')[4:]).values

    # print summary stats for daily returns
    for policy in policies:
        nm = policy.__name__

        # mean daily return
        print('Mean daily return under',nm,'for',stock_name+':',str(round(np.mean(rets[policy],axis=0),5)))

        # standard deviation of daily return
        print('Standard deviation of daily return under',nm,'for',stock_name+':',round(np.std(rets[policy],axis=0),3))

        # information ratio of daily return
        checkhist(rets[policy].values,bls)
        pr = np.mean(rets[policy].values)
        br = np.mean(bls)
        te = np.std(rets[policy].values-bls)
        ir = round((pr-br)/(te)*np.sqrt(len(bls)),2)
        print('Information Ratio against S&P 500 under',nm,'strategy for',stock_name+':',ir)

        # sharpe ratio of daily return
        dat = rets[policy].loc[rfi].values # need to correct dates to line up with risk free return
        checkhist(dat,rfr)
        rp = np.mean(dat)
        br = np.mean(rfr)
        sd = np.std(rfr-dat)
        sr = round((rp-br)/(sd)*np.sqrt(len(rfr)),2)
        print('Sharpe Ratio against five-year treasury yield under',nm,'strategy for',stock_name+':',sr)
        print('Note: only used dates when five-year treasury yields were available in calculating RFR for Sharpe Ratio')
        print('\n')

    for policy1 in policies:
        p1 = rets[policy1].loc[rfi].values # filter to dates with five-year treasury yields available
        n1 = policy1.__name__

        # independent samples t-test vs. risk-free return
        checkhist(p1,rfr)
        t = ttest_ind(p1,rfr,equal_var=True)
        gr = t[0] > 0
        n2 = 'rfr'
        p = round(t[1],3)/2 # make one-sided
        if gr:
            print('T-test for difference of mean returns in',n1,'and',n2,'finds',n1,'>',n2,'with p-value',round(p,3))
        else:
            print('T-test for difference of mean returns in',n2,'and',n1,'finds',n2,'>',n1,'with p-value',round(p,3))

        # levene test vs. risk-free return
        l = levene(rets[policy1].values,bls)
        p = round(l[1],3)
        gr = np.std(rets[policy1].values) > np.std(bls)
        n2 = 'bls'
        if gr:
            print('Levene test for difference of variances (volatility) in',n1,'and',n2,'finds p-value of',round(p,3),'with',n1,'showing more volatility')
        else:
            print('Levene test for difference of variances (volatility) in',n1,'and',n2,'finds p-value of',round(p,3),'with',n2,'showing more volatility')
        print('\n')

        for policy2 in policies:
            if policy1 != policy2: #and hash(policy1) <= hash(policy2) - not necessary
                p1 = rets[policy1].values # no longer need to filter to dates with five-year treasury yields available
                p2 = rets[policy2].values
                checkhist(p1,p2)
                n2 = policy2.__name__

                # independent samples t-test
                t = ttest_ind(p1,p2,equal_var=True)
                gr = t[0] > 0
                p = round(t[1],3)/2 # make one-sided
                if gr:
                    print('T-test for difference of mean returns in',n1,'and',n2,'finds',n1,'>',n2,'with p-value',round(p,3))
                else:
                    print('T-test for difference of mean returns in',n2,'and',n1,'finds',n2,'>',n1,'with p-value',round(p,3))

                # levene test
                l = levene(p1,p2)
                p = round(l[1],5)
                gr = np.std(p1) > np.std(p2)
                if gr:
                    print('Levene test for difference of variances (volatility) in',n1,'and',n2,'finds p-value of',round(p,3),'with',n1,'showing more volatility')
                else:
                    print('Levene test for difference of variances (volatility) in',n1,'and',n2,'finds p-value of',round(p,3),'with',n2,'showing more volatility')
                print('\n')
            print('\n')

    # TODO: add any additional desired visualizations
    plt.show()

if __name__ == '__main__':

    stocks = ticker
    for stock in [stocks]:
        return_stats(stock=stock)
