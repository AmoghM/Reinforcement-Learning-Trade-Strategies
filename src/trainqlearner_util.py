import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_process as d
import pandas_datareader.data as web  # fetch stock data
import seaborn as sns


def initialize_q_mat(all_states, all_actions):
    '''
    Initialize Q-table
    Inputs:
    all_states: a list of all the states values
    all_actions: a dictionary of all possible actions to take
    Output:
    q_mat: randomly initialized Q-table
    '''
    states_size = len(all_states)
    actions_size = len(all_actions)

    q_mat = np.random.rand(states_size, actions_size)/1e9
    q_mat = pd.DataFrame(q_mat, columns=all_actions.keys())

    q_mat['states'] = all_states
    q_mat.set_index('states', inplace=True)

    return q_mat

def act(state, q_mat, threshold, actions_size=3):
    '''
    Taking an action based on different strategies:
    either random pick actions or take the actions
    with the highest future return
    Inputs:
    state(str)
    q_mat(dataframe): Q-table
    threshold(float): the percentage of picking a random action
    action_size(int): number of possible actions
    Output:
    action(int)
    '''
    if np.random.uniform(0, 1) < threshold:  # go random
        action = np.random.randint(low=0, high=actions_size)
    else:
        action = np.argmax(q_mat.loc[state].values)
    return action


def get_return_since_entry(bought_history, current_adj_close):
    '''
    Calculate the returns of current share holdings.
    Inputs:
    bought_history(list)
    current_adj_close(float)
    current_day(int)
    Output:
    return_since_entry(float)
    '''
    return_since_entry = 0.

    for b in bought_history:
        return_since_entry += (current_adj_close - b)
    return return_since_entry


# In[36]:


def visualize_results(actions_history, returns_since_entry):
    '''
    Visualize the trading results with 2 plots
    The upper plot shows the return since entry
    The lower plot shows the action signal
    Inputs:
    actions_history(dict): has everydays' actions and close price
    returns_since_entry(list): contains every day's return since entry
    Output:
    None
    '''
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    ax1.plot(returns_since_entry)

    days, prices, actions = [], [], []
    for d, p, a in actions_history:
        days.append(d)
        prices.append(p)
        actions.append(a)

    # ax2.figure(figsize=(20,10))
    ax2.plot(days, prices, label='normalized adj close price')
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
        # ax2.annotate(all_actions[a], xy=(d,p), xytext=(d-.2, p+0.001), color=color, arrowprops=dict(arrowstyle='->',connectionstyle='arc3'))
    ax2.scatter(hold_d, hold_p, color='blue', label='hold')
    ax2.scatter(buy_d, buy_p, color='green', label='buy')
    ax2.scatter(sell_d, sell_p, color='red', label='sell')
    ax2.legend()

def get_invested_capital(actions_history, returns_since_entry):
    '''
    Calculate the max capital being continously invested by the trader
    Input:
    actions_history(dict): has everydays' actions and close price
    returns_since_entry(list): contains every day's return since entry
    Output:
    return_invest_ratio(float)
    '''
    invest = []
    total = 0
    return_invest_ratio = None
    for i in range(len(actions_history)):
        a = actions_history[i][2]
        p = actions_history[i][1]

        try:
            next_a = actions_history[i+1][2]
        except:
            break
        if a == 1:
            total += p
            if next_a != 1 or (i == len(actions_history)-2 and next_a == 1):
                invest.append(total)
                total = 0
    if invest:
        return_invest_ratio = returns_since_entry[-1]/max(invest)
        print('invested capital {}, return/invest ratio {}'.format(max(invest),
                                                                   return_invest_ratio))
    else:
        print('no buy transactions, invalid training')
    return return_invest_ratio

def get_base_return(data):
    '''
    Calculate the benchmark returns of a given stock
    Input:
    data(dataframe): containing normalized close price and state
    Output:
    return/invest ratio(float)
    '''
    start_price, _ = data[0]
    end_price, _ = data[-1]
    return (end_price - start_price)/start_price

def train_q_learning(train_data, q, alpha, epsilon, epsilon_decay, gamma, episodes,commission,sell_penalty):
    episode = 0
    '''
    Train a Q-table
    Inputs:
    train_data(dataframe)
    q(dataframe): initial Q-table
    epsilon(float): threshold of which action strategy to take
    alpha(float): proportion to weight future expected return vs. current return
    gamma(float): discount percentage on the future return
    commission(float): amount charged for stock transaction
    Output:
    q(dataframe): Updated Q-table
    actions_history(dict): has everydays' actions and close price
    returns_since_entry(list): contains every day's return since entry
    '''
    # create framework for episode-to-episode Q table change tracking; will track MSE between episodes
    q_cur = q.copy()
    errs = []
    episode_decile = episodes//10
    
    for ii in range(episodes):
        episode +=1
        if episode == 1 or episode%episode_decile == 0 or episode == episodes:
            print('Training episode {}'.format(episode))
        actions_history = []
        num_shares = 0
        bought_history = []
        returns_since_entry = [0]
        days = [0]
        
        # add convergence tracking for episode 1
        if episode == 1:
            errs_1 = []
            q_cur_1 = q.copy()
        
        for i, val in enumerate(train_data):
            current_adj_close, state = val
            try:
                next_adj_close, next_state = train_data[i+1]
            except:
                break

            if len(bought_history) > 0:
                returns_since_entry.append(get_return_since_entry(
                    bought_history, current_adj_close))
            else:
                returns_since_entry.append(returns_since_entry[-1])

            # decide action
            '''
            if alpha > 0.1:
                alpha = alpha/(i+1)
            '''
            epsilon*=epsilon_decay
            action = act(state, q, threshold=epsilon, actions_size=3)

            # get reward
            if action == 0:  # hold
                if num_shares > 0:
                    prev_adj_close, _ = train_data[i-1]
                    future = next_adj_close - current_adj_close
                    past = current_adj_close - prev_adj_close
                    reward = past
                else:
                    reward = 0

            if action == 1:  # buy
                reward = 0-commission
                num_shares += 1
                bought_history.append((current_adj_close))

            if action == 2:  # sell
                if num_shares > 0:
                    bought_price = bought_history[0]
                    reward = (current_adj_close - bought_price) - commission
                    bought_history.pop(0)
                    num_shares -= 1

                else:
                    reward = 0 - sell_penalty
            actions_history.append((i, current_adj_close, action))

            # update q table
            q.loc[state, action] = (
                1.-alpha)*q.loc[state, action] + alpha*(reward+gamma*(q.loc[next_state].max()))
            
            q_last_1 = q_cur_1.copy()
            q_cur_1 = q.copy()
            
            # add convergence tracking for episode 1
            if episode == 1:
                MSE_1 = np.sum(np.square(q_cur_1-q_last_1).values)
                errs_1 += [MSE_1]
            
        # add convergence tracking for episode 1
        if episode == 1:
            plt.figure(figsize=(14,8))
            plt.title('Q Table Stabilization Within Episode 1',size=25)
            plt.xlabel('Day Number',size=20)
            plt.ylabel('Mean Squared Difference Between Current & Last QTable',size=14)
            x_axis = np.array([i+1 for i in range(len(errs_1))])
            plt.plot(x_axis,errs_1)
            plt.show()
            
        # calculate MSE between epsiodes
        q_last = q_cur.copy()
        q_cur = q.copy()
            
        # update MSE tracking
        MSE = np.sum(np.square(q_cur - q_last).values)
        
        # plot irregularities
        if episode > 1:
            if MSE > errs[-1]*3:

                print('Episode ' + str(episode) + ' showed irregularity. MSE was ' + str(MSE) + '. Showing big 10 biggest jumps in QTable below.')
 
                q_diff = (q_cur - q_last).copy()
                q_diff['colsum'] = q_diff.sum(axis=1)
                q_diff = q_diff.sort_values('colsum',ascending=False).iloc[:10]
                print(q_diff.drop(columns=['colsum']))
                print('\n\n\n\n')
          
        errs += [MSE]
            
    print('End of Training!')
    
    # plot MSE
    plt.figure(figsize=(14,8))
    plt.title('Q Table Stabilization By Episode',size=25)
    plt.xlabel('Episode Number',size=20)
    plt.ylabel('Mean Squared Difference Between Current & Last QTable',size=14)
    x_axis = np.array([i+1 for i in range(len(errs))])
    plt.plot(x_axis,errs)
    
    # plot MSE for episodes 1-10
    if len(errs) >= 10:
        # plot MSE
        errs_new = errs[:10]
        plt.figure(figsize=(14,8))
        plt.title('Q Table Stabilization By Episode (Episodes 1-10)',size=25)
        plt.xlabel('Episode Number',size=20)
        plt.ylabel('Mean Squared Difference Between Current & Last QTable',size=14)
        x_axis = np.array([i+1 for i in range(len(errs_new))])
        plt.plot(x_axis,errs_new)
        
    # plot MSE for episodes 11-end if possible
    if len(errs) >= 10:
        # plot MSE
        errs_new = errs[11:]
        plt.figure(figsize=(14,8))
        plt.title('Q Table Stabilization By Episode (Episodes 11-End)',size=25)
        plt.xlabel('Episode Number',size=20)
        plt.ylabel('Mean Squared Difference Between Current & Last QTable',size=14)
        x_axis = np.array([i+11 for i in range(len(errs_new))])
        plt.plot(x_axis,errs_new)
      

    return q, actions_history, returns_since_entry

def trainqlearner(start_date, end_date, ticker,alpha=0.01, epsilon=0.2, epsilon_decay = .99995, gamma=0.95, episodes=500,commission=0,sell_penalty=0):

    # Split the data into train and test data set
    train_df = d.get_stock_data(ticker, start_date, end_date)

    # Action Definition (= Q table columns)
    all_actions = {0: 'hold', 1: 'buy', 2: 'sell'}

    # create_df = normalized predictors norm_bb_width, norm_adj_close, norm_close_sma_ratio
    train_df = d.create_df(train_df, 3)

    # get_states = States Dictionary after discretizing by converting continuous values to integer state
    percent_b_states_values, close_sma_ratio_states_value, mrdr_value = d.get_states(
        train_df)

    # Create_state_df =  Add state information to the DF
    train_df = d.create_state_df(
        train_df, percent_b_states_values, close_sma_ratio_states_value,mrdr_value)
    #train_df = d.create_state_df(train_df, None, percent_b_states_values, close_sma_ratio_states_value)

    # Return a list of strings representing the combination of all the states
    all_states = d.get_all_states(
        percent_b_states_values, close_sma_ratio_states_value, mrdr_value)
    # all_states = d.get_all_states(None, percent_b_states_values, close_sma_ratio_states_value)
    states_size = len(all_states)

    # Preparation of the Q Table
    q_init = initialize_q_mat(all_states, all_actions)/1e9
    
    train_data = np.array(train_df[['norm_adj_close', 'state']])
    
    q, train_actions_history, train_returns_since_entry = train_q_learning(
        train_data, q_init, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay,gamma=gamma, episodes=episodes,commission=commission,sell_penalty=sell_penalty)

    # Specify quantiles
    BB_quantiles = percent_b_states_values
    SMA_ratio_quantiles = close_sma_ratio_states_value
    MRDR_quantiles = mrdr_value

    return q, percent_b_states_values, SMA_ratio_quantiles, MRDR_quantiles
