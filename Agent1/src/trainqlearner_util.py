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


    plt.show()


def get_epsi_decay(epsilon, total_rep, end_value):
    tots = int(total_rep * 0.90)
    return np.exp((1 / tots) * (np.log(end_value / epsilon)))



def train_q_learning(train_data, q, cash_states_values, shares_states_values, gamma, episodes, sh, alp, epsi):
    '''
    Train a Q-table 
    Inputs:
    train_data(dataframe)
    q(dataframe): initial Q-table
    alpha(float): threshold of which action strategy to take
    gamma(float): discount percentage on the future return
    Output:
    q(dataframe): Updated Q-table
    actions_history(dict): has everydays' actions and close price
    returns_since_entry(list): contains every day's return since entry
    '''
    # actions_history = []
    # num_shares = 0
    # bought_history = []
    # returns_since_entry = [0]
    # cash = 100000

    episode = 0
    q_cur = q.copy()
    errs = []
    episode_decile = episodes//10
    epsilon = epsi
    total_rep = int((len(train_data)-1) * episodes)
    epsi_decay = get_epsi_decay(epsi, total_rep, 0.05)
    alpha = alp
    count = 0
    for ii in range(episodes):

        episode += 1
        if episode == 1 or episode%episode_decile == 0 or episode == episodes:
            print('Training episode {}'.format(episode))
            print('Epsilon: ' + str(epsilon))


        actions_history = []
        cash = 100000
        num_shares = 0

        current_portfolio_value = []

        #add convergence tracking for episode 1
        if episode == 1:
            errs_1 = []
            q_cur_1 = q.copy()



        for i, val in enumerate(train_data):
            current_adj_close, state = val
            try:
                next_adj_close, next_state = train_data[i+1]
            except:
                break

            count += 1
            current_cash_state = d.value_to_state(cash, cash_states_values)
            current_share_state = d.value_to_state(num_shares, shares_states_values)
            state = state + current_cash_state + current_share_state



            if count == (total_rep * 0.90):
                print('90% reached')
                print(epsilon)
                print(count)

            if ii == 0 and i == 0:
                epsilon = epsi
            elif count >= (total_rep * 0.90):
                epsilon = epsilon
            else:
                epsilon *= epsi_decay
              
            action = act(state, q, threshold=epsilon, actions_size=3)
            
            # get reward
            if action == 0: # hold
                if num_shares > 0:
                    next_cash = cash # no change
                    reward = (cash + num_shares*next_adj_close) - (cash + num_shares*current_adj_close)                
                else:
                    reward = 0

            if action == 1: # buy
                if cash > sh*current_adj_close:
                  next_cash = cash - sh*current_adj_close
                  # reward = (cash - current_adj_close + ((num_shares+1)*next_adj_close)) - (cash + num_shares*current_adj_close)
                  reward = (next_cash + ((num_shares+sh)*next_adj_close)) - (cash + num_shares*current_adj_close)
                  num_shares += sh
                  cash = next_cash
                else: 
                  reward = 0
            
            if action == 2: # sell
                if num_shares > 0:
                    next_cash = cash + sh*current_adj_close
                    # reward = (cash + current_adj_close + ((num_shares-1)*next_adj_close)) - (cash + num_shares*current_adj_close)
                    reward = (next_cash + ((num_shares-sh)*next_adj_close)) - (cash + num_shares*current_adj_close)
                    num_shares -= sh
                    cash = next_cash
                else:
                    reward = 0

            #NEXT using cash and share

            #next_cash_state = value_to_state(next_cash,cash_states_values)
            ## Use 'cash' instead as affect 'current'
            next_cash_state = d.value_to_state(cash,cash_states_values)
            next_share_state = d.value_to_state(num_shares, shares_states_values)
            ## Note: cash and num_share are automatically updated in at the end of the Action code block
            next_state = next_state + next_cash_state + next_share_state

            actions_history.append((i, current_adj_close, action))
            
            # update q table
            q.loc[state, action] = (1.-alpha)*q.loc[state, action] + alpha*(reward+gamma*(q.loc[next_state].max()))

            current_portfolio_value.append(cash + num_shares*next_adj_close)

            # ---- (tentative) start of q-table info plotting/output -----
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
    print('Train_shape:' + str(len(train_data)))
    print('Count' + str(count))
    print('Total_rep' + str(total_rep))
    print('Epsilon' + str(epsilon))
    print('Epsi_decay' + str(epsi_decay))
    print('Alpha' + str(alpha))



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

    return q, actions_history, current_portfolio_value




#def trainqlearner(start_date, end_date, ticker,alpha=0.01, epsilon=0.2, epsilon_decay = .99995, gamma=0.95, episodes=500,commission=0,sell_penalty=0):
def trainqlearner(ticker, start_date, end_date, window, gamma, episodes, sh, alp, epsi):
    # Split the data into train and test data set
    train_df = d.get_stock_data(ticker, start_date, end_date)

    # Action Definition (= Q table columns)
    all_actions = {0: 'hold', 1: 'buy', 2: 'sell'}

    # create_df = normalized predictors norm_bb_width, norm_adj_close, norm_close_sma_ratio
    train_df = d.create_df(train_df, window)

    # get_states = States Dictionary after discretizing by converting continuous values to integer state
    percent_b_states_values, close_sma_ratio_states_value = d.get_states(
        train_df)


    # Create_state_df =  Add state information to the DF
    train_df = d.create_state_df(
        train_df, percent_b_states_values, close_sma_ratio_states_value)

    #train_df = d.create_state_df(train_df, None, percent_b_states_values, close_sma_ratio_states_value)

    cash_states_values, shares_states_values = d.create_cash_and_holdings_quantiles()
    train_df.to_csv("data/train_dqn_data.csv")

    # Return a list of strings representing the combination of all the states
    all_states = d.get_all_states(percent_b_states_values, close_sma_ratio_states_value, cash_states_values, shares_states_values)
    # all_states = d.get_all_states(None, percent_b_states_values, close_sma_ratio_states_value)

    states_size = len(all_states)

    # Preparation of the Q Table

    q = initialize_q_mat(all_states, all_actions)/1e5
    
    train_data = np.array(train_df[['Adj Close', 'state']])


    
    q, train_actions_history, train_returns_since_entry = train_q_learning(train_data, q, cash_states_values, shares_states_values, gamma, episodes, sh, alp, epsi)


    return q, train_actions_history, train_returns_since_entry, percent_b_states_values, close_sma_ratio_states_value, cash_states_values, shares_states_values
