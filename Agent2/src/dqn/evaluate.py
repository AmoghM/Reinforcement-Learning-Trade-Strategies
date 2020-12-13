import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from agent.agent import Agent
from functions import *
import pandas as pd

# stock_name = '^HSI_2018'
window_size = 5

'''
agent = Agent(window_size, True)
data, adj_close = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
episode_count = 1000
'''
action_map = {0:"HOLD", 1: "BUY", 2:"SELL"}
def DQN(stock_table=None, money= None, inc= None, original_shares= None, commission= None):

	window_size=3
	cash, num_shares = 100_000, 0
	sh = 50
	agent = Agent(window_size)
	data, adj_close, date = getStockDataVec(path = "../data/test_dqn_data.csv")
	l = len(adj_close) - 1
	batch_size = 32
	episode_count = 5



	closes = []
	buys = 	[]
	sells = []


	final_vals, actions, shares, cashes, dates = [], [], [], [], []

	episode_count=1
	for e in range(episode_count):
		closes = []
		buys = []
		sells = []


		state = getState(data, 0, window_size + 1)
		total_profit = 0
		agent.inventory = []


		# capital = 100000
		for t in range(l):
			action = agent.act(state)
			# action = np.random.randint(0, 3)
			closes.append(data[t])

			# sit
			next_state = getState(data, t + 1, window_size + 1)
			reward = 0
			'''
			if action == 1: # buy
				if capital > adj_close[t]:
					agent.inventory.append(adj_close[t])
					buys.append(adj_close[t])
					sells.append(None)
					capital -= adj_close[t]
				else:
					buys.append(None)
					sells.append(None)
	
			elif action == 2: # sell
				if len(agent.inventory) > 0:
					bought_price = agent.inventory.pop(0)
					reward = max(adj_close[t] - bought_price, 0)
					total_profit += adj_close[t] - bought_price
					buys.append(None)
					sells.append(adj_close[t])
					capital += adj_close[t]
				else:
					buys.append(None)
					sells.append(None)
			elif action == 0:
				buys.append(None)
				sells.append(None)
			'''

			next_adj_close = adj_close[t+1]
			current_adj_close = adj_close[t]

			# get reward
			if action == 0:  # hold
				if num_shares > 0:
					next_cash = cash  # no change
					reward = (cash + num_shares * next_adj_close) - (cash + num_shares * current_adj_close)
				else:
					reward = 0

			if action == 1:  # buy
				if cash > sh * current_adj_close:
					next_cash = cash - sh * current_adj_close
					# reward = (cash - current_adj_close + ((num_shares+1)*next_adj_close)) - (cash + num_shares*current_adj_close)
					reward = (next_cash + ((num_shares + sh) * next_adj_close)) - (cash + num_shares * current_adj_close)
					num_shares += sh
					cash = next_cash
				else:
					reward = 0

			if action == 2:  # sell
				if num_shares > 0:
					next_cash = cash + sh * current_adj_close
					# reward = (cash + current_adj_close + ((num_shares-1)*next_adj_close)) - (cash + num_shares*current_adj_close)
					reward = (next_cash + ((num_shares - sh) * next_adj_close)) - (cash + num_shares * current_adj_close)
					num_shares -= sh
					cash = next_cash
				else:
					reward = 0



			done = True if t == l - 1 else False
			agent.memory.push(state, action, next_state, reward)
			state = next_state

			'''
			if done:
				print("--------------------------------")
				print(" Total Profit: " + formatPrice(total_profit))
				print(" Total Shares: ", )
				print("--------------------------------")
			'''
			if done:
				print("--------------------------------")
				print("Total Profit: " + formatPrice(total_profit))
				print("Total Reward: ", reward)
				print("Total shares: ", num_shares)
				print("Total cash: ",  cash)
				print("--------------------------------")


			cur_cash, cur_shares = cash, num_shares
			final_vals.append(cur_cash + (cur_shares * adj_close[t]))
			cashes.append(cur_cash)
			actions.append(action_map[action])
			shares.append(num_shares)
			dates.append(date[t])

		cashes = pd.Series(cashes,index = pd.to_datetime(dates))
		shares = pd.Series(shares,index = pd.to_datetime(dates))
		actions = pd.Series(actions,index = pd.to_datetime(dates))
		final_vals = pd.Series(final_vals,index = pd.to_datetime(dates))

		results = {'final_vals': final_vals, 'actions': actions, 'shares': shares, 'cash': cashes}


	return results

if __name__=="__main__":
	res = DQN()
	dic = pd.DataFrame(res)
	print(dic)