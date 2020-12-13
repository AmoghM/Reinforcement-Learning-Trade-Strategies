from dqn.agent.agent import Agent
from dqn.agent.memory import Transition, ReplayMemory
from dqn.functions import *
import sys
import torch


device = torch.device("cpu")

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
#     print("Running on the GPU")
# else:
#     device = torch.device("cpu")
#     print("Running on the CPU")

# if len(sys.argv) != 4:
# 	print("Usage: python train.py [stock] [window] [episodes]")
# 	exit()

# stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

window_size, episode_count = 2, 100
cash, num_shares, sh = 100_000, 0, 50
state_size = 2
agent = Agent(state_size)
data, adj_close, date = getStockDataVec(path="../../data/train_dqn_data.csv")
l = len(adj_close) - 1


for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):

		action = agent.act(state)
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		'''
		# sit
		if num_shares>0:
			reward = (adj_close[t+1] - adj_close[t]) * num_shares
		else:
			reward = 0

		if action == 1: # buy
			# agent.inventory.append(data[t])
			agent.inventory.append(adj_close[t])
			num_shares+=1
			reward = 0
			# print("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			# reward = max(data[t] - bought_price, 0)
			# total_profit += data[t] - bought_price
			# print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

			reward = (adj_close[t]-bought_price)*num_shares
			num_shares-=1
			total_profit += adj_close[t] - bought_price
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

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("Total Reward: ", reward)
			print("Total shares: ", num_shares)
			print("Total cash: ",  cash)
			print("--------------------------------")

		agent.optimize()

	if e % 10 == 0:
		agent.target_net.load_state_dict(agent.policy_net.state_dict())
		torch.save(agent.policy_net, "models/policy_model")
		torch.save(agent.target_net, "models/target_model")