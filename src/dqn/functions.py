import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def getStockDataVec(key=None,path=None):
	vec, states, date = [], [], []
	lines = open(path, "r").read().splitlines()
	for line in lines[5:]:
		row = line.split(",")
		close = row[6]
		if close != 'null':
			date.append(row[0])
			vec.append(float(row[6]))
			states.append(list(map(float, row[11:13])))

	return states, vec, date

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	return np.array([data[t]])
