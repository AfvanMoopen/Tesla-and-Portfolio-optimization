import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import pandas_datareader
import datetime

import pandas_datareader.data as web
start = datetime.datetime(2012, 1, 3)
end = datetime.date.today()
apple_raw = web.DataReader("AAPL", 'morningstar', start, end)
aapl = apple_raw.loc['AAPL']['Close']
aapl.head()
amzn_raw = web.DataReader("AMZN", 'morningstar', start, end)
amzn = amzn_raw.loc['AMZN']['Close']
tsla_raw = web.DataReader("TSLA", 'morningstar', start, end)
tsla = tsla_raw.loc['TSLA']['Close']
ibm_raw = web.DataReader("IBM", 'morningstar', start, end)
ibm = ibm_raw.loc['IBM']['Close']
stocks = pd.concat([aapl,tsla,ibm,amzn],axis=1)
stocks.columns = ['aapl','tsla','ibm','amzn']
stocks.head()
mean_daily_ret = stocks.pct_change(1).mean()
mean_daily_ret
stocks.pct_change(1).corr()
stock_normed = stocks/stocks.iloc[0]
stock_normed.plot(figsize=(12,8))
stock_daily_ret = stocks.pct_change(1)
stock_daily_ret.head()
log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()
log_ret.hist(bins=100,figsize=(10,5));
plt.tight_layout()
log_ret.mean() * 252
log_ret.cov()
log_ret.cov()*252 # multiply by days
num_runs = 10000

all_weights = np.zeros((num_runs,len(stocks.columns)))
ret_arr = np.zeros(num_runs)
vol_arr = np.zeros(num_runs)
sharpe_arr = np.zeros(num_runs)

for ind in range(num_runs):

    # Create Random Weights
    weights = np.array(np.random.random(4))

    # Rebalance Weights
    weights = weights / np.sum(weights)
    
    # Save Weights
    all_weights[ind,:] = weights

    # Expected Return
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

    # Expected Variance
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
    
sharpe_arr.max()
sharpe_arr.argmax()

all_weights[9642,:]
print(stocks.columns)
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

print('Return with Maximum SR')
print(max_sr_ret)
print('Volality with Maximum SR')
print(max_sr_vol)

plt.figure(figsize=(14,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Add red dot for max SR
plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')
def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])
from scipy.optimize import minimize

# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
init_guess = [0.25,0.25,0.25,0.25]
frontier_y = np.linspace(0,0.3,150)
def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1] ##Grab the 2nd item which is volatility
frontier_volatility = []

for possible_return in frontier_y:
    # function for return. 
    cons = ({'type':'eq','fun': check_sum},
            {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
    
    frontier_volatility.append(result['fun'])
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Add frontier line
plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)
