'''
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on January, 24, 2013

@contact: ski@sankhe.com
@summary: Sharpe Ratio
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da



# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np



print "Pandas Version", pd.__version__


def simulate(startdate, enddate, symbols,allocation):
    '''Simulate Function    
    @summary Returns the range of possible returns with upper and lower bounds on the portfolio participation
    @param startdate: 
    @param enddate: 
    @param symbols: List of Symbols
    @param allocation:Allocations to the equities at the beginning of the simulation (e.g., 0.2, 0.3, 0.4, 0.1)
    @return tuple containing (vol, daily_ret, sharpe, cum_ret)
    Standard deviation of daily returns of the total portfolio
    Average daily return of the total portfolio
    Sharpe ratio (Always assume you have 252 trading days in an year. And risk free rate = 0) of the total portfolio
    Cumulative return of the total portfolio
    vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, ['GOOG','AAPL','GLD','XOM'], [0.2,0.3,0.4,0.1])
   
    Here are some notes and assumptions:
    When we compute statistics on the portfolio value, we include the first day.
    We assume you are using the data provided with QSTK. If you use other data your results may turn out different from ours. Yahoo's online data changes every day. We could not build a consistent "correct" answer based on "live" Yahoo data.
    Assume 252 trading days/year.
    '''
    
    d_data = getYahooData(startdate, enddate, symbols)
    cum_ret, vol, daily_ret, sharpe = getPortfolioStats(d_data, allocation)
    return (vol, daily_ret, sharpe, cum_ret)

def getPortfolioStats(d_data, allocation):
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    
    #Normalize the prices according to the first day. The first row for each stock should have a value of 1.0 at this point.
    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]
   
    #Multiply each colunpmn by the allocation to the corresponding equity. 
    allocatedprice =  na_normalized_price * allocation
    
    #Sum each row for each day. That is your cumulative daily portfolio value.
    cum_daily_port_value = allocatedprice.sum(axis=1)
    cum_ret = cum_daily_port_value[-1]
    
    #daily return
    daily_port_returns = cum_daily_port_value.copy()
    tsu.returnize0(daily_port_returns)
    
    vol = np.std(daily_port_returns)
    daily_ret = np.average(daily_port_returns)
    
    sharpe = tsu.get_sharpe_ratio(daily_port_returns)
    #tsu.sharpeRatio(cum_daily_port_value)
    return cum_ret, vol, daily_ret, sharpe

def getYahooData(startdate, enddate, symbols):
    """
    @summary Returns the adjusted closing prices
    @param startdate: 
    @param enddate: 
    @param symbols: List of Symbols
    @return yahoo data 
    """
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    # Read in adjusted closing prices for the equities.
    ldf_data = c_dataobj.get_data(ldt_timestamps,symbols, ls_keys)
    
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    return d_data

def optimize(dt_start, dt_end,symbols):
    '''Optimze'''
    
    optimalsr =0
    a = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    a = np.arange(0,1.01,0.01)
    d_data = getYahooData(dt_start, dt_end, symbols)
    

    for x in range(len(a)):
        for y in range(len(a)):
            for z in range(len(a)):
                for w in range(len(a)):
                    b = [a[x], a[y], a[z], a[w]]
                    if np.sum(b) == 1 :
                        data_copy = d_data.copy()
                        cum_ret, vol, daily_ret, sharpe = getPortfolioStats(data_copy, b)
                        if optimalsr < sharpe :
                            optimalsr = sharpe
                            optimalacc = b
                            print "Max Found:",(b,sharpe)
    print "Optimal for symbols:",symbols
    return(optimalsr,optimalacc)

def main():
    ''' Main Function'''

    # List of symbols
    
    #Start Date: January 1, 2011
    #End Date: December 31, 2011
    #Symbols: ['AAPL', 'GLD', 'GOOG', 'XOM']
    #Optimal Allocations: [0.4, 0.4, 0.0, 0.2]
    #Sharpe Ratio: 1.02828403099
    #Volatility (stdev of daily returns):  0.0101467067654
    #Average Daily Return:  0.000657261102001
    #Cumulative Return:  1.16487261965

    ls_symbols = ["AAPL"]

    # Start and End date of the charts
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    
    vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ['AAPL', 'GLD', 'GOOG', 'XOM'], [0.4, 0.4, 0.0, 0.2])
    print "Sharpe ratio:",sharpe
    print "Vol:",vol
    print "Avg Daily Return:",daily_ret
    print "Cumulative Return:",cum_ret
    
    #Start Date: January 1, 2010
    #End Date: December 31, 2010
    #Symbols: ['AXP', 'HPQ', 'IBM', 'HNZ']
    #Optimal Allocations:  [0.0, 0.0, 0.0, 1.0]
    #Sharpe Ratio: 1.29889334008
    #Volatility (stdev of daily returns): 0.00924299255937
    #Average Daily Return: 0.000756285585593
    #Cumulative Return: 1.1960583568

    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)
       
    vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ['AXP', 'HPQ', 'IBM', 'HNZ'], [0.0, 0.0, 0.0, 1.0])
    
    print "Sharpe ratio:",sharpe
    print "Vol:",vol
    print "Avg Daily Return:",daily_ret
    print "Cumulative Return:",cum_ret
    
    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)

    optimalsr, optimal_alloc = optimize(dt_start, dt_end,['BRCM', 'TXN', 'AMD', 'ADI'])      
    optimalsr, optimal_alloc = optimize(dt_start, dt_end,['BRCM', 'ADBE', 'AMD', 'ADI'])  
    
if __name__ == '__main__':
    main()
    
