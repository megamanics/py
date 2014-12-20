'''
(c) 2011, 2012 Sankhe LLC
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on sept, 21, 2013

@author: ski sankhe
@contact: ski@sankhe.com
@summary: Event Profiler Home Work:
Part 2: Create an event study profile of a specific "known" event on S&P 500 stocks, and compare its impact on two groups of stocks.
The event is defined as when the actual close of the stock price drops below $5.00, more specifically, when:
price[t-1] >= 5.0
price[t] < 5.0
an event has occurred on date t. Note that just because the price is below 5 it is not an event for every day that it is below 5, 
only on the day it first drops below 5.

Example output

For the $5.0 event with S&P500 in 2012, we find 176 events. Date Range = 1st Jan,2008 to 31st Dec, 2009.
For the $5.0 event with S&P500 in 2008, we find 326 events. Date Range = 1st Jan,2008 to 31st Dec, 2009.
The PDF chart will report fewer events than you might count in your code because the events at the very beginning and end cannot be part of the results 
as there will be missing data.

IMPORTANT : It it always important to remove NAN from price data, specially for the S&P 500 from 2008. 
Use the code below after reading the data to get the correct results.
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
We will talk about other methods for "scrubbing" data later on in the course.
'''


import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import time as ti
import matplotlib.pyplot as plt
    

"""
Accepts a list of symbols along with start and end date
Returns the Event Matrix which is a pandas Datamatrix
Event matrix has the following structure :
    |IBM |GOOG|XOM |MSFT| GS | JP |
(d1)|nan |nan | 1  |nan |nan | 1  |
(d2)|nan | 1  |nan |nan |nan |nan |
(d3)| 1  |nan | 1  |nan | 1  |nan |
(d4)|nan |  1 |nan | 1  |nan |nan |
...................................
...................................
Also, d1 = start date
nan = no information about any event.
1 = status bit(positively confirms the event occurence)
"""
print "Pandas Version", pd.__version__

def get_price_events(dt_start,dt_end,symbollist,price):
    ''' Finding the event dataframe modified for homework 2 and the quiz '''
    ls_symbols, ls_keys, ldf_data = getData(dt_start,dt_end,symbollist)
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    df_close = d_data['actual_close']
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN
    df_test = df_close.copy()
    df_yesterday = df_close >= price
    df_today = df_close < price
    df_test.values[1:,:] = df_yesterday.values[0:-1,:] & df_today.values[1:,:]
    df_test.iloc[0] = False
    df_bool = df_test == True
    df_events[df_bool] = 1
    return (d_data,df_events)
    
def generateNPTrades(df_events,df_dataframe,sellAfterDays):
    '''Generates Transactions from Event Matrix'''
    df_data = df_dataframe['actual_close']
    
    df_transactions       = copy.deepcopy(df_events)
    df_transactions.fillna(0,inplace=True)                               # fillna to zero
           
    df_bool = df_transactions == 1
    df_transactions[df_bool]  = 100                                      # Buy on eventpe

    df_transactions_close = copy.deepcopy(df_transactions)
    df_transactions_close = df_transactions_close.shift(sellAfterDays)   # Shift 5 trading days
    df_transactions_close.fillna(0,inplace=True)
    
    df_fulltransacions = df_transactions - df_transactions_close
    return df_fulltransacions

def generateTrades(df_events,df_dataframe):
    '''Generates Transactions from Event Matrix'''
    df_data = df_dataframe['actual_close']
    df_transactions = copy.deepcopy(df_events)
    df_bool = df_transactions == 1
    df_transactions[df_bool]  = 100                                      # Buy on eventpe
    end_date = max(df_transactions.index)
    for s in df_bool:
        for d in df_bool[s].index:
            if (df_bool.ix[d,s]):
                selldate = d + dt.timedelta(1)
                trading_days = 0
                sell_price = np.NaN               
                #print "sell after 5 days:",(selldate,s,sell_price,df_bool.ix[d,s])
                while ((selldate <= end_date) & (trading_days <= 5)):
                    try:
                        sell_price = np.NaN
                        sell_price = df_data.ix[selldate,s]
                        #df_transactions.ix[selldate,s] = np.NaN # remove any existing buy signals so there are no two buys in the same time period.
                        trading_days += 1
                        #print "Increase trading Day:", (s,trading_days,selldate,sell_price)
                    except KeyError, e:
                        a = 1
                        #print 'Not a trading day   :', (s,trading_days,selldate,sell_price)
                    if ((trading_days < 5)):
                        selldate += dt.timedelta(1)
                try:
                    if np.isnan(df_transactions.ix[selldate,s]):
                        df_transactions.ix[selldate,s] = -100                   # Sell after 5 days (initialise if nan)
                    else:
                        #print "Double Event Occurred:", df_transactions.ix[selldate,s]
                        df_transactions.ix[selldate,s] += - 100  # Sell after 5 days (if event occured then this will be 100)
                        print "After Double Event Occurred:", (s,trading_days,selldate,sell_price,df_transactions.ix[selldate,s])
                except KeyError, e:
                    print "Couldnt sell: keep open position",(s,selldate)
                #try:
                    #print "Sell on Fifth T day  :",s,trading_days,d,selldate,df_transactions.ix[selldate,s]
                #except KeyError, e:
                    #print "KeyError:", (s,d,selldate)
    #df_transactions.values[df_bool+5]  = -100                           # Buy on eventpe
    #df_transactions.values[5:,:]   -= df_transactions.values[0:-1,:]    # Sell after 5 days
    return df_transactions
    

def find_events(ls_symbols, d_data):
    ''' Finding the event dataframe '''
    df_close = d_data['actual_close']
    ts_market = df_close['SPY']

    print "Finding Events"

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN
    
    df_events8 = copy.deepcopy(df_close)
    df_events8 = df_events * np.NAN    

    # Time stamps for the event range
    ldt_timestamps = df_close.index
    event  = 0
    event8 = 0
    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            
            #f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
            #f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
            #f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
            #f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

            # Event is found if the symbol is down more then 3% while the
            # market is up more then 2%
            #The event is defined as when the actual close of the stock price drops below $5.00, more specifically, when:
            #price[t-1] >= 5.0
            #price[t] < 5.0
            
            if f_symprice_today < 8.0 and f_symprice_yest >= 8.0:
                event8 += 1
                df_events8[s_sym].ix[ldt_timestamps[i]] = 1
                #print (event8,s_sym,ldt_timestamps[i],f_symprice_yest,f_symprice_today)            
            
            if f_symprice_today < 6.0 and f_symprice_yest >= 6.0:
                df_events[s_sym].ix[ldt_timestamps[i]] = 1
                event += 1
                #print (event,s_sym,ldt_timestamps[i],f_symprice_yest,f_symprice_today)
                
    print "Events below six dollars:", event
    print "Events below eight dollars:", event8
    return (df_events, df_events8)

def processEvent(year, ls_keys, ldf_data, ls_symbols):
    start = ti.time()
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    (df_events,df_events8 )= find_events(ls_symbols, d_data)
    t3 = ti.time()
    print 'Found Events in %f' %(t3-start)
    
    filename5 = "6dollarprice_" + year + ".pdf"
    filename8 = "8dollarprice_" + year + ".pdf"
    
    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20, 
                     s_filename=filename5, b_market_neutral=True, b_errorbars=True, s_market_sym='SPY')
    #ep.eventprofiler(df_events8, d_data, i_lookback=20, i_lookforward=20, 
    #                 s_filename=filename8, b_market_neutral=True, b_errorbars=True, s_market_sym='SPY')
    
    t4 = ti.time()
    print 'Study Created in %f' %(t4-t3)

def getData(dt_end, dt_start,symbollist):
    start = ti.time()
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list(symbollist)
    
    t1 = ti.time()
    print 'Got Symbols in %f' %(t1-start)
    ls_symbols.append('SPY')

    #ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ls_keys = ['actual_close','close']
    print "Getting Data ", symbollist
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    t2 = ti.time()
    print 'Got Data in %f' %(t2-t1)
    return ls_symbols, ls_keys, ldf_data


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

def getPortfolioStats(na_price, allocation):
    # Getting the numpy ndarray or pandas dataframe of close prices.
    
    #Normalize the prices according to the first day. The first row for each stock should have a value of 1.0 at this point.
    # Normalizing the prices to start at 1 and see relative returns
    if type(na_price) == type(pd.DataFrame()):
        na_price = na_price.values
        na_normalized_price = na_price / na_price[0,:]
    else:
        na_normalized_price = na_price / na_price[0]
   
    #Multiply each colunpmn by the allocation to the corresponding equity. 
    allocatedprice =  na_normalized_price * allocation
    
    #Sum each row for each day. That is your cumulative daily portfolio value.
    cum_daily_port_value = allocatedprice
    cum_ret = cum_daily_port_value[-1]
    
    #daily return
    daily_port_returns = cum_daily_port_value.copy()
    tsu.returnize0(daily_port_returns)
    
    vol = np.std(daily_port_returns)
    daily_ret = np.average(daily_port_returns)
    
    sharpe = tsu.get_sharpe_ratio(daily_port_returns)
    #tsu.sharpeRatio(cum_daily_port_value)
    return cum_ret, vol, daily_ret, sharpe, na_normalized_price

def getYahooData(startdate, enddate, symbols):
    """
    @summary Returns the adjusted closing prices
    @param startdate: 
    @param enddate: 
    @param symbols: List of Symbols
    @return yahoo data 
    """
    start = ti.time()
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

    # Filling the data for NAN forward fill followed by backward fill to prevent glancing into future
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    end = ti.time()
    print 'Got Data in %s' %(end-start)
    return d_data

def getData(dt_start,dt_end,symbollist):
    start = ti.time()
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj    = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list(symbollist)
    
    t1 = ti.time()
    print 'Got Symbols in %f' %(t1-start)
    ls_symbols.append('SPY')

    #ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ls_keys = ['actual_close','close']
    print "Getting Data ", symbollist
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    t2 = ti.time()
    print 'Got Data in %f' %(t2-t1)
    return ls_symbols, ls_keys, ldf_data

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

def date_parse(year,month,day):    
    return dt.datetime(int(year),int(month),int(day),16)

def readCSV(filename):
    #np.read_csv(filename, names=['year', 'month', 'day', 'symbol', 'trans', 'lot'], 
    #              parse_dates={'datetime':['year', 'month', 'day']},
    #              date_parser=dtparse)
    np_trans1 = np.loadtxt(filename,dtype='i,i,i,S5,S5,i',delimiter=',',comments='#',skiprows=0)
    #np_trans2 = pd.read_csv(filename, names=['year', 'month', 'day', 'symbol', 'trans', 'lot'],parse_dates={'datetime':['year', 'month', 'day']})
    np_trans  = np.loadtxt(filename,dtype='str',delimiter=',',comments='#',skiprows=0)    
    return np_trans

def find_events(ls_symbols, d_data):
    ''' Finding the event dataframe '''
    df_close = d_data['actual_close']
    ts_market = df_close['SPY']

    print "Finding Events"

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN
    
    df_events8 = copy.deepcopy(df_close)
    df_events8 = df_events * np.NAN    

    # Time stamps for the event range
    ldt_timestamps = df_close.index
    event  = 0
    event8 = 0
    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest  = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            
            #f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
            #f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
            #f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
            #f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

            # Event is found if the symbol is down more then 3% while the
            # market is up more then 2%
            #The event is defined as when the actual close of the stock price drops below $5.00, more specifically, when:
            #price[t-1] >= 5.0
            #price[t] < 5.0
            
            if f_symprice_today < 8.0 and f_symprice_yest >= 8.0:
                event8 += 1
                df_events8[s_sym].ix[ldt_timestamps[i]] = 1
                #print (event8,s_sym,ldt_timestamps[i],f_symprice_yest,f_symprice_today)            
            
            if f_symprice_today < 6.0 and f_symprice_yest >= 6.0:
                df_events[s_sym].ix[ldt_timestamps[i]] = 1
                event += 1
                #print (event,s_sym,ldt_timestamps[i],f_symprice_yest,f_symprice_today)
                
    print "Events below six dollars:", event
    print "Events below eight dollars:", event8
    return (df_events, df_events8)

def processEvent(year, ls_keys, ldf_data, ls_symbols):
    start = ti.time()
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    (df_events,df_events8 )= find_events(ls_symbols, d_data)
    t3 = ti.time()
    print 'Found Events in %f' %(t3-start)
    
    filename5 = "6dollarprice_" + year + ".pdf"
    filename8 = "8dollarprice_" + year + ".pdf"
    
    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20, 
                     s_filename=filename5, b_market_neutral=True, b_errorbars=True, s_market_sym='SPY')
    #ep.eventprofiler(df_events8, d_data, i_lookback=20, i_lookforward=20, 
    #                 s_filename=filename8, b_market_neutral=True, b_errorbars=True, s_market_sym='SPY')
    
    t4 = ti.time()
    print 'Study Created in %f' %(t4-t3)
    
def plotChart():
    # Plotting the prices with x-axis=timestamps
    plt.clf()
    plt.plot(ldt_timestamps, na_port_total, label='Portfolio')
    plt.plot(ldt_timestamps, na_market, label='SPY')
    plt.legend()
    plt.ylabel('Returns')
    plt.xlabel('Date')
    plt.savefig('homework1.pdf', format='pdf')

def convertTrans2Portfolio(investment,df_trans,ldf_dataframe):    
    ldf_data = ldf_dataframe['close'] # use adjusted close for analysing
    df_alloc = copy.deepcopy(ldf_data)
    df_alloc[:] = 0
    
    df_cash = np.sum(ldf_data,axis=1)
    df_cash.fill(investment)
    
    for trans_dt in sorted(df_trans.index):
        for sym in df_trans:      
            lot   = df_trans.ix[trans_dt,sym]
            if ((not np.isnan(lot)) & (lot <> 0)):
                price = ldf_data.ix[trans_dt,sym]
                costOfTransaction = lot * price
                cashBefore = df_cash.ix[trans_dt]
                df_alloc.ix[trans_dt :,sym] += lot
                df_cash.ix[trans_dt] -= costOfTransaction
                cashAfter = df_cash.ix[trans_dt]
                df_cash.ix[trans_dt :] = cashAfter
                #if (lot <> 0):
                    #print trans_dt,lot,sym,price,costOfTransaction,cashBefore,cashAfter
    
    #close open positions:
    for sym in df_trans:
        openpositions = df_alloc.ix[trans_dt,sym]
        if not np.isnan(openpositions):
            if (openpositions != 0):
                price = ldf_data.ix[trans_dt,sym]
                costOfTransaction = -openpositions * price
                cashBefore = df_cash.ix[trans_dt]
                df_alloc.ix[trans_dt :,sym] += -openpositions #close out open position
                df_cash.ix[trans_dt] -= costOfTransaction
                cashAfter = df_cash.ix[trans_dt]
                df_cash.ix[trans_dt :] = cashAfter
                #print trans_dt,-openpositions,sym,price,costOfTransaction,cashBefore,cashAfter
                
    df_portfoliovalue = df_alloc * ldf_data    
    df_netliquidation = np.sum(df_portfoliovalue,axis=1)
    df_netliquidation += df_cash   
    return df_netliquidation

def analyzeTransactions(investment,df_trans,ldf_dataframe,benchmark):    
    ldf_data = ldf_dataframe['close'] # use adjusted close for analysing
    df_alloc = copy.deepcopy(ldf_data)
    df_cash = np.sum(ldf_data,axis=1)
    df_cash.fill(investment)

    dt_min = min(ldf_data.index)
    dt_max = max(ldf_data.index)
    ldf_benchmark =  getYahooData(dt_min, dt_max, benchmark)
    ldf_benchmark_close = copy.deepcopy(ldf_benchmark['close'])    
    
    for keys in df_alloc:
        df_alloc[keys] = 0
    
    for trans_dt in sorted(df_trans.index):
        for sym in df_trans:      
            lot   = df_trans.ix[trans_dt,sym]
            if ((not np.isnan(lot)) & (lot <> 0)):
                price = ldf_data.ix[trans_dt,sym]
                costOfTransaction = lot * price
                cashBefore = df_cash.ix[trans_dt]
                df_alloc.ix[trans_dt :,sym] += lot
                df_cash.ix[trans_dt] -= costOfTransaction
                cashAfter = df_cash.ix[trans_dt]
                df_cash.ix[trans_dt :] = cashAfter
                if (lot <> 0):
                    print trans_dt,lot,sym,price,costOfTransaction,cashBefore,cashAfter
    
    #close open positions:
    for sym in df_trans:
        openpositions = df_alloc.ix[trans_dt,sym]
        if not np.isnan(openpositions):
            if (openpositions != 0):
                price = ldf_data.ix[trans_dt,sym]
                costOfTransaction = -openpositions * price
                cashBefore = df_cash.ix[trans_dt]
                df_alloc.ix[trans_dt :,sym] += -openpositions #close out open position
                df_cash.ix[trans_dt] -= costOfTransaction
                cashAfter = df_cash.ix[trans_dt]
                df_cash.ix[trans_dt :] = cashAfter
                print trans_dt,-openpositions,sym,price,costOfTransaction,cashBefore,cashAfter

    
    df_portfoliovalue = df_alloc * ldf_data    
    df_netliquidation = np.sum(df_portfoliovalue,axis=1)
    df_netliquidation += df_cash   
    #print df_netliquidation.ix[max(ldt_timestamps) :] 
    #print df_netliquidation.ix[dt.datetime(2011,3,28,16)]
    
    cum_ret,  vol,  daily_ret,  sharpe,  na_normalized_price  = getPortfolioStats(df_netliquidation,   [1])    
    cum_ret1, vol1, daily_ret1, sharpe1, na_normalized_price1 = getPortfolioStats(ldf_benchmark_close, [1])
       
    print "Final Value of the portfolio      = ", df_netliquidation.ix[max(df_netliquidation.index)]
    print "Sharpe Ratio of Fund              = ", sharpe
    print "Sharpe Ratio of benchmark         = ", sharpe1
    print "Total Return of Fund              = ", cum_ret
    print "Total Return of benchmark         = ", cum_ret1
    print "Standard Deviation of Fund        = ", vol
    print "Standard Deviation of benchmark   = ", vol1 
    print "Average Daily Return of Fund      = ", daily_ret
    print "Average Daily Return of benchmark = ", daily_ret1    
          
    # Plotting the prices with x-axis=timestamps
    plt.clf()
    plt.plot(na_normalized_price.index, na_normalized_price,  label='portfolio')
    plt.plot(na_normalized_price.index, na_normalized_price1, label='benchmark')
    plt.legend()
    plt.ylabel('Returns')
    plt.xlabel('Date')
    fileName = "Return_" + str(cum_ret) + '_marketSim.pdf'
    plt.savefig(fileName, format='pdf')        

def comparePortfolios(ls_portfolio):
    plt.clf()
    for portlabel in ls_portfolio:
        cum_ret,  vol,  daily_ret,  sharpe,  na_normalized_price  = getPortfolioStats(ls_portfolio[portlabel],[1])
        print "Portfolio Name                    = ", portlabel
        print "Final Value of the portfolio      = ", ls_portfolio[portlabel].ix[max(ls_portfolio[portlabel].index)]
        print "Sharpe Ratio of Fund              = ", sharpe
        print "Total Return of Fund              = ", cum_ret
        print "Standard Deviation of Fund        = ", vol
        print "Average Daily Return of Fund      = ", daily_ret    
        portfolio_label = portlabel + " Shrp:" +str(sharpe) + " Rtn:" + str(cum_ret)
        plt.plot(na_normalized_price.index, na_normalized_price,label=portfolio_label)
    plt.legend()
    plt.ylabel('Returns')
    plt.xlabel('Date')
    fileName = "PortfolioCompare_marketSim.pdf"
    plt.savefig(fileName, format='pdf')        
    
def analyzeTrades(tradeFile,benchmark):
    '''    #orders.csv
    2011,1,10,AAPL,Buy,1500,
    2011,1,13,AAPL,Sell,1500,
    2011,1,13,IBM,Buy,4000,
    2011,1,26,GOOG,Buy,1000,
    2011,2,2,XOM,Sell,4000,
    2011,2,10,XOM,Buy,4000,
    2011,3,3,GOOG,Sell,1000,
    2011,3,3,IBM,Sell,2200,
    2011,5,3,IBM,Buy,1500,
    2011,6,3,IBM,Sell,3300,
    2011,6,10,AAPL,Buy,1200,
    2011,8,1,GOOG,Buy,55,
    2011,8,1,GOOG,Sell,55,
    2011,12,20,AAPL,Sell,1200,

    #The final value of the portfolio using the sample file is -- 2011,12,20,1133860
    #Details of the Performance of the portfolio :
    #Data Range :  2011-01-10 16:00:00  to  2011-12-20 16:00:00
    #Sharpe Ratio of Fund : 1.21540462111
    #Sharpe Ratio of $SPX : 0.0183391412227
    #Total Return of Fund : 1.13386
    #Total Return of $SPX : 0.97759401457
    #Standard Deviation of Fund :   0.00717514512699
    #Standard Deviation of $SPX :   0.0149090969828
    #Average Daily Return of Fund : 0.000549352749569
    #Average Daily Return of $SPX : 1.72238432443e-05
    
    #The other sample file is orders2.csv that you can use to test your code, and compare with others.
    2011,1,14,AAPL,Buy,1500,
    2011,1,19,AAPL,Sell,1500,
    2011,1,19,IBM,Buy,4000,
    2011,1,31,GOOG,Buy,1000,
    2011,2,4,XOM,Sell,4000,
    2011,2,11,XOM,Buy,4000,
    2011,3,2,GOOG,Sell,1000,
    2011,3,2,IBM,Sell,2200,
    2011,5,23,IBM,Buy,1500,
    2011,6,2,IBM,Sell,3300,
    2011,6,10,AAPL,Buy,1200,
    2011,8,9,GOOG,Buy,55,
    2011,8,11,GOOG,Sell,55,
    2011,12,14,AAPL,Sell,1200,
    #The final value of the portfolio using the sample file is -- 2011,12,14, 1078753
    #Data Range :  2011-01-14 16:00:00  to  2011-12-14 16:00:00
    #Sharpe Ratio of Fund : 0.788988545538
    #Sharpe Ratio of $SPX :-0.177204632551
    #Total Return of Fund : 1.078753
    #Total Return of $SPX : 0.937041848381
    #Standard Deviation of Fund :   0.00708034656073
    #Standard Deviation of $SPX :   0.0149914504972
    #Average Daily Return of Fund : 0.000351904599618
    #Average Daily Return of $SPX :-0.000167347202139    
'''
    np_transactions = readCSV(tradeFile)
    ls_symbols =  sorted(set(np_transactions[:,3]))    
    ls_dates   = list()
    for a in np_transactions:
        ls_dates.append(dt.datetime(int(a[0]),int(a[1]),int(a[2])))
    dt_min = min(ls_dates)
    dt_max = max(ls_dates) + dt.timedelta(1)
    
    #notrans = len(np_transactions)
    #df_transactions = pd.DataFrame(np.random.randn(notrans,4),ls_dates,columns=['sym', 'trans', 'lot'])
    #df_transactions.describe()
    #print df_transactions
    
    ldf_data      =  getYahooData(dt_min, dt_max, ls_symbols)
    ldf_benchmark =  getYahooData(dt_min, dt_max, benchmark)
    ldt_timestamps = du.getNYSEdays(dt_min, dt_max, dt.timedelta(hours=16))
    
    ldf_close           = ldf_data['close']
    ldf_benchmark_close = copy.deepcopy(ldf_benchmark['close'])
    
    
    df_alloc = copy.deepcopy(ldf_data['close'])
    df_cash = np.sum(ldf_close,axis=1)
    df_cash.fill(1000000)
    
    for keys in df_alloc:
        df_alloc[keys] = 0
        
    # We need closing prices so the timestamp should be hour_s=16.
    dt_cob = dt.timedelta(hours=16)
    
    for a in np_transactions:
        trans_dt = dt.datetime(int(a[0]),int(a[1]),int(a[2]),16) 
        a[6] = trans_dt
    
    for a in np_transactions:
        trans_dt = dt.datetime(int(a[0]),int(a[1]),int(a[2]),16) 
        lot = int(a[5])
        sym = a[3]
        transType = str(a[4]).lower()
        price = ldf_close.ix[trans_dt,sym]
        costOfTransaction = lot * price
        cashBefore = df_cash.ix[trans_dt]
        if (transType.find('buy') > -1):
            #df_alloc[a[3]][trans_dt] = a[5]
            df_alloc.ix[trans_dt :,sym] += lot
            df_cash.ix[trans_dt] -= costOfTransaction
        else:
            #df_alloc[a[3]][trans_dt] = -lot
            df_alloc.ix[trans_dt :,sym] -= lot
            df_cash.ix[trans_dt :] += costOfTransaction
        cashAfter = df_cash.ix[trans_dt]
        df_cash.ix[trans_dt :] = cashAfter
        print trans_dt,transType,lot,sym,price,costOfTransaction,cashBefore,cashAfter
    
    df_portfoliovalue = df_alloc * ldf_close
    
    df_netliquidation = np.sum(df_portfoliovalue,axis=1)
    #print df_cash.ix[dt.datetime(2011,12,19,16) :] 
    #print df_netliquidation.ix[dt_max - dt.timedelta(2) :] 
    df_netliquidation += df_cash   
    #print df_netliquidation.ix[max(ldt_timestamps) :] 
    #print dt.datetime(2011,11,9,16)
    #print df_netliquidation.ix[dt.datetime(2011,11,9,16)]
    #print dt.datetime(2011,3,28,16)
    #print df_netliquidation.ix[dt.datetime(2011,3,28,16)]
    
    cum_ret,  vol,  daily_ret,  sharpe,  na_normalized_price  = getPortfolioStats(df_netliquidation,   [1])
    cum_ret1, vol1, daily_ret1, sharpe1, na_normalized_price1 = getPortfolioStats(ldf_benchmark_close, [1])
       
    print "Final Value of the portfolio      = ", df_netliquidation.ix[max(ldt_timestamps)]
    print "Sharpe Ratio of Fund              = ", sharpe
    print "Sharpe Ratio of benchmark         = ", sharpe1
    print "Total Return of Fund              = ", cum_ret
    print "Total Return of benchmark         = ", cum_ret1
    print "Standard Deviation of Fund        = ", vol
    print "Standard Deviation of benchmark   = ", vol1 
    print "Average Daily Return of Fund      = ", daily_ret
    print "Average Daily Return of benchmark = ", daily_ret1
    
    # Plotting the prices with x-axis=timestamps
    plt.clf()
    plt.plot(ldt_timestamps, na_normalized_price, label='portfolio')
    plt.plot(ldt_timestamps, na_normalized_price1, label='benchmark')
    plt.legend()
    plt.ylabel('Returns')
    plt.xlabel('Date')
    fileName = tradeFile + '_marketSim.pdf'
    plt.savefig(fileName, format='pdf')    
    

if __name__ == '__main__':
    t0 = ti.time()


    #dt_start = dt.datetime(2008, 1, 1)
    #dt_end   = dt.datetime(2009, 12, 31)
    #ls_symbols, ls_keys, ldf_data = getData(dt_end, dt_start,"sp5002012")
    #processEvent("2012",ls_keys, ldf_data, ls_symbols)

    #ls_symbols, ls_keys, ldf_data = getData(dt_end, dt_start,"sp5002008")
    #processEvent("2008",ls_keys, ldf_data, ls_symbols)    
    #$5.0 event with sp5002012, 176     Date Range = 1st Jan,2008 to 31st Dec, 2009.
    #$8.0 event with sp5002012, 375     Date Range = 1st Jan,2008 to 31st Dec, 2009.    
    #$5.0 event with sp5002008, 326     Date Range = 1st Jan,2008 to 31st Dec, 2009.
    #$6.0 event with sp5002008,         Date Range = 1st Jan,2008 to 31st Dec, 2009.
    #$8.0 event with sp5002008, 527     Date Range = 1st Jan,2008 to 31st Dec, 2009.
    
    #On simulating the $5 event the output is:
    #The final value of the portfolio using the sample file is -- 2009,12,28,54824.0    
    #Details of the Performance of the portfolio    
    #Data Range :  2008-01-03 16:00:00  to  2009-12-28 16:00:00    
    #Sharpe Ratio of Fund : 0.527865227084
    #Sharpe Ratio of $SPX : -0.184202673931    
    #Total Return of Fund :  1.09648
    #Total Return of $SPX : 0.779305674563    
    #Standard Deviation of Fund :  0.0060854156452
    #Standard Deviation of $SPX : 0.022004631521    
    #Average Daily Return of Fund :  0.000202354576186
    #Average Daily Return of $SPX : -0.000255334653467
    
    #analyzeTrades('orders.csv' ,['$SPX'])
    #analyzeTrades('orders2.csv',['$SPX'])
    #analyzeTrades('orders3.csv',['$SPX'])
    #analyzeTrades('orders4.csv',['$SPX'])
    
    dt_start = dt.datetime(2008, 1, 1)
    #dt_start = dt.datetime(2008, 8, 4)
    #dt_end = dt.datetime(2008, 1, 31)
    dt_end = dt.datetime(2009, 12, 31)
    #dt_end = dt.datetime(2008, 10, 2)
    
    #df_transactions  = generateTrades(df_events,df_data)
    #analyzeTransactions(50000,df_transactions ,df_data,['$SPX'])    
    #analyzeTransactions(50000,df_transactions1,df_data,['$SPX'])
    
    ls_portfolio      = list()
    ls_portfolioLabel = list()
    
    dt_start            = dt.datetime(2008, 1, 1)
    dt_end              = dt.datetime(2009, 12, 31)    

    ls_portfolioLabel.append("portfolio 5")    
    (df_data,df_events) = get_price_events(dt_start,dt_end,"sp5002012",5.0)
    df_transactions     = generateNPTrades(df_events,df_data,5)   
    df_port             = convertTrans2Portfolio(50000, df_transactions, df_data)
    ls_portfolio.append(df_port)
    
    ls_portfolioLabel.append("portfolio 10") 
    (df_data,df_events) = get_price_events(dt_start,dt_end,"sp5002012",10.0)
    df_transactions     = generateNPTrades(df_events,df_data,5)   
    df_port             = convertTrans2Portfolio(50000, df_transactions, df_data)
    ls_portfolio.append(df_port)

    ls_portfolioLabel.append("portfolio 9") 
    (df_data,df_events) = get_price_events(dt_start,dt_end,"sp5002012",9.0)
    df_transactions     = generateNPTrades(df_events,df_data,5)   
    df_port             = convertTrans2Portfolio(50000, df_transactions, df_data)
    ls_portfolio.append(df_port)
    
    ls_portfolioLabel.append("portfolio 7") 
    (df_data,df_events) = get_price_events(dt_start,dt_end,"sp5002012",7.0)
    df_transactions     = generateNPTrades(df_events,df_data,5)   
    df_port             = convertTrans2Portfolio(50000, df_transactions, df_data) 
    ls_portfolio.append(df_port)

    ls_portfolioLabel.append("price drop 8") 
    (df_data,df_events) = get_price_events(dt_start,dt_end,"sp5002012",8.0)
    df_transactions     = generateNPTrades(df_events,df_data,5)   
    df_port             = convertTrans2Portfolio(50000, df_transactions, df_data) 
    ls_portfolio.append(df_port)

    ls_portfolioLabel.append("price drop 6") 
    (df_data,df_events) = get_price_events(dt_start,dt_end,"sp5002012",6.0)
    df_transactions     = generateNPTrades(df_events,df_data,5)   
    df_port             = convertTrans2Portfolio(50000, df_transactions, df_data) 
    ls_portfolio.append(df_port)
    
    comparePortfolios(dict(zip(ls_portfolioLabel,ls_portfolio)))
    
    end = ti.time()
    print 'complete: %f' %(end-t0)

