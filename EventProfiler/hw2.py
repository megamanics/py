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


def find_events2(ls_symbols, d_data):
    ''' Finding the event dataframe
    modified for homework 2 and the quiz '''
    df_close = d_data['actual_close']
    print "Finding Events"
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN
    df_test = df_close.copy()
    df_yesterday = df_close >= 8.0
    df_today = df_close < 8.0
    df_test.values[1:,:] = df_yesterday.values[0:-1,:] & df_today.values[1:,:]
    df_test.iloc[0] = False
    df_bool = df_test == True
    df_events[df_bool] = 1
    return df_events

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
    ls_keys = ['actual_close']
    print "Getting Data ", symbollist
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    t2 = ti.time()
    print 'Got Data in %f' %(t2-t1)
    return ls_symbols, ls_keys, ldf_data

if __name__ == '__main__':
    t0 = ti.time()
    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)
    #ls_symbols, ls_keys, ldf_data = getData(dt_end, dt_start,"sp5002012")
    #processEvent("2012",ls_keys, ldf_data, ls_symbols)

    ls_symbols, ls_keys, ldf_data = getData(dt_end, dt_start,"sp5002008")
    processEvent("2008",ls_keys, ldf_data, ls_symbols)    
    #$5.0 event with sp5002012, 176     Date Range = 1st Jan,2008 to 31st Dec, 2009.
    #$8.0 event with sp5002012, 375     Date Range = 1st Jan,2008 to 31st Dec, 2009.    
    #$5.0 event with sp5002008, 326     Date Range = 1st Jan,2008 to 31st Dec, 2009.
    #$6.0 event with sp5002008,      Date Range = 1st Jan,2008 to 31st Dec, 2009.
    #$8.0 event with sp5002008, 527     Date Range = 1st Jan,2008 to 31st Dec, 2009.

