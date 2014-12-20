'''
    (c) Julian Berengut 2013

    Genetic algortithm with simulated annealing.
    Used to optimise the fitness of a portfolio using historical data.
    
    Things that could be optimised:
    - Random part of population breeding: Portfolio.breed()
    - Start temperature and number of annealing steps: genetic_annealing()
    - Starting population: main()
    - Number of anneals: main()
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import math
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from operator import itemgetter, attrgetter

# Cache data
print "Caching historical data."
dt_start = dt.datetime(2011, 1, 1)
dt_end   = dt.datetime(2011, 12, 31)
dt_timeofday = dt.timedelta(hours=16)
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

c_dataobj = da.DataAccess('Yahoo')
ls_all_syms = c_dataobj.get_all_symbols()

# Keys to be read from the data, it is good to read everything in one go.
#ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
ls_keys = ['close']
    
# Reading the data, now d_data is a dictionary with the keys above.
# Timestamps and symbols are the ones that were specified before.
ldf_data = c_dataobj.get_data(ldt_timestamps, ls_all_syms, ls_keys)
d_data = dict(zip(ls_keys, ldf_data))
close_data = dict(filter(lambda t: not np.isnan(np.sum(t[1])), \
                         zip(ls_all_syms, d_data['close'].values.transpose())))


class Portfolio:
    '''A portfolio made of stocks and their allocations'''
    def __init__(self, stocks, alloc):
        if len(stocks) != len(alloc):
            print "Bad init: ", stocks, alloc
        self.symbols = np.array(stocks)
        self.allocations = np.array(alloc)
        self.prune_and_normalize(self.MAX_PORTFOLIO_SIZE)

    def prune_and_normalize(self, num_stocks = 0):
        # Replace negative allocations with zero
        self.allocations = np.array([max(x, 0.0) for x in self.allocations])
        # Remove stocks with allocations smaller than 1.e-6
        new_length = sum(x > self.MIN_ALLOCATION for x in self.allocations)
        if num_stocks != 0:
            new_length = min(num_stocks, new_length)
        if new_length & new_length < len(self.allocations):
            temp_tuple = sorted(zip(self.symbols, self.allocations), key=itemgetter(1))
            temp_tuple = temp_tuple[-new_length:]
            a, b = zip(*temp_tuple)
            self.symbols = np.array(a)
            self.allocations = np.array(b)
        # Normalise
        norm = np.sum(self.allocations)
        self.allocations = self.allocations/norm

    def breed(self, other, random_factor):
        '''Merge two portfolios together'''
        selfset = set(self.symbols)
        otherset = set(other.symbols)
        stocks = []
        alloc = []
        for key in selfset - otherset:
            starting_alloc = 0.5 * self.allocations[np.where(self.symbols == key)[0]]
            stocks.append(key)
            alloc.append(np.random.normal(starting_alloc, starting_alloc * random_factor))
        for key in otherset - selfset:
            starting_alloc = 0.5 * other.allocations[np.where(other.symbols == key)[0]]
            stocks.append(key)
            alloc.append(np.random.normal(starting_alloc, starting_alloc * random_factor))
        for key in otherset & selfset:
            starting_alloc = 0.5 * (self.allocations[np.where(self.symbols == key)[0]] \
                                    + other.allocations[np.where(other.symbols == key)[0]] )
            diff = 0.5 * (self.allocations[np.where(self.symbols == key)[0]] \
                          - other.allocations[np.where(other.symbols == key)[0]])
            stocks.append(key)
            if diff > self.MIN_ALLOCATION:
                alloc.append(np.random.normal(starting_alloc, diff))
            else:
                alloc.append(starting_alloc[0])
        return Portfolio(stocks, alloc)

    def size(self):
        return len(self.symbols)

    MAX_PORTFOLIO_SIZE = 10
    MIN_ALLOCATION = 1.e-4


def simulate(portfolio):
    ''' Show the result of a portfolio (stock symbols & allocations) over the date range given.

        It is assumed that the stocks are "bought and held" with the allocations correct on start_date.
        PRE: close data for all stocks is held in the global dictionary close_data
        POST: return four floating-point numbers for our portfolio simulation
            volatility (std. dev. of daily return)
            average daily return
            sharpe ratio
            cumulative return
    '''
    # sanity checks
    if portfolio.size() == 0:
        print "No symbols found"

    # Use close_data object to speed things up
    d_data = []
    for key in portfolio.symbols:
        d_data.append(close_data[key])
    d_data = np.array(d_data)
    d_data = d_data.transpose()

    # Portfolio buy in (per dollar) at open
    #purchase = stock_allocations/d_data['open'].values[0]
    purchase = portfolio.allocations/d_data[0]
    portfolio_data = d_data * purchase
    portfolio_total = sum(portfolio_data.transpose())

    #Calculate daily returns
    daily_return = portfolio_total[0] - 1.0
    daily_return = np.append(daily_return, portfolio_total[1:]/portfolio_total[0:-1] - 1)

    volatility = np.std(daily_return)
    average_daily_return = np.mean(daily_return)
    #sharpe_ratio = math.sqrt(len(ldt_timestamps)) * average_daily_return/volatility
    sharpe_ratio = math.sqrt(252.) * average_daily_return/volatility
    cumulative_return = portfolio_total[-1]

    return volatility, average_daily_return, sharpe_ratio, cumulative_return


def genetic_annealing(starter_list):
    ''' From the starting population, generate 10 times as many children with random parentage
        and some extra randomness in the stock allocations, given by temperature.
        Repeat many times, gradually reducing the temperature (annealing).
    '''        
    n = len(starter_list)
    population = starter_list
    # Number of children to generate from first population
    num_children = n * 10
    # Temperature of anneal
    for temperature in np.arange(3.0, 0.0, -0.06):
        # I included some random population. Don't know if it makes a difference
        random_pop = []
        for port in population:
            random_pop += generate_random_portfolios(1, port.symbols, 10)
        population = population + random_pop
        parents_1 = np.random.choice(population, num_children, replace=True)
        parents_2 = np.random.choice(population, num_children, replace=True)
        for parents in zip(parents_1, parents_2):
            if parents[0] != parents[1]:
                population.append(Portfolio.breed(parents[0], parents[1], temperature))
        # Get fitness function for entire population
        fitness = []
        for portfolio in population:
            volatility, mean_daily_return, sharpe, cumulative_return = simulate(portfolio)
            fitness.append(sharpe)
        # Remove unfit members of population
        temp_tuple = zip(population, fitness)
        temp_tuple = sorted(temp_tuple, key=itemgetter(1))
        temp_tuple = temp_tuple[-n:]
        a, b = zip(*temp_tuple)
        population = list(a)
        print "Temp: ", temperature, "  Sharpe: ", b[-1]

    return population


def generate_random_portfolios(num_portfolios, stock_symbols, portfolio_size):
    '''Generate random portfolios from all stock symbols with random allocations.'''
    list = []
    n = len(stock_symbols)
    r = min(portfolio_size, n)
    for i in range(num_portfolios):
        stocks = np.random.choice(stock_symbols, r, replace=False)
        alloc = np.random.random(r)
        list.append(Portfolio(stocks, alloc))
    return list


def main():
    n = len(close_data.keys())
    # Number of symbols per portfolio
    r = 10
    # Get number of starting portfolios required so that there is high likelihood
    # that all stock symbols are included somewhere
    num_port = math.trunc(math.log(0.05/n) / math.log(math.fabs(n-r)/n))

    port_list = []
    best_port = []
    best_sharpe = []

    # Repeat annealing three times, keeping 10 best porfolios each time
    for i in range(3):
        print "\nAnneal ", i+1
        port_list = port_list + generate_random_portfolios(num_port, close_data.keys(), r)
        port_list = genetic_annealing(port_list)
        # Store best portfolio
        optimal = port_list[len(port_list)-1]
        volatility, mean_daily_return, sharpe, cumulative_return = simulate(optimal)
        best_port.append(optimal)
        best_sharpe.append(sharpe)
        print "Sharpe Ratio: ", sharpe
        print zip(optimal.symbols, optimal.allocations)
        # Keep 10 best for next anneal
        port_list = port_list[-10:]

    for i in range(len(best_sharpe)):
        print "\nSharpe Ratio: ", best_sharpe[i], ":"
        print zip(best_port[i].symbols, best_port[i].allocations)

main()
