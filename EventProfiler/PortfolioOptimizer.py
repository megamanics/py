#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__="Stuart Gordon Reid"
__date__ ="$31 Aug 2013 9:49:30 AM$"

import QSTK.qstkutil.DataAccess as date_access
import QSTK.qstkutil.qsdateutil as date_util
import QSTK.qstkutil.tsutil as time_series
import matplotlib.pyplot as plot
import datetime as date_time
import numpy
import pandas
import random
        
class Security: 
    weight = 3.14159265359
    symbol = "NOTSET"
    
    def __init__(self, sym, wght):
        self.weight = wght
        self.symbol = sym

class Portfolio:
    similarity_score = 1.0
    securities = []
    
    start_date = date_time.MAXYEAR
    end_date = date_time.MAXYEAR

    timeseries =  None
    yahoo_connect = None
    data_dictionary = None
    
    def __init__(self, start, end):
        self.start_date = start
        self.end_date = end
        self.calculateTimeSeries()
        self.generateDictionary()
        return
    
    def setup(self):
        self.calculateTimeSeries()
        self.generateDictionary()
    
    def addSecurity(self, security):
        self.securities.append(security)
        self.setup()
    
    def setEndDate(self, date):
        self.end_date = date
        
    def setStartDate(self, date):
        self.start_date = date
    
    def calculateTimeSeries(self):
        timeofday = date_time.timedelta(hours=16)
        self.timeseries = date_util.getNYSEdays(self.start_date, self.end_date, timeofday)
    
    def setSimilarityScore(self, score):
        self.similarity_score = score

    def getSimilarityScore(self):
        return self.similarity_score
    
    def generateDictionary(self):
        self.yahoo_connect = date_access.DataAccess('Yahoo')
        keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
        symbols = self.getSymbols()
        #print symbols
        raw_data = self.yahoo_connect.get_data(self.timeseries, symbols, keys)
        self.data_dictionary = dict(zip(keys, raw_data))
        
    def getSymbols(self):
        symbols = []
        for security in self.securities:
            symbols.append(security.symbol)
        #print symbols
        return symbols   
    
    def getWeights(self):
        weights = []
        for security in self.securities:
            weights.append(security.weight)
        #print weights
        return weights   
    
    def priceSeriesPDF(self, key, name):
        closing_prices = self.data_dictionary[key].values  
        plot.clf()
        plot.plot(self.timeseries, closing_prices)
        plot.legend(self.getSymbols())
        plot.ylabel('Adjusted Close')
        plot.xlabel('Date')
        plot.savefig(name, format='pdf')
        
    def dailyReturns(self):
        weights = self.getWeights()
        closing_prices = self.data_dictionary['close'].values
        normalized_prices = closing_prices / closing_prices[0, :]
        daily_rets = numpy.sum(normalized_prices * weights,axis=1)
        return time_series.returnize0(daily_rets)

    def average(self, returns):
        return numpy.average(returns)
    
    def stdev(self, returns):
        return numpy.std(returns)
    
    def sharpe(self, returns):
        return (self.average(returns) / self.stdev(returns)) * numpy.sqrt(252)

    def culumativeReturn(self):
        closing_prices = self.data_dictionary['close'].values
        start_prices = closing_prices[0]
        close_prices = closing_prices[len(closing_prices)-1]
        return 1 + numpy.sum(((close_prices - start_prices) / start_prices) * self.getWeights())
        
    def setWeights(self, weights):
        c = 0
        for sec in self.securities:
            sec.weight = weights[c]
            c = c + 1

class PSOPortfolioOptimizer:
    swarm = []
    swarm_size = 30
    iterations = 3000
    portfolio = None
    gbest_index = 0
    
    def __init__(self, iter, s_size, port):
        self.swarm_size = s_size
        self.portfolio = port
        self.iterations = iter

    def algorithm(self):
        self.initialize()
        for iteration in range(self.iterations):
            gbest = self.getGlobalBest()
            for particleIndex in range(self.swarm_size):
                self.moveParticle(particleIndex, gbest)
                restart = self.similarityScore(particleIndex, gbest)
                if restart > 0.49 and restart < 0.51:
                    if particleIndex != self.gbest_index:
                        self.restartParticle(particleIndex)
        return self.getGlobalBest()

    def initialize(self):
        self.swarm = []
        particle_size = len(self.portfolio.getWeights())
        for i in range(self.swarm_size):
            particle = []
            for j in range(particle_size):
                particle.append(random.random())
            self.swarm.append(particle)
    
    def moveParticle(self, particleIndex, gbest):
        for i in range(len(self.swarm[particleIndex])):
            velocity = (0.5 - self.similarityScore(particleIndex, gbest)) / 4.0
            if velocity > 0:
                self.swarm[particleIndex][i] = self.swarm[particleIndex][i] + velocity;
            else:
                self.swarm[particleIndex][i] = self.swarm[particleIndex][i] - velocity;
            #if(self.swarm[particleIndex][i] < 0):
            #    self.swarm[particleIndex][i] = 0.0

    def similarityScore(self, particleIndex, gbest):
        similarity = 0.0
        for i in range(len(self.swarm[particleIndex])):
            score = self.swarm[particleIndex][i] / (self.swarm[particleIndex][i] + gbest[i])
            similarity = similarity + score
        return similarity / len(self.swarm[particleIndex])
    
    def restartParticle(self, particleIndex):
        for i in range(len(self.swarm[particleIndex])):
            self.swarm[particleIndex][i] = random.random()  
        return
    
    def transform(self, particle):
        sum_total = numpy.sum(particle)
        for i in range(len(particle)):
            particle[i] = particle[i] / sum_total
        return particle
    
    def getGlobalBest(self):
        gbest_particle = self.swarm[0]
        best_sharpe = 0.0
        count = 0
        for particle in self.swarm:
            particle = self.transform(particle)
            self.portfolio.setWeights(particle)
            daily_rets = self.portfolio.dailyReturns()
            sharpe_ratio = self.portfolio.sharpe(daily_rets)
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                gbest_particle = particle
                self.gbest_index = count
            count =  count + 1
        return gbest_particle

def main():
    AAPL = Security('AAPL', 0.1)
    GOOG = Security('GOOG', 0.1)
    GLD = Security('GLD', 0.1)
    XOM = Security('XOM', 0.1)
    AXP = Security('AXP', 0.1)
    HPQ = Security('HPQ', 0.1) 
    BRCM = Security('BRCM', 0.1)
    TXN = Security('TXN', 0.1)
    IBM = Security('IBM', 0.1)
    HNZ = Security('HNZ', 0.1)
    MSFT = Security('MSFT', 0.1)
    BAC = Security('BAC', 0.1)
    GE = Security('GE', 0.1)
    WMT = Security('WMT', 0.1)
    GS = Security('GS', 0.1)
    JPM = Security('JPM', 0.1) 
    PFE = Security('PFE', 0.1)
    TWX = Security('TWX', 0.1)
    MRK = Security('MRK', 0.1)
    VZ = Security('VZ', 0.1)

    date_start = date_time.datetime(2011, 1, 1)
    date_end = date_time.datetime(2011, 12, 31)
    portfolio = Portfolio(date_start, date_end)

    portfolio.addSecurity(AAPL)
    portfolio.addSecurity(GOOG)
    portfolio.addSecurity(GLD)
    portfolio.addSecurity(XOM)
    portfolio.addSecurity(BRCM)
    portfolio.addSecurity(TXN)
    portfolio.addSecurity(IBM)
    portfolio.addSecurity(HNZ)
    portfolio.addSecurity(AXP)
    portfolio.addSecurity(HPQ)
    #portfolio.addSecurity(MSFT)
    #portfolio.addSecurity(BAC)
    #portfolio.addSecurity(VZ)
    #portfolio.addSecurity(GE)
    #portfolio.addSecurity(WMT)
    #portfolio.addSecurity(GS)
    #portfolio.addSecurity(JPM)
    #portfolio.addSecurity(PFE)
    #portfolio.addSecurity(TWX)
    #portfolio.addSecurity(MRK)
    
    daily_rets = portfolio.dailyReturns()
    
    print "STARTING PSO OPTIMIZER"
    
    weights = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
    portfolio.setWeights(weights)
    daily_rets = portfolio.dailyReturns()
    equal = portfolio.sharpe(daily_rets)
    
    for i in range(5):
        optimizer = PSOPortfolioOptimizer(1000, 40, portfolio)
        weights = optimizer.algorithm()
        portfolio.setWeights(weights)
        daily_rets = portfolio.dailyReturns()
        optimal = portfolio.sharpe(daily_rets)
        print "Candidate optimal sharpe:",optimal,",",equal
        print "Optimal weights:", weights
        
    """
    print "Weights = ", portfolio.getWeights()
    print "Average daily return = ", portfolio.average(daily_rets)
    print "Standard deviation of returns = ", portfolio.stdev(daily_rets)
    print "Sharpe ratio = ", portfolio.sharpe(daily_rets)
    print "Cumulative return = ", portfolio.culumativeReturn()
    """
    
    """
    print "Weights = ", portfolio.getWeights()
    print "Average daily return = ", portfolio.average(daily_rets)
    print "Standard deviation of returns = ", portfolio.stdev(daily_rets)
    print "Sharpe ratio = ", portfolio.sharpe(daily_rets)
    print "Cumulative return = ", portfolio.culumativeReturn()
    portfolio.priceSeriesPDF('close', 'optimized.pdf')
    """
    
    return
    
if  __name__ =='__main__':
    main()
