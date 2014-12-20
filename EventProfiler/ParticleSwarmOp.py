import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import numpy as np
import pylab as pl
import random
import datetime
import time

def getDataColumns():
    return ['open', 'high', 'low', 'close', 'volume', 'actual_close']

def getVendorSource():
    return 'Yahoo'

def replaceNaN(fields, data, value): # Filling the data for NAN
    for key in fields:
        data[key] = data[key].fillna(method='ffill')
        data[key] = data[key].fillna(method='bfill')
        data[key] = data[key].fillna(value)

def validAllocation(allocations):
    return np.sum(allocations) == 1.0

def GetClosingPrices(startDate, endDate, symbols):
    tradingDays = du.getNYSEdays(startDate, endDate, dt.timedelta(hours=16))
    yahooDAO = da.DataAccess(getVendorSource(), cachestalltime=0)
    fields = getDataColumns()
    symbolData = yahooDAO.get_data(tradingDays, symbols, fields)
    data = dict(zip(fields, symbolData))
    replaceNaN(fields, data, 1.0)
    return data['close'].values, tradingDays

class Particle:

    def __init__(self, position, fitness, velocity, bestPosition, bestFitness):
        self.position = position
        self.fitness = fitness
        self.velocity = velocity
        self.bestPosition = bestPosition
        self.bestFitness = bestFitness

    def toString(self):
        value = 'Position: %s \n' % ', '.join(map(str, self.position))
        value += 'Fitness: %f \n' % self.fitness
        value += 'Velocity: %s \n' % ', '.join(map(str, self.velocity))
        value += 'Best Position: %s \n' % ', '.join(map(str, self.bestPosition))
        value += 'Best Fitness: %f \n' % self.bestFitness
        return value

class ParticleSwarmOptimizer:
    
    dim = 10
    minX = 0
    maxX = 1
    minV = -1.0
    maxV = maxX
    swarm = []
    sharpeHistory = []  
    startTime = 0
    endTime = 0
    
    def __init__(self, particles, iterations, symbols, cumRet, tradingDays):
        self.particles = particles
        self.iterations = iterations
        self.symbols = symbols
        self.dim = len(symbols)
        self.bestGlobalPosition = [0] * self.dim
        self.bestGlobalFitness = 0
        self.cumRet = cumRet
        self.tradingDays = tradingDays

    def objectiveFunction(self, cumRet, allocation, tradingDays):
        individualFundInvestment = cumRet * allocation
        totalFundInvestment = np.sum(individualFundInvestment, axis=1)
        #fundCumRet = totalFundInvestment / totalFundInvestment[0]
        fundDailyRet = (totalFundInvestment / np.insert(totalFundInvestment[0:-1], 0, totalFundInvestment[0], 0)) - 1
        aveDailyRet = np.average(fundDailyRet)
        volatility = np.std(fundDailyRet)
        sharpeRatio = np.sqrt(len(tradingDays)) * aveDailyRet / volatility
        return sharpeRatio

    def dynamicInertia(self, count):
        if count < 1000:
            return 0.9
        elif count < 2000:
            return 0.7
        elif count < 4000:
            return 0.5
        elif count < 5000:
            return 0.4

    def optimize(self):
        print 'Starting optimization for symbols: %s' % ', '.join(self.symbols)
        self.startTime = time.time()
        iteration = 0
        
        # initialize
        for p in range(self.particles):
            randomPosition = []
            for i in range(self.dim):
                randomPosition.append((self.maxX - self.minX) * random.uniform(0, 1) + self.minX)
                   
            fitness = self.objectiveFunction(self.cumRet, randomPosition, self.tradingDays)
            
            randomVelocity = []
            for i in range(self.dim):
                randomVelocity.append((abs(self.maxX - self.minX) - (-1.0 * abs(self.maxX - self.minX))) * random.uniform(0, 1) + (-1.0 * abs(self.maxX - self.minX)))
                
            particle = Particle(randomPosition, fitness, randomVelocity, randomPosition, fitness)
            self.swarm.append(particle)
        
            if particle.fitness > self.bestGlobalFitness:
                self.bestGlobalFitness = particle.bestFitness
                self.bestGlobalPosition = list(particle.position)
        
        w = 0.729;    # inertia weight
        c1 = 1.49445; # cognitive/local weight
        c2 = 1.49445; # social/global weight
        r1 = 0        # cognitive randomization
        r2 = 0        # social randomization
             
        sharpeHistory = []     
    
        # activity
        while iteration < self.iterations:
            iteration += 1
            newFitness = 0
                 
            for i in range(self.particles):
                currentParticle = self.swarm[i]
                newVelocity = []
                newPosition = []
                for j in range(self.dim):
                    r1 = random.uniform(0,1)
                    r2 = random.uniform(0,1)
                    
                    newVelocity.append((w * currentParticle.velocity[j]) + \
                        (c1 * r1 * (currentParticle.bestPosition[j] - currentParticle.position[j])) + \
                        (c2 * r2 * (self.bestGlobalPosition[j] - currentParticle.position[j])))
                    newVelocity[j] = max([self.minV, min([newVelocity[j], self.maxV])])
                        
                currentParticle.velocity = list(newVelocity)
                
                for j in range(self.dim):
                    newPosition.append(currentParticle.position[j] + newVelocity[j])
                    newPosition[j] = max([self.minX, min([newPosition[j], self.maxX])])
                        
                currentParticle.position = list(newPosition)
                newFitness = self.objectiveFunction(self.cumRet, newPosition, self.tradingDays)
                currentParticle.fitness = newFitness
                
                if newFitness > currentParticle.bestFitness:
                    currentParticle.bestPosition = list(newPosition)
                    currentParticle.bestFitness = newFitness
                    
                if newFitness > self.bestGlobalFitness:
                    print 'better fitness found, iteration %d' % (iteration)
                    self.bestGlobalPosition = list(newPosition)
                    self.bestGlobalFitness = newFitness
                    sharpeHistory.append(newFitness)
            
            w = self.dynamicInertia(iteration)
                    
        self.endtime = time.time()

    def printResults(self):
        s = datetime.datetime.fromtimestamp(self.startTime).strftime('%Y-%m-%d %H:%M:%S')
        e = datetime.datetime.fromtimestamp(self.endtime).strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------'
        print 'Particle Swarm Optimization results:'
        print 'Start time %s, end time %s' % (s, e)
        print 'Particles: %d, iterations: %d' % (self.particles, self.iterations)
        print 'Symbols: %s' % ', '.join(self.symbols)
        print 'final best global fitness = %.12f' % self.bestGlobalFitness  
        #print 'final best global position %s' % ', '.join(map(str, pso.bestGlobalPosition))     
        psoAllocation = np.array(self.bestGlobalPosition) / (sum(self.bestGlobalPosition) * 1.0)
        print 'final best pso allocation: %s' % ', '.join(map(str, psoAllocation))  
        #print 'solution history: %s' % ', '.join(map(str, pso.sharpeHistory))             

psoList = []
particles = 50
iterations = 5000

start = du.dt.date(2011, 1, 1)
end = du.dt.date(2011, 12, 31)    
symbols = ['MA', '$VIX', 'MCD', 'GWW', 'SH', 'OKE', 'TJX', 'UST', 'ISRG', 'PM'] #['ROST', '$VIX', 'MA', 'OKE', 'PM'] #['TJX', 'MCD', 'EP', 'LO', 'HUM'] #['AAPL', 'GLD', 'GOOG', 'XOM'] #['XOM', 'INTC', 'ADBE', 'AMD', 'ADI', 'GE', 'GLD', 'GOOG'] #['AAPL', 'GLD', 'GE', 'INTC'] #['BRCM', 'ADBE', 'AMD', 'ADI'] #['AAPL', 'GLD', 'GOOG', 'XOM']
closingPrices, tradingDays = GetClosingPrices(start, end, symbols)
cumReturn = closingPrices / closingPrices[0]

for i in range(10):
    pso = ParticleSwarmOptimizer(particles, iterations, symbols, cumReturn, tradingDays)
    psoList.append(pso)
    pso.optimize()

for p in psoList:
    p.printResults()


psoList = []
symbols = ['ROST', '$VIX', 'MA', 'OKE', 'PM']
closingPrices, tradingDays = GetClosingPrices(start, end, symbols)
cumReturn = closingPrices / closingPrices[0]
for i in range(10):
    pso = ParticleSwarmOptimizer(particles, iterations, symbols, cumReturn, tradingDays)
    psoList.append(pso)
    pso.optimize()

for p in psoList:
    p.printResults()
