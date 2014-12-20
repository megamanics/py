'''
from pyevolve import *
import math
 
error_accum = Util.ErrorAccumulator()
 
# This is the functions used by the GP core,
# Pyevolve will automatically detect them
# and the they number of arguments
def gp_add(a, b): return a+b
def gp_sub(a, b): return a-b
def gp_mul(a, b): return a*b
def gp_sqrt(a):   return math.sqrt(abs(a))
 
def eval_func(chromosome):
   global error_accum
   error_accum.reset()
   code_comp = chromosome.getCompiledCode()
 
   for a in xrange(0, 5):
	  for b in xrange(0, 5):
		 # The eval will execute a pre-compiled syntax tree
		 # as a Python expression, and will automatically use
		 # the "a" and "b" variables (the terminals defined)
		 evaluated	 = eval(code_comp)
		 target		= math.sqrt((a*a)+(b*b))
		 error_accum += (target, evaluated)
   return error_accum.getRMSE()
 
def main_run():
   genome = GTree.GTreeGP()
   genome.setParams(max_depth=5, method="ramped")
   genome.evaluator.set(eval_func)
 
   ga = GSimpleGA.GSimpleGA(genome)
   # This method will catch and use every function that
   # begins with "gp", but you can also add them manually.
   # The terminals are Python variables, you can use the
   # ephemeral random consts too, using ephemeral:random.randint(0,2)
   # for example.
   ga.setParams(gp_terminals	   = ['a', 'b'],
				gp_function_prefix = "gp")
   # You can even use a function call as terminal, like "func()"
   # and Pyevolve will use the result of the call as terminal
   ga.setMinimax(Consts.minimaxType["minimize"])
   ga.setGenerations(1000)
   ga.setMutationRate(0.08)
   ga.setCrossoverRate(1.0)
   ga.setPopulationSize(2000)
   ga.evolve(freq_stats=5)
 
   print ga.bestIndividual()
 
if __name__ == "__main__":
   main_run()
'''

from pyevolve import *
import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

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

#symbols = ["BRCM","TXN","AMD","ADI"] 
#startdate = dt.datetime(2010, 1, 1) 
#enddate = dt.datetime(2010, 12, 31)

def eval_func(chromosome): 
score = 0.0 
total = 0 

for value in chromosome: 
    total+=value 

# Only allow 100% (ej. 0.1,0.1,0.9,0.3 is not valid) 

if total == 100: 
    a = float(chromosome[0]) / 10 
    b = float(chromosome[1]) / 10 
    c = float(chromosome[2]) / 10 
    d = float(chromosome[3]) / 10 

    # simulate fitness 

    vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, symbols,[a,b,c,d], d_data) 
    # Perhaps this needs to be modified allowing sharpe > 1.0 to evolve ????
    score =sharpe 
else: 
    score = 0 
if score < 0: 
    score = 0 
return score 

genome = G1DList.G1DList(4) 
genome.evaluator.set(eval_func) 
ga = GSimpleGA.GSimpleGA(genome) 
ga.setGenerations(100) 
ga.setMutationRate(0.1) 
ga.setPopulationSize(200) 
ga.evolve(freq_stats=10) 
print "Best: ",ga.bestIndividual() 
a= float(ga.bestIndividual()[0])/10 
b= float(ga.bestIndividual()[1])/10 
c= float(ga.bestIndividual()[2])/10 
d= float(ga.bestIndividual()[3])/10 
print "Best portfolio: ",a,b,c,d 
vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, symbols,[a,b,c,d], d_data) 
print "Sharpe Ratio: ", sharpe 
print "Volatility: ", vol 
print "Average Daily Return: ", daily_ret 
print "Cumulative Return: ", cum_ret 

#And some results: 

#pop=100,gen=100,mut=0.3 

#5.3 2.5 0.4 1.8 1.03351726163 
#2.8 2.8 2.2 2.2 0.753716050956 
#4.7 3.3 0.4 1.6 1.03870047842 
#4.5 4.9 0.0 0.6 1.11427675579 
#5.2 2.7 0.9 1.2 0.973746641939 

#pop=100,gen=100,mut=0.1 

#3.6 4.8 0.0 1.6 1.09678108047 
#8.0 1.2 0.1 0.7 1.07238583571 
#4.6 3.9 0.2 1.3 1.07317235297 
#4.0 3.9 0.5 1.6 1.02378701548 
#3.7 3.0 1.4 1.9 0.884972901212 

#pop=200,gen=300,mut=0.2 

#4.6 3.4 0.9 1.1 0.975215274238 
#5.5 3.2 0.6 0.7 1.02395697763 
#6.1 2.6 0.1 1.2 1.08184148222 
#3.8 3.4 1.1 1.7 0.932973462634 
#3.5 3.0 0.7 2.8 0.970189864457 

#here, a typical output, where the optimal allocations using the brute force method are [0.4 0.6 0 0]: 

#Start Date 2010-01-01 00:00:00 
#End Date: 2010-12-31 00:00:00 
#Symbols: ['BRCM', 'TXN', 'AMD', 'ADI'] 
#Brute Force Best Output Allocations: [0.40000000000000002, 0.60000000000000009, 0.0, 0.0] 
#Sharpe Ratio: 1.12334024545 
#Volatility: 0.0174466092415 
#Average Daily Return: 0.00123458808756 
#Cumulative Return: 1.31223266453 
#Gen. 0 (0.00%): Max/Min/Avg Fitness(Raw) [0.00(0.00)/0.00(0.00)/0.00(0.00)] 
#Gen. 10 (10.00%): Max/Min/Avg Fitness(Raw) [1.20(1.07)/0.00(0.25)/1.05(1.00)] 
#Gen. 20 (20.00%): Max/Min/Avg Fitness(Raw) [1.18(1.07)/0.00(0.25)/1.02(0.98)] 
#Gen. 30 (30.00%): Max/Min/Avg Fitness(Raw) [1.21(1.07)/0.00(0.25)/1.07(1.01)] 
#Gen. 40 (40.00%): Max/Min/Avg Fitness(Raw) [1.21(1.07)/0.00(0.25)/1.06(1.01)] 
#Gen. 50 (50.00%): Max/Min/Avg Fitness(Raw) [1.21(1.07)/0.00(0.25)/1.06(1.01)] 
#Gen. 60 (60.00%): Max/Min/Avg Fitness(Raw) [1.18(1.07)/0.00(0.25)/1.03(0.99)] 
#Gen. 70 (70.00%): Max/Min/Avg Fitness(Raw) [1.15(1.07)/0.00(0.25)/0.98(0.96)] 
#Gen. 80 (80.00%): Max/Min/Avg Fitness(Raw) [1.19(1.07)/0.00(0.25)/1.05(0.99)] 
#Gen. 90 (90.00%): Max/Min/Avg Fitness(Raw) [1.18(1.07)/0.00(0.23)/1.02(0.98)] 
#Gen. 100 (100.00%): Max/Min/Avg Fitness(Raw) [1.21(1.07)/0.00(0.25)/1.06(1.01)] 
#Total time elapsed: 4.843 seconds. 
