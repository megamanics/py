'''
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on January, 24, 2013

@author: Sourabh Bajaj
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


def main():
    ''' Main Function'''
    a = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    legal = 0
    for x in range(len(a)):
        for y in range(len(a)):
            for z in range(len(a)):
                for w in range(len(a)):
                    b = [a[x], a[y], a[z], a[w]]
                    if np.sum(b) == 1 :
                        legal =+ 1
                        print b
    print "No of Legal Allocations:",
    ls_aloc = [[i*0.1, j*0.1, k*0.1, l*0.1] for i in range(11) for j in range(11) for k in range(11) for l in range(11) if i+j+k+l == 10]                     
    print len(ls_aloc)
                          
if __name__ == '__main__':
    main()
    
