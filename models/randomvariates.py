"""
Author: Justin Mao-Jones
implementation of random number generators
as of 9/1/2016, only LogGamma RNG implemented
"""

import math
import random

def randomLogGamma(beta,seed=None):
    """
    Generate Log-Gamma variates
    p(x|beta) = exp(beta*x - x)/gamma(beta)
    RNG derived from G. Marsaglia and W. Tsang. A simple method for generating gamma variables. ACM Transactions on Mathematical Software, 26(3):363-372, 2000.
    See http://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/ for reference.
    """
    if seed!=None:
        random.seed(seed)
    assert beta > 0, "beta=%s must be greater than 0" % beta
    beta0 = beta
    if beta0 < 1:
        beta = beta+1
    d = beta-1.0/3.0
    cinv = 3.0*(d**0.5)
    
    while True:
        Z = random.normalvariate(0,1)
        if Z > -cinv:
            logU = math.log(random.uniform(0,1))
            val = 1+Z/cinv
            V = val**3.0
            logV = 3*math.log(val)
            if logU < 0.5*(Z**2.0)+d-d*V+d*logV:
                # 1.5*math.log(9) = 3.2958368660043
                logX = -0.5*math.log(d) + 3.0*math.log(cinv+Z)-3.2958368660043
                break
    if beta0 < 1:
        logU = math.log(random.uniform(0,1))
        logX = logX + logU/beta0
    return logX


def unittest_randomLogGamma():
    import numpy as np
    import scipy.special
    import scipy.stats

    def logpdf(x,beta):
        beta = np.array(beta)
        return x*beta-np.exp(x)-scipy.special.gammaln(beta)

    def entropy_numerical(x,beta):
        return -logpdf(x,beta).mean()

    beta = np.exp(np.arange(-5,5))
    n = 200000
    for b in beta:
        x = np.array([randomLogGamma(b) for i in range(n)])
        approx = entropy_numerical(x,b)    
        actual = scipy.stats.loggamma.entropy(b)
        diff = np.abs(approx-actual)
        print 'beta=%0.4f, approx=%0.4f, actual=%0.4f, |diff|=%0.4f' % (b,approx,actual,diff)

if __name__ == '__main__':
    unittest_randomLogGamma()
