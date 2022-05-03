import numpy as np

def linear(x,m,c):
    return m*x + c


def gauss(x,*params):
    
    g = 0
    
    for i in range(0,len(params),3):
        g += params[i] * np.exp(-(x-params[i+1])**2/(2*params[i+2]**2))
        
    return g




