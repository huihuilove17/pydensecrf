'''
python file for simple test

'''
from numba import jit
import numpy as np

@jit(nopython=True)
def test():
    lis = [np.arange(10).reshape(2,5),np.arange(100).reshape(10,10)]
    val = lis[0.0][2,1]
    return val

if __name__ == '__main__':
    _ = test()