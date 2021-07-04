import numpy as np
import matplotlib.pyplot as pl


def root(a, n, q=1.0, x0=1.0, niter=100):
    xvals = np.zeros(niter)
    xvals[0] = x0
    for i in range(1, niter):
        x = xvals[i - 1]
        xvals[i] = (a * x * x + q * x) / (q + x ** (n + 1))
        print(x)
    return xvals

def main():
    a = 68
    n = 3
    q = 2e2
    x0 = 4.0
    xvals = root(a, n, q, x0, niter=100)
    pl.plot(xvals)
    pl.show()
if __name__ == '__main__':
    main()
