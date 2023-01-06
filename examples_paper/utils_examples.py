import numpy as np

def normalize(x):
    return x / np.sum(x)


def gauss(grid, mu, sig):
    return np.exp(-0.5 * ((grid-mu) / sig) ** 2)


def generate_synthetic_measure(n, m):
    x = np.linspace(0.2, 0.4, num=n)
    a = np.zeros_like(x)
    a[:n // 2] = 2.
    a[n // 2:] = 3.
    y = np.linspace(0.45, 0.95, num=m)
    a = normalize(a)
    b = normalize(gauss(y, 0.6, 0.03)
                  + gauss(y, 0.7, 0.03)
                  + gauss(y, 0.8, 0.03))
    return a, x, b, y
