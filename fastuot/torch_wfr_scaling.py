import torch

def euclidean_conic_correlation(x, y, rho=1.):
    C = (x[:,None] - y[:,None]) ** 2
    return (-C / (2 * rho)).exp()

def scaling_loop(A, B, Om, a, b):
    B_ = B * Om**2
    R = a / B_.sum(dim=1)
    A = B_ * R[:,None]
    A_ = A * Om**2
    C = b / A_.sum(dim=0)
    B = A_ * C[None,:]
    return A, B

if __name__ == '__main__':
    import numpy as np


    def normalize(x):
        return x / np.sum(x)


    def gauss(grid, mu, sig):
        return np.exp(-0.5 * ((grid - mu) / sig) ** 2)

    def generate_measure(n, m):
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

    np.random.seed(6)
    n, m = 50, 50
    a, x, b, y = generate_random_measure(n, m)