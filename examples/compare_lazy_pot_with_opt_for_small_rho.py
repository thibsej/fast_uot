import numpy as np
import matplotlib.pyplot as plt

from fastuot.uot1d import lazy_potential
from fastuot.cvxpy_uot import dual_via_cvxpy

if __name__ == '__main__':
    # generate data
    n = int(15)
    m = int(16)
    # np.random.seed(0)
    normalize = lambda p: p / np.sum(p)
    a = normalize(np.random.uniform(size=n))
    b = normalize(np.random.uniform(size=m))
    x = np.sort(np.random.uniform(size=n))
    y = np.sort(np.random.uniform(size=m))

    # params
    p = 1.5
    C = np.abs(x[:, None] - y[None, :]) ** p
    rho = np.amin(C)

    result, constr, fr, gr = dual_via_cvxpy(a, b, x, y, p, rho, tol=1e-10)
    fr, gr = fr.value, gr.value
    plt.imshow(np.log(C - fr[:, None] - gr[None, :]))
    plt.title('CVXPY')
    plt.show()

    fc, gc = lazy_potential(x, y, p, diagonal=False)
    plt.imshow(np.log(C - fc[:, None] - gc[None, :]))
    plt.title('LAZY POT W/O DIAGONAL MOVES')
    plt.show()

    fc, gc = lazy_potential(x, y, p, diagonal=True)
    plt.imshow(np.log(C - fc[:, None] - gc[None, :]))
    plt.title('LAZY POT WITH DIAGONAL MOVES')
    plt.show()