import numpy as np

if __name__ == '__main__':
    N, M = 2, 2
    p = 2
    random = False
    # np.random.seed(0)

    if random:
        x, y = np.sort(np.random.normal(size=N)), \
               np.sort(np.random.normal(loc=0.2, size=M))
        C = np.abs(x[:, None] - y[None, :]) ** p
    else:
        C = np.array([[0., 1.], [2., 0.]])

    print("Cost matrix\n", C)

    f1 = np.array([C[0, 0], C[1, 1] - C[0, 1] + C[0, 0]])
    g1 = np.array([0., C[0, 1] - C[0, 0]])
    print("\n Dual constraint tightness for right-up move\n",
          C - f1[:, None] - g1[None, :])

    f2 = np.array([C[0, 0], C[1, 0]])
    g2 = np.array([0., C[1, 1] - C[1, 0]])
    print("\n Dual constraint tightness for down-right move\n",
          C - f2[:, None] - g2[None, :])

    f, g = 0.5 * (f1 + f2), 0.5 * (g1 + g2)
    print("\n Dual constraint tightness for diagonal move\n",
          C - f[:, None] - g[None, :])
