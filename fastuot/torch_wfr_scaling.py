import torch
import progressbar

# def euclidean_conic_correlation(x, y, rho=1.):
#     C = torch.ones([x.size()[0] + 1, y.size()[0] + 1])
#     C[1:, 1:] = (x[:,None] - y[None,:]) ** 2
#     return (-C / (2 * rho)).exp()

def euclidean_conic_correlation(x, y, rho=1.):
    C = (x[:,None] - y[None,:]) ** 2
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
    import matplotlib.pyplot as plt


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
    n, m = 1000, 1500
    a, x, b, y = generate_measure(n, m)
    a, x = torch.from_numpy(a), torch.from_numpy(x)
    b, y = torch.from_numpy(b), torch.from_numpy(y)
    C = (x[:,None] - y[None,:]) ** 2
    Om = euclidean_conic_correlation(x, y, rho=1.)
    # A = torch.zeros_like(Om)
    # A[1:,1:] = Om[1:,1:]
    # B = torch.zeros_like(Om)
    # B[1:, 1:] = Om[1:, 1:]
    A=B=Om
    for i in progressbar.progressbar(range(500)):
        A, B = scaling_loop(A, B, Om, a, b)
    # print(A)
    # print(B)
    plt.imshow(A.data.numpy())
    plt.show()
    plt.imshow(B.data.numpy())
    plt.show()
    A1, A2 =A.sum(dim=1), A.sum(dim=0)
    B1, B2 =B.sum(dim=1), B.sum(dim=0)
    # print((A1-a).abs())
    # print((B2 - b).abs())
    # print(A.sum())
    # print(B.sum())
    plt.plot(a.data.numpy(), c='r')
    plt.plot(B1.data.numpy(), c='b')
    plt.show()
    plt.plot(b.data.numpy(), c='r')
    plt.plot(A2.data.numpy(), c='b')
    plt.show()
    f, g = -(B1 / a).log(), -(A2 / b).log()
    D = C - f[:,None] - g[None,:]
    # print(D)
    print(D.min())
    plt.plot(a.data.numpy(), c='r')
    plt.plot((B1*((-0.5 * D).min().exp())).data.numpy(), c='b')
    plt.show()
    plt.plot(b.data.numpy(), c='r')
    plt.plot((A2*((-0.5 * D).min().exp())).data.numpy(), c='b')
    plt.show()
