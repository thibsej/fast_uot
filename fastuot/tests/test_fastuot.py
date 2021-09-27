import pytest
import numpy as np
from fastuot.uot1d import rescale_potentials, dual_loss, init_greed_uot, \
    solve_uot

p = 1.5


@pytest.mark.parametrize('seed,rho,rho2,mass', [(a, b, c, d)
                                                for a in [1, 2, 3, 4, 5, 6, 7]
                                                for b in [0.1, 1.0, 10.0]
                                                for c in [0.1, 1.0, 10.0]
                                                for d in [0.5, 1., 2.]])
def test_rescale_potential_same_mass(seed, rho, rho2, mass):
    n = int(15)
    m = int(16)
    np.random.seed(seed)
    normalize = lambda p: p / np.sum(p)
    a = normalize(np.random.uniform(size=n))
    a = mass * a
    b = normalize(np.random.uniform(size=m))
    f = np.random.normal(size=a.shape[0])
    g = np.random.normal(size=b.shape[0])
    transl = rescale_potentials(f, g, a, b, rho, rho2)
    A, B = a * np.exp(-(f + transl) / rho), b * np.exp(-(g - transl) / rho2)
    assert np.allclose(np.sum(A), np.sum(B), atol=1e-10)


@pytest.mark.parametrize('seed,rho,rho2,mass', [(a, b, c, d)
                                                for a in [1, 2, 3, 4, 5, 6, 7]
                                                for b in [0.1, 1.0, 10.0]
                                                for c in [0.1, 1.0, 10.0]
                                                for d in [0.5, 1., 2.]])
def test_rescale_potential_increase_score(seed, rho, rho2, mass):
    n = int(15)
    m = int(16)
    np.random.seed(seed)
    normalize = lambda p: p / np.sum(p)
    a = normalize(np.random.uniform(size=n))
    a = mass * a
    b = normalize(np.random.uniform(size=m))
    f = np.random.normal(size=a.shape[0])
    g = np.random.normal(size=b.shape[0])
    score1 = dual_loss(f, g, a, b, rho, rho2=rho2)
    transl = rescale_potentials(f, g, a, b, rho, rho2)
    score2 = dual_loss(f + transl, g - transl, a, b, rho, rho2=rho2)
    assert score1 <= score2 + 1e-16


@pytest.mark.parametrize('seed,rho,rho2,mass', [(a, b, c, d)
                                                for a in [1, 2, 3, 4, 5, 6, 7]
                                                for b in [0.1, 1.0, 10.0]
                                                for c in [0.1, 1.0, 10.0]
                                                for d in [0.5, 1., 2.]])
def test_init_greed_is_feasible(seed, rho, rho2, mass):
    n = int(15)
    m = int(16)
    np.random.seed(seed)
    normalize = lambda p: p / np.sum(p)
    a = normalize(np.random.uniform(size=n))
    a = mass * a
    b = normalize(np.random.uniform(size=m))
    x = np.sort(np.random.uniform(size=n))
    y = np.sort(np.random.uniform(size=m))
    ft, gt = init_greed_uot(a, b, x, y, p, rho, rho2=rho2)
    T = np.abs(x[:, None] - y[None, :]) ** p + 1e-15 > (
                ft[:, None] + gt[None, :])
    assert np.all(T)


@pytest.mark.parametrize('seed,rho,rho2,mass,niter', [(a, b, c, d, e)
                                                      for a in
                                                      [1, 2, 3, 4, 5, 6, 7]
                                                      for b in [0.1, 1.0, 10.0]
                                                      for c in [0.1, 1.0, 10.0]
                                                      for d in [0.5, 1., 2.]
                                                      for e in
                                                      [1, 10, 50, 500]])
def test_pot_fw_is_feasible(seed, rho, rho2, mass, niter):
    n = int(15)
    m = int(16)
    np.random.seed(seed)
    normalize = lambda p: p / np.sum(p)
    a = normalize(np.random.uniform(size=n))
    a = mass * a
    b = normalize(np.random.uniform(size=m))
    x = np.sort(np.random.uniform(size=n))
    y = np.sort(np.random.uniform(size=m))
    ft, gt = init_greed_uot(a, b, x, y, p, rho, rho2=rho2)
    _, _, _, f, g, _ = solve_uot(a, b, x, y, p, rho, rho2=rho2, niter=niter,
                                    tol=1e-6,
              greed_init=True, line_search='default', stable_lse=True)
    T = np.abs(x[:, None] - y[None, :]) ** p + 1e-16 > (
                ft[:, None] + gt[None, :])
    assert np.all(T)
