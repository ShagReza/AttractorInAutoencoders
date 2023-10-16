import numpy as np
import matplotlib.pyplot as plt


def newton(z, f, fprime, max_iter=100, tol=1e-6):
    """The Newton-Raphson method."""
    for i in range(max_iter):
        step = f(z)/fprime(z)
        if abs(step) < tol:
            return i, z
        z -= step
    return i, z


def plot_newton_basins(p, pprime, n=200, extent=[-1,1,-1,1], cmap='jet'):
    
    """Shows basin of attraction for convergence to each root using the Newton-Raphson method."""
    root_count = 0
    roots = {}

    m = np.zeros((n,n))
    xmin, xmax, ymin, ymax = extent
    for r, x in enumerate(np.linspace(xmin, xmax, n)):
        for s, y in enumerate(np.linspace(ymin, ymax, n)):
            z = x + y*1j
            root = np.round(newton(z, p, pprime)[1], 1)
            if not root in roots:
                roots[root] = root_count
                root_count += 1
            m[r, s] = roots[root]
    plt.imshow(m.T, cmap=cmap, extent=extent)


f = lambda x: x**3 - 1
fprime = lambda x: 3*x**2

plot_newton_basins(f, fprime)


















delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
               origin='lower', extent=[-3, 3, -3, 3],
               vmax=abs(Z).max(), vmin=-abs(Z).max())

plt.show()