import numpy as np
from matplotlib import pyplot as plt
from inspect import signature


def J(a, b):
    """a function with three arguments"""
    return (a - 1)**2 + (b + 1.5)**2 - 0.1*a*b


def dJ(a, b):
    """function's derivative"""
    return np.array([2*(a - 1) - 0.1*b, 2*(b + 1.5) - 0.1*a])


def grad(f, point, e=1e-3):
    """numerically computed gradient"""
    return np.array([(f(*(point+eps)) - f(*(point-eps)))/(2*e)
                     for eps in np.identity(len(point)) * e])


def find_min(f, x0, alpha=0.1, cond=1e-5, verbose=False):
    """return min by gradient descent"""
    x = x0
    i = 0
    while True:
        i += 1
        x_new = x - alpha * grad(f, x)
        delta = np.linalg.norm(x_new - x)
        if delta < cond:
            break
        x = x_new
        if verbose:
            print(x_new)
    if verbose:
        print(f"Iterations {i}")
    return x_new


# plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([J(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_wireframe(X, Y, Z, rstride=20, cstride=20)
plt.savefig("f3.pdf")

# check if the gradients (analytical, numerical) are the same
x = np.array([0, 0])
print(f"Check gradients, numeric: {grad(J, x)} analytical: {dJ(*x)}")

# gradient descent
n = len(signature(J).parameters)
theta0 = np.zeros(n)
theta = find_min(J, theta0)
print("Minimum at parameters: " + ", ".join(f"{t:.3}" for t in theta))
