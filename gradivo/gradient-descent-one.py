import numpy as np
import matplotlib.pyplot as plt


def J(x):
    return (x - 6.5)**2 + 3


def derivative(f, x, eps=1e-3):
    return (f(x+eps) - f(x-eps)) / (2*eps)


def find_min(f, x0=0, alpha=0.1, cond=1e-3, verbose=True):
    """return min by gradient descent"""
    x = x0
    i = 0
    while True:
        i += 1
        x_new = x - alpha * derivative(f, x)
        delta = abs(x_new - x)
        if delta < cond:
            break
        x = x_new
    if verbose:
        print(f"Iterations {i}")
    return x_new


# plot a function
a = np.arange(0, 10, 0.1)
y = J(a)
plt.plot(a, y, "k-")
plt.savefig("f.pdf")

# find a minimum by gradient descent
theta = find_min(J, cond=1e-3)
print(theta)

