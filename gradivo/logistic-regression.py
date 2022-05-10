import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def generate_data():
    """Generates a two-feature, binary class data set where
    the classes are (almost) linearly separated, with some overlap."""
    mean1, mean2 = [4, 6.5], [5, 5.5]
    cov1, cov2 = [[1, 0], [0, 1]], [[1, 0.7], [0.7, 1]]
    n1, n2 = 50, 50
    data1 = np.random.multivariate_normal(mean1, cov1, n1)
    data2 = np.random.multivariate_normal(mean2, cov2, n2)
    x = np.concatenate((data1, data2))
    y = np.array([0 if i < n1 else 1 for i in range(n1+n2)])
    x = np.column_stack((np.ones(len(x)), x))
    return x, y


def scatter_plot(x, y, theta, file_name="logreg.pdf"):
    """Plots the data and its logistic regression model."""
    plt.plot(x[y == 0, 1], x[y == 0, 2], "sy")
    plt.plot(x[y == 1, 1], x[y == 1, 2], "sb")
    # x_e = np.array([[1, x[:, 1].min()], [1, x[:, 1].max()]])
    # plt.plot(x_e[:, -1], x_e.dot(theta))

    n = 50
    xr = np.linspace(x[:, 1].min(), x[:, 1].max(), n, endpoint=True)
    yr = np.linspace(x[:, 2].min(), x[:, 2].max(), n, endpoint=True)
    xg, yg = np.meshgrid(xr, yr, indexing="ij")
    z = np.zeros((len(xr), len(yr)))

    for i in range(len(xr)):
        for j in range(len(yr)):
            xi = np.array([1, xg[i, j], yg[i, j]], ndmin=2)
            z[i, j] = h(theta, expand(xi))

    cs = plt.contour(xg, yg, z, [0.1, 0.4, 0.5, 0.6, 0.9], colors="k")
    plt.clabel(cs, inline=1, fontsize=10)

    plt.savefig(file_name)
    plt.close()


def expand(x, order=1):
    # we already have order of 0 and 1
    for o in range(2, order+1):
        for i in range(o+1):
            # print "x^%d y^%d" % (i, o - i)
            x = np.c_[x, (x[:, 1] ** i) * (x[:, 2] ** (o - i))]
    return x


def h(theta, x):
    """Logistic function"""
    return 1. / (1 + np.exp(-x.dot(theta)))


def j(theta, x, y):
    """Log likelihood"""
    return y.dot(np.log(h(theta, x))) + (1-y).dot(np.log(1-h(theta, x)))


def grad_theta(theta, x, y):
    """Gradient of log likelihood"""
    return (y - h(theta, x)).dot(x)


def grad_approx(f, thetas):
    """Approximate computation of cost function gradient."""
    e = 1e-5
    return np.array([(f(thetas + row) - f(thetas - row)) / (2 * e)
                     for row in np.identity(len(thetas)) * e])


def bfgs(x, y):
    """Implements logistic regression, returns parameters of the
    fitted model."""
    theta0 = np.zeros(x.shape[1]).T
    res = minimize(lambda theta, x=x, y=y: -j(theta, x, y),
                   theta0,
                   method='BFGS',
                   jac=lambda theta, x=x, y=y: -grad_theta(theta, x, y),
                   tol=0.00001)
    print("iterations:", res.njev)
    return res.x


data_x, data_y = generate_data()
print("optimization ...")
theta = bfgs(data_x, data_y)

print("plotting ...")
scatter_plot(data_x, data_y, theta)

p = np.zeros(data_x.shape[1]).T
grad_approx(lambda t, x=data_x, y=data_y: j(t, x, y), p)
grad_theta(p, data_x, data_y)
