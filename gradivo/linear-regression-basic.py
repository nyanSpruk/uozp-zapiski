import numpy as np
from matplotlib import pyplot as plt


def scatter_plot(X, y, theta, fname="lr-scatter.pdf"):
    """Plot the data and its linear regression model."""
    plt.clf()
    plt.plot(X[:, 1], y, "o")
    x_e = np.array([[1, X[:, 1].min()], [1, X[:, 1].max()]])
    plt.plot(x_e[:, -1], x_e.dot(theta))
    plt.savefig(fname)
    plt.close()


def gradient_descent(X, y, alpha=0.001, epochs=1000, trace=False):
    """For a matrix x and vector y return a linear regression model."""
    theta = np.zeros(X.shape[1])
    for i in range(epochs):
        theta = theta - alpha * (X.dot(theta) - y).dot(X)
        if trace and (i % 10 == 0):
            scatter_plot(X, y, theta, fname="tmp/%03d.pdf" % i)
    return theta


data = np.loadtxt("linear-0.txt")
X_train = data[:, 0:-1]
X_train = np.column_stack((np.ones(len(X_train)), X_train))
y_train = data[:, -1]
theta_star = gradient_descent(X_train, y_train)

scatter_plot(X_train, y_train, theta_star)
