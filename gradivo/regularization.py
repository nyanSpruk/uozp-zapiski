import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score

# load the data, prepare it for sklearn
data = pd.read_excel("bodyfat.xlsx")
m = data.to_numpy()
X, y = m[:, 1:], m[:, 0]

labels = data.columns.values[1:]

# normalization
# think about why it has no effect with plain (unregularized) linear regression
# and why the cross-validated accuracy changes between raw and normalized data
# when we use regularization
X = X - X.mean(axis=0)
X /= X.std(axis=0)

# selection of the model
# model_type = LinearRegression
model_type, model_name = \
    [(Ridge, "Ridge Regression"), (Lasso, "Lasso Regression")][0]

# cross validation
# for the start, to figure out if we are doing well
# and to compare between normalized and raw data
cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = LinearRegression()
y_hat = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
r2 = r2_score(y, y_hat)
print(f"R2 for unregularized regression: {r2:.3}")

# choice of regularization parameters
alphas = np.logspace(-4, 5, 20)

# how does the accuracy change with regularization strength?
# we cross-validate to assess the accuracy and plot
# the accuracy(regularization strength) graph
r2s = []
for alpha in alphas:
    model = model_type(alpha=alpha)
    y_hat = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    r2 = r2_score(y, y_hat)
    r2s.append(r2)

index_max = np.argmax(r2s)
best_alpha = alphas[index_max]
print(f"Best regularization strength: {best_alpha:.2}")

fig = plt.figure()
ax = plt.gca()
ax.plot(alphas, r2s, "o-")
ax.set_xscale('log')
plt.xlabel("regularization strength")
plt.ylabel("cross validated r2")
plt.savefig("fig-accuracy-vs-regularization.pdf")
plt.clf()

# select best-ranked features for specific degree of regularization
alpha = 0.1
model = model_type(alpha=alpha)
fitted = model.fit(X, y)
coef = np.abs(fitted.coef_)

k = 5 # number of best-rank features to select
ind = np.argpartition(coef, -k)[-k:]

# compute coefficients for the regularization path
cs = []
for alpha in alphas:
    model = model_type(alpha=alpha)
    fitted = model.fit(X, y)
    cs.append(fitted.coef_)
res = np.stack(cs)

# plot the regularization path for selected features
fig = plt.figure()
ax = plt.gca()
for i in ind:
    ax.plot(alphas, res[:, i], "o-", label=labels[i])
ax.legend(loc="upper right")
ax.set_xscale('log')
plt.xlabel("regularization strength")
plt.ylabel("feature weight")
plt.savefig("fig-regularization-path.pdf")
plt.clf()
