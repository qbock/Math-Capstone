import csv
from LDA import *
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

y = []
X = []
codes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

with open('./IRIS/iris.data', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        data = np.array(row)
        X.append(data[:4])
        if data[4] == 'Iris-setosa':
            y.append(0)
        elif data[4] == 'Iris-versicolor':
            y.append(1)
        else:
            y.append(2)

X = np.array(X).astype('float64')

# train_X = np.concatenate((np.concatenate((X[0:40], X[50:90]), 0), X[100:140]), 0)
# train_y = np.concatenate((np.concatenate((y[0:40], y[50:90]), 0), y[100:140]), 0)
# valid_X = np.concatenate((np.concatenate((X[40:50], X[90:100]), 0), X[140:150]), 0)
# valid_y = np.concatenate((np.concatenate((y[40:50], y[90:100]), 0), y[140:150]), 0)
#
# lda_full = LDA(X, y)
# lda_train = LDA(train_X, train_y)
#
# print('Validation Error for model trained on all the data is ' + str(error(lda_full, valid_X, valid_y)))
# print('Validation Error for model trained on the training data is ' + str(error(lda_train, valid_X, valid_y)))


sigma = 5
mu_1, mu_2 = -5, 5
num_bins = 30

fig, ax = plt.subplots()

# normal_1, normal_2 = rnd.normal(mu_1, sigma, 100), rnd.normal(mu_2, sigma, 100)

np.random.seed(19680801)

x_1 = mu_1 + sigma * np.random.randn(437)
x_2 = mu_2 + sigma * np.random.randn(437)

# the histogram of the data
n_1, bins_1, patches_1 = ax.hist(x_1, num_bins, density=1, alpha=0.75)

n_2, bins_2, patches_2 = ax.hist(x_2, num_bins, density=1, alpha=0.75)

# add a 'best fit' line
y_1 = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins_1 - mu_1))**2))
y_2 = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins_2 - mu_2))**2))
ax.plot(bins_1, y_1, color='blue')
ax.plot(bins_2, y_2, color='darkorange')
ax.axvline(x=0, color='black')

plt.show()
