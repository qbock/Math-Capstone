from Tree import *
import csv

# ****************** IRIS DATA *********************

# xy = []
# codes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#
# with open('./IRIS/iris.data', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(reader):
#         data = np.array(row)
#         if data[4] == 'Iris-setosa':
#             data = np.concatenate((data[:4], [0]), axis=0)
#         elif data[4] == 'Iris-versicolor':
#             data = np.concatenate((data[:4], [1]), axis=0)
#         else:
#             data = np.concatenate((data[:4], [2]), axis=0)
#         xy.append(data)
#
# xy = np.array(xy).astype('float64')
#
# tree = ClassificationTree(xy, 1, 20, 10)
#
# print_tree(tree.root)

# ****************** WINE DATA *********************

y = []
X = []

with open('./winequality-red.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    headers = next(reader, None)
    for i, row in enumerate(reader):
        data = np.array(row)
        X.append(data[:11])
        y.append(data[11:])

X = np.array(X).astype('float64')
y = np.array(y).astype('float64')

xy = np.concatenate((X, y), axis=1)

tree = RegressionTree(xy, 10, 10, 10)

print_tree(tree.root)
