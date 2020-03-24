from sklearn import svm
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import csv
import numpy as np
import matplotlib.pyplot as plt
import math


def error_classification(clf, X, y):
    num_correct = 0
    for i in range(X.shape[0]):
        toPredict = X[i:i+1]
        pred = clf.predict(toPredict)
        # print(str(X[i:i+1]) + " predicted: " + str(pred) + " actual: " + str(y[i]) + '\n')
        if pred == y[i]:
            num_correct += 1
    return ((X.shape[0] - num_correct)/X.shape[0])*100


def error_regression(clf, X, y):
    sum = 0
    for i in range(X.shape[0]):
        toPredict = X[i:i + 1]
        pred = clf.predict(toPredict)
        # print(str(X[i:i+1]) + " predicted: " + str(pred) + " actual: " + str(y[i]) + '\n')
        sum += np.abs(pred - y[i])
    return sum/X.shape[0]

# ******************************* IRIS ************************************

IRIS_y = []
IRIS_X = []
codes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

with open('./IRIS/iris.data', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        data = np.array(row)
        IRIS_X.append(data[:4])
        if data[4] == 'Iris-setosa':
            IRIS_y.append(0)
        elif data[4] == 'Iris-versicolor':
            IRIS_y.append(1)
        else:
            IRIS_y.append(2)

IRIS_X = np.array(IRIS_X).astype('float64')

IRIS_train_X = np.concatenate((np.concatenate((IRIS_X[0:40], IRIS_X[50:90]), 0), IRIS_X[100:140]), 0)
IRIS_train_y = np.concatenate((np.concatenate((IRIS_y[0:40], IRIS_y[50:90]), 0), IRIS_y[100:140]), 0)
IRIS_valid_X = np.concatenate((np.concatenate((IRIS_X[40:50], IRIS_X[90:100]), 0), IRIS_X[140:150]), 0)
IRIS_valid_y = np.concatenate((np.concatenate((IRIS_y[40:50], IRIS_y[90:100]), 0), IRIS_y[140:150]), 0)

# ~~~~~~~~ LDA ~~~~~~~~~~
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(IRIS_valid_X, IRIS_valid_y)

# ~~~~~~~~ Tree ~~~~~~~~~~
tree_classifier = tree.DecisionTreeClassifier(random_state=0)
tree_classifier.fit(IRIS_valid_X, IRIS_valid_y)

# ~~~~~~~~ Bagging ~~~~~~~~~~
bagging_classifier = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
                                       n_estimators=10, random_state=0).fit(IRIS_valid_X, IRIS_valid_y)

# ~~~~~~~~ Random Forest ~~~~~~~~~~
RandomForest_Classifier = RandomForestClassifier(max_depth=2, random_state=0)
RandomForest_Classifier.fit(IRIS_valid_X, IRIS_valid_y)

# ~~~~~~~~ SVM ~~~~~~~~~~
svm_classifier = svm.SVC()
svm_classifier.fit(IRIS_valid_X, IRIS_valid_y)

print('IRIS Validation Error lda: ' + str(error_classification(lda_classifier, IRIS_train_X, IRIS_train_y)))
print('IRIS Validation Error svm: ' + str(error_classification(svm_classifier, IRIS_train_X, IRIS_train_y)))
print('IRIS Validation Error tree: ' + str(error_classification(tree_classifier, IRIS_train_X, IRIS_train_y)))
print('IRIS Validation Error bagging: ' + str(error_classification(bagging_classifier, IRIS_train_X, IRIS_train_y)))
print('IRIS Validation Error random forest: ' + str(error_classification(RandomForest_Classifier, IRIS_train_X, IRIS_train_y)))


# ******************************* WINE ************************************

y = []
X = []

with open('./winequality-red.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    headers = next(reader, None)
    for i, row in enumerate(reader):
        data = np.array(row)
        X.append(data[:11])
        y.append(data[11])

Wine_X = np.array(X).astype('float64')
Wine_y = np.array(y).astype('float64')

Wine_train_X = Wine_X[0:1280]
Wine_train_y = Wine_y[0:1280]
Wine_valid_X = Wine_X[1280:1599]
Wine_valid_y = Wine_y[1280:1599]

# ~~~~~~~~ Tree ~~~~~~~~~~
tree_classifier = tree.DecisionTreeRegressor(max_depth=5, random_state=0)
tree_classifier.fit(Wine_train_X, Wine_train_y)

# ~~~~~~~~ Random Forest ~~~~~~~~~~
RandomForest_Classifier = RandomForestRegressor(max_depth=4, random_state=0)
RandomForest_Classifier.fit(Wine_train_X, Wine_train_y)

print('IRIS Validation Error tree: ' + str(error_regression(tree_classifier, Wine_valid_X, Wine_valid_y)))
print('IRIS Validation Error random forest: ' + str(error_regression(RandomForest_Classifier, Wine_valid_X, Wine_valid_y)))

# ******************************* Banknote ************************************

y = []
X = []

with open('./data_banknote_authentication.txt', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    headers = next(reader, None)
    for i, row in enumerate(reader):
        data = np.array(row)
        X.append(data[:4])
        y.append(data[4])

BN_X = np.array(X).astype('float64')
BN_y = np.array(y).astype('float64')

BN_train_X = BN_X[0:1098]
BN_train_y = BN_y[0:1098]
BN_valid_X = BN_X[1098:1371]
BN_valid_y = BN_y[1098:1371]

# ~~~~~~~~ LDA ~~~~~~~~~~
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(BN_train_X, BN_train_y)

# ~~~~~~~~ Tree ~~~~~~~~~~
tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(BN_train_X, BN_train_y)

# ~~~~~~~~ Bagging ~~~~~~~~~~
bagging_classifier = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
                                       n_estimators=100, random_state=0).fit(BN_train_X, BN_train_y)

# ~~~~~~~~ Random Forest ~~~~~~~~~~
RandomForest_Classifier = RandomForestClassifier(max_depth=5, random_state=0)
RandomForest_Classifier.fit(BN_train_X, BN_train_y)

# ~~~~~~~~ SVM ~~~~~~~~~~
svm_classifier = svm.SVC()
svm_classifier.fit(BN_train_X, BN_train_y)

print('IRIS Validation Error lda: ' + str(error_classification(lda_classifier, BN_valid_X, BN_valid_y)))
print('IRIS Validation Error svm: ' + str(error_classification(svm_classifier, BN_valid_X, BN_valid_y)))
print('IRIS Validation Error tree: ' + str(error_classification(tree_classifier, BN_valid_X, BN_valid_y)))
print('IRIS Validation Error bagging: ' + str(error_classification(bagging_classifier, BN_valid_X, BN_valid_y)))
print('IRIS Validation Error random forest: ' + str(error_classification(RandomForest_Classifier, BN_valid_X, BN_valid_y)))

# ******************************* Cancer ************************************

Cancer_y = []
Cancer_X = []
codes = ['no-recurrence-events', 'recurrence-events']

with open('./breast-cancer.data', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        data = np.array(row)
        toAppend = []

        # take care of y
        if data[0] == codes[0]:
            Cancer_y.append(0)
        else:
            Cancer_y.append(1)

        # take care of x
        if data[1] == '10-19':
            toAppend.append(0)
        elif data[1] == '20-29':
            toAppend.append(1)
        elif data[1] == '30-39':
            toAppend.append(2)
        elif data[1] == '40-49':
            toAppend.append(3)
        elif data[1] == '50-59':
            toAppend.append(4)
        elif data[1] == '60-69':
            toAppend.append(5)
        elif data[1] == '70-79':
            toAppend.append(6)
        elif data[1] == '80-89':
            toAppend.append(7)
        else:
            toAppend.append(8)

        if data[2] == 'lt40':
            toAppend.append(0)
        elif data[2] == 'ge40':
            toAppend.append(1)
        else:
            toAppend.append(2)

        if data[3] == '0-4':
            toAppend.append(0)
        elif data[3] == '5-9':
            toAppend.append(1)
        elif data[3] == '10-14':
            toAppend.append(2)
        elif data[3] == '15-19':
            toAppend.append(3)
        elif data[3] == '20-24':
            toAppend.append(4)
        elif data[3] == '25-29':
            toAppend.append(5)
        elif data[3] == '30-34':
            toAppend.append(6)
        elif data[3] == '35-39':
            toAppend.append(7)
        elif data[3] == '40-44':
            toAppend.append(8)
        elif data[3] == '45-49':
            toAppend.append(9)
        elif data[3] == '50-54':
            toAppend.append(10)
        else:
            toAppend.append(11)

        if data[4] == '0-2':
            toAppend.append(0)
        elif data[4] == '3-5':
            toAppend.append(1)
        elif data[4] == '6-8':
            toAppend.append(2)
        elif data[4] == '9-11':
            toAppend.append(3)
        elif data[4] == '12-14':
            toAppend.append(4)
        elif data[4] == '15-17':
            toAppend.append(5)
        elif data[4] == '18-20':
            toAppend.append(6)
        elif data[4] == '21-23':
            toAppend.append(7)
        elif data[4] == '24-26':
            toAppend.append(8)
        elif data[4] == '27-29':
            toAppend.append(9)
        elif data[4] == '30-32':
            toAppend.append(10)
        elif data[4] == '33-35':
            toAppend.append(11)
        else:
            toAppend.append(12)

        if data[5] == 'yes':
            toAppend.append(0)
        else:
            toAppend.append(1)

        toAppend.append(int(data[6]))

        if data[7] == 'right':
            toAppend.append(0)
        else:
            toAppend.append(1)

        if data[8] == 'left-up':
            toAppend.append(0)
        elif data[8] == 'left-low':
            toAppend.append(1)
        elif data[8] == 'right-up':
            toAppend.append(2)
        elif data[8] == 'right-low':
            toAppend.append(3)
        else:
            toAppend.append(4)

        if data[9] == 'yes':
            toAppend.append(0)
        else:
            toAppend.append(1)
        Cancer_X.append(toAppend)


Cancer_X = np.array(Cancer_X).astype('float64')
Cancer_y = np.array(Cancer_y).astype('float64')

Cancer_train_X = np.concatenate((Cancer_X[0:160], Cancer_X[201:269]), 0)
Cancer_train_y = np.concatenate((Cancer_y[0:160], Cancer_y[201:269]), 0)
Cancer_valid_X = np.concatenate((Cancer_X[160:201], Cancer_X[269:285]), 0)
Cancer_valid_y = np.concatenate((Cancer_y[160:201], Cancer_y[269:285]), 0)

# ~~~~~~~~ LDA ~~~~~~~~~~
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(Cancer_train_X, Cancer_train_y)

# ~~~~~~~~ Tree ~~~~~~~~~~
tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(Cancer_train_X, Cancer_train_y)

# path = tree_classifier.cost_complexity_pruning_path(Cancer_train_X, Cancer_train_y)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
# clfs = []
# for ccp_alpha in ccp_alphas:
#     clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
#     clf.fit(Cancer_train_X, Cancer_train_y)
#     clfs.append(clf)
#
# train_scores = [clf.score(Cancer_train_X, Cancer_train_y) for clf in clfs]
# test_scores = [clf.score(Cancer_valid_X, Cancer_valid_y) for clf in clfs]

# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train",
#         drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker='o', label="test",
#         drawstyle="steps-post")
# ax.legend()
# plt.show()


# lowest = math.inf
# place = 0
# for i, score in enumerate(test_scores):
#     if score < lowest:
#         lowest = score
#         place = i
#
# print(place)
# print(lowest)
# print(ccp_alphas[place])

# ~~~~~~~~ Bagging ~~~~~~~~~~
bagging_classifier = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1),
                                       n_estimators=100, random_state=0).fit(Cancer_train_X, Cancer_train_y)

# ~~~~~~~~ Random Forest ~~~~~~~~~~
RandomForest_Classifier = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=0)
RandomForest_Classifier.fit(Cancer_train_X, Cancer_train_y)

# ~~~~~~~~ SVM ~~~~~~~~~~
svm_classifier = svm.SVC()
svm_classifier.fit(Cancer_train_X, Cancer_train_y)

print('IRIS Validation Error lda: ' + str(error_classification(lda_classifier, Cancer_valid_X, Cancer_valid_y)))
print('IRIS Validation Error svm: ' + str(error_classification(svm_classifier, Cancer_valid_X, Cancer_valid_y)))
print('IRIS Validation Error tree: ' + str(error_classification(tree_classifier, Cancer_valid_X, Cancer_valid_y)))
print('IRIS Validation Error bagging: ' + str(error_classification(bagging_classifier, Cancer_valid_X, Cancer_valid_y)))
print('IRIS Validation Error random forest: ' + str(error_classification(RandomForest_Classifier, Cancer_valid_X, Cancer_valid_y)))
