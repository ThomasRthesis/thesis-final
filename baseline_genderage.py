import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn import metrics

# Loading data
X = np.load('data/images_vgg16.pkl')
y = np.load('data/info.pkl')

X = X[:6113]
y = y[:6113]
X = X.reshape(X.shape[0], 25088)
print('shape X: ', X.shape)
print('shape y: ', y.shape)

# Preprocessing age & gender
gender = pd.get_dummies(y['gender'])
print('shape shape dummies:', gender.shape)
X = np.hstack((X, gender))
print('shape after adding gender:', X.shape)

age = pd.to_numeric(y['age'], errors='coerce')
age.fillna(age.mean(), inplace=True)
age_bins = [0, 13, 20, 37, 66, 100]
age = pd.cut(age, age_bins, labels=False)
print(age[:5])
print('dummies age:', np.bincount(age))
age = pd.get_dummies(age)

X = np.hstack((X, age))
print('shape after adding age:', X.shape)

y = y['followers_count']
quantiles = [0, 0.15, 0.85, 1]
num_classes = len(quantiles)-1
bins = y.quantile(q=quantiles)
bins = list(bins)
bins[0] -= 1 #otherwise it will not pick up the lowest number
y = pd.cut(y, bins, labels=False).as_matrix()
print('shape y: ' , y.shape)
print('bins:', bins)
print('bincount:', np.bincount(y))

# Test 80%, val 15%, test 5%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.25, random_state=1)

print('shape X_train:', X_train.shape)
print('shape y_train:', y_train.shape)
print('shape X_val:', X_val.shape)
print('shape y_val:', y_val.shape)
print('shape X_test:', X_test.shape)
print('shape y_test:', y_test.shape)
print(y_val[:5])

parameters = {
    }

gs = GridSearchCV(
     GradientBoostingClassifier(n_estimators=200, verbose=1),

     parameters, n_jobs=-1, verbose=1,
     # Seed matches others. Validation set is bigger
     cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=1))
print('fitting....')
gs = gs.fit(X_train, y_train)

print("Best score:", gs.best_score_)
print()
for parameter in sorted(parameters.keys()):
      print("%s: %r" % (parameter, gs.best_params_[parameter]))

# Predict & evaluate
y_pred_val = gs.predict(X_val)
y_pred_test = gs.predict(X_test)
print("Accuracy score on validation set:", metrics.accuracy_score(y_val, y_pred_val), end="\n\n")
print("Accuracy score on test set:", metrics.accuracy_score(y_test, y_pred_test), end="\n\n")
#print(metrics.classification_report(y_val, y_pred_val), end="\n\n")
print(metrics.confusion_matrix(y_val, y_pred_val))
