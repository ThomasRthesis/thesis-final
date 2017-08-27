import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn import metrics

from keras import backend as K
K.set_image_dim_ordering('tf')

# Loading data
X = pd.read_pickle('data/images.pkl')
y = pd.read_pickle('data/info.pkl')
y = y['followers_count']
"""
y2 = pd.read_pickle('data/gender2_complete.pkl')
X2 = y2['image']
y2 = y2['followers_count']

y3 = pd.read_pickle('data/gender_complete.pkl')
X3 = y3['image']
y3 = y3['followers_count']

y4 = pd.read_pickle('data/volkova_complete.pkl')
X4 = y4['image']
y4 = y4['followers_count']

X = pd.concat([X, X2, X3, X4])
y = pd.concat([y, y2, y3, y4])
"""
print('shape y: ', y.shape)
print('shape X: ', X.shape)
X = X.as_matrix()
X = np.stack([array for array in X])
X = preprocess_input(X.astype('float16'))
print('shape X: ', X.shape)

# load VGG16
base_model = VGG16(weights='imagenet', include_top=True, input_tensor=Input(shape=(224,224,3)))
print('vgg16 model loaded')

datagen = ImageDataGenerator(
        rotation_range=60,
        zoom_range=0.2,
        horizontal_flip=True,
        )

batch_size = 32

predict_generator = datagen.flow(X, batch_size=batch_size, shuffle=False)

# Go over the images twice in this generator
X_features = base_model.predict(X)
#X_features = base_model.predict_generator(predict_generator, steps=len(X)// (batch_size/2) +2, verbose=1)
print('vgg features done')
print('shape X: ', X_features.shape)

#X_shape = X_features.shape[0]
#X_features = X_features.reshape(X_shape, 25088)
#print('shape X_features:', X_features.shape)
#print('predict X_features done')

# Preprocessing age & gender
gender = pd.get_dummies(y['gender']).as_matrix()
print('shape dummies:', gender.shape)
X_features = np.hstack((X_features, gender))
print('shape after adding gender:', X.shape)

age = pd.to_numeric(y['age'], errors='coerce')
age.fillna(age.mean(), inplace=True)
age_bins = [0, 13, 20, 37, 66, 100]
age = pd.cut(age, age_bins, labels=False)
print(age[:5])
print('dummies age:', np.bincount(age))
age = pd.get_dummies(age).as_matrix()

X_features = np.hstack((X_features, age))
print('shape after adding age:', X.shape)
print('shape y:', y.shape)

quantiles = [x/100 for x in range(0, 101, 5)]
num_classes = len(quantiles)-1
bins = y.quantile(q=quantiles)
bins = list(bins)
bins[0] -= 1 #otherwise it will not pick up the lowest number
y = pd.cut(y, bins, labels=False)
print('shape y: ' , y.shape)
print('bins:', bins)
print('bincount:', np.bincount(y))

# Test 80%, val 15%, test 5%
X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.25, random_state=1)

print('shape X_train:', X_train.shape)
print('shape y_train:', y_train.shape)
print('shape X_val:', X_val.shape)
print('shape y_val:', y_val.shape)
print('bincount y_val:', np.bincount(y_val))
print('shape X_test:', X_test.shape)
print('shape y_test:', y_test.shape)
print('bincount y_test:', np.bincount(y_test))
print(y_val[:5])

parameters = {
#    'kernel': ('linear', 'sigmoid', 'poly'),
#    'decision_function_shape': ('ovr', 'ovo'),
#     'degree': (1, 3, 5, 10),
#    'C': (0.5, 1, 1.5, 2),
    }

gs = GridSearchCV(
     SVC(decision_function_shape='ovr', kernel='linear', C=1.5, class_weight='balanced'),
     parameters, n_jobs=-1, verbose=1,
     # seed matches others. Validation set is bigger
     cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=1))

start = time.time()
print('fitting....')
gs = gs.fit(X_train, y_train)
end = time.time()
print('training time', end - start)

print("Best score:", gs.best_score_)
print()
for parameter in sorted(parameters.keys()):
      print("%s: %r" % (parameter, gs.best_params_[parameter]))

# Predict & evaluate
y_pred_val = gs.predict(X_val)
start = time.time()
y_pred_test = gs.predict(X_test)
end = time.time()

fs = open('info.txt', 'w')
fs.write('Acc val: ' + str(metrics.accuracy_score(y_val, y_pred_val)) + '\n')
fs.write('MAE val:  ' + str(metrics.mean_absolute_error(y_val, y_pred_val)) + '\n')
fs.write('CF val:  ' + str(metrics.confusion_matrix(y_val, y_pred_val)) + '\n')
fs.write('Acc test: ' + str(metrics.accuracy_score(y_test, y_pred_test)) + '\n')
fs.write('MAE test:  ' + str(metrics.mean_absolute_error(y_test, y_pred_test)) + '\n')
fs.write('CF test:  ' + str(metrics.confusion_matrix(y_test, y_pred_test)) + '\n')
fs.close()
