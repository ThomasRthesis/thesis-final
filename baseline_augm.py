import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input

from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn import metrics

from keras import backend as K
K.set_image_dim_ordering('tf')

# Loading data
X = pd.read_pickle('data/images.pkl')
y = pd.read_pickle('data/info.pkl')

print('shape y: ', y.shape)
print('shape X: ', X.shape)
X = X.as_matrix()
X = np.stack([array for array in X])
X = preprocess_input(X.astype('float16'))
print('shape X: ', X.shape)

# load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
print('vgg16 model loaded')

datagen = ImageDataGenerator(
        rotation_range=60,
        zoom_range=0.2,
        horizontal_flip=True,
        )

batch_size = 32

predict_generator = datagen.flow(X, batch_size=batch_size, shuffle=False)

# Go over the images twice in this generator
X_features = base_model.predict_generator(predict_generator, steps=len(X)// (batch_size/2) +2, verbose=1)
print('vgg representation done')
print('shape X: ', X_features.shape)

X_shape = X_features.shape[0]
X_features = X_features.reshape(X_shape, 25088)
print('shape X_features:', X_features.shape)
print('predict X_features done')

y = y['followers_count']
y = pd.concat([y, y])
print('shape y:', y.shape)

quantiles = [0, 0.15, 0.85, 1]
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
print('shape X_test:', X_test.shape)
print('shape y_test:', y_test.shape)
print(y_val[:5])

parameters = {
    }

gs = GridSearchCV(
     GradientBoostingClassifier(n_estimators=200),
     parameters, n_jobs=-1, verbose=1,
     # seed matches others. Validation set is bigger
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
#print(metrics.confusion_matrix(y_val, y_pred_val))
