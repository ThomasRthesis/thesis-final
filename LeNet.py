"""
Simple CNN loading each different dataset & augmentation
"""

import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy

from keras import backend as K
K.set_image_dim_ordering('tf')

from sklearn.model_selection import train_test_split

# Loading & stacking dataframes
X = pd.read_pickle('data/images.pkl')
y = pd.read_pickle('data/info.pkl')

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
X = X.as_matrix()
X = np.stack([array for array in X])

#y = y['followers_count']
y = pd.concat([y, y2, y3, y4])
print('shape X: ', X.shape)
print('shape y: ', y.shape)

quantiles = [0, 0.15, 0.85, 1]
num_classes = len(quantiles)-1

bins = y.quantile(q=quantiles)
bins = list(bins)
bins[0] -= 1 #otherwise it will not pick up the lowest number
y = pd.cut(y, bins, labels=False).as_matrix()
y_class = y
print('shape y: ' , y.shape)
print('bins:', bins)
print('bincount:', np.bincount(y))

# Hot Encoding
y = y.tolist()
y = to_categorical(y, num_classes)
print(y[:5])

# Computer class weights
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_class), y_class)
class_weight = {i:j for i,j in zip(np.unique(y_class), class_weight)}

# Test 80%, val 15%, test 5%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.25, random_state=1)

del X
del y

print('shape X_train:', X_train.shape)
print('shape y_train:', y_train.shape)
print('shape X_val:', X_val.shape)
print('shape y_val:', y_val.shape)
print('shape X_test:', X_test.shape)
print('shape y_test:', y_test.shape)

# Building the model
# Image preprocessing
datagen = ImageDataGenerator(
        featurewise_center=True
)

# Building the network
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

def top_k(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mae', top_k])

batch_size = 16

# Checkpoint saving & early stopping
early_stop = EarlyStopping(monitor='val_top_k', min_delta=0, patience=10, mode='max', verbose=1)
fp_checkpoint = 'weights/best_simple_cnn.h5'
checkpoint = ModelCheckpoint(fp_checkpoint, monitor='val_top_k', verbose=1,
             mode='max', save_best_only=True)

# Fit zero-center
datagen.fit(X_train)

generator = datagen.flow(X_train, y_train, seed=1, batch_size=batch_size)

# Training
model.fit_generator(
        generator,
        steps_per_epoch=len(X_train) // (batch_size/1) +1, # +number to finish each round over the data
	    class_weight=class_weight,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint])

model.save_weights('weights/final_weights_simple_cnn.h5')

# Predict & evaluate
y_pred_val = model.predict(X_val, batch_size=batch_size, verbose=1)
y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=1)

y_val = np.argmax(y_val, axis=1)
y_test= np.argmax(y_test, axis=1)
y_pred_val = np.argmax(y_pred_val, axis=1)
y_pred_test = np.argmax(y_pred_test, axis=1)

print('Accuracy score on validation set:', metrics.accuracy_score(y_val, y_pred_val))
print('Accuracy score on test set:', metrics.accuracy_score(y_test, y_pred_test))
