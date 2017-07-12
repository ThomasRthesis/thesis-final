import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical
from sklearn.utils import class_weight

from keras import applications
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from sklearn import metrics

from keras import backend as K
K.set_image_dim_ordering('tf')

from sklearn.model_selection import train_test_split

import gc

# Loading data
X = pd.read_pickle('data/images_stacked.pkl')
y = pd.read_pickle('data/info_stacked.pkl')
print('loaded pickles')

#X = X.as_matrix()
#X = np.stack([array for array in X])
#X = X.astype('float32')
#X = preprocess_input(X)
#print('preprocessing done')

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
y = y.tolist()
y = to_categorical(y, num_classes)
print(y[:5])

# Compute class weights
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_class), y_class)
class_weight = {i:j for i,j in zip(np.unique(y_class),class_weight)}

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
print('shape y_test:', y_test)

# Build VGG16
model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
print('vgg16 model loaded')

# Bottleneck features block
top_model = Sequential()
#top_model.add(Flatten(input_shape=X_train.shape[1:]))
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(512))
top_model.add(Activation('relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes))
top_model.add(Activation('softmax'))

# Load bottleneck features weights, otherwise expected performance to be worse
top_model.load_weights('weights/best_bottleneck_features.h5') # weights from checkpoint

model = Model(input=model.input, output=top_model(model.output))

# Freeze layers from VGG16 network
for layer in model.layers[:25]:
    layer.trainable = False

def top_k(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

# Slow training
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['mae', top_k])

batch_size = 32

# Checkpoint saving & early stopping
early_stop = EarlyStopping(monitor='val_top_k', min_delta=0, patience=10)
fp_checkpoint = 'weights/best_fine-tuned.h5'
checkpoint = ModelCheckpoint(fp_checkpoint, monitor='val_top_k', verbose=1, save_best_only=True, mode='max')

# Training
model.fit(X_train, y_train,
        batch_size=batch_size,
        class_weight=class_weight,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint])

model.save_weights('weights/fine-tuned.h5')

# Predict & evaluate
y_pred_val = model.predict(X_val, batch_size=batch_size, verbose=1)
y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=1)

y_val = np.argmax(y_val, axis=1)
y_test= np.argmax(y_test, axis=1)
y_pred_val = np.argmax(y_pred_val, axis=1)
y_pred_test = np.argmax(y_pred_test, axis=1)

print('Accuracy score on validation set:', metrics.accuracy_score(y_val, y_pred_val))
print('Accuracy score on test set:', metrics.accuracy_score(y_test, y_pred_test))
print(metrics.confusion_matrix(y_val, y_pred_val))
