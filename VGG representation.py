import pandas as pd
import numpy as np

from keras.layers import Input
from keras import applications
from keras.applications.imagenet_utils import preprocess_input

import pickle

# Loading pickles
X = pd.read_pickle('data/images.pkl')
y = pd.read_pickle('data/info.pkl')
y = y['followers_count']
print(X.shape)

D2 = pd.read_pickle('data/gender2_complete.pkl')
X2 = D2['image']
y2 = D2['followers_count']
print(X2.shape)

D3 = pd.read_pickle('data/gender_complete.pkl')
X3 = D3['image']
y3 = D3['followers_count']
print(X3.shape)

D4 = pd.read_pickle('data/volkova_complete.pkl')
X4 = D4['image']
y4 = D4['followers_count']
print(X4.shape)

X = pd.concat([X, X2, X3, X4])
y = pd.concat([y, y2, y3, y4])
print('loaded and merged pickles')

X = X.as_matrix()
X = np.stack([array for array in X])
X = X.astype('float16')
print('Shape X: ', X.shape)

X = preprocess_input(X)
print('preprocessing done')

pickle.dump(X, open('data/images_stacked.pkl', 'wb'), protocol=4)

# VGG16
model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
print('VGG16 model loaded')

X = model.predict(X, verbose=1)
print('Predicting done')

# Saving
X.dump('data/images_vgg16.pkl')
y.to_pickle('data/info_stacked.pkl')
print('pickles saved')
