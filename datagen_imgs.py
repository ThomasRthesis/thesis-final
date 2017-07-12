import os

import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Keeping the directory clean(!)
path = 'imgs/'
imgs = [ img for img in os.listdir(path+'.') if img.endswith(".jpeg") ]
for img in imgs:
    os.remove(path+img)

# Needed to load all images, then select some
# It loads as array
X = pd.read_pickle('data/images.pkl')
X = X.as_matrix()

img_list = [x for x in range(5, 71, 5)]
for num in img_list:
    img = X[num]
    img = Image.fromarray(img)
    img.save('imgs/img_'+str(num)+'.png')

X = X[5]
X = X.reshape((1,) + X.shape)

datagen = ImageDataGenerator(
        rotation_range=60,
        zoom_range=0.2,
        horizontal_flip=True,
)

# Break after 20 images, so it will not loop forever
i = 0
for batch in datagen.flow(X, batch_size=1,
                          save_to_dir='imgs', save_prefix='img1', save_format='jpeg'):
    i += 1
    if i > 20:
        break

