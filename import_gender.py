import pymongo

from PIL import Image
import requests
from io import BytesIO

from pandas.io.json import json_normalize
import pandas as pd
import numpy as np

from db_config import db_connect

db = db_connect()

profiles = db.twitter_gender
data = profiles.find({'default_profile_image':False,
                      'annotations': {'$exists':False}
                      },
                     {'profile_image_url_https':1,
                      'friends_count':1,
                      'followers_count':1,
                      'id_str':1,
                      'noisy_gender':1,
                      '_id':0})
print('data imported')

data = json_normalize(list(data))
print(data.columns)
print('df flattened')
print('shape data:', data.shape)

data = data.rename(columns={'noisy_gender': 'gender',
                            'profile_image_url_https': 'urls',
                            'id_str': 'id'})

# Separate urls and request normal image size
data['urls'].replace(regex=True, inplace=True, to_replace=r'_normal', value=r'')

# New df for images
images = data[['urls', 'id']]
images = data.rename(columns={'urls':'image'})
for col in images.columns:
    if col != 'image':
        images.drop(col, axis=1, inplace=True)

count_success = 0
count_fail = 0

for num, url in enumerate(data['urls']):
    # In case url fails
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224), Image.ANTIALIAS) # size pretrained vgg16
        img = img.convert('RGB')
        img = np.asarray(img)
        images['image'][num] = img
        count_success += 1
    except:
        images['image'][num] = np.nan
        count_fail += 1
    if (count_success+count_fail) % 10 == 0:
        print(count_success, ' + ', count_fail, ' = ' , count_success + count_fail)

print('succes:', count_success)
print('fail:', count_fail)

data.set_index(data['id'].values, inplace=True)
images.set_index(data['id'].values, inplace=True)
print(images.columns)
data['image'] = images
print(data.columns)
data.dropna(axis=0, subset=['image'], inplace=True)
print('shape data:', data.shape)

data['image'].to_pickle('data/gender_images.pkl')
data.to_pickle('data/gender_complete.pkl')
print('everything saved')
