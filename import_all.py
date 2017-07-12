"""
This imports every profile, not just faces. Slightly larger profiles as result.

It overwrites the same data pickles to save space.
"""

import pymongo
from db_config import db_connect

from pandas.io.json import json_normalize
import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO

db = db_connect()

# Importing data
profiles = db.twitter_gender
data = profiles.find({'annotations': {'$exists': True},
                             'default_profile_image': False},
                            {'annotations':1, 'followers_count':1,
                             'friends_count':1, 'id':1, '_id':0})

print('data imported')

# Flatten & merge & clean
def clean_df(df, col_name):
    df.drop(df.filter(regex=col_name), axis=1, inplace=True)
    return df

def update_nans(df, col_name, col_name2):
    columns = [col for col in df.columns if col_name2 in col]
    print('update_nans info', col_name)
    print(columns)
    for col in columns:
        df[col_name].update(df[col])
    return df

data = json_normalize(data)
print(data.columns)
print('df flattened')
print('shape data: ', data.shape)

clean_df(data, 'socioe')

data = data.rename(index=str, columns={'annotations.thomas.image': 'image',
                                       'annotations.thomas.gender': 'gender',
                                       'annotations.thomas.age': 'age',
                                       'annotations.thomas.bot': 'bot',
                                       'annotations.thomas.face': 'face',
                                       'annotations.thomas.signal': 'signal'
                                       })
print('renamed columns')

for col in ['image', 'gender', 'age', 'bot', 'face']:
    update_nans(data, col, '.'+col)
print('updated columnns')

clean_df(data, 'chris')
clean_df(data, 'vannesa')
clean_df(data, 'thomas')

# manual cleaning
def manual_clean(df, col, incorrect, correct):
    mask = df[col] == incorrect
    df.loc[mask, col] = correct

manual_clean(data, 'age', '227', 27)
manual_clean(data, 'gender', '0', 'o')
manual_clean(data, 'gender', '%2525252525252B', '-')
manual_clean(data, 'gender', '%25252525252B', '-')
manual_clean(data, 'signal', ' image', 'image')
manual_clean(data, 'signal', 'iamge description', 'image description')
manual_clean(data, 'signal', '32', '')
manual_clean(data, 'signal', 'jandhandle name', 'handle name')
manual_clean(data, 'signal', 'twee', 'tweets')
manual_clean(data, 'signal', 'image escription', 'image description')
manual_clean(data, 'signal', 'image twwets', 'image tweets')
manual_clean(data, 'signal', 'escription', 'description')
manual_clean(data, 'signal', 'name weets', 'name tweets')
manual_clean(data, 'signal', 'image handle descripition', 'image handle description')
manual_clean(data, 'signal', 'unage', 'image')
manual_clean(data, 'signal', 'twet', 'tweets')
manual_clean(data, 'signal', 'o%2Cimage handle', 'image handle')
print('manual cleaning done')

for col in ['age', 'followers_count', 'friends_count']:
    data[col] = data[col].apply(pd.to_numeric, errors='coerce')
print('converted columns to numeric')

# Drop unannotated profiles, e.g. if no sensible info about profile was visible
data.drop(data[(data['bot']==False) & (data['gender']=='-')].index, inplace=True)

# Drop profiles with low amount of followers and 'friends', because --
# unsure if these were active on Twitter at all or dummy accounts, etc.
#data.drop(data[(data['followers_count']<100) | (data['friends_count']<100)].index, inplace=True)

print('shape data: ', data.shape)

# Create images and convert to numpy array for Tensorflow format
# Tensorflow images require shape: (nb_sample, height=150, width=150, channel=3)
def bin2array(img):
    if isinstance(img, bytes) == False:
        return np.nan
    try:
        img = BytesIO(img)
        img = Image.open(img)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = img.convert('RGB')
        img = np.asarray(img)
        return img
    except:
        return np.nan

data['image'] = data['image'].apply(bin2array)
data = data.dropna(subset=['image'])
print('images are converted')
print('shape data: ', data.shape)
print(data.columns)

data.to_pickle('data/complete.pkl')
print(data[:5])

print('shape data: ', data.shape)

# Save df as pickle in data folder
data['image'].to_pickle('data/images.pkl')
data[['gender', 'age', 'signal',
      'bot', 'followers_count',
      'friends_count', 'id']].to_pickle('data/info.pkl')
print('everything saved')
