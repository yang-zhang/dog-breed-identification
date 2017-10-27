
# coding: utf-8

# [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

# In[45]:

from secrets import KAGGLE_USER, KAGGLE_PW
import utils_ds

import pandas as pd
import numpy as np
import os

from IPython.display import Image


# In[84]:

competition_name = 'dog-breed-identification'

dir_data = '/opt/notebooks/data/' + competition_name


# ### download

# In[9]:

get_ipython().system('mkdir -p $dir_data/raw ')


# In[10]:

get_ipython().magic('cd $dir_data/raw')


# In[11]:

get_ipython().system('kg config -g -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')
get_ipython().system('kg download')


# ### unzip

# In[86]:

get_ipython().system('mkdir $dir_data/preprocessed')


# In[12]:

get_ipython().magic('cp $dir_data/raw/*.zip $dir_data/preprocessed/')


# In[13]:

get_ipython().system("unzip '$dir_data/preprocessed/*.zip' -d $dir_data/preprocessed")


# In[14]:

get_ipython().magic('ls $dir_data/preprocessed/train/ -l | wc -l')
get_ipython().magic('ls $dir_data/preprocessed/test/ -l | wc -l')


# In[78]:

get_ipython().system('mkdir $dir_data/preprocessed/test/unknown')


# In[96]:

mv $dir_data/preprocessed/test/*.jpg $dir_data/preprocessed/test/unknown


# ### move samples to validation folder

# In[17]:

get_ipython().system('mkdir $dir_data/preprocessed/valid ')


# In[18]:

utils_ds.move_sample(
    dir_source=dir_data + '/preprocessed/train',
    dir_destin=dir_data + '/preprocessed/valid',
    file_type='jpg',
    n=2000)


# In[19]:

get_ipython().magic('ls $dir_data/preprocessed/valid/ -l | wc -l')


# ### copy samples to sample folder

# In[20]:

get_ipython().system('mkdir $dir_data/sample')


# In[21]:

get_ipython().system('mkdir $dir_data/sample/train $dir_data/sample/test $dir_data/sample/valid')


# In[97]:

get_ipython().system('mkdir $dir_data/sample/test/unknown')


# In[100]:

mv $dir_data/sample/test/*.jpg $dir_data/sample/test/unknown


# In[22]:

utils_ds.copy_sample(
    dir_source=dir_data + '/preprocessed/train',
    dir_destin=dir_data + '/sample/train',
    file_type='jpg',
    n=1200)


# In[23]:

utils_ds.copy_sample(
    dir_source=dir_data + '/preprocessed/test/unknown',
    dir_destin=dir_data + '/sample/test/unknown',
    file_type='jpg',
    n=1200)


# In[24]:

utils_ds.copy_sample(
    dir_source=dir_data + '/preprocessed/valid',
    dir_destin=dir_data + '/sample/valid',
    file_type='jpg',
    n=1000)


# In[25]:

get_ipython().magic('ls $dir_data/sample/train/ -l | wc -l')
get_ipython().magic('ls $dir_data/sample/test/ -l | wc -l')
get_ipython().magic('ls $dir_data/sample/valid/ -l | wc -l')


# ### make results dirs

# In[26]:

mkdir $dir_data/preprocessed/results


# In[27]:

mkdir $dir_data/sample/results


# ### rearrange data into labeled folders

# In[28]:

labels = pd.read_csv(dir_data+'/preprocessed/labels.csv')


# In[29]:

breeds = set(labels.breed)


# #### train

# In[30]:

for breed in breeds:
    os.mkdir(dir_data+'/preprocessed/train/'+breed)


# In[31]:

for row in labels.iterrows():
    id_=row[1]['id']
    breed=row[1]['breed']
    try:
        os.rename(dir_data+'/preprocessed/train/%s.jpg' % id_, 
              dir_data+'/preprocessed/train/%s/%s.jpg' % (breed, id_))
    # pic is in valid folder
    except OSError: 
        pass


# #### valid

# In[70]:

for breed in breeds:
    os.mkdir(dir_data+'/preprocessed/valid/'+breed)


# In[71]:

for row in labels.iterrows():
    id_=row[1]['id']
    breed=row[1]['breed']
    try:
        os.rename(dir_data+'/preprocessed/valid/%s.jpg' % id_, 
              dir_data+'/preprocessed/valid/%s/%s.jpg' % (breed, id_))
    # pic is in train folder
    except OSError: 
        pass


# #### train in sample

# In[34]:

for breed in breeds:
    os.mkdir(dir_data+'/sample/train/'+breed)


# In[35]:

for row in labels.iterrows():
    id_=row[1]['id']
    breed=row[1]['breed']
    try:
        os.rename(dir_data+'/sample/train/%s.jpg' % id_, 
              dir_data+'/sample/train/%s/%s.jpg' % (breed, id_))
    # pic is in valid folder
    except OSError: 
        pass


# #### valid in sample

# In[36]:

for breed in breeds:
    os.mkdir(dir_data+'/sample/valid/'+breed)


# In[37]:

for row in labels.iterrows():
    id_=row[1]['id']
    breed=row[1]['breed']
    try:
        os.rename(dir_data+'/sample/valid/%s.jpg' % id_, 
              dir_data+'/sample/valid/%s/%s.jpg' % (breed, id_))
    # pic is in train folder
    except OSError: 
        pass


# ### Verify

# In[76]:

breed = np.random.choice(list(breeds))
imgs = os.listdir(dir_data+'/preprocessed/valid/'+breed)
img = np.random.choice(imgs)
print(breed)
Image(dir_data+'/preprocessed/valid/%s/%s' % (breed, img))


# In[ ]:



