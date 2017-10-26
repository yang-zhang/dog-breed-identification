
# coding: utf-8

# [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

# In[60]:

from secrets import KAGGLE_USER, KAGGLE_PW
import utils_ds

import pandas as pd
import os


# In[89]:

competition_name = 'dog-breed-identification'

dir_data = '/opt/notebooks/data/' + competition_name


# ### download

# In[5]:

get_ipython().system('mkdir -p $dir_data/raw ')


# In[6]:

get_ipython().magic('cd $dir_data/raw')


# In[7]:

get_ipython().system('kg config -g -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')
get_ipython().system('kg download')


# ### unzip

# In[13]:

get_ipython().system('mkdir $dir_data/preprocessed')


# In[14]:

get_ipython().magic('cp $dir_data/raw/*.zip $dir_data/preprocessed/')


# In[ ]:

get_ipython().system("unzip '$dir_data/preprocessed/*.zip' -d $dir_data/preprocessed")


# In[16]:

get_ipython().magic('ls $dir_data/preprocessed/train/ -l | wc -l')
get_ipython().magic('ls $dir_data/preprocessed/test/ -l | wc -l')


# In[15]:

rm $dir_data/preprocessed/*.zip


# In[20]:

ls $dir_data/preprocessed/


# ### move samples to validation folder

# In[30]:

get_ipython().system('mkdir $dir_data/preprocessed/valid ')


# In[32]:

utils_ds.move_sample(
    dir_source=dir_data + '/preprocessed/train',
    dir_destin=dir_data + '/preprocessed/valid',
    file_type='jpg',
    n=2000)


# In[33]:

get_ipython().magic('ls $dir_data/preprocessed/valid/ -l | wc -l')


# ### copy samples to sample folder

# In[34]:

get_ipython().system('mkdir $dir_data/sample')


# In[36]:

get_ipython().system('mkdir $dir_data/sample/train $dir_data/sample/test $dir_data/sample/valid')


# In[37]:

utils_ds.copy_sample(
    dir_source=dir_data + '/preprocessed/train',
    dir_destin=dir_data + '/sample/train',
    file_type='jpg',
    n=200)


# In[38]:

utils_ds.copy_sample(
    dir_source=dir_data + '/preprocessed/test',
    dir_destin=dir_data + '/sample/test',
    file_type='jpg',
    n=200)


# In[39]:

utils_ds.copy_sample(
    dir_source=dir_data + '/preprocessed/valid',
    dir_destin=dir_data + '/sample/valid',
    file_type='jpg',
    n=50)


# In[88]:

get_ipython().magic('ls $dir_data/sample/train/ -l | wc -l')
get_ipython().magic('ls $dir_data/sample/test/ -l | wc -l')
get_ipython().magic('ls $dir_data/sample/valid/ -l | wc -l')


# ### make results dirs

# In[41]:

mkdir $dir_data/preprocessed/results


# In[42]:

mkdir $dir_data/sample/results


# ### rearrange data into labeled folders

# In[43]:

labels = pd.read_csv(dir_data+'/preprocessed/labels.csv')


# In[44]:

breeds = set(labels.breed)


# #### train

# In[61]:

for breed in breeds:
    os.mkdir(dir_data+'/preprocessed/train/'+breed)


# In[81]:

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

# In[82]:

for breed in breeds:
    os.mkdir(dir_data+'/preprocessed/valid/'+breed)


# In[83]:

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

# In[84]:

for breed in breeds:
    os.mkdir(dir_data+'/sample/train/'+breed)


# In[85]:

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

# In[86]:

for breed in breeds:
    os.mkdir(dir_data+'/sample/valid/'+breed)


# In[87]:

for row in labels.iterrows():
    id_=row[1]['id']
    breed=row[1]['breed']
    try:
        os.rename(dir_data+'/sample/valid/%s.jpg' % id_, 
              dir_data+'/sample/valid/%s/%s.jpg' % (breed, id_))
    # pic is in train folder
    except OSError: 
        pass

