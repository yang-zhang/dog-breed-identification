
# coding: utf-8

# In[52]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import display
from IPython.display import Image

import os
from os import listdir
from datetime import datetime

from keras.preprocessing import image

from keras.applications import vgg16 
from keras.applications import xception
from keras.applications import inception_v3

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

from secrets import KAGGLE_USER, KAGGLE_PW


# ## Setup

# In[2]:

seed = 2014
batch_size = 32


# In[29]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/all_train'
raw_dir = '/opt/notebooks/data/' + competition_name + '/raw'

!mkdir $data_dir
!cp $raw_dir/*.zip $data_dir

!ls $data_dir

!unzip $data_dir/*.zip -d $data_dir

!rm $data_dir/*.zip

labels = pd.read_csv(data_dir+'/labels.csv')
breeds = set(labels.breed)

for breed in breeds:
    os.mkdir(data_dir+'/train/'+breed)
for row in labels.iterrows():
    id_=row[1]['id']
    breed=row[1]['breed']
    os.rename(data_dir+'/train/%s.jpg' % id_, 
          data_dir+'/train/%s/%s.jpg' % (breed, id_))

!mkdir $data_dir/test/unknown

mv $data_dir/test/*.jpg $data_dir/test/unknown
# In[65]:

get_ipython().system('mkdir $data_dir/results')


# In[56]:

gen = image.ImageDataGenerator()


# In[57]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, batch_size=batch_size)


# In[58]:

y_encode = batches.classes


# In[62]:

def preprocess_batches(batches, mdl):
    while True:
        try:
            batch = batches.next()
            imgs = batch[0]
            imgs = np.apply_along_axis(mdl.preprocess_input, 0, imgs)
            yield batch
        except StopIteration:
            break


# ## Xception

# ### Extract Xception bottleneck features

# In[63]:

batches = gen.flow_from_directory(data_dir+'/train', target_size=(299, 299), shuffle=False)
batches_preprocessed = preprocess_batches(batches, xception)


# In[ ]:

xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

nb_batches = math.ceil(batches.n/batch_size)
bf_x = xception_bottleneck.predict_generator(batches_preprocessed, 
                                           steps=nb_batches,
                                           verbose=1)

np.save(data_dir+'/results/bf_x', bf_x)


# In[9]:

bf_x = np.load(data_dir+'/results/bf_x.npy')


# ## Inception

# ### Extract Inception bottleneck features

# In[67]:

batches = gen.flow_from_directory(data_dir+'/train', target_size=(299, 299), shuffle=False)
batches_preprocessed = preprocess_batches(batches, inception_v3)


# In[68]:

inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

nb_batches = math.ceil(batches.n/batch_size)
bf_i = inception_bottleneck.predict_generator(batches_preprocessed, 
                                           steps=nb_batches,
                                           verbose=1)

np.save(data_dir+'/results/bf_i', bf_i)


# In[17]:

bf_i = np.load(data_dir+'/results/bf_i.npy')


# ## Stack

# ### LogReg on all bottleneck features

# In[69]:

X = np.hstack([bf_x, bf_i])
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
logreg.fit(X, y_encode)


# ## Test

# ### Predict test data

# In[79]:

test_ids = [file.split('.')[0] for file in listdir(data_dir+'/test/unknown')]


# In[120]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)


# In[121]:

test_ids[:3]


# In[122]:

batches_test.filenames[:3]


# In[123]:

nb_batches_test = math.ceil(batches_test.n/batch_size)


# In[ ]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)
batches_test_preprocessed = preprocess_batches(batches_test, xception)

bf_x_test = xception_bottleneck.predict_generator(batches_test_preprocessed, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[162]:

np.save(data_dir+'/results/bf_x_test', bf_x_test)


# In[72]:

bf_x_test = np.load('/opt/notebooks/data/dog-breed-identification/preprocessed/results/bf_x_test.npy')


# In[ ]:

bf_x_test = np.load(data_dir+'/results/bf_x_test.npy')


# In[ ]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)
batches_test_preprocessed = preprocess_batches(batches_test, inception_v3)

bf_i_test = inception_bottleneck.predict_generator(batches_test_preprocessed, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[161]:

np.save(data_dir+'/results/bf_i_test', bf_i_test)


# In[73]:

bf_i_test = np.load('/opt/notebooks/data/dog-breed-identification/preprocessed/results/bf_i_test.npy')


# In[ ]:

bf_i_test = np.load(data_dir+'/results/bf_i_test.npy')


# In[74]:

X_test = np.hstack([bf_x_test, bf_i_test])


# In[75]:

test_probs = logreg.predict_proba(X_test)


# In[76]:

test_probs.shape


# ### Make test submission file

# In[80]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), test_probs]))


# In[81]:

labels = pd.read_csv(data_dir+'/labels.csv')


# In[82]:

cols = ['id']+sorted(labels.breed.unique())


# In[83]:

subm.columns = cols
description = 'vgg_xception_inception_stack_on_logistic_all_train_data'
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[84]:

get_ipython().system('kg config -g -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')
get_ipython().system('kg submit $submission_file_name -u $KAGGLE_USER -p $KAGGLE_PW -m $description')


# Your submission scored 0.30054
