
# coding: utf-8

# In[46]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import display
from IPython.display import Image

from os import listdir
from datetime import datetime

from keras.preprocessing import image

from keras.applications import xception
from keras.applications import inception_v3
from keras.utils.np_utils import to_categorical

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

from secrets import KAGGLE_USER, KAGGLE_PW


# ### Get batches

# In[2]:

seed = 2014
batch_size = 32


# In[3]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'


# In[4]:

gen = image.ImageDataGenerator()


# In[5]:

batches = gen.flow_from_directory(data_dir+'/train', target_size=(299, 299), shuffle=False, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', target_size=(299, 299), shuffle=False, batch_size=batch_size)


# In[12]:

y_encode = batches.classes
y_val_encode = batches_val.classes


# In[6]:

def preprocess_batches(batches):
    while True:
        try:
            batch = batches.next()
            imgs = batch[0]
            imgs = np.apply_along_axis(xception.preprocess_input, 0, imgs)
            yield batch
        except StopIteration:
            break


# ### Extract Xception bottleneck features

# In[7]:

batches_preprocessed = preprocess_batches(batches)
batches_val_preprocessed = preprocess_batches(batches_val)


# In[8]:

xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling='avg')


# In[9]:

nb_batches = math.ceil(batches.n/batch_size)
bf_x = xception_bottleneck.predict_generator(batches_preprocessed, 
                                           steps=nb_batches,
                                           verbose=1)

nb_batches_val = math.ceil(batches_val.n/batch_size)
bf_val_x = xception_bottleneck.predict_generator(batches_val_preprocessed, 
                                           steps=nb_batches_val,
                                           verbose=1)


# ### LogReg on Xception bottleneck features

# In[13]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
logreg.fit(bf_x, y_encode)

valid_probs = logreg.predict_proba(bf_val_x)
valid_preds = logreg.predict(bf_val_x)


# In[14]:

log_loss(y_val_encode, valid_probs)


# In[15]:

accuracy_score(y_val_encode, valid_preds)


# ### Extract Inception bottleneck features

# In[27]:

batches = gen.flow_from_directory(data_dir+'/train', target_size=(299, 299), shuffle=False)
batches_val = gen.flow_from_directory(data_dir+'/valid', target_size=(299, 299), shuffle=False)
batches_preprocessed = preprocess_batches(batches)
batches_val_preprocessed = preprocess_batches(batches_val)


# In[ ]:

inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')


# In[ ]:

bf_i = inception_bottleneck.predict_generator(batches_preprocessed, 
                                           steps=nb_batches,
                                           verbose=1)
bf_val_i = inception_bottleneck.predict_generator(batches_val_preprocessed, 
                                           steps=nb_batches_val,
                                           verbose=1)


# ### LogReg on Inception bottleneck features

# In[ ]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
logreg.fit(bf_i, y_encode)


# In[ ]:

valid_probs = logreg.predict_proba(bf_val_i)
valid_preds = logreg.predict(bf_val_i)


# In[ ]:

log_loss(y_val_encode, valid_probs)


# In[ ]:

accuracy_score(y_val_encode, valid_preds)


# ### LogReg on all bottleneck features

# In[37]:

bf_x_val = bf_val


# In[39]:

X = np.hstack([bf_x, bf_i])
V = np.hstack([bf_x_val, bf_val_i])
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
logreg.fit(X, y_encode)
valid_probs = logreg.predict_proba(V)
valid_preds = logreg.predict(V)


# In[40]:

log_loss(y_val_encode, valid_probs)


# ### predict test data

# In[220]:

test_ids = [file.split('.')[0] for file in listdir(data_dir+'/test/unknown')]


# In[221]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)


# In[228]:

test_ids[:3]


# In[230]:

batches_test.filenames[:3]


# In[98]:

nb_batches_test = math.ceil(batches_test.n/batch_size)


# In[ ]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)
batches_test_preprocessed = preprocess_batches(batches_test)

bf_x_test = xception_bottleneck.predict_generator(batches_test_preprocessed, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[ ]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)
batches_test_preprocessed = preprocess_batches(batches_test)

bf_i_test = inception_bottleneck.predict_generator(batches_test_preprocessed, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[174]:

X_test = np.hstack([bf_x_test, bf_i_test])
test_probs = logreg.predict_proba(X_test)


# ### Make test submission file

# In[202]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), test_probs]))


# In[127]:

labels = pd.read_csv(data_dir+'/labels.csv')


# In[128]:

cols = ['id']+sorted(labels.breed.unique())


# In[129]:

subm.columns = cols
description = 'beluga_batch'
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[216]:

get_ipython().system('kg config -g -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')
get_ipython().system('kg submit $submission_file_name -u $KAGGLE_USER -p $KAGGLE_PW -m $description')


# Your submission scored 0.30091
