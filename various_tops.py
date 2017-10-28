
# coding: utf-8

# In[92]:

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


# In[3]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'


# In[11]:

gen = image.ImageDataGenerator()


# In[12]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, batch_size=batch_size)


# In[13]:

y_encode = batches.classes
y_val_encode = batches_val.classes


# In[8]:

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
batches = gen.flow_from_directory(data_dir+'/train', target_size=(299, 299), shuffle=False)
batches_val = gen.flow_from_directory(data_dir+'/valid', target_size=(299, 299), shuffle=False)

y_encode = batches.classes
y_val_encode = batches_val.classes

batches_preprocessed = preprocess_batches(batches, xception)
batches_val_preprocessed = preprocess_batches(batches_val, xception)

xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

nb_batches = math.ceil(batches.n/batch_size)
bf_x = xception_bottleneck.predict_generator(batches_preprocessed, 
                                           steps=nb_batches,
                                           verbose=1)

nb_batches_val = math.ceil(batches_val.n/batch_size)
bf_val_x = xception_bottleneck.predict_generator(batches_val_preprocessed, 
                                           steps=nb_batches_val,
                                           verbose=1)

np.save(data_dir+'/results/bf_x', bf_x)
np.save(data_dir+'/results/bf_val_x', bf_val_x)
# In[8]:

bf_x = np.load(data_dir+'/results/bf_x.npy')
bf_val_x = np.load(data_dir+'/results/bf_val_x.npy')


# ### LogReg on Xception bottleneck features

# In[30]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
logreg.fit(bf_x, y_encode)

valid_probs = logreg.predict_proba(bf_val_x)
valid_preds = logreg.predict(bf_val_x)


# In[28]:

log_loss(y_val_encode, valid_probs)


# In[29]:

accuracy_score(y_val_encode, valid_preds)


# ### Other models on Xception bottleneck features

# In[97]:

mdl = RandomForestClassifier()
mdl.fit(bf_x, y_encode)


# In[98]:

valid_probs = mdl.predict_proba(bf_val_x)
valid_preds = mdl.predict(bf_val_x)


# In[99]:

log_loss(y_val_encode, valid_probs)


# In[100]:

accuracy_score(y_val_encode, valid_preds)


# ## Inception

# ### Extract Inception bottleneck features
batches = gen.flow_from_directory(data_dir+'/train', target_size=(299, 299), shuffle=False)
batches_val = gen.flow_from_directory(data_dir+'/valid', target_size=(299, 299), shuffle=False)
batches_preprocessed = preprocess_batches(batches, inception_v3)
batches_val_preprocessed = preprocess_batches(batches_val, inception_v3)

inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

bf_i = inception_bottleneck.predict_generator(batches_preprocessed, 
                                           steps=nb_batches,
                                           verbose=1)

bf_val_i = inception_bottleneck.predict_generator(batches_val_preprocessed, 
                                           steps=nb_batches_val,
                                           verbose=1)

np.save(data_dir+'/results/bf_i', bf_i)
np.save(data_dir+'/results/bf_val_i', bf_val_i)
# In[17]:

bf_i = np.load(data_dir+'/results/bf_i.npy')
bf_val_i = np.load(data_dir+'/results/bf_val_i.npy')


# ### LogReg on Inception bottleneck features

# In[18]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
logreg.fit(bf_i, y_encode)


# In[19]:

valid_probs = logreg.predict_proba(bf_val_i)
valid_preds = logreg.predict(bf_val_i)


# In[20]:

log_loss(y_val_encode, valid_probs)


# In[21]:

accuracy_score(y_val_encode, valid_preds)


# ### Other models on Inception bottleneck features

# In[88]:

mdl = RandomForestClassifier()
mdl.fit(bf_i, y_encode)


# In[89]:

valid_probs = mdl.predict_proba(bf_val_i)
valid_preds = mdl.predict(bf_val_i)


# In[90]:

log_loss(y_val_encode, valid_probs)


# In[91]:

accuracy_score(y_val_encode, valid_preds)


# ## Stack

# ### LogReg on all bottleneck features

# In[22]:

X = np.hstack([bf_x, bf_i])
V = np.hstack([bf_val_x, bf_val_i])
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
logreg.fit(X, y_encode)


# In[23]:

valid_probs = logreg.predict_proba(V)
valid_preds = logreg.predict(V)


# In[24]:

log_loss(y_val_encode, valid_probs)


# In[25]:

accuracy_score(y_val_encode, valid_preds)


# ## Test

# ### Predict test data

# In[119]:

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


# In[125]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(224, 224), 
                                       shuffle=False)
batches_test_preprocessed = preprocess_batches(batches_test, vgg16)

bf_v_test = vgg_bottleneck.predict_generator(batches_test_preprocessed, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[163]:

np.save(data_dir+'/results/bf_v_test', bf_v_test)


# In[ ]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)
batches_test_preprocessed = preprocess_batches(batches_test, xception)

bf_x_test = xception_bottleneck.predict_generator(batches_test_preprocessed, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[162]:

np.save(data_dir+'/results/bf_x_test', bf_x_test)


# In[ ]:

batches_test = gen.flow_from_directory(data_dir+'/test', target_size=(299, 299), 
                                       shuffle=False)
batches_test_preprocessed = preprocess_batches(batches_test, inception_v3)

bf_i_test = inception_bottleneck.predict_generator(batches_test_preprocessed, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[161]:

np.save(data_dir+'/results/bf_i_test', bf_i_test)


# In[148]:

X_test = np.hstack([bf_v_test, bf_x_test, bf_i_test])
test_probs = logreg.predict_proba(X_test)


# ### Make test submission file

# In[151]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), test_probs]))


# In[152]:

labels = pd.read_csv(data_dir+'/labels.csv')


# In[153]:

cols = ['id']+sorted(labels.breed.unique())


# In[154]:

subm.columns = cols
description = 'vgg_xception_inception_stack_on_logistic'
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[155]:

get_ipython().system('kg config -g -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')
get_ipython().system('kg submit $submission_file_name -u $KAGGLE_USER -p $KAGGLE_PW -m $description')


# Your submission scored 
