
# coding: utf-8

# In[1]:

import math
import os
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.layers import Input, Lambda, Dense, Dropout, Flatten
from keras.models import Model, Sequential

from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam

from keras.applications import xception, inception_v3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from secrets import KAGGLE_USER, KAGGLE_PW


# In[2]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'

batch_size = 16
target_size=(299, 299)

def add_preprocess(base_model, preprocess_func, inputs_shape=(299, 299, 3)):
    inputs = Input(shape=inputs_shape)
    x = Lambda(preprocess_func)(inputs)
    outputs = base_model(x)
    model = Model(inputs, outputs)
    return model


# ### train

# emsemble a few augmentation training data

# In[3]:

base_model_x = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
model_x = add_preprocess(base_model_x, xception.preprocess_input)


# In[4]:

base_model_i = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
model_i = add_preprocess(base_model_i, inception_v3.preprocess_input)


# In[5]:

batches = image.ImageDataGenerator().flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = image.ImageDataGenerator().flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_test = image.ImageDataGenerator().flow_from_directory(data_dir+'/test', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)
nb_batches_test = math.ceil(batches_test.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)


# In[6]:

# bf_val_x = model_x.predict_generator(batches_val, steps=nb_batches_val, verbose=1)
# np.save(data_dir+'/results/bf_val_x', bf_val_x)
bf_val_x = np.load(data_dir+'/results/bf_val_x.npy')

# bf_x_test = model_x.predict_generator(batches_test, steps=nb_batches_test, verbose=1)
# np.save(data_dir+'/results/bf_x_test', bf_x_test)
bf_x_test = np.load(data_dir+'/results/bf_x_test.npy')


# In[18]:

# bf_val_i = model_i.predict_generator(batches_val, steps=nb_batches_val, verbose=1)
# np.save(data_dir+'/results/bf_val_i', bf_val_i)
bf_val_i = np.load(data_dir+'/results/bf_val_i.npy')

# bf_i_test = model_i.predict_generator(batches_test, steps=nb_batches_test, verbose=1)
# np.save(data_dir+'/results/bf_i_test', bf_i_test)
bf_i_test = np.load(data_dir+'/results/bf_i_test.npy')


# In[8]:

gen = image.ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.1,
                               shear_range=0.15, 
                               zoom_range=0.1, 
                               channel_shift_range=10., 
                               horizontal_flip=True)


# In[45]:

preds = []
nb_runs = 20
for i in range(nb_runs):
    print("i:", i)
    batches = gen.flow_from_directory(data_dir+'/train', target_size=target_size, batch_size=batch_size, shuffle=False)
    y = to_categorical(batches.classes)
    bf_x = model_x.predict_generator(batches, steps=nb_batches, verbose=1)
    
    batches = gen.flow_from_directory(data_dir+'/train', target_size=target_size, batch_size=batch_size, shuffle=False)
    bf_i = model_i.predict_generator(batches, steps=nb_batches, verbose=1)
    
    lm = Sequential([Dense(120, activation='softmax', input_shape=(2048+2048,))])
    lm.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
    lm.fit(np.hstack([bf_x, bf_i]), y, epochs=15, batch_size=nb_batches, 
           validation_data=(np.hstack([bf_val_x, bf_val_i]), y_val))
    
    pred = lm.predict(np.hstack([bf_x_test, bf_i_test]), batch_size=batch_size, verbose=1)
    preds.append(pred)


# In[46]:

pred_ensemble = np.stack(preds).mean(axis=0)


# ### predict

# In[47]:

test_ids = [f.split('/')[1].split('.')[0] for f in batches_test.filenames]


# In[48]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), pred_ensemble]))
labels = pd.read_csv(data_dir+'/labels.csv')
cols = ['id']+sorted(labels.breed.unique())
subm.columns = cols


# In[49]:

description = 'xception_inception_ensemble_%d_data_aug' % nb_runs
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[50]:

get_ipython().system('kg config -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')


# In[51]:

get_ipython().system('kg submit $submission_file_name -m $description')


# In[ ]:



