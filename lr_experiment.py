
# coding: utf-8

# In[2]:

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

from keras.applications import xception
from keras.applications import inception_v3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from secrets import KAGGLE_USER, KAGGLE_PW


# In[3]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'

gen = image.ImageDataGenerator()

batch_size = 16
target_size=(299, 299)

def add_preprocess(base_model, preprocess_func, inputs_shape=(299, 299, 3)):
    inputs = Input(shape=inputs_shape)
    x = Lambda(preprocess_func)(inputs)
    outputs = base_model(x)
    model = Model(inputs, outputs)
    return model


# In[4]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)


# In[5]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

model_x = add_preprocess(base_model, xception.preprocess_input)

# bf_x=model_x.predict_generator(batches, steps=nb_batches, verbose=1)
# np.save(data_dir+'/results/bf_x', bf_x)
bf_x = np.load(data_dir+'/results/bf_x.npy')
# bf_val_x=model_x.predict_generator(batches_val, steps=nb_batches_val, verbose=1)
# np.save(data_dir+'/results/bf_val_x', bf_val_x)
bf_val_x = np.load(data_dir+'/results/bf_val_x.npy')


# In[7]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(bf_x, y_encode)
valid_probs = logreg.predict_proba(bf_val_x)
valid_preds = logreg.predict(bf_val_x)
print('logloss:', log_loss(y_val_encode, valid_probs))
print('accuracy:', accuracy_score(y_val_encode, valid_preds))


# In[9]:

lm = Sequential([Dense(120, activation='softmax', input_shape=(2048,))])
lm.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
lm.fit(bf_x, y, epochs=15, batch_size=nb_batches, validation_data=(bf_val_x, y_val))


# In[15]:

lm = Sequential([Dense(120, activation='softmax', input_shape=(2048,))])
lm.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
lm.fit(bf_x, y, epochs=50, batch_size=nb_batches, validation_data=(bf_val_x, y_val))


# In[17]:

lm = Sequential([Dense(120, activation='softmax', input_shape=(2048,))])
lm.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
lm.fit(bf_x, y, epochs=50, batch_size=nb_batches, validation_data=(bf_val_x, y_val))
lm.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
lm.fit(bf_x, y, epochs=5, batch_size=nb_batches, validation_data=(bf_val_x, y_val))


# In[ ]:



