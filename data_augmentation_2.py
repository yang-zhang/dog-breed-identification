
# coding: utf-8

# In[72]:

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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from secrets import KAGGLE_USER, KAGGLE_PW


# In[2]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'

gen = image.ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.1,
                               shear_range=0.15, 
                               zoom_range=0.1, 
                               channel_shift_range=10., 
                               horizontal_flip=True)
batch_size = 16
target_size=(299, 299)

def add_preprocess(base_model, preprocess_func, inputs_shape=(299, 299, 3)):
    inputs = Input(shape=inputs_shape)
    x = Lambda(preprocess_func)(inputs)
    outputs = base_model(x)
    model = Model(inputs, outputs)
    return model


# ### train

# #### pretrained

# In[48]:

batches = image.ImageDataGenerator().flow_from_directory(data_dir+'/train', target_size=target_size, batch_size=batch_size)
batches_val = image.ImageDataGenerator().flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

# model_x = add_preprocess(base_model, xception.preprocess_input)
# bf_x=model_x.predict_generator(batches, steps=nb_batches, verbose=1)
# np.save(data_dir+'/results/bf_x', bf_x)
bf_x = np.load(data_dir+'/results/bf_x.npy')
# bf_val_x=model_x.predict_generator(batches_val, steps=nb_batches_val, verbose=1)
# np.save(data_dir+'/results/bf_val_x', bf_val_x)
bf_val_x = np.load(data_dir+'/results/bf_val_x.npy')

lm = Sequential([Dense(120, activation='softmax', input_shape=(2048,))])
lm.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
lm.fit(bf_x, y, epochs=15, batch_size=nb_batches, validation_data=(bf_val_x, y_val))


# In[49]:

lm.summary()


# In[50]:

lm.layers[0]


# #### augmentation

# In[62]:

batches = gen.flow_from_directory(data_dir+'/train', target_size=target_size, batch_size=batch_size)
batches_val = image.ImageDataGenerator().flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)


# In[77]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
inputs = Input(shape=(299, 299, 3))
x = Lambda(xception.preprocess_input)(inputs)
x = base_model(x)
outputs = Dense(120, activation='softmax', name='predictions')(x)
model_ft = Model(inputs, outputs)
for layer in base_model.layers:
    layer.trainable = False


# In[78]:

model_ft.summary()


# In[79]:

model_ft.layers[3].set_weights(lm.layers[0].get_weights())


# In[80]:

model_ft.compile(optimizer=Adam(lr=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])


# In[81]:

# verify that the starting point is already good model
model_ft.evaluate_generator(batches_val, steps=nb_batches_val)


# In[82]:

model_ft.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=10,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )


# ### predict

# In[9]:

batches_test = image.ImageDataGenerator().flow_from_directory(data_dir+'/test', shuffle=False, target_size=target_size, batch_size=batch_size)


# In[10]:

nb_batches_test = math.ceil(batches_test.n/batch_size)


# In[11]:

pred = model_ft.predict_generator(batches_test, steps=nb_batches_test, verbose=1)


# In[18]:

test_ids = [f.split('/')[1].split('.')[0] for f in batches_test.filenames]


# In[20]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), pred]))
labels = pd.read_csv(data_dir+'/labels.csv')
cols = ['id']+sorted(labels.breed.unique())
subm.columns = cols


# In[24]:

description = 'xception_data_augmentation'
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[12]:

description = 'xception_data_augmentation'


# In[8]:

get_ipython().system('kg config -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')


# In[13]:

get_ipython().system('kg submit $submission_file_name -m $description')


# In[ ]:



