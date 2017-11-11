
# coding: utf-8

# - https://keras.io/applications/
# - https://github.com/yang-zhang/courses/blob/scratch/deeplearning1/nbs/lesson2.ipynb

# In[52]:

import math
import os
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.layers import Input, Lambda, Dense, Dropout, Flatten
from keras.models import Model, Sequential, load_model

from keras.utils import to_categorical
from keras.optimizers import RMSprop, SGD

from keras.applications import xception

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from secrets import KAGGLE_USER, KAGGLE_PW


# In[53]:

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


# ### train

# #### first fine-tune last layer

# In[60]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
inputs = Input(shape=(299, 299, 3))
x = Lambda(xception.preprocess_input)(inputs)
x = base_model(x)
outputs = Dense(120, activation='softmax', name='predictions')(x)
model_ft = Model(inputs, outputs)


# In[61]:

for layer in base_model.layers:
    layer.trainable = False


# In[62]:

batches = gen.flow_from_directory(data_dir+'/train', target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)


# In[63]:

model_ft.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[66]:

latest_filename


# In[69]:

no_of_epochs = 10
for epoch in range(no_of_epochs):
    print ("Running epoch: %d" % epoch)
    model_ft.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=1,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )
    latest_filename = data_dir+'/results/ft_top_%d_%s.h5' %  (epoch,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    )
    print(latest_filename)
    model_ft.save(latest_filename)


# In[76]:

model_ft = load_model(data_dir+'/results/ft_top_3_2017-11-10-20-57.h5')


# #### then fine-tune more layers

# In[9]:

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)


# In[77]:

# we chose to train the top 2 xception blocks, i.e. we will freeze
# the first 115 layers and unfreeze the rest:
for layer in base_model.layers[:116]:
   layer.trainable = False
for layer in base_model.layers[116:]:
   layer.trainable = True


# In[79]:

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model_ft.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
                   
no_of_epochs = 10
histories = []
for epoch in range(no_of_epochs):
    print ("Running epoch: %d" % epoch)
    hist = model_ft.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=1,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )
    histories.append(hist)
    latest_filename = data_dir+'/results/ft_%d_%s.h5' %  (epoch,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    )
    print(latest_filename)
    model_ft.save(latest_filename)


# ### predict

# In[86]:

batches_test = gen.flow_from_directory(data_dir+'/test', shuffle=False, target_size=target_size, batch_size=batch_size)


# In[87]:

nb_batches_test = math.ceil(batches_test.n/batch_size)


# In[88]:

pred = model_ft.predict_generator(batches_test, steps=nb_batches_test, verbose=1)


# In[89]:

test_ids = [f.split('/')[1].split('.')[0] for f in batches_test.filenames]


# In[90]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), pred]))
labels = pd.read_csv(data_dir+'/labels.csv')
cols = ['id']+sorted(labels.breed.unique())
subm.columns = cols


# In[91]:

description = 'xception_data_finetune_more_layers'
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[48]:

get_ipython().system('kg config -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')


# In[49]:

get_ipython().system('kg submit $submission_file_name -m $description')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



