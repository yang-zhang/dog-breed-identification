
# coding: utf-8

# - https://keras.io/applications/
# - https://github.com/yang-zhang/courses/blob/scratch/deeplearning1/nbs/lesson2.ipynb
# - http://localhost:8887/notebooks/git/dog-breed-identification/fine_tune_2.ipynb
# - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# In[10]:

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

from keras.applications import xception, inception_v3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from secrets import KAGGLE_USER, KAGGLE_PW


# In[18]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'
batch_size = 16


# ### train

# #### first fine-tune last layer
# create the base pre-trained model
base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

# add preprocessing at the bottom
inputs = Input(shape=(299, 299, 3))
x = Lambda(xception.preprocess_input)(inputs)
x = base_model(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer 
predictions = Dense(120, activation='softmax')(x)
# this is the model we will train
model = Model(inputs, predictions)
# In[36]:

# create the base pre-trained model
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# add preprocessing at the bottom
inputs = Input(shape=(299, 299, 3))
x = Lambda(inception_v3.preprocess_input)(inputs)
x = base_model(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer 
predictions = Dense(120, activation='softmax')(x)
# this is the model we will train
model = Model(inputs, predictions)


# In[37]:

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


# In[38]:

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[39]:

gen = image.ImageDataGenerator()
batches = gen.flow_from_directory(data_dir+'/train', target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)


# In[ ]:

model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=5,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val)


# In[ ]:

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=5,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val)


# In[ ]:




# In[ ]:




# In[ ]:




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




# In[11]:

# create the base pre-trained model
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)


# In[ ]:



