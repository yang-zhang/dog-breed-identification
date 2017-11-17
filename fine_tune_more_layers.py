
# coding: utf-8

# In[2]:

# import os
# os.environ["KERAS_BACKEND"] = "theano"


# - https://keras.io/applications/
# - https://github.com/yang-zhang/courses/blob/scratch/deeplearning1/nbs/lesson2.ipynb
# - http://localhost:8887/notebooks/git/dog-breed-identification/fine_tune_2.ipynb
# - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# In[4]:

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


# In[2]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'
batch_size = 16
target_size = (299, 299)
nb_classes = 120


# In[3]:

def get_batch_data(data_dir, target_size=target_size, batch_size=batch_size):
    gen = image.ImageDataGenerator()
    batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
    batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)
    nb_batches = math.ceil(batches.n/batch_size)
    nb_batches_val = math.ceil(batches_val.n/batch_size)
    return batches, batches_val, nb_batches, nb_batches_val


# ### output of bottleneck model with preprocessed input

# In[4]:

# create the base pre-trained model
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')


# In[5]:

# add preprocessing at the front
inputs = Input(shape=(299, 299, 3))
x = Lambda(inception_v3.preprocess_input)(inputs)
x = base_model(x)
mdl_preprocess_base_model = Model(inputs, x)


# In[ ]:

batches, batches_val, nb_batches, nb_batches_val = get_batch_data(data_dir)

base_model_output = mdl_preprocess_base_model.predict_generator(batches, steps=nb_batches, verbose=1)
np.save(data_dir+'/results/base_model_output_{}'.format(base_model.name) , base_model_output)

base_model_output_val = model_base_model_output.predict_generator(batches_val, steps=nb_batches_val, verbose=1)
np.save(data_dir+'/results/base_model_output_val_{}'.format(base_model.name) , base_model_output_val)


# In[6]:

base_model_output = np.load(data_dir+'/results/base_model_output_{}.npy'.format(base_model.name))
base_model_output_val = np.load(data_dir+'/results/base_model_output_val_{}.npy'.format(base_model.name))


# ### finetune - 1 dense layer

# In[7]:

mdl_single_dense_on_base_model = Sequential(
    [Dense(nb_classes, 
           activation='softmax', 
           input_shape=(base_model_output.shape[1],)
          )]
)
mdl_single_dense_on_base_model.compile(optimizer=RMSprop(), 
                                       loss='categorical_crossentropy', 
                                       metrics=['accuracy'])

batches, batches_val, nb_batches, nb_batches_val = get_batch_data(data_dir)
y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)

mdl_single_dense_on_base_model.fit(base_model_output, 
                                   y, 
                                   epochs=15, 
                                   batch_size=nb_batches, 
                                   validation_data=(base_model_output_val, y_val))


# ### finetune - 2 dense layers

# In[8]:

mdl_2_dense_on_base_model = Sequential(
    [Dense(1024, activation='relu', input_shape=(base_model_output.shape[1],)),
    Dense(nb_classes, activation='softmax',)])

mdl_2_dense_on_base_model.compile(optimizer=RMSprop(),
                                  loss='categorical_crossentropy', 
                                  metrics=['accuracy'])

batches, batches_val, nb_batches, nb_batches_val = get_batch_data(data_dir)
y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)

mdl_2_dense_on_base_model.fit(base_model_output, 
                                   y, 
                                   epochs=5, 
                                   batch_size=nb_batches, 
                                   validation_data=(base_model_output_val, y_val))


# ### finetune some conv layers

# In[29]:

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


# In[30]:

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


# In[31]:

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[32]:

gen = image.ImageDataGenerator()
batches = gen.flow_from_directory(data_dir+'/train', shuffle=True, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)


# In[33]:

model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=3,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val)


# In[34]:

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)


# In[35]:

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in base_model.layers[:249]:
   layer.trainable = False
for layer in base_model.layers[249:]:
   layer.trainable = True


# In[36]:

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


# In[37]:

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
no_of_epochs = 50
for epoch in range(no_of_epochs):
    print ("Running epoch: %d" % epoch)
    batches = gen.flow_from_directory(data_dir+'/train', shuffle=True, target_size=target_size, batch_size=batch_size)
    batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)
    model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=1,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )
    latest_filename = data_dir+'/results/ft_top_%d_%s.h5' %  (epoch,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    )
    print(latest_filename)
    model.save(latest_filename)


# In[40]:

ls /opt/notebooks/data/dog-breed-identification/preprocessed/results/ft_top_5_2017-11-16-14-56.h5


# In[6]:

model = load_model('/opt/notebooks/data/dog-breed-identification/preprocessed/results/ft_top_5_2017-11-16-14-56.h5')


# ### predict

# In[47]:

batches_test = gen.flow_from_directory(data_dir+'/test', shuffle=False, target_size=target_size, batch_size=batch_size)


# In[48]:

nb_batches_test = math.ceil(batches_test.n/batch_size)


# In[49]:

pred = model.predict_generator(batches_test, steps=nb_batches_test, verbose=1)


# In[50]:

test_ids = [f.split('/')[1].split('.')[0] for f in batches_test.filenames]


# In[51]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), pred]))
labels = pd.read_csv(data_dir+'/labels.csv')
cols = ['id']+sorted(labels.breed.unique())
subm.columns = cols


# In[52]:

description = 'inception_data_finetune_more_layers'
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[53]:

get_ipython().system('kg config -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')


# In[54]:

get_ipython().system('kg submit $submission_file_name -m $description')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



