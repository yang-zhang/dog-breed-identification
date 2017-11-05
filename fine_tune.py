
# coding: utf-8

# In[53]:

import math
import os
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.layers import Input, Lambda, Dense, Dropout, Flatten
from keras.models import Model, Sequential

from keras.utils import to_categorical
from keras.optimizers import RMSprop

from keras.applications import xception
from keras.applications import inception_v3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from secrets import KAGGLE_USER, KAGGLE_PW

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


# ### Feed Xception output to logistic regression

# #### Xception works okay with logistic regression

# In[22]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

model_x = add_preprocess(base_model, xception.preprocess_input)

model_x.summary()

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)

#bf_x=model_x.predict_generator(batches, steps=nb_batches, verbose=1)

# np.save(data_dir+'/results/bf_x', bf_x)

bf_x = np.load(data_dir+'/results/bf_x.npy')

#bf_val_x=model_x.predict_generator(batches_val, steps=nb_batches_val, verbose=1)

# np.save(data_dir+'/results/bf_val_x', bf_val_x)

bf_val_x = np.load(data_dir+'/results/bf_val_x.npy')

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(bf_x, y_encode)

valid_probs = logreg.predict_proba(bf_val_x)
valid_preds = logreg.predict(bf_val_x)


# In[13]:

log_loss(y_val_encode, valid_probs)


# In[14]:

accuracy_score(y_val_encode, valid_preds)


# In[ ]:




# In[ ]:




# #### Xception works okay with logistic regression - even without preprocess

# In[107]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

bf_x_no_prep=model_x.predict_generator(batches, steps=nb_batches, verbose=1)

bf_val_x_no_prep=model_x.predict_generator(batches_val, steps=nb_batches_val, verbose=1)

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(bf_x_no_prep, y_encode)

valid_probs = logreg.predict_proba(bf_val_x_no_prep)
valid_preds = logreg.predict(bf_val_x_no_prep)


# In[112]:

log_loss(y_val_encode, valid_probs)


# In[113]:

accuracy_score(y_val_encode, valid_preds)


# ### Finetune 
# - https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson2.ipynb
# - https://github.com/fchollet/keras/issues/3465

# #### Use xception model output as input of a one-dense-layer model - works okay

# In[ ]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

model_x = add_preprocess(base_model, xception.preprocess_input)

model_x.summary()

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes

y = to_categorical(batches.classes)
y_val = to_categorical(batches_val.classes)

#bf_x=model_x.predict_generator(batches, steps=nb_batches, verbose=1)

# np.save(data_dir+'/results/bf_x', bf_x)

bf_x = np.load(data_dir+'/results/bf_x.npy')

#bf_val_x=model_x.predict_generator(batches_val, steps=nb_batches_val, verbose=1)

# np.save(data_dir+'/results/bf_val_x', bf_val_x)

bf_val_x = np.load(data_dir+'/results/bf_val_x.npy')

lm = Sequential([Dense(120, activation='softmax', input_shape=(2048,))])

lm.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[125]:

lm.fit(bf_x, y, epochs=20, batch_size=nb_batches, validation_data=(bf_val_x, y_val))


# ### make a model to combine the above steps

# In[140]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False
inputs = Input(shape=(299, 299, 3))
x = Lambda(xception.preprocess_input)(inputs)
x = base_model(x)
outputs = Dense(120, activation='softmax')(x)
model = Model(inputs, outputs)


# In[141]:

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[142]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=3,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val,
                    shuffle=False
                   )


# In[ ]:




# #### make a model without preprocessing

# In[118]:

# create the base pre-trained model
base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

# add a global spatial average pooling layer
x = base_model.output
# x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# and a logistic layer 
predictions = Dense(120, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[119]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)
nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=3,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )


# In[ ]:




# In[ ]:




# #### make a model to combine the above steps - attempt1

# In[83]:

top_model = Sequential()
top_model.add(Dense(120, activation='softmax', input_shape=(2048,)))


# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(120, activation='sigmoid'))
# top_model.load_weights(top_model_weights_path)

model_ft = Model(inputs= model_x.input, outputs = top_model(base_model.output))


# In[27]:

model_ft.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[29]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=3,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )


# #### make a model to combine the above steps - attempt2

# In[98]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
inputs = Input(shape=(299, 299, 3))
x = Lambda(xception.preprocess_input)(inputs)
x = base_model(x)
outputs = Dense(120, activation='softmax', name='predictions')(x)
model_ft = Model(inputs, outputs)
for layer in base_model.layers:
    layer.trainable = False


# In[101]:

model_ft.layers


# In[103]:

for layer in model_ft.layers[:-1]:
    layer.trainable=False


# In[104]:

model_ft.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[105]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)


# In[106]:

# train the model on the new data for a few epochs
model_ft.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=3,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )


# #### make a  model to combine the above steps - fail...

# In[9]:

model_x.summary()


# In[10]:

base_model.summary()


# In[11]:

x = model_x.output
x = Dense(120, activation='softmax', name='predictions')(x)
model = Model(input=model_x.input, output=x)


# In[12]:

model.summary()


# In[13]:

for layer in model.layers:
    print(layer.trainable)


# In[14]:

for layer in model.layers[1:-1]:
    layer.trainable = False


# In[15]:

model.summary()


# In[16]:

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[17]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes


# In[18]:

# train the model on the new data for a few epochs
model.fit_generator(batches, 
                    steps_per_epoch=nb_batches, 
                    epochs=3,
                    validation_data=batches_val,
                    validation_steps=nb_batches_val
                   )


# In[ ]:



