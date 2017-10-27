
# coding: utf-8

# [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

# In[1]:

get_ipython().magic('matplotlib inline')
from __future__ import print_function, division
from secrets import KAGGLE_USER, KAGGLE_PW
import utils_ds

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import BatchNormalization, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2


# In[4]:

competition_name = 'dog-breed-identification'
dir_data = '/opt/notebooks/data/' + competition_name
path = '/opt/notebooks/data/'+competition_name+'/sample/'
batch_size=32

batches = utils_ds.get_batches(path+'train', batch_size=batch_size)

val_batches = utils_ds.get_batches(path+'valid', batch_size=batch_size)


# In[5]:

batches.nb_sample


# ### Linear model

# In[6]:

model = Sequential([
    BatchNormalization(axis=1, input_shape=(3, 224, 224)),
    Flatten(),
    Dense(120, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(batches, batches.nb_sample, nb_epoch=1, validation_data=val_batches, 
                   nb_val_samples=val_batches.nb_sample)


# In[24]:

model.summary()


# In[25]:

120*3*224*224


# In[27]:

reslt = model.predict_generator(batches, batches.n)

reslt[np.random.choice(reslt.shape[0]),:]


# In[61]:

model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(120, activation='softmax')
    ])
model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=5, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# #### with L2 regularization

# In[93]:

model = Sequential([
    BatchNormalization(axis=1, input_shape=(3, 224, 224)),
    Flatten(),
    Dense(120, activation='softmax', W_regularizer=l2(1))
])
model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(batches, batches.nb_sample, nb_epoch=10, validation_data=val_batches, 
                   nb_val_samples=val_batches.nb_sample)


# In[95]:

model.optimizer.lr = 0.001
model.fit_generator(batches, batches.nb_sample, nb_epoch=5, validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


# ### Single hidden layer

# In[ ]:

model = Sequential([
     BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(120, activation='softmax')
    ])
model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=15, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# #### with l2 regularization

# In[97]:

model = Sequential([
     BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(100, activation='relu', W_regularizer=l2(10)),
        BatchNormalization(),
        Dense(120, activation='softmax', W_regularizer=l2(10))
    ])
model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=20, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# ### single conv layer

# In[104]:

model = Sequential([
    BatchNormalization(axis=1, input_shape=(3,224,224)),
    Conv2D(32, 3, 3, activation='relu'),
    BatchNormalization(axis=1),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(100, activation='relu'),
    BatchNormalization(),
    Dense(120, activation='softmax')
])
model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, 
                     nb_val_samples=val_batches.nb_sample)


# In[105]:

model.optimizer.lr = 0.001
model.fit_generator(batches, batches.nb_sample, nb_epoch=4, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# ### double conv layer

# In[107]:

model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Conv2D(32,3,3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3,3)),
        Conv2D(64,3,3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3,3)),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(120, activation='softmax')
    ])
model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, 
                     nb_val_samples=val_batches.nb_sample)


# In[108]:

model.optimizer.lr = 0.001
model.fit_generator(batches, batches.nb_sample, nb_epoch=4, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# In[109]:

model.optimizer.lr = 0.01
model.fit_generator(batches, batches.nb_sample, nb_epoch=4, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# ### vgg

# In[5]:

import sys
sys.path.append('/opt/notebooks/git/courses/deeplearning1/nbs')


# In[144]:

import vgg16


# In[145]:

vgg=vgg16.Vgg16()


# In[146]:

vgg.finetune(batches)


# In[147]:

vgg.model.optimizer.lr = 0.01


# In[148]:

no_of_epochs = 5


# In[149]:

for epoch in range(no_of_epochs):
    print('Running epoch %d' % epoch)
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename='vgg_ft%d.h5' % epoch
    vgg.model.save_weights(path+'results/' + latest_weights_filename)


# In[150]:

ls $path/results


# ### vgg without dropout

# In[6]:

from utils import vgg_ft
model = vgg_ft(120)
model.load_weights(path+'results/vgg_ft4.h5')
layers = model.layers
last_conv_idx = [index for index,layer in enumerate(layers) 
                     if type(layer) is Conv2D][-1]
conv_layers = layers[:last_conv_idx+1]
conv_model = Sequential(conv_layers)
# Dense layers - also known as fully connected or 'FC' layers
fc_layers = layers[last_conv_idx+1:]


# In[7]:

val_features = conv_model.predict_generator(val_batches, val_batches.nb_sample)
trn_features = conv_model.predict_generator(batches, batches.nb_sample)


# In[16]:

from keras.utils.np_utils import to_categorical as onehot

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)


# In[17]:

def proc_wgts(layer): return [o/2 for o in layer.get_weights()]
opt = RMSprop(lr=0.00001, rho=0.7)
def get_fc_model():
    model = Sequential([
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(120, activation='softmax')
        ])

    for l1,l2 in zip(model.layers, fc_layers): l1.set_weights(proc_wgts(l2))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
fc_model = get_fc_model()
fc_model.fit(trn_features, trn_labels, nb_epoch=8, 
             batch_size=batch_size, validation_data=(val_features, val_labels))


# In[ ]:




# In[ ]:



