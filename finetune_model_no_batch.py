
# coding: utf-8

# In[84]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os.path import join


from keras.preprocessing import image
from keras.applications import xception
from keras.layers import Input, Lambda, Dense
from keras.models import Model, Sequential

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[42]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/unzipped/'


# In[58]:

num_classes = 10


# In[60]:

labels = pd.read_csv(data_dir+'labels.csv')

selected_breed_list = labels.breed.value_counts().index.values[:num_classes]
selected_breed_list = list(selected_breed_list)
labels = labels[labels.breed.isin(selected_breed_list)]


# In[61]:

y_train = pd.get_dummies(labels.breed).values


# In[62]:

le = LabelEncoder()
y_train_encode = le.fit_transform(labels.breed.values)


# In[63]:

rnd = np.random.random(len(labels))

train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
ytr = y_train[train_idx]
yv = y_train[valid_idx]


# ### Use output of bottleneck xception from preprocessed input

# In[64]:

x_train = np.empty((len(labels), 299, 299, 3), dtype='float32')
def read_img(img_id, train_or_test, size):
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img

for i, img_id in enumerate(labels['id']):
    img = read_img(img_id, 'train', (299, 299))
    x = xception.preprocess_input(img)
    x_train[i] = x

Xtr = x_train[train_idx]
Xv = x_train[valid_idx]


# In[67]:

xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling='avg')


# In[68]:

train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)


# #### feed to logistic reg - ok

# In[55]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(train_x_bf, y_train_encode[train_idx])

valid_probs = logreg.predict_proba(valid_x_bf)
valid_preds = logreg.predict(valid_x_bf)


# In[56]:

log_loss(yv, valid_probs)


# In[57]:

accuracy_score(y_train_encode[valid_idx], valid_preds)


# #### feed to dense softmax layer - ok

# In[86]:

lm = Sequential([Dense(num_classes, activation='softmax', input_shape=(2048,))])
lm.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[87]:

lm.fit(train_x_bf, ytr, epochs=10, batch_size=32, validation_data=(valid_x_bf, yv))


# ### put preprocess step in a model - xception - dense softmax layer - ok 

# In[92]:

x_train = np.empty((len(labels), 299, 299, 3), dtype='float32')
def read_img(img_id, train_or_test, size):
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img

for i, img_id in enumerate(labels['id']):
    img = read_img(img_id, 'train', (299, 299))
    x_train[i] = img


# In[ ]:

Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
inputs = Input(shape=(299, 299, 3))
x = Lambda(xception.preprocess_input)(inputs)
outputs = base_model(x)
model = Model(inputs, outputs)


# In[94]:

train_x_bf = model.predict(Xtr, batch_size=32, verbose=1)
valid_x_bf = model.predict(Xv, batch_size=32, verbose=1)


# In[95]:

lm = Sequential([Dense(num_classes, activation='softmax', input_shape=(2048,))])
lm.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[96]:

lm.fit(train_x_bf, ytr, epochs=10, batch_size=32, validation_data=(valid_x_bf, yv))


# ### make into one model - no longer works

# In[80]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False
inputs = Input(shape=(299, 299, 3))
x = Lambda(xception.preprocess_input)(inputs)
x = base_model(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[82]:

model.fit(x=Xtr, y=ytr, batch_size=32, epochs=3, verbose=1, validation_data=(Xv, yv))


# In[ ]:



