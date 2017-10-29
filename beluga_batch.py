
# coding: utf-8

# In[123]:

import math
import os
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.layers import Input, Lambda
from keras.models import Model

from keras.applications import xception
from keras.applications import inception_v3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from secrets import KAGGLE_USER, KAGGLE_PW


# In[26]:

competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'


# In[27]:

gen = image.ImageDataGenerator()


# In[80]:

batch_size = 32
target_size=(299, 299)


# In[63]:

def add_preprocess(base_model, preprocess_func, inputs_shape=(299, 299, 3)):
    inputs = Input(shape=inputs_shape)
    x = Lambda(preprocess_func)(inputs)
    outputs = base_model(x)
    model = Model(inputs, outputs)
    return model


# ### Xception

# In[89]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes


# In[64]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')


# In[65]:

model_x = add_preprocess(base_model, xception.preprocess_input)


# In[66]:

bf_x=model_x.predict_generator(batches, steps=nb_batches, verbose=1)


# In[67]:

bf_val_x=model_x.predict_generator(batches_val, steps=nb_batches_val, verbose=1)


# In[68]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(bf_x, y_encode)


# In[69]:

valid_probs = logreg.predict_proba(bf_val_x)
valid_preds = logreg.predict(bf_val_x)


# In[70]:

log_loss(y_val_encode, valid_probs)


# In[71]:

accuracy_score(y_val_encode, valid_preds)


# ### Inception

# In[90]:

batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=target_size, batch_size=batch_size)

nb_batches = math.ceil(batches.n/batch_size)
nb_batches_val = math.ceil(batches_val.n/batch_size)

y_encode = batches.classes
y_val_encode = batches_val.classes


# In[81]:

base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')


# In[82]:

model_i = add_preprocess(base_model, inception_v3.preprocess_input)


# In[83]:

bf_i = model_i.predict_generator(batches, steps=nb_batches, verbose=1)


# In[84]:

bf_val_i = model_i.predict_generator(batches_val, steps=nb_batches_val, verbose=1)


# In[85]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(bf_i, y_encode)


# In[86]:

valid_probs = logreg.predict_proba(bf_val_i)
valid_preds = logreg.predict(bf_val_i)


# In[87]:

log_loss(y_val_encode, valid_probs)


# In[88]:

accuracy_score(y_val_encode, valid_preds)


# ### LogReg on all bottleneck features

# In[95]:

X = np.hstack([bf_x, bf_i])
V = np.hstack([bf_val_x, bf_val_i])
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X, y_encode)
valid_probs = logreg.predict_proba(V)
valid_preds = logreg.predict(V)


# In[96]:

log_loss(y_val_encode, valid_probs)


# In[97]:

accuracy_score(y_val_encode, valid_preds)


# ### predict test data

# In[101]:

test_ids = [file.split('.')[0] for file in os.listdir(data_dir+'/test/unknown')]


# In[102]:

test_ids[:3]


# In[103]:

batches_test = gen.flow_from_directory(data_dir+'/test', shuffle=False, target_size=target_size, batch_size=batch_size)
batches_test.filenames[:3]


# In[104]:

batches_test = gen.flow_from_directory(data_dir+'/test', shuffle=False, target_size=target_size, batch_size=batch_size)
nb_batches_test = math.ceil(batches_test.n/batch_size)


# In[105]:

bf_x_test = model_x.predict_generator(batches_test, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[106]:

batches_test = gen.flow_from_directory(data_dir+'/test', shuffle=False, target_size=target_size, batch_size=batch_size)
nb_batches_test = math.ceil(batches_test.n/batch_size)


# In[107]:

bf_i_test = model_i.predict_generator(batches_test, 
                                           steps=nb_batches_test,
                                           verbose=1)


# In[108]:

X_test = np.hstack([bf_x_test, bf_i_test])
test_probs = logreg.predict_proba(X_test)


# ### Make test submission file

# In[111]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), test_probs]))


# In[112]:

labels = pd.read_csv(data_dir+'/labels.csv')


# In[113]:

cols = ['id']+sorted(labels.breed.unique())


# In[118]:

subm.columns = cols
description = 'beluga_batch_lambda_preprocess'
submission_file_name = data_dir+'/results/%s_%s.csv' % (description,
                                                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
                                                       )
subm.to_csv(submission_file_name, index=False)


# ### submit

# In[124]:

get_ipython().system('kg config -g -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')


# In[125]:

get_ipython().system('kg submit $submission_file_name -u $KAGGLE_USER -p $KAGGLE_PW -m $description')


# Your submission scored 
