
# coding: utf-8

# In[73]:

import math

from keras.models import Model
from keras.layers import Input, Lambda
from keras.applications import xception
from keras.preprocessing import image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score


# In[31]:

base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')


# In[56]:

inputs = Input(shape=(299,299, 3))


# In[57]:

x = Lambda(xception.preprocess_input)(inputs)


# In[59]:

outputs = base_model(x)


# In[60]:

model = Model(inputs, outputs)


# In[ ]:

model.compile()


# In[102]:

batch_size = 32
competition_name = 'dog-breed-identification'
data_dir = '/opt/notebooks/data/' + competition_name + '/preprocessed'
gen = image.ImageDataGenerator()
batches = gen.flow_from_directory(data_dir+'/train', shuffle=False, target_size=(299, 299), batch_size=batch_size)
batches_val = gen.flow_from_directory(data_dir+'/valid', shuffle=False, target_size=(299, 299), batch_size=batch_size)
y_encode = batches.classes
y_val_encode = batches_val.classes


# In[103]:

nb_batches = math.ceil(batches.n/batch_size)
bf_x=model.predict_generator(batches, steps=nb_batches, verbose=1)


# In[104]:

nb_batches_val = math.ceil(batches_val.n/batch_size)
bf_val_x=model.predict_generator(batches_val, steps=nb_batches_val, verbose=1)


# In[105]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(bf_x, y_encode)


# In[106]:

valid_probs = logreg.predict_proba(bf_val_x)
valid_preds = logreg.predict(bf_val_x)


# In[108]:

log_loss(y_val_encode, valid_probs)


# In[109]:

accuracy_score(y_val_encode, valid_preds)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[32]:

inputs = Input(shape=(224, 224, 3))


# In[33]:

inputs


# In[16]:

y = base_model(inputs)


# In[17]:

model = Model(inputs=inputs, outputs=y)


# In[18]:

model.summary()


# In[21]:

base_model.summary()


# In[ ]:




# In[ ]:




# In[2]:

vgg16.preprocess_input


# In[ ]:

vgg_bottleneck = vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')


# In[3]:

inputs = Input()
Lambda(vgg16.preprocess_input, 
       output_shape=(3, 224, 224)
      )


# In[ ]:



