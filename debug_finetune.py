
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[2]:

X = np.random.random((10000, 2))


# In[3]:

X.shape


# In[4]:

M = np.dot(X, [[2, 3, 1], 
               [4, 2, 1]]) + 1


# In[5]:

M.shape


# In[6]:

Y = np.dot(M, [[1], 
               [5], 
               [2]]) + 1


# In[7]:

Y.shape


# ## base model

# In[8]:

base_model = Sequential([
    Dense(3, input_shape=(2,))
])


# In[9]:

base_model.summary()


# In[10]:

base_model.compile(optimizer='sgd', loss='mse')
base_model.fit(X, M, epochs=10, verbose=0)


# In[11]:

base_model.evaluate(X, M)


# freeze base_model weights

# In[12]:

for layer in base_model.layers:
    layer.trainable = False


# In[13]:

for layer in base_model.layers:
    print(layer.get_weights())


# In[14]:

pred_mid = base_model.predict(X)


# In[15]:

base_model.layers[0].get_weights()


# ## base_model output as dense layer input

# In[16]:

lm = Sequential([
    Dense(1, input_shape=(3,))
])
lm.compile(optimizer='sgd', loss='mse')


# In[17]:

lm.fit(pred_mid, Y, epochs=5, verbose=1)


# In[18]:

for layer in lm.layers:
    print(layer.get_weights())


# In[19]:

pred1 = lm.predict(pred_mid)


# In[20]:

lm.evaluate(pred_mid, Y)


# ## put in one model 

# In[21]:

x = base_model.output
predictions = Dense(1)(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='sgd', loss='mse')


# In[22]:

model.fit(X, Y, epochs=5, verbose=1)


# In[23]:

pred2 = model.predict(X)


# In[24]:

model.evaluate(X, Y)


# In[25]:

for layer in model.layers:
    print(layer.get_weights())


# In[26]:

base_model.layers[0].get_weights()


# In[27]:

np.all(model.layers[1].get_weights()[0] == base_model.layers[0].get_weights()[0])


# In[28]:

np.all(model.layers[1].get_weights()[1] == base_model.layers[0].get_weights()[1])


# In[29]:

model.layers[2].get_weights()


# In[30]:

lm.layers[0].get_weights()


# ### output of model at each layer

# In[31]:

# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# Testing
layer_outs = [func([X, 1.]) for func in functors]


# In[32]:

np.allclose(X, layer_outs[0])


# In[33]:

np.all(pred_mid==layer_outs[1])


# In[34]:

np.all(pred2==layer_outs[2])

