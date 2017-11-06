
# coding: utf-8

# In[46]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[47]:

X = np.random.random((10000, 2))


# In[48]:

X.shape


# In[49]:

M = np.dot(X, [[2, 3, 1], 
               [4, 2, 1]]) + 1


# In[50]:

M.shape


# In[51]:

Y = np.dot(M, [[1], 
               [5], 
               [2]]) + 1


# In[52]:

Y.shape


# In[53]:

n = X.shape[0]


# In[54]:

rnd = np.random.random(n)
idx_trn = rnd < 0.8
idx_val = rnd >=0.8


# In[55]:

X_trn = X[idx_trn]
X_val = X[idx_val]
Y_trn = Y[idx_trn]
Y_val = Y[idx_val]
M_trn = M[idx_trn]
M_val = M[idx_val]


# ## base model

# In[56]:

base_model = Sequential([
    Dense(3, input_shape=(2,))
])


# In[57]:

base_model.summary()


# In[58]:

base_model.compile(optimizer='sgd', loss='mse')
base_model.fit(X_trn, M_trn, epochs=5, verbose=1, validation_data=(X_val, M_val))


# freeze base_model weights

# In[59]:

for layer in base_model.layers:
    layer.trainable = False


# In[60]:

for layer in base_model.layers:
    print(layer.get_weights())


# In[61]:

pred_mid_trn = base_model.predict(X_trn)
pred_mid_val = base_model.predict(X_val)


# ## base_model output as dense layer input

# In[65]:

lm = Sequential([
    Dense(1, input_shape=(3,))
])
lm.compile(optimizer='sgd', loss='mse')


# In[66]:

lm.fit(pred_mid_trn, Y_trn, epochs=10, verbose=1, validation_data=(pred_mid_val, Y_val))


# In[67]:

for layer in lm.layers:
    print(layer.get_weights())


# ## put in one model 

# In[70]:

x = base_model.output
predictions = Dense(1)(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='sgd', loss='mse')


# In[71]:

model.fit(X_trn, Y_trn, epochs=10, verbose=1, validation_data=(X_val, Y_val))


# In[72]:

for layer in model.layers:
    print(layer.get_weights())


# In[73]:

base_model.layers[0].get_weights()


# In[74]:

np.all(model.layers[1].get_weights()[0] == base_model.layers[0].get_weights()[0])


# In[75]:

np.all(model.layers[1].get_weights()[1] == base_model.layers[0].get_weights()[1])


# In[76]:

model.layers[2].get_weights()


# In[77]:

lm.layers[0].get_weights()


# ### output of model at each layer from X_trn

# In[103]:

# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# Testing
layer_outs = [func([X_trn, 1.]) for func in functors]


# In[104]:

np.allclose(X_trn, layer_outs[0])


# In[105]:

np.all(pred_mid_trn == layer_outs[1])


# In[106]:

pred_trn = model.predict(X_trn)


# In[107]:

pred_trn


# In[108]:

np.all(pred_trn == layer_outs[2])


# ### output of model at each layer from X_val

# In[88]:

# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# Testing
layer_outs = [func([X_val, 1.]) for func in functors]


# In[89]:

np.allclose(X_val, layer_outs[0])


# In[90]:

np.all(pred_mid_val == layer_outs[1])


# In[98]:

pred_val = model.predict(X_val)


# In[99]:

np.all(pred_val == layer_outs[2])


# In[ ]:



