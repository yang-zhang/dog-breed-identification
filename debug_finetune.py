
# coding: utf-8

# In[141]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[142]:

X = np.random.random((1000, 2))


# In[143]:

X.shape


# In[144]:

M = np.dot(X, [[2, 3, 1], 
               [4, 2, 1]]) + 1


# In[145]:

M.shape


# In[146]:

Y = np.dot(M, [[1], 
               [5], 
               [2]]) + 1


# In[147]:

Y.shape


# ## base model

# In[151]:

base_model = Sequential([
    Dense(3, input_shape=(2,))
])


# In[152]:

base_model.summary()


# In[153]:

base_model.compile(optimizer='sgd', loss='mse')
base_model.fit(X, M, epochs=10, verbose=0)


# In[154]:

base_model.evaluate(X, M)


# freeze base_model weights

# In[155]:

for layer in base_model.layers:
    layer.trainable = False


# In[156]:

for layer in base_model.layers:
    print(layer.get_weights())


# In[157]:

pred_mid = base_model.predict(X)


# In[158]:

base_model.layers[0].get_weights()


# In[159]:

pred_mid


# ## base_model output as dense layer input

# In[9]:

lm = Sequential([
    Dense(1, input_shape=(3,))
])
lm.compile(optimizer='sgd', loss='mse')


# In[161]:

lm.fit(pred_mid, Y, epochs=10, verbose=1)


# In[162]:

for layer in lm.layers:
    print(layer.get_weights())


# In[164]:

pred1 = lm.predict(pred_mid)


# In[165]:

lm.evaluate(pred_mid, Y)


# ## put in one model 

# In[167]:

x = base_model.output
predictions = Dense(1)(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='sgd', loss='mse')


# In[168]:

model.fit(X, Y, epochs=10, verbose=1)


# In[169]:

pred2 = model.predict(X)


# In[170]:

model.evaluate(X, Y)


# In[171]:

for layer in model.layers:
    print(layer.get_weights())


# In[172]:

base_model.layers[0].get_weights()


# In[173]:

np.all(model.layers[1].get_weights()[0] == base_model.layers[0].get_weights()[0])


# In[174]:

np.all(model.layers[1].get_weights()[1] == base_model.layers[0].get_weights()[1])


# In[175]:

model.layers[2].get_weights()


# In[176]:

lm.layers[0].get_weights()


# ### output of model at each layer

# In[177]:

# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# Testing
layer_outs = [func([X, 1.]) for func in functors]
print(layer_outs)


# In[178]:

np.allclose(X, layer_outs[0])


# In[179]:

np.all(pred_mid==layer_outs[1])


# In[180]:

np.all(pred2==layer_outs[2])


# In[ ]:



