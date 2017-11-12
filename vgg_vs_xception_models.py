
# coding: utf-8

# In[1]:

from keras.applications import vgg16, xception


# In[13]:

vgg = vgg16.VGG16()


# In[14]:

len(vgg.layers)


# In[15]:

vgg.summary()


# In[10]:

vgg_bottleneck = vgg16.VGG16(include_top=False, pooling='avg')


# In[16]:

len(vgg_bottleneck.layers)


# In[11]:

vgg_bottleneck.summary()


# In[4]:

xcptn = xception.Xception()


# In[5]:

xcptn.summary()


# In[17]:

len(xcptn.layers)


# In[8]:

xcptn_bottleneck =xception.Xception(include_top=False, pooling='avg')


# In[18]:

len(xcptn_bottleneck.layers)


# In[9]:

xcptn_bottleneck.summary()


# In[ ]:




# In[ ]:



