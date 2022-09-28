#!/usr/bin/env python
# coding: utf-8

# # Mnist classification with NNs
# A first example of a simple Neural Network, applied to a well known dataset.

# In[44]:


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import utils
import numpy as np


# Let us load the mnist dataset

# In[45]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[46]:


print(x_train.shape)
print("pixel range is [{},{}]".format(np.min(x_train),np.max(x_train)))


# We normalize the input in the range [0,1]

# In[47]:


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train,(60000,784))
x_test = np.reshape(x_test,(10000,784))


# tf.keras.utils.to_categorical()

# In[48]:


print(y_train[0])
y_train_cat = utils.to_categorical(y_train)
print(y_train_cat[0])
y_test_cat = utils.to_categorical(y_test)


# Our first Netwok just implements logistic regression

# In[49]:


xin = Input(shape=(784))
res = Dense(10,activation='softmax')(xin)

mynet = Model(inputs=xin,outputs=res)


# In[50]:


mynet.summary()


# Now we need to compile the network.
# In order to do it, we need to pass two mandatory arguments:
# 
# 
# *   the **optimizer**, in charge of governing the details of the backpropagation algorithm
# *   the **loss function**
# 
# Several predefined optimizers exist, and you should just choose your favourite one. A common choice is Adam, implementing an adaptive lerning rate, with momentum
# 
# Optionally, we can specify additional metrics, mostly meant for monitoring the training process.
# 

# In[53]:


mynet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# Finally, we fit the model over the trianing set. 
# 
# Fitting, just requires two arguments: training data e ground truth, that is x and y. Additionally we can specify epochs, batch_size, and many additional arguments.
# 
# In particular, passing validation data allow the training procedure to measure loss and metrics on the validation set at the end of each epoch.

# In[ ]:


mynet.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))


# In[56]:


xin = Input(shape=(784))
x = Dense(128,activation='relu')(xin)
res = Dense(10,activation='softmax')(x)

mynet2 = Model(inputs=xin,outputs=res)


# In[57]:


mynet2.summary()


# In[58]:


mynet2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[60]:


mynet2.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))


# In[ ]:





# An amazing improvment. WOW!

# # Exercises
# 
# 1.   Add additional Dense layers and check the performance of the network
# 2.   Replace 'reu' with different activation functions
# 3. Adapt the network to work with the so called sparse_categorical_crossentropy
# 4. the fit function return a history of training, with temporal sequences for all different metrics. Make a plot.
# 
# 
