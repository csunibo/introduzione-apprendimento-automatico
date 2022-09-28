#!/usr/bin/env python
# coding: utf-8

# This notebook is meant to introduce convolutional layers, with special emphasis on the relation between the dimension of the input tensor, the kernel size, the stride, the number of filters and the dimension of the output tensor.

# In[ ]:


from keras.layers import Input, Conv2D, ZeroPadding2D, Dense, Flatten
from keras.models import Model
from keras import metrics
from keras.datasets import mnist


# We run the example over the mnist data set. Keras provides a very friendly access to several renowed databases, comprising mnist, cifar10, cifar100, IMDB and many others. See https://keras.io/api/datasets/ for documentation

# In[ ]:


import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Mnist images are grayscale images with pixels in the range [0,255].
# We pass to floats, and normalize them in the range [0,1].

# In[ ]:


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# Bidimensional convolutions expect input with three dimensions (plus an additional batchsize dimension): width, height, channels. 
# Since mnist digits have only two dimensions (being in grayscale), we need to extend them with an additional dimension.

# In[ ]:


(n,w,h) = x_train.shape
x_train = x_train.reshape(n,w,h,1)
(n,w,h) = x_test.shape
x_test = x_test.reshape(n,w,h,1)
print(x_train.shape)
print(x_test.shape)


# Mnist labels are integers in the range [0,9]. Since the network will produce probabilities for each one of these categories, if we want to compare it with the ground trouth probability using categorical crossentropy, that is the traditional choice, we should change each integer in its categorical description, using e.g. the "to_categorical" function in utils.
# 
# Alternatively, we can use the so called "sparse categorical crossentropy" loss function https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy that allows us to directly compare predictions with labels.

# In[ ]:


#y_train = keras.utils.to_categorical(y_train)
#y_test = keras.utils.to_categorical(y_test)


# Let us come to the convolutional network. We define a simple network composed by three convolutional layers, followed by a couple of Dense layers.

# In[ ]:


xin = Input(shape=(28,28,1))
x = Conv2D(16,(3,3),strides=(2,2),padding='valid')(xin)
x = Conv2D(32,(3,3),strides=(2,2),padding='valid')(x)
x = Conv2D(32,(3,3),strides=(2,2),padding='valid')(x)
x = Flatten()(x)
x = Dense(64, activation ='relu')(x)
res = Dense(10,activation = 'softmax')(x)

mynet = Model(inputs=xin,outputs=res)


# Let's have a look at the summary

# In[ ]:


mynet.summary()


# In valid mode, no padding is applied. 
# Along each axis, the output dimension O is computed from the input dimension I using the formula O=(I-K)/S +1, where K is the kernel dimension and S is the stride.
# 
# For all layers, K=3 and S=2. So, for the first conv we pass from dimension 28
# to dimension (28-3)/2+1 = 13, then to dimension (13-3)/2+1 = 6 and finally to dimension (6-3)/2+1 = 2. 
# 
# Exercise: modify "valid" to "same" and see what happens.
# 
# The second important point is about the number of parameters.
# You must keep in mind that a kernel of dimension K1 x K2 has an actual dimension K1 x K2 x CI, where CI is number of input channels: in other words the kernel is computing at the same time spatial and cross-channel correlations.
# 
# So, for the first convolution, we have 3 x 3 x 1 + 1 = 10 parameters for each filter (1 for the bias), and since we are computing 16 filters, the number of parameters is 10 x 16 = 160.
# 
# For the second convolution, each filter has 3 x 3 x 16 + 1 = 145 parameters, ans since we have 32 filters, the total number of parameters is 145 x 32 = 4640.
# 
# 

# Let us come to training.
# 
# In addition to the optimizer and the loss, we also pass a "metrics" argument. Metrics are additional functions that are not directly used for training, but allows us to monitor its advancement. For instance, we use accuracy, in this case (sparse, because we are using labels, and cateogrical because we have multiple categories).

# In[ ]:


mynet.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=[metrics.SparseCategoricalAccuracy()])


# In[ ]:


mynet.fit(x_train,y_train, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test))

