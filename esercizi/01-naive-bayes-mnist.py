#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

#load dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


# In[ ]:


def show_samples(samples):
    n = np.shape(samples)[0]
    plt.figure(figsize=(n, 2))
    for i in range(n):
        # display original
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(samples[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

show_samples(x_test[10:20])


# We shall work with a discrete version of Naive Bayes, so let us discretize the input, e.g. by thresholding.

# In[ ]:


x_train_discr = x_train >.5
x_test_discr = x_test >.5

show_samples(x_test_discr[10:20])


# In[ ]:


#compute the cardinality of all categories

card = np.zeros((10,1))
for i in range(0,10):
    card[i] = np.sum(y_train == i)

print("card = ", card)


# We can also plot these frequency as an histogram

# In[ ]:


def plot_hist(a,bins=10,title=None):
  plt.figure(tight_layout=True)
  plt.hist(a,bins=bins)
  if title:
    plt.title(title)
  plt.show()

plot_hist(y_train,title="Data for categories")


# For each category, and each pixel we compute how frequently it is set to 1 in the training set.

# In[ ]:


Freq = np.zeros((10,28*28))
for i in range(0,10):
    Freq[i] = np.sum(x_train_discr * np.expand_dims(y_train == i,axis=1), axis=0)

#we add one to ensure it is not 0 (we need to compute logs)
Freq += 1  #we assume Freq < card


# In[ ]:


def freq_ij_by_category(i,j,data,labels):
  ldata = (data*(np.expand_dims(y_train+1,axis=1))+10)%11
  #category 10 is for 0
  print(np.min(ldata),np.max(ldata))
  print(ldata.shape)
  plot_hist(ldata[:,i*28+j],bins=11,title="Occurrences of 1 for each category. Cat 10 is for 0")

freq_ij_by_category(4,10,x_train_discr,y_train)


# We pass to probabilities, by MLE

# In[ ]:


#probabilities to be 1
Prob1 = Freq/card
Prob0 = 1-Prob1
print("Prob1, Prob0 shape = ", Prob1.shape, Prob0.shape)

# passing to logs;
logProb1 = np.log(Prob1)
logProb0 = np.log(Prob0)
assert (logProb1 <= 0).all() & (logProb0 <= 0).all()


# In[89]:


def classify(img):
    d_img = img > .5
    logp = np.ones(10)
    #loglikelihood
    logp = np.sum(logProb1 * d_img + logProb0*(1-d_img),axis=-1)
    return(np.argmax(logp))

for i in range(100):
    prediction = classify(x_test[i])
    true = y_test[i]
    print("true = {}, predicted = {}".format(true,prediction))
    if not true == prediction:
        #we show imagae in case of missclassification
        show_samples(np.expand_dims(x_test_discr[i],axis=0))
    x = input()
    if x == 'q':
      break

