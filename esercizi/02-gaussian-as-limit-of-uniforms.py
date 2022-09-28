#!/usr/bin/env python
# coding: utf-8

# This is a simple demo to show that a Gaussian distribution can be understood as a sum of uniform distibutions.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# In[ ]:


We compute 100000 numbers as a sum of 12 integers sampled according to a uniform distribution in the range [0,1].
The mean of a sum of distributions is equal to the sum of the means, so the expected mean is 12*.5=6.
We shift each number of -6 to recenter the distributio on the origin.
The variance of a sum of distributions is equal to the sum of variances. 
Since the variance of a uniform distribution is 1/12 (prove it by exercise), the variance of
the sum is 1. 


# In[ ]:


no = 100000
all = []
for i in range(0,no):
    x = 0
    for j in range(0,12):
        x += np.random.rand()
    all.append(x-6)


# Now we draw an histogram of our distribution and compare it with the Gaussian.

# In[5]:


n, bins, patches = plt.hist(all, 100, density=True)
#now draw the gaussian

x = np.linspace(-4, 4, 100)
plt.plot(x,norm.pdf(x,0,1),linewidth=3)
plt.show()


# WOW!
