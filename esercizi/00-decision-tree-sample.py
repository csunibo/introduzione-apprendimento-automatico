#!/usr/bin/env python
# coding: utf-8

# # A simple example of Decision Tree
# 
# The script contains a simple example of classification using Decision Trees. It is meant to illustrate the _feature importances attribute, and the possibility to visualize the tree.

# In[ ]:


from sklearn import tree


# The dataset is extremely simple: each input is a sequence of 4 boolean values X0, X1,X2 and X3. However, it has been conceived to put into "trouble" the decision tree since the output Y depends from the comparison of two inputs, namely Y is X2==X3

# In[ ]:


X = [[0,0,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [1,1,0,1], [0,1,1,0], [0,1,1,1]]
Y = [0,0,1,1,0,0,1,1,1,0]


# In[ ]:


clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)


# In[ ]:


clf = clf.fit(X, Y)


# Let us predict a new result.

# In[ ]:


print(clf.predict([[1,1,1,1]]))


# One of the most interesting features of decision trees is the possibility to visualize the claissifier as an actual tree.

# In[ ]:


tree.export_graphviz(clf, out_file='tree.dot')
get_ipython().system('dot -T png tree.dot -o tree.png')


# In[ ]:


from IPython.display import Image
Image('tree.png') #, width=100, height=100)


# In this case, the information gain of individual features is not a
# good selection policy!
# 
# Another interesting functionality is the possibility to compute the "importance" of each feature.

# In[ ]:


print(clf.feature_importances_)


# Even if policy selection was not particulalry good, the classifier still understands that features X2 and X3 are "more relevant" than others.
# 
# This beacuse importance is calculated ex post for each
# decision tree, by the amount that each attribute split-point
# improves the performance measure, weighted by the number of
# observations the node is responsible for.
