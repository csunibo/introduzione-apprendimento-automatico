from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.backend as K
import time
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=True)
model.summary()

#an examples of classification
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x0 = image.img_to_array(img)

x = np.expand_dims(x0, axis=0)
print("shape = {}, range=[{},{}]".format(x.shape,np.min(x[0]),np.max(x[0])))

preds = model.predict(x)
print("label = {}".format(np.argmax(preds)))
print('Predicted:', decode_predictions(preds, top=3)[0])

#xd = scipy.misc.toimage(x[0])
xd = image.array_to_img(x[0])
imageplot = plt.imshow(xd)
plt.show()

#start of the fooling code


# this is the placeholder for the input images
input_img = model.input

# build a loss function that maximizes the activation of a wrong category

pred = model.output
print(pred.shape)

output_index = 3 
expected = np.zeros(1000)
expected[output_index] = 1
expected = K.variable(expected)
loss = K.categorical_crossentropy(model.output[0],expected)
#loss = K.mean(K.square(expected - model.output[0]), axis=-1)

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
loss_grads = K.function([input_img], [loss, grads])

input_img_data = np.copy(x)

#step = 1

# run gradient ascent for 50 steps
for i in range(50):
    print("iteration n. {}".format(i))
    res = model.predict(input_img_data)[0]
    print("elephant prediction: {}".format(res[386]))
    print("tiger shark prediction: {}".format(res[3]))
    time.sleep(1)
    loss_value, grads_value = loss_grads([input_img_data])
    #print(grads_value.shape)
    #print("loss = {}".format(loss_value))
    ming = np.min(grads_value)
    maxg = np.max(grads_value)
    #print("min grad = {}".format(ming))
    #print("max grad = {}".format(maxg))
    scale = 1/(maxg-ming)
    #brings gradients to a sensible value
    input_img_data -= grads_value * scale

input()
preds = model.predict(input_img_data)
print("label = {}".format(np.argmax(preds)))

print('Predicted:', decode_predictions(preds, top=3)[0])
img = input_img_data[0]
img = image.array_to_img(img)

plt.figure(figsize=(10,5))
ax = plt.subplot(1, 2, 1)
plt.title("elephant")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.imshow(xd)
ax = plt.subplot(1, 2, 2)
plt.imshow(img)
plt.title("tiger shark")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

imageplot = plt.imshow(img)
plt.show()



