import numpy as np
from keras.layers import Input, LSTM, Conv1D, Dense, Lambda
from keras.models import Model
import keras.backend as K

batchsize = 100
seqlen = 10

def generator(batchsize,seqlen):
    #init_carry = np.random.randint(2,size=batchsize)
    init_carry = np.zeros(batchsize)
    carry = init_carry
    while True:
      a = np.random.randint(2,size=(batchsize,seqlen,2))
      res = np.zeros((batchsize,seqlen))
      for t in range(0,seqlen):
        sum = a[:,t,0]+a[:,t,1] + carry
        res[:,t] = sum % 2
        carry = sum // 2
      yield (a, res)

for i in range (0,10):
    a, res = next(generator(1, 1))
    print(a,res)

def gen_model():
    xa = Input(shape=(None,2))
    x = Conv1D(8,1,activation='relu')(xa)
    x = Conv1D(4,1,activation='relu')(x)
    x = LSTM(4,activation=None, return_sequences=True)(x)
    x = Dense(1,activation='sigmoid')(x)
    out = Lambda(lambda x:K.squeeze(x,2))(x)
    comp = Model(inputs=xa, outputs=out)
    return comp

comp = gen_model()
comp.summary()

comp.compile(optimizer='adam',loss='mse')

comp.load_weights("weights/lstm.h5")

#comp.fit_generator(generator(batchsize,seqlen), steps_per_epoch=100, epochs=10)
#comp.save_weights("weights/lstm.h5")

example,res = next(generator(1,10))
predicted = np.array([int(np.rint(x)) for x in comp.predict(example)[0]])

print("a1        = {}".format(example[0][:,0].astype(int)))
print("a2        = {}".format(example[0][:,1].astype(int)))
print("expected  = {}".format(res[0].astype(int)))
print("predicted = {}".format(predicted))
