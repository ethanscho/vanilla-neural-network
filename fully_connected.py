from random import seed
from random import random
import numpy as np

def init_fc(input_dim, output_dim, batch_size):
    w = 2 * np.random.rand(input_dim, output_dim) - 1
    b = np.random.rand(batch_size, output_dim)
    return w, b

def softmax(x):
    ps = np.empty(x.shape)
    for i in range(x.shape[0]):
        ps[i,:]  = np.exp(x[i,:])
        ps[i,:] /= np.sum(ps[i,:])
    return ps

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)

# Load MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

BATCH_SIZE = 10
batch = mnist.train.next_batch(BATCH_SIZE)
x = batch[0]
y = batch[1]

fc_w1, fc_b1 = init_fc(784, 10, BATCH_SIZE)

for i in range(100):
    h = np.dot(x, fc_w1)
    z = sigmoid(h)

    loss = y - z
    loss = np.square(np.mean(loss ** 2))
    dz = loss * derivative_sigmoid(z) * 0.001

    fc_w1 += np.dot(x.T, dz)

print z

