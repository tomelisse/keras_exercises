from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import h5py

def main():
    ''' linear '''
    x_train = np.arange(100)
    y_train = 3*x_train + 0.6*np.random.randn(*x_train.shape) - 0.3 + 2 + 0.4*np.random.randn(*x_train.shape) - 0.2
    # _, ax = plt.subplots()
    # ax.plot(x_train, y_train, 'o')
    # plt.show()
    model = Sequential()
    model.add(Dense(input_dim=1, output_dim=1, init='uniform', activation='linear'))
    # model.add(Dense(1, input_shape = (1, ), init='uniform', activation='linear'))
    model.compile(optimizer = 'sgd', loss = 'mse')
    model.fit(x_train, y_train, epochs = 100)
    x_pred = [2,4,6]
    print model.predict(x_pred)


def prepare_dataset():
    ''' convert the original dara to .storage file '''
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
    with h5py.File('data/mnist.storage', 'w') as f:
        train = f.create_group('train')
        train.create_dataset('images', data = mnist.train.images)
        train.create_dataset('labels', data = mnist.train.labels)
       
        test = f.create_group('test')
        test.create_dataset('images', data = mnist.test.images)
        test.create_dataset('labels', data = mnist.test.labels)

class LossHistory(Callback):
    ''' custom callback for recording losses for each batch '''
    def on_train_begin(self, logs = {}):
        self.losses = []

    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))

def mnist():
    # prepare_dataset()
    with h5py.File('data/mnist.storage') as f:
        x_train = f['train/images'][:]
        y_train = f['train/labels'][:]
        x_test  = f['test/images'][:]
        y_test  = f['test/labels'][:]
    model = Sequential()
    model.add(Dense(units = y_train.shape[1], input_shape = (x_train.shape[1], ), activation = 'softmax'))
    model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')

    history = LossHistory()
    model.fit(x_train, y_train, batch_size = 5500, epochs = 2, callbacks = [history])
    model.evaluate(x_test, y_test)

    _, ax = plt.subplots()
    ax.plot(history.losses, 'o')
    plt.show()
    

if __name__ == '__main__':
    # main()
    mnist()
