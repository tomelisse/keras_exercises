''' recreating deep mnist for experts in keras '''
''' https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5 '''
from keras.models import Model
from keras import layers as kl
import tensorflow as tf
import dataset
import h5py

def mnist():
    # run just once:
    # dataset.prepare_dataset()
    with h5py.File('data/mnist.storage') as f:
        x_train = f['train/images'][:]
        y_train = f['train/labels'][:]
        x_test  = f['test/images'][:]
        y_test  = f['test/labels'][:]

    # reshape
    x_train = tf.reshape(x_train, [-1, 28, 28, 1])

    images = kl.Input(shape = (28, 28, 1))

    # images = kl.Input(shape = (784, ))
    # # input reshape
    # reshaped = kl.Reshape((28, 28, 1))

    # 1st convolution
    convoluted1 = kl.Conv2D(32, (5, 5), activation = 'relu')(images)
    # convoluted1 = kl.Conv2D(32, (5, 5), activation = 'relu')(reshaped)
    pooled1     = kl.MaxPooling2D((2, 2))(convoluted1)

    # 2nd convolution
    convoluted2 = kl.Conv2D(64, (5, 5), activation = 'relu')(pooled1)
    pooled2     = kl.MaxPooling2D((2, 2))(convoluted2)

    # 1st fully-connected
    flattened    = kl.Flatten()(pooled2)
    fced1        = kl.Dense(1024, activation = 'relu')(flattened)

    # 2nd fully-connected
    predictions  = kl.Dense(10, activation = 'softmax')(fced1)

    model = Model(inputs = images, outputs = predictions)
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy')
    model.fit(x_train, y_train, steps_per_epoch = 10)

    print 'Done'


if __name__ == '__main__':
    mnist()
