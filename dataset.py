import h5py

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


