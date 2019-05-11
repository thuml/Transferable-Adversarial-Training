import numpy as np
def shuffle(X, Y):
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffle = X[permutation]
    Y_shuffle = Y[permutation]
    shuffles = {"X_shuffle": X_shuffle, "Y_shuffle": Y_shuffle}
    return shuffles

def get_mini_batches(X, Y, mini_batch_size):
    shuffles = shuffle(X, Y)
    num_examples = shuffles["X_shuffle"].shape[0]
    num_complete =  num_examples // mini_batch_size
    mini_batches = []
    for i in range(num_complete):
        mini_batches.append([shuffles["X_shuffle"][i * mini_batch_size:(i + 1) * mini_batch_size], shuffles["Y_shuffle"][i * mini_batch_size:(i + 1) * mini_batch_size]])

    if 0 == num_examples % mini_batch_size:
        pass
    else:
        mini_batches.append([shuffles["X_shuffle"][num_complete * mini_batch_size:], shuffles["Y_shuffle"][num_complete * mini_batch_size:]])
    return mini_batches