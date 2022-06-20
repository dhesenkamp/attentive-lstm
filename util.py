import tensorflow as tf
import numpy as np


def create_batch(df, num_classes=5, samples=25):
    """
    Utility function to create batches for MMD-NCA triplet loss model based on Coskun et al. (2018).
    Code heavily borrows from https://github.com/xrenaa/Human-Motion-Analysis-with-Deep-Metric-Learning/blob/master/train.py
    Default params based on Coskun et al.

    Args:
        df: dataframe from which to create dataset (json)
        num_classes (int): number of classes to consider for negative examples. Default: 5
        samples (int): number of samples per category. Default: 25
    Returns:
        batch (np.array): batch with (num_classes+2 * samples) many samples
    """

    batch = []
    classes = []

    # iterate through dataset and collect classes
    for key in df:
        classes.append(key)
    
    for _ in range(num_classes+1):
        # assign positive and negative classes from entirety of classes
        pos_class = np.random.choice(classes)
        neg_class_1 = np.random.choice(classes)
        neg_class_2 = np.random.choice(classes)
        neg_class_3 = np.random.choice(classes)
        neg_class_4 = np.random.choice(classes)
        neg_class_5 = np.random.choice(classes)

    # reassign if class has less than specified nr of samples or same class is chosen multiple times
    while len(df[pos_class]) < 2:
        pos_class = np.random.choice(classes)
    while (neg_class_1 == pos_class or len(df[neg_class_1]) < 2):
        neg_class_1 = np.random.choice(classes)
    while ((neg_class_2 == pos_class) or (neg_class_2 == neg_class_1) or (len(df[neg_class_2]) < 2)):
        neg_class_2 = np.random.choice(classes)
    while ((neg_class_3 == pos_class) or (neg_class_3 == neg_class_1) or (neg_class_3 == neg_class_2) or (len(df[neg_class_3]) < 2)):
        neg_class_3 = np.random.choice(classes)
    while ((neg_class_4 == pos_class) or (neg_class_4 == neg_class_1) or (neg_class_4 == neg_class_2) or (neg_class_4 == neg_class_3) or (len(df[neg_class_4]) < 2)):
        neg_class_4 = np.random.choice(classes)
    while ((neg_class_5 == pos_class) or (neg_class_5 == neg_class_1) or (neg_class_5 == neg_class_2) or (neg_class_5 == neg_class_3) or (neg_class_5 == neg_class_4) or (len(df[neg_class_5]) < 2)):
        neg_class_5 = np.random.choice(classes)
    
    # create array of 25 random samples, repeat for all selected classes
    # anchor class
    arr = np.arange(df[pos_class].shape[0])
    np.random.shuffle(arr)
    for i in range(samples):
        if i == 0:
            _batch = df[pos_class][arr[i]]
        else:
            _batch = np.concatenate((_batch, df[pos_class][arr[i]]), axis = 0)
    
    # positive class
    arr = np.arange(df[pos_class].shape[0])
    np.random.shuffle(arr)
    for i in range(25):
        _batch = np.concatenate((_batch, df[pos_class][arr[i]]), axis = 0)

    # negative class 1    
    arr = np.arange(df[neg_class_1].shape[0])
    np.random.shuffle(arr)
    for i in range(25):
        _batch = np.concatenate((_batch, df[neg_class_1][arr[i]]), axis = 0)

    # negative class 2
    arr = np.arange(df[neg_class_2].shape[0])
    np.random.shuffle(arr)
    for i in range(25):
        _batch = np.concatenate((_batch, df[neg_class_2][arr[i]]), axis = 0)

    # negative class 3
    arr = np.arange(df[neg_class_3].shape[0])
    np.random.shuffle(arr)
    for i in range(25):
        _batch = np.concatenate((_batch, df[neg_class_3][arr[i]]), axis = 0)
    
    # negative class 4
    arr = np.arange(df[neg_class_4].shape[0])
    np.random.shuffle(arr)
    for i in range(25):
        _batch = np.concatenate((_batch, df[neg_class_4][arr[i]]), axis = 0)
    
    # negative class 5
    arr = np.arange(df[neg_class_5].shape[0])
    np.random.shuffle(arr)
    for i in range(25):
        _batch = np.concatenate((_batch, df[neg_class_5][arr[i]]), axis = 0)
    
    # make tf dataset?
    batch.append(_batch)
        
    return batch


def kernel_function(x, x_prime, sigma=[1,2,4,8,16]):
    """
    Gaussian kernel function
    
    Args:
        x, x_prime: two IID samples from a distribution
        sigma (list): list with kernel parameters to compute from
    Returns:
        k_sum (float): sum over all kernels
    """
    k = []
    for s in sigma:
        k.append(tf.math.exp(-tf.math.pow((x - x_prime), 2) / 2* s**2))
    
    k_sum = tf.math.reduce_sum(k)
    
    return k_sum.numpy()


def mmd(X, Y):
    """
    Calculate maximum mean discrepancy between distributions X and Y.
    See Coskun et al. (2018), chapter 4 'Loss Function' for details, esp. formula 7.

    Args:
        X (array): samples from distribution X 
        Y (array): samples from distribution Y
    Returns:
        mmd (float) = maximum mean discrepancy
    """
    first_term = tf.math.divide(1, len(X)**2) * [kernel_function(i, j) for i in X for j in X]
    second_term = tf.math.divide(2, len(X)*len(Y)) * [kernel_function(i, j) for i in X for j in Y]
    third_term = tf.math.divide(1, len(Y)**2) * [kernel_function(i, j) for i in Y for j in Y]

    mmd = tf.math.reduce_sum(first_term) - tf.math.reduce_sum(second_term) + tf.math.reduce_sum(third_term)
    return mmd.numpy()