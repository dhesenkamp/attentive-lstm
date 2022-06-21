import tensorflow as tf
import numpy as np


def create_batch(df, num_classes=5, samples=25):
    """
    Utility function to create batches for MMD-NCA triplet loss model based on Coskun et al. (2018).
    Collects all available classes, then randomly selects a positive and num_classes negative ones. Finally draws 25 samples
    randomly from the classes and concatenates everything to a batch.

    Code heavily borrows from https://github.com/xrenaa/Human-Motion-Analysis-with-Deep-Metric-Learning/blob/master/train.py
    Default params based on Coskun et al.: https://arxiv.org/abs/1807.11176v2

    Args:
        df: dataframe from which to create dataset (json)
        num_classes (int): number of classes to consider for negative examples. Default: 5
        samples (int): number of samples per category. Default: 25
    Returns:
        batch (np.array): batch with (num_classes+2 * samples) many samples
    """

    batch = []

    # iterate through dataset and collect classes
    classes = [key for key in df]
    
    # assign classes
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
    arr = np.arange(df[pos_class].shape[0]) # get indices for samples of this class
    np.random.shuffle(arr) # shuffle
    anchor = [df[pos_class][arr[i]] for i in range(samples)] # select 25 samples
    batch.append(anchor)
    
    # positive class
    arr = np.arange(df[pos_class].shape[0])
    np.random.shuffle(arr)
    pos = [df[pos_class][arr[i]] for i in range(samples)]
    batch.append(pos)

    # negative class 1    
    arr = np.arange(df[neg_class_1].shape[0])
    np.random.shuffle(arr)
    neg_1 = [df[neg_class_1][arr[i]] for i in range(samples)]
    batch.append(neg_1)

    # negative class 2
    arr = np.arange(df[neg_class_2].shape[0])
    np.random.shuffle(arr)
    neg_2 = [df[neg_class_2][arr[i]] for i in range(samples)]
    batch.append(neg_2)

    # negative class 3
    arr = np.arange(df[neg_class_3].shape[0])
    np.random.shuffle(arr)
    neg_3 = [df[neg_class_3][arr[i]] for i in range(samples)]
    batch.append(neg_3)
    
    # negative class 4
    arr = np.arange(df[neg_class_4].shape[0])
    np.random.shuffle(arr)
    neg_4 = [df[neg_class_4][arr[i]] for i in range(samples)]
    batch.append(neg_4)
    
    # negative class 5
    arr = np.arange(df[neg_class_5].shape[0])
    np.random.shuffle(arr)
    neg_5 = [df[neg_class_5][arr[i]] for i in range(samples)]
    batch.append(neg_5)
    
    # batch now has shape [nr_classes, nr_samples, feat_vector]
    # feat_vector itself should have shape [time_steps, nr_joints, coordinates]
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