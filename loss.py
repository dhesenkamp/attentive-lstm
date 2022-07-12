import tensorflow as tf


def kernel_function(x, x_prime, sigma=[1,2,4,8,16]):
    """
    Gaussian kernel function.
    
    Args:
        x, x_prime: two IID samples from a distribution
        sigma (list): list with kernel parameters to compute from
    Returns:
        k_sum (float): sum over all kernels
    """
    k = [tf.math.exp(-tf.math.pow((x - x_prime), 2) / 2* s**2) for s in sigma]
    k_sum = tf.math.reduce_sum(k)
    
    return k_sum


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
    return mmd


def mmd_nca_loss(input):
    """
    // Needs adaptation to actual dataset/training function //
    
    MMD-NCA loss based on Coskun et al. (2018).
    nominator is the MMD between anchor and positive sample, denominator is the sum over MMDs between anchor and all negative samples.

    Args:
        input (list): list of sequences/embeddings to calculate MMD-NCA from. Has to be of the following order:
            input[0] = anchor embedding
            input[1] = positive embedding
            input[2:] = variable number of negative embeddings
    Returns:
        mmd_nca: MMD-NCA loss
    """
    if type(input) != list:
        raise TypeError('Var `input` has to be of type list.')
    if len(input) < 3:
        raise ValueError('Var `input` has to contain at least 3 embeddings.')

    anchor = input[0]
    mmd_list = [tf.math.exp(- mmd(anchor, i)) for i in input[1:]]
    
    nominator = mmd_list[0]
    denominator = sum(mmd_list[1:])

    mmd_nca = nominator / denominator
    
    return mmd_nca