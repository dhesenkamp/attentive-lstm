import tensorflow as tf


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
    **Needs adaptation to actual dataset/training function**
    
    MMD-NCA loss based on Coskun et al. (2018)

    Args:
        input: model output for 7 classes of motion sequences
    Returns:
        mmd_nca: MMD-NCA loss
    """
    x_anchor, x_pos, x_neg1, x_neg2, x_neg3, x_neg4, x_neg5 = input

    nominator = tf.math.exp(- mmd(x_anchor, x_pos))
    denom_1 = tf.math.exp(- mmd(x_anchor, x_neg1))
    denom_2 = tf.math.exp(- mmd(x_anchor, x_neg2))
    denom_3 = tf.math.exp(- mmd(x_anchor, x_neg3))
    denom_4 = tf.math.exp(- mmd(x_anchor, x_neg4))
    denom_5 = tf.math.exp(- mmd(x_anchor, x_neg5))
    
    denominator = denom_1 + denom_2 + denom_3 + denom_4 + denom_5
    mmd_nca = nominator / denominator
    
    return mmd_nca