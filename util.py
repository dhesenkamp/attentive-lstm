import tensorflow as tf
import numpy as np
import os
import re
import cdflib


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
        samples (int): number of samples per category. Ultimately corresponds to the batch size, as this
            is how many samples will be fed to the model per epoch. Default: 25
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
    arr = np.arange(len(df[pos_class])) # get indices for samples of this class
    np.random.shuffle(arr) # shuffle
    anchor = [df[pos_class][arr[i]] for i in range(samples)] # select 25 samples
    batch.append(anchor)
    
    # positive class
    arr = np.arange(len(df[pos_class]))
    np.random.shuffle(arr)
    pos = [df[pos_class][arr[i]] for i in range(samples)]
    batch.append(pos)

    # negative class 1    
    arr = np.arange(len(df[neg_class_1]))
    np.random.shuffle(arr)
    neg_1 = [df[neg_class_1][arr[i]] for i in range(samples)]
    batch.append(neg_1)

    # negative class 2
    arr = np.arange(len(df[neg_class_2]))
    np.random.shuffle(arr)
    neg_2 = [df[neg_class_2][arr[i]] for i in range(samples)]
    batch.append(neg_2)

    # negative class 3
    arr = np.arange(len(df[neg_class_3]))
    np.random.shuffle(arr)
    neg_3 = [df[neg_class_3][arr[i]] for i in range(samples)]
    batch.append(neg_3)
    
    # negative class 4
    arr = np.arange(len(df[neg_class_4]))
    np.random.shuffle(arr)
    neg_4 = [df[neg_class_4][arr[i]] for i in range(samples)]
    batch.append(neg_4)
    
    # negative class 5
    arr = np.arange(len(df[neg_class_5]))
    np.random.shuffle(arr)
    neg_5 = [df[neg_class_5][arr[i]] for i in range(samples)]
    batch.append(neg_5)
    
    # batch now has shape [nr_classes, nr_samples, feat_vector]
    # feat_vector itself should have shape [time_steps, nr_joints * coordinates]
    return tf.convert_to_tensor(value=batch, dtype=tf.float32)


def get_paths(path, file_format):
    """Collect all paths to given file format.
    
    Args:
        path (str):
        file_format (str):
    Returns:
        paths
    """
    paths = []
    for root, dirs, files in os.walk(path):
        for f in files:
            #print(os.path.relpath(os.path.join(root, f), "."))
            if file_format in f:
                paths.append(os.path.relpath(os.path.join(root, f), "."))
    return paths


def collect_data(path='.', file_format='.cdf'):
    
    coordinates = []
    labels = []
    paths = get_paths(path=path, file_format=file_format)

    for p in paths:
        # get 2D data
        cdf_file = cdflib.CDF(p)
        coord = cdf_file.varget('Pose').squeeze()
        coordinates.append(coord)

        # get motion name
        filename = os.path.basename(p)
        motion = re.split('[ .\d+]', filename)[0]
        labels.append(motion)
    
    return coordinates, labels


def downsample(data, factor):
    """Downsample sequences by specified factor.

    Example: sequence with 150 frames/steps. Downsampling by factor 3 means only keeping every 3rd frame, 
    resulting in a sequence with 50 frames.

    Args:
        data (array): 
        factor (int): factor by which to downsample the sequences
    Returns:
        data (array): downsampled sequences
    """
    for i in range(len(data)):
        data[i] = data[i][::factor]
    
    return data


def cut_sequences(data, length):
    """Cut given sequences down to specified length.
    
    Args:
        data
        length (int): 
    Returns:
        data
    Raises:
        ValueError
    """
    min_length = min([len(elem) for elem in data])
    if length > min_length:
        raise ValueError('Var `length` has to be <= length of the shortest element in `data`.')

    data = [elem[:length] for elem in data]

    return data


def prepare_data(data, labels, batch_size=64, normalize=True, one_hotify=True, add_noise=False, **kwargs):
    """Prepare motion data.

    Preprocesses motion sequences and optionally augments data, increasing dataset size two-fold.

    Args:
        data:
        labels:
        batch_size (int): Size of mini-batches. For mini-batching to have a positive impact on training behaviour,
            batch_size should be at least 16, but preferably in the range of 64-128. Default: 64
        normalize (Bool):
        one_hotify (Bool):
        add_noise (Bool):

    Keyword Args:
        noise_factor (float): standard deviation to use for noise, which is sampled from a normal distribution with mean 0. Default: 2.0
    
    Returns:
        ds (tf.data.Dataset): TensorFlow dataset instance
    """
    norm = tf.keras.layers.Normalization()
    norm.adapt(np.asarray(data))

    ds = tf.data.Dataset.from_tensor_slices((data, int_labels))

    if one_hotify:
        ds = ds.map(lambda x,y: (x, tf.one_hot(y, depth=NR_CLASSES)))
    if add_noise:
        ds_noise = ds.map(lambda x,y: (x+tf.random.normal(shape=x.shape, mean=0, stddev=kwargs['noise_factor']),y))
        ds.concatenate(ds_noise)
    if normalize:
        ds = ds.map(lambda x,y: (tf.squeeze(norm(x)), y))

    ds = ds.shuffle(512).batch(batch_size)#, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return ds