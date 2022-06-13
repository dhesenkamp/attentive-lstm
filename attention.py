import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class SimpleAttention(Model):
    """
    Simple attention mechanism, based on https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e
    """

    def __init__(self):
        """Constructor"""

        super(SimpleAttention).__init__()

        self.dense = Dense(units=1, activation='tanh')
    
    
    def call(self, input):
        """
        Feed input through dense layer + softmax to obtain attention-adjusted weights.
        Input comes directly from LSTM and is of shape (?, seq len, hidden units), output is 
        of shape (?, seq len, 1).

        Args:
            input (tensor): output from previous LSTM layer, i.e. embedding of the input sequence
        Returns:
            a (tensor): attention-adjusted weights
            output_sum (float): sum over inputs, weighted by attention
        """

        e = self.dense(input)
        a = tf.nn.softmax(e)
        output = a * input
        output_sum = tf.math.sum(output, axis=1)

        return a, output_sum


class SelfAttention(Model):
    """
    Self-attention mechanism as per Lin et al. (2017), adapted for Coskun et al. (2018)
    https://arxiv.org/abs/1703.03130
    """

    def __init__(self):
        super(SelfAttention, self).__init__()

        # 2 weight matrices á 200x10 and 10x1 (numbers from Coskun et al.)
        # init weight matrices with uniform dist with zero mean and 0.001 sd
        self.d_a = 10 # treat as hyperparam for weight matrix size
        self.W_s1
        self.W_s2


    def call(self, input):
        """ 
        Compute r:
            1. Transpose state sequence S
            2. Multiply by weight matrix W_s1
            3. Feed through tanh
            4. Multiply by weight matrix W_s2
            5. Normalize by feeding through softmax
        
        Compute a_i:
            6. Normalize (e.g. softmax)
            7. Negative log

        Final embedding E:
            8. Multiply scores with sequence A x S
        """
        
        # r
        state_sequence_T = tf.transpose(input)
        r = tf.nn.tanh(tf.matmul(self.W_s1, state_sequence_T))
        r = tf.matmul(r, self.W_s2)

        # a
        a = tf.math.exp(r)
        a = tf.nn.softmax(a)
        a = tf.math.negative(tf.math.log(a))

        # E = A x S
        embedding = tf.matmul(a, input)
        
        return embedding
