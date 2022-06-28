import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense


class SimpleAttention(Layer):
    """
    Simple attention mechanism, based on https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e
    """

    def __init__(self):
        """Constructor"""

        super(SimpleAttention, self).__init__()
        self.dense = Dense(units=1, activation='tanh')
    
    
    def call(self, input):
        """
        Feed input through dense layer + softmax to obtain attention-adjusted weights.
        Input comes directly from LSTM and is of shape (batch size, seq len, hidden units), output is 
        of shape (batch size, seq len, 1).

        Args:
            input (tensor): output from previous LSTM layer, i.e. embedding of the input sequence
        Returns:
            a (tensor): attention-adjusted weights
            output (float): sum over inputs, weighted by attention
        """

        e = self.dense(input)
        a = tf.nn.softmax(e)
        attn_adj_input = a * input
        output = tf.math.reduce_sum(attn_adj_input, axis=1)

        return a, output


class SelfAttention(Layer):
    """
    Self-attention mechanism as per Lin et al. (2017), adapted for Coskun et al. (2018)
    https://arxiv.org/abs/1703.03130
    """

    def __init__(self, r, lstm_units):
        """
        Constructor. Initialize weight matrices for attention.
        
        Args:
            r (int): number of time steps to pay attention to
            lstm_units (int): number of hidden units of the preceeding LSTM. Required to get weight matrix in the correct shape
        """
        
        super(SelfAttention, self).__init__()

        # 2 weight matrices á 200x10 and 10x1 (numbers from Coskun et al.)
        # init weight matrices with uniform dist with zero mean and 0.001 sd
        self.d_a = 10 # treat as hyperparam for weight matrix size
        self.W_s1 = self.add_weight(shape=[self.d_a, lstm_units*2], trainable=True)
        self.W_s2 = self.add_weight(shape=[r, self.d_a], trainable=True)


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
            8. Multiply scores A with sequence S
        
        Args:
            input: embedding from the LSTM
        Returns:
            embedding: final, attention-adjusted embedding 
        """
        
        # r
        r = tf.nn.tanh(self.W_s1 @ tf.transpose(input, perm=[0, 2, 1]))
        r = self.W_s2 @ r

        # a_i = -log(exp r_i / sum exp r)
        #a = tf.math.exp(r)
        a = tf.nn.softmax(r)
        a = tf.math.negative(tf.math.log(a))

        # E = A x S
        embedding = tf.matmul(a, input)
        
        return embedding