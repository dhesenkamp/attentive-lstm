import tensorflow as tf
from tensorflow.keras import Model

class SelfAttention(Model):
    """
    Self-attention mechanism as per Lin et al. (2017), adapted for Coskun et al. (2018)
    https://arxiv.org/abs/1703.03130
    """

    def __init__(self):
        super(SelfAttention, self).__init__()

        # 2 weight matrices á 200x10 and 10x1 (numbers from Coskun et al.)
        # init weight matrices with uniform dist with zero mean and 0.001 sd
        self.d_a = 10 # arbitrary hyper param to set weight matrix size
        self.W_s1
        self.W_s2

        self.tanh = tf.nn.tanh()
        self.softmax = tf.nn.softmax()


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
        r = self.tanh(tf.matmul(self.W_s1, state_sequence_T))
        r = tf.matmul(r, self.W_s2)

        # a_i
        # E
        
        return 
