import tensorflow as tf
from tensorflow.keras import Model
from tf.keras.layers import Dense, LSTM, Bidirectional, BatchNormalization, Dropout
from tensorflow_addons.rnn import LayerNormLSTMCell
from attention import SimpleAttention, SelfAttention


class AttentiveLSTM(Model):
    """Self-attentive LSTM based on Coskun et al. (2018)"""


    def __init__(self, hidden_units=128):
        super(AttentiveLSTM, self).__init__()

        self.attention_layer = SelfAttention()
        self.all_layers = [
            Bidirectional(LSTM(units=hidden_units, return_sequences=True)), #output of shape seq_len x hidden_units*2(bidir)
            # possibly use this instead
            #LayerNormLSTMCell(units=hidden_units)

            BatchNormalization(),
            Dropout(),
            
            #self-attention
            self.attention_layer(),

            BatchNormalization(),
            Dropout(),
            Dense(units=320, activation='relu'),

            BatchNormalization(),
            Dropout(),
            Dense(units=320, activation='relu'),

            BatchNormalization(),
            Dense(units=128),

            #l2 norm 
        ]


    def call(self, input):
        pass
