import tensorflow as tf
from tensorflow.keras import Model
from tf.keras.layers import Dense, LSTM, Bidirectional, BatchNorm, Dropout
from tensorflow_addons.rnn import LayerNormLSTMCell
from attention import SelfAttention


class AttentiveLSTM(Model):


    def __init__(self, hidden_units=128):
        super(AttentiveLSTM, self).__init__()

        self.attention_layer = SelfAttention()
        self.all_layers = [
            Bidirectional(LSTM(units=hidden_units)),
            # possibly use this instead
            #LayerNormLSTMCell(units=hidden_units)

            BatchNorm(),
            Dropout(),
            
            #self-attention
            self.attention_layer(),

            BatchNorm(),
            Dropout(),
            Dense(units=320, activation='relu'),

            BatchNorm(),
            Dropout(),
            Dense(units=320, activation='relu'),

            BatchNorm(),
            Dense(units=128),

            #l2 norm 
        ]


    def call(self, input):
        pass
